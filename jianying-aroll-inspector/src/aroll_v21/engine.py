from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from aroll_final_repeat_gate import build_final_repeat_gate_report
from aroll_text_normalize import normalize_text
from aroll_v21.compiler import FinalTimelineCompiler
from aroll_v21.decision import (
    DeepSeekSemanticPlanner,
    SemanticAdjudicationDecision,
    SemanticAdjudicationDecisionType,
    SemanticAdjudicationProvider,
    SemanticDecisionPlanner,
)
from aroll_v21.decision.deepseek_semantic_planner import (
    BATCH_METADATA_FIELDS,
    SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
    SEMANTIC_BATCH_PROVIDER_FAILED_CODE,
    SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE,
)
from aroll_v21.decision.final_target_repeat_resolver import FinalTargetRepeatResolver
from aroll_v21.decision.semantic_adjudication import request_from_final_target_payload
from aroll_v21.decision.semantic_contracts import (
    SemanticAdjudicationMode,
    SemanticAdjudicationRequest,
    SemanticAdjudicationResult,
    SemanticIssueSeverity,
    SemanticIssueType,
    SemanticRoutingDecision,
    semantic_contract_to_dict,
)
from aroll_v21 import engine_validation as engine_validation_helpers
from aroll_v21.engine_artifacts import write_run_artifacts
from aroll_v21.engine_report_compaction import _compact_runtime_report_payload, _resolved_semantic_decision_rows
from aroll_v21.engine_summary import build_run_summary, _normalize_effective_speed_prewrite_placeholder
from aroll_v21.evidence import CandidateEvidenceBuilder
from aroll_v21.ingest import DraftIngest
from aroll_v21.ir.models import Blocker, BlockerReport, RunReport
from aroll_v21.quality import VisualPacingNormalizer
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
from aroll_v21.quality.pipeline import QualityPipeline, QualityPipelineHooks
from aroll_v21.quality.quality_audit import build_quality_snapshot, build_timeline_mutation
from aroll_v21.quality.quality_gate import build_quality_gate_report
from aroll_v21.quality.repeat_span_repair import self_repair_aborted_phrase_candidate
from aroll_v21.render import SubtitleRenderer
from aroll_v21.validate import ReadOnlyValidators
from aroll_v21.writer import CaptionMaterialWriter


FINAL_TARGET_PROVIDER_FAILURE_CODE = SEMANTIC_BATCH_PROVIDER_FAILED_CODE
AUTO_PROVIDER_ROUTING_SKIPPED_CODE = "V21_AUTO_PROVIDER_ROUTING_SKIPPED_REQUIRED_REQUEST"
FINAL_TARGET_PROVIDER_BLOCKER_CODES = {
    "FINAL_TARGET_REPEAT_SEMANTIC_DECISION_REQUIRED",
    "FINAL_TARGET_REPEAT_REQUIRES_HUMAN_REVIEW",
    "FINAL_TARGET_REPEAT_HIGH_FATAL_KEEP_ALL_REJECTED",
    FINAL_TARGET_PROVIDER_FAILURE_CODE,
    AUTO_PROVIDER_ROUTING_SKIPPED_CODE,
    "SEMANTIC_DECISION_NOT_PROVIDED",
    SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
    SEMANTIC_BATCH_PROVIDER_FAILED_CODE,
    SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE,
    "DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS",
    "SEMANTIC_DECISION_SCHEMA_INVALID",
}
FORBIDDEN_SEMANTIC_PROVIDER_FIELDS = {
    "source_start_us",
    "source_end_us",
    "target_start_us",
    "target_end_us",
    "material_id",
    "source_material_id",
    "source_segment_id",
    "segment_id",
    "final_timeline",
    "final_edl",
    "edl",
    "draft_content",
}


@dataclass(frozen=True)
class ArollRunInput:
    draft_data: dict[str, Any] = field(default_factory=dict)
    word_timeline: list[dict[str, Any]] = field(default_factory=list)
    subtitles: list[dict[str, Any]] = field(default_factory=list)
    source_segments: list[dict[str, Any]] | None = None
    source_materials: list[dict[str, Any]] | None = None
    text_materials: list[dict[str, Any]] | None = None
    text_segments: list[dict[str, Any]] | None = None
    postwrite_materials: list[dict[str, Any]] | None = None
    ingest_blockers: list[Blocker] = field(default_factory=list)
    ingest_metadata: dict[str, Any] = field(default_factory=dict)
    postwrite_mode: Literal[
        "auto",
        "simulated",
        "simulated_write",
        "actual_decrypt",
        "unavailable",
        "skipped_for_sacrificial_draft",
    ] = "auto"
    mode: Literal["dry-run", "write", "verify-only"] = "dry-run"


@dataclass(frozen=True)
class _IngestStageResult:
    source_graph: Any
    blockers: list[Blocker]
    blocked_report: RunReport | None = None


@dataclass(frozen=True)
class _DecisionStageResult:
    repeat_clusters: Any
    decision_plan: Any
    blocked_report: RunReport | None = None


@dataclass(frozen=True)
class _CompileStageResult:
    final_timeline: list[Any]
    blocked_report: RunReport | None = None


@dataclass(frozen=True)
class _QualityStageResult:
    final_timeline: list[Any]
    captions: list[Any]
    visual_pacing_report: dict[str, Any]
    final_visible_repair_report: dict[str, Any]
    quality_mutations: list[dict[str, Any]]


@dataclass(frozen=True)
class _WriterStageResult:
    material_write_plan: dict[str, Any]
    blocked_report: RunReport | None = None


@dataclass(frozen=True)
class _ValidationStageResult:
    validator_report: dict[str, Any]
    validator_blockers: list[Blocker]


class ArollEngine:
    def __init__(
        self,
        *,
        ingest: DraftIngest | None = None,
        evidence_builder: CandidateEvidenceBuilder | None = None,
        deepseek_planner: DeepSeekSemanticPlanner | None = None,
        semantic_provider: SemanticAdjudicationProvider | None = None,
        semantic_mode: str = "auto",
        compiler: FinalTimelineCompiler | None = None,
        renderer: SubtitleRenderer | None = None,
        writer: CaptionMaterialWriter | None = None,
        validators: ReadOnlyValidators | None = None,
        visual_pacing: VisualPacingNormalizer | None = None,
    ) -> None:
        self.ingest = ingest or DraftIngest()
        self.evidence_builder = evidence_builder or CandidateEvidenceBuilder()
        self.decision_planner = SemanticDecisionPlanner(
            deepseek_planner=deepseek_planner,
            semantic_provider=semantic_provider,
            semantic_mode=semantic_mode,
        )
        self.semantic_provider = semantic_provider
        self.deepseek_planner = deepseek_planner
        self.compiler = compiler or FinalTimelineCompiler()
        self.renderer = renderer or SubtitleRenderer()
        self.writer = writer or CaptionMaterialWriter()
        self.validators = validators or ReadOnlyValidators()
        self.visual_pacing = visual_pacing or VisualPacingNormalizer()

    def run(self, inputs: ArollRunInput) -> RunReport:
        ingest_stage = self._run_ingest_stage(inputs)
        if ingest_stage.blocked_report is not None:
            return ingest_stage.blocked_report

        source_graph = ingest_stage.source_graph
        blockers = ingest_stage.blockers
        decision_stage = self._run_decision_stage(source_graph, blockers)
        if decision_stage.blocked_report is not None:
            return decision_stage.blocked_report

        repeat_clusters = decision_stage.repeat_clusters
        decision_plan = decision_stage.decision_plan
        compile_stage = self._run_compile_stage(source_graph, repeat_clusters, decision_plan, blockers)
        if compile_stage.blocked_report is not None:
            return compile_stage.blocked_report

        quality_stage = self._run_quality_stage(
            final_timeline=compile_stage.final_timeline,
            source_graph=source_graph,
            decision_plan=decision_plan,
            blockers=blockers,
        )
        writer_stage = self._run_writer_stage(
            source_graph=source_graph,
            repeat_clusters=repeat_clusters,
            decision_plan=decision_plan,
            final_timeline=quality_stage.final_timeline,
            captions=quality_stage.captions,
            blockers=blockers,
        )
        if writer_stage.blocked_report is not None:
            return writer_stage.blocked_report

        validation_stage = self._run_validation_stage(
            inputs=inputs,
            source_graph=source_graph,
            decision_plan=decision_plan,
            final_timeline=quality_stage.final_timeline,
            captions=quality_stage.captions,
            material_write_plan=writer_stage.material_write_plan,
            visual_pacing_report=quality_stage.visual_pacing_report,
            final_visible_repair_report=quality_stage.final_visible_repair_report,
            blockers=blockers,
        )
        return self._build_final_run_report(
            inputs=inputs,
            source_graph=source_graph,
            repeat_clusters=repeat_clusters,
            decision_plan=decision_plan,
            final_timeline=quality_stage.final_timeline,
            captions=quality_stage.captions,
            material_write_plan=writer_stage.material_write_plan,
            validator_report=validation_stage.validator_report,
            validator_blockers=validation_stage.validator_blockers,
            blockers=blockers,
        )

    def _run_ingest_stage(self, inputs: ArollRunInput) -> _IngestStageResult:
        source_graph = self.ingest.build_source_graph(
            draft_data=inputs.draft_data,
            word_timeline=inputs.word_timeline,
            subtitles=inputs.subtitles,
            source_segments=inputs.source_segments,
            source_materials=inputs.source_materials,
            text_materials=inputs.text_materials,
            text_segments=inputs.text_segments,
        )
        blockers: list[Blocker] = list(inputs.ingest_blockers) + list(source_graph.invariant_report.blockers)
        fatal_ingest_blocker = any(blocker.severity == "fatal" for blocker in inputs.ingest_blockers)
        blocked_report: RunReport | None = None
        if fatal_ingest_blocker or not source_graph.invariant_report.single_source_graph_ok:
            blocked_report = self._blocked(
                source_graph=source_graph,
                blockers=blockers,
                summary={"stage": "ingest", "ingest_metadata": inputs.ingest_metadata},
            )
        return _IngestStageResult(source_graph=source_graph, blockers=blockers, blocked_report=blocked_report)

    def _run_decision_stage(self, source_graph, blockers: list[Blocker]) -> _DecisionStageResult:
        repeat_clusters = self.evidence_builder.build(source_graph)
        decision_plan = self.decision_planner.plan(repeat_clusters)
        self._harden_modifier_redundancy_semantic_requests(decision_plan)
        self._refresh_semantic_adjudication_report(decision_plan)
        blocked_report: RunReport | None = None
        if decision_plan.blocked:
            consistency_blockers = self._semantic_request_consistency_blockers(decision_plan, {})
            if consistency_blockers:
                decision_plan.blockers.extend(consistency_blockers)
            blockers.extend(decision_plan.blockers)
            blocked_report = self._blocked(
                source_graph=source_graph,
                repeat_clusters=repeat_clusters,
                decision_plan=decision_plan,
                blockers=blockers,
                summary={"stage": "decision"},
            )
        return _DecisionStageResult(
            repeat_clusters=repeat_clusters,
            decision_plan=decision_plan,
            blocked_report=blocked_report,
        )

    def _run_compile_stage(self, source_graph, repeat_clusters, decision_plan, blockers: list[Blocker]) -> _CompileStageResult:
        pre_compile_decision_blocker_count = len(decision_plan.blockers)
        final_timeline, compiler_blockers = self.compiler.compile(source_graph, decision_plan)
        if self._route_final_target_repeat_semantic_requests(decision_plan):
            final_timeline, provider_resolved_blockers = FinalTargetRepeatResolver().resolve(final_timeline, decision_plan)
            compiler_blockers.extend(provider_resolved_blockers)
        self._harden_modifier_redundancy_semantic_requests(decision_plan)
        self._block_final_modifier_keep_all(decision_plan)
        self._refresh_semantic_adjudication_report(decision_plan)
        new_decision_blockers = decision_plan.blockers[pre_compile_decision_blocker_count:]
        blockers.extend(compiler_blockers)
        blocked_report: RunReport | None = None
        if compiler_blockers or any(blocker.severity == "fatal" for blocker in new_decision_blockers):
            blockers.extend(decision_plan.blockers)
            blocked_report = self._blocked(
                source_graph=source_graph,
                repeat_clusters=repeat_clusters,
                decision_plan=decision_plan,
                final_timeline=final_timeline,
                blockers=blockers,
                summary={"stage": "compiler"},
            )
        return _CompileStageResult(final_timeline=final_timeline, blocked_report=blocked_report)

    def _run_quality_stage(
        self,
        *,
        final_timeline,
        source_graph,
        decision_plan,
        blockers: list[Blocker],
    ) -> _QualityStageResult:
        result = QualityPipeline(
            QualityPipelineHooks(
                render_captions=lambda timeline, graph: self.renderer.render(timeline, graph),
                visual_pacing_normalize=self.visual_pacing.normalize,
                repair_final_visible_caption_issues=repair_final_visible_caption_issues,
                drop_deterministic_self_repair_aborted_segments=self._drop_deterministic_self_repair_aborted_segments,
                drop_final_target_aborted_caption_restarts=self._drop_final_target_aborted_caption_restarts,
                record_quality_mutation=self._record_quality_mutation,
                accept_pending_visual_pacing_recheck=self._accept_pending_visual_pacing_recheck,
                combined_final_visible_repair_report=self._combined_final_visible_repair_report,
                reconcile_late_final_target_repeat_semantics=self._reconcile_late_final_target_repeat_semantics,
                quality_mutation_report_fields=self._quality_mutation_report_fields,
                final_timeline_state_signature=self._final_timeline_state_signature,
                final_visible_state_signature=self._final_visible_state_signature,
                sync_semantic_gate_with_final_output=self._sync_semantic_gate_with_final_output,
                refresh_semantic_adjudication_report=self._refresh_semantic_adjudication_report,
            )
        ).run(
            final_timeline=final_timeline,
            source_graph=source_graph,
            decision_plan=decision_plan,
            blockers=blockers,
        )
        return _QualityStageResult(
            final_timeline=result.final_timeline,
            captions=result.captions,
            visual_pacing_report=result.visual_pacing_report,
            final_visible_repair_report=result.final_visible_repair_report,
            quality_mutations=result.quality_mutations,
        )

    def _run_writer_stage(
        self,
        *,
        source_graph,
        repeat_clusters,
        decision_plan,
        final_timeline,
        captions,
        blockers: list[Blocker],
    ) -> _WriterStageResult:
        material_write_plan, writer_blockers = self.writer.build_write_plan(source_graph, captions)
        blockers.extend(writer_blockers)
        blocked_report: RunReport | None = None
        if writer_blockers:
            blocked_report = self._blocked(
                source_graph=source_graph,
                repeat_clusters=repeat_clusters,
                decision_plan=decision_plan,
                final_timeline=final_timeline,
                captions=captions,
                material_write_plan=material_write_plan,
                blockers=blockers,
                summary={"stage": "writer"},
            )
        return _WriterStageResult(material_write_plan=material_write_plan, blocked_report=blocked_report)

    def _run_validation_stage(
        self,
        *,
        inputs: ArollRunInput,
        source_graph,
        decision_plan,
        final_timeline,
        captions,
        material_write_plan: dict[str, Any],
        visual_pacing_report: dict[str, Any],
        final_visible_repair_report: dict[str, Any],
        blockers: list[Blocker],
    ) -> _ValidationStageResult:
        validator_report = self.validators.run(
            source_graph=source_graph,
            decision_plan=decision_plan,
            final_timeline=final_timeline,
            captions=captions,
            material_write_plan=material_write_plan,
            visual_pacing_report=visual_pacing_report,
            postwrite_materials=inputs.postwrite_materials,
            postwrite_mode=inputs.postwrite_mode,
        )
        validator_report["final_visible_caption_repair_report"] = final_visible_repair_report
        validator_report = self._attach_final_caption_visible_repeat_gate(validator_report, captions)
        final_visible_semantic_changed = self._merge_final_visible_repeat_semantic_requests(decision_plan, validator_report)
        if self._route_final_visible_repeat_semantic_requests(decision_plan):
            final_visible_semantic_changed = True
        if final_visible_semantic_changed:
            self._refresh_semantic_adjudication_report(decision_plan)
            self._refresh_validator_semantic_gate_after_request_merge(validator_report, decision_plan)
        consistency_blockers = self._semantic_request_consistency_blockers(decision_plan, validator_report)
        if consistency_blockers:
            blockers.extend(consistency_blockers)
            decision_plan.blockers.extend(consistency_blockers)
        validator_blockers: list[Blocker] = []
        if not validator_report.get("validator_report_ok"):
            validator_blockers = self._validator_blockers(validator_report)
            blockers.extend(validator_blockers)
        return _ValidationStageResult(validator_report=validator_report, validator_blockers=validator_blockers)

    def _build_final_run_report(
        self,
        *,
        inputs: ArollRunInput,
        source_graph,
        repeat_clusters,
        decision_plan,
        final_timeline,
        captions,
        material_write_plan: dict[str, Any],
        validator_report: dict[str, Any],
        validator_blockers: list[Blocker],
        blockers: list[Blocker],
    ) -> RunReport:
        blocking_blockers = [blocker for blocker in blockers if blocker.severity == "fatal" or (inputs.mode == "write" and blocker.severity == "write_blocker")]
        semantic_write_allowed = bool(decision_plan.semantic_unresolved_count == 0 and decision_plan.write_allowed)
        semantic_adjudication_report = decision_plan.semantic_adjudication_report or {}
        validator_write_allowed = bool(validator_report.get("validator_report_ok")) and not any(
            blocker.severity == "fatal" for blocker in validator_blockers
        )
        writer_fallback_count = int(material_write_plan.get("writer_fallback_count") or 0)
        ready_for_write = bool(semantic_write_allowed and validator_write_allowed and writer_fallback_count == 0 and not blocking_blockers)
        blocker_report = BlockerReport(
            blocked=bool(blocking_blockers),
            blockers=blockers,
            summary={
                "mode": inputs.mode,
                "speech_timeline_provider": str(inputs.ingest_metadata.get("speech_timeline_provider") or ""),
                "speech_timeline_granularity": str(inputs.ingest_metadata.get("speech_timeline_granularity") or ""),
                "speech_timeline_precision": str(inputs.ingest_metadata.get("speech_timeline_precision") or ""),
                "speech_timeline_can_cut_inside_caption": bool(inputs.ingest_metadata.get("speech_timeline_can_cut_inside_caption")),
                "word_timeline_count": int(inputs.ingest_metadata.get("word_timeline_count") or len(source_graph.words)),
                "single_source_graph_ok": source_graph.invariant_report.single_source_graph_ok,
                "all_final_segments_have_word_ids": all(bool(segment.word_ids) for segment in final_timeline),
                "all_captions_derived_from_final_timeline": bool(
                    validator_report.get("subtitle_coverage_validator", {}).get("all_captions_derived_from_final_timeline")
                ),
                "all_materials_from_canonical_template": bool(material_write_plan.get("canonical_caption_template_id")),
                "no_writer_fallback": bool(material_write_plan.get("no_writer_fallback")),
                "writer_fallback_count": writer_fallback_count,
                "semantic_unresolved_count": decision_plan.semantic_unresolved_count,
                "semantic_adjudication_gate_passed": bool(semantic_adjudication_report.get("semantic_adjudication_gate_passed")),
                "semantic_request_count": int(semantic_adjudication_report.get("semantic_request_count") or 0),
                "semantic_request_unresolved_count": int(semantic_adjudication_report.get("semantic_request_unresolved_count") or 0),
                "fatal_semantic_issue_count": int(semantic_adjudication_report.get("fatal_semantic_issue_count") or 0),
                "deepseek_provider_configured": bool(semantic_adjudication_report.get("deepseek_provider_configured")),
                "deepseek_provider_called_count": int(semantic_adjudication_report.get("deepseek_provider_called_count") or 0),
                "deepseek_provider_error": str(semantic_adjudication_report.get("deepseek_provider_error") or ""),
                "deepseek_batch_enabled": bool(semantic_adjudication_report.get("deepseek_batch_enabled")),
                "deepseek_batch_request_count": int(semantic_adjudication_report.get("deepseek_batch_request_count") or 0),
                "deepseek_batch_attempt_count": int(semantic_adjudication_report.get("deepseek_batch_attempt_count") or 0),
                "deepseek_batch_retry_count": int(semantic_adjudication_report.get("deepseek_batch_retry_count") or 0),
                "deepseek_batch_issue_count": int(semantic_adjudication_report.get("deepseek_batch_issue_count") or 0),
                "deepseek_batch_resolved_count": int(semantic_adjudication_report.get("deepseek_batch_resolved_count") or 0),
                "deepseek_batch_unresolved_count": int(semantic_adjudication_report.get("deepseek_batch_unresolved_count") or 0),
                "deepseek_batch_missing_issue_ids": list(semantic_adjudication_report.get("deepseek_batch_missing_issue_ids") or []),
                "deepseek_batch_error": str(semantic_adjudication_report.get("deepseek_batch_error") or ""),
                "commit_reused_semantic_cache": bool(semantic_adjudication_report.get("commit_reused_semantic_cache")),
                "semantic_cache_input_hash": str(semantic_adjudication_report.get("semantic_cache_input_hash") or ""),
                "semantic_cache_issue_count": int(semantic_adjudication_report.get("semantic_cache_issue_count") or 0),
                "semantic_cache_resolved_count": int(semantic_adjudication_report.get("semantic_cache_resolved_count") or 0),
                "semantic_cache_unresolved_count": int(semantic_adjudication_report.get("semantic_cache_unresolved_count") or 0),
                "deepseek_provider_skipped_count": int(semantic_adjudication_report.get("deepseek_provider_skipped_count") or 0),
                "deepseek_provider_skipped_reasons": dict(semantic_adjudication_report.get("deepseek_provider_skipped_reasons") or {}),
                "semantic_decision_cache_used": bool(semantic_adjudication_report.get("semantic_decision_cache_used")),
                "semantic_auto_route_count": int(semantic_adjudication_report.get("semantic_auto_route_count") or 0),
                "semantic_local_decision_count": int(semantic_adjudication_report.get("semantic_local_decision_count") or 0),
                "semantic_provider_required_count": int(semantic_adjudication_report.get("semantic_provider_required_count") or 0),
                "deterministic_baseline_refused_count": int(semantic_adjudication_report.get("deterministic_baseline_refused_count") or 0),
                "requires_human_review": decision_plan.requires_human_review,
                "semantic_write_allowed": semantic_write_allowed,
                "validator_write_allowed": validator_write_allowed,
                "write_allowed": ready_for_write,
                "ready_for_write": ready_for_write,
                "READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT": ready_for_write,
                "dry_run_continued_for_discovery": bool(
                    inputs.mode == "dry-run" and decision_plan.dry_run_continued_for_discovery and final_timeline
                ),
            },
        )
        return RunReport(
            status="blocked" if blocking_blockers else "ok",
            source_graph=source_graph,
            repeat_clusters=repeat_clusters,
            decision_plan=decision_plan,
            final_timeline=final_timeline,
            captions=captions,
            material_write_plan=material_write_plan,
            validator_report=validator_report,
            postwrite_report=validator_report.get("postwrite_material_validator") or {},
            blocker_report=blocker_report,
            decision_trace=decision_plan.decision_trace,
        )

    def _record_quality_mutation(
        self,
        quality_mutations: list[dict[str, Any]],
        *,
        phase: str,
        rule_name: str,
        before_timeline,
        before_captions,
        after_timeline,
        after_captions,
        source_graph,
        action: dict[str, Any] | None = None,
        before_visual_pacing_report: dict[str, Any] | None = None,
        after_visual_pacing_report: dict[str, Any] | None = None,
        enforce_regression_guard: bool = True,
    ) -> dict[str, Any] | None:
        if self._final_visible_state_signature(before_timeline, before_captions) == self._final_visible_state_signature(
            after_timeline,
            after_captions,
        ):
            no_mutation: dict[str, Any] | None = None
            return no_mutation
        before = build_quality_snapshot(
            final_timeline=list(before_timeline),
            captions=list(before_captions),
            source_graph=source_graph,
            visual_pacing_report=before_visual_pacing_report,
        )
        after = build_quality_snapshot(
            final_timeline=list(after_timeline),
            captions=list(after_captions),
            source_graph=source_graph,
            visual_pacing_report=after_visual_pacing_report,
        )
        mutation = build_timeline_mutation(
            phase=phase,
            rule_name=rule_name,
            before=before,
            after=after,
            action=action,
        ).to_report()
        if not enforce_regression_guard and not bool(mutation.get("accepted")):
            mutation["accepted"] = True
            mutation["audit_only"] = True
            mutation["audit_only_rejection_reason"] = str(mutation.get("rejection_reason") or "")
            mutation["rejection_reason"] = ""
        quality_mutations.append(mutation)
        return mutation

    def _accept_pending_visual_pacing_recheck(self, mutation: dict[str, Any] | None) -> None:
        if mutation is None or bool(mutation.get("accepted")):
            return
        if str(mutation.get("rejection_reason") or "") not in {
            "blocking_short_segment_count_increased",
            "visual_blocker_introduced",
        }:
            return
        before = mutation.get("before") if isinstance(mutation.get("before"), dict) else {}
        after = mutation.get("after") if isinstance(mutation.get("after"), dict) else {}
        if int(after.get("final_visible_fatal_count") or 0) > int(before.get("final_visible_fatal_count") or 0):
            return
        before_alignment = set(before.get("caption_alignment_blocker_codes") or [])
        after_alignment = set(after.get("caption_alignment_blocker_codes") or [])
        introduced_alignment = after_alignment - before_alignment
        after_visual = set(after.get("visual_blocker_codes") or [])
        pending_short_caption_alignment = introduced_alignment <= {"V21_CAPTION_TOO_SHORT"} and (
            "V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN" in after_visual
            or int(after.get("blocking_short_segment_count") or 0) > int(before.get("blocking_short_segment_count") or 0)
        )
        if introduced_alignment and not pending_short_caption_alignment:
            return
        mutation["accepted"] = True
        mutation["audit_only"] = True
        mutation["pending_visual_pacing_recheck"] = True
        mutation["audit_only_rejection_reason"] = str(mutation.get("rejection_reason") or "")
        mutation["rejection_reason"] = ""

    def _accept_pending_final_visible_repair(self, mutation: dict[str, Any] | None) -> None:
        if mutation is None or bool(mutation.get("accepted")):
            return
        if str(mutation.get("rejection_reason") or "") != "final_visible_fatal_count_increased":
            return
        before = mutation.get("before") if isinstance(mutation.get("before"), dict) else {}
        after = mutation.get("after") if isinstance(mutation.get("after"), dict) else {}
        before_visual = set(before.get("visual_blocker_codes") or [])
        after_visual = set(after.get("visual_blocker_codes") or [])
        if after_visual - before_visual:
            return
        if int(after.get("blocking_short_segment_count") or 0) > int(before.get("blocking_short_segment_count") or 0):
            return
        before_alignment = set(before.get("caption_alignment_blocker_codes") or [])
        after_alignment = set(after.get("caption_alignment_blocker_codes") or [])
        if after_alignment - before_alignment:
            return
        mutation["accepted"] = True
        mutation["audit_only"] = True
        mutation["pending_final_visible_repair"] = True
        mutation["audit_only_rejection_reason"] = str(mutation.get("rejection_reason") or "")
        mutation["rejection_reason"] = ""

    def _combined_final_visible_repair_report(
        self,
        reports: list[dict[str, Any]],
        *,
        visual_pacing_rerun_after_final_repair_count: int,
        max_cycle_exhausted: bool,
        cycle_stop_reason: str = "",
        quality_mutations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        quality_mutations = list(quality_mutations or [])
        if not reports:
            return {
                "final_visible_repair_enabled": True,
                "final_visible_repair_attempted": False,
                "final_visible_repair_success": True,
                "final_visible_repair_action_count": 0,
                "final_visible_repair_actions": [],
                "final_visible_repair_visual_pacing_rerun_count": int(
                    visual_pacing_rerun_after_final_repair_count
                ),
                "final_visible_repair_cycle_count": 0,
                "final_visible_repair_max_cycle_exhausted": bool(max_cycle_exhausted),
                "final_visible_repair_stop_reason": str(cycle_stop_reason or ""),
                **self._quality_mutation_report_fields(quality_mutations),
            }
        combined = dict(reports[-1])
        all_actions: list[dict[str, Any]] = []
        all_unresolved: list[dict[str, Any]] = []
        for report in reports:
            all_actions.extend(
                [dict(row) for row in list(report.get("final_visible_repair_actions") or []) if isinstance(row, dict)]
            )
            all_unresolved.extend(
                [dict(row) for row in list(report.get("final_visible_repair_unresolved") or []) if isinstance(row, dict)]
            )
        combined["final_visible_repair_attempted"] = bool(
            combined.get("final_visible_repair_attempted")
            or all_actions
            or any(bool(report.get("final_visible_repair_attempted")) for report in reports)
        )
        combined["final_visible_repair_success"] = bool(combined.get("final_visible_repair_success")) and not bool(max_cycle_exhausted)
        combined["final_visible_repair_action_count"] = len(all_actions)
        combined["final_visible_repair_actions"] = all_actions
        combined["final_visible_repair_unresolved"] = all_unresolved
        combined["final_visible_repair_cycle_count"] = len(reports)
        combined["final_visible_repair_visual_pacing_rerun_count"] = int(visual_pacing_rerun_after_final_repair_count)
        combined["final_visible_repair_max_cycle_exhausted"] = bool(max_cycle_exhausted)
        combined["final_visible_repair_passes_executed_total"] = sum(
            int(report.get("final_visible_repair_passes_executed") or 0)
            for report in reports
        )
        if max_cycle_exhausted:
            stop_reason = str(cycle_stop_reason or "max_repair_cycles_exhausted")
            combined["final_visible_repair_stop_reason"] = stop_reason
            all_unresolved.append(
                {
                    "reason": stop_reason,
                    "cycle_count": len(reports),
                    "visual_pacing_rerun_count": int(visual_pacing_rerun_after_final_repair_count),
                }
            )
        elif cycle_stop_reason:
            combined["final_visible_repair_stop_reason"] = str(cycle_stop_reason)
        combined.update(self._quality_mutation_report_fields(quality_mutations))
        return combined

    def _reconcile_late_final_target_repeat_semantics(
        self,
        final_timeline,
        captions,
        source_graph,
        decision_plan,
        quality_mutations: list[dict[str, Any]],
    ):
        blockers: list[Blocker] = []
        max_passes = 3
        for pass_index in range(max_passes):
            before_signature = self._final_timeline_state_signature(final_timeline)
            before_payload_ids = {
                str(payload.get("cluster_id") or payload.get("issue_id") or "")
                for payload in decision_plan.semantic_request_payloads
                if isinstance(payload, dict)
            }
            timeline_before = list(final_timeline)
            captions_before = list(captions)
            resolver = FinalTargetRepeatResolver()
            final_timeline, resolver_blockers = resolver.resolve(final_timeline, decision_plan)
            blockers.extend(resolver_blockers)
            timeline_changed = self._final_timeline_state_signature(final_timeline) != before_signature
            if timeline_changed:
                captions = self.renderer.render(final_timeline, source_graph)
                mutation = self._record_quality_mutation(
                    quality_mutations,
                    phase="final_target_repeat.late_semantic_reconcile",
                    rule_name="FinalTargetRepeatResolver.resolve",
                    before_timeline=timeline_before,
                    before_captions=captions_before,
                    after_timeline=final_timeline,
                    after_captions=captions,
                    source_graph=source_graph,
                    action={"pass_index": pass_index + 1},
                )
                self._accept_pending_final_visible_repair(mutation)
                if mutation is not None and not bool(mutation.get("accepted")):
                    final_timeline = timeline_before
                    captions = captions_before
                    blockers.append(
                        Blocker(
                            code="V21_LATE_FINAL_TARGET_RECONCILE_REGRESSION_REVERTED",
                            message="late final-target semantic reconciliation introduced a quality regression and was reverted",
                            layer="decision",
                            severity="write_blocker",
                            context={
                                "pass_index": pass_index + 1,
                                "rejection_reason": str(mutation.get("rejection_reason") or ""),
                            },
                        )
                    )
                    break

            provider_decision_added = self._route_final_target_repeat_semantic_requests(decision_plan)
            if provider_decision_added:
                continue

            after_payload_ids = {
                str(payload.get("cluster_id") or payload.get("issue_id") or "")
                for payload in decision_plan.semantic_request_payloads
                if isinstance(payload, dict)
            }
            new_payload_ids = after_payload_ids - before_payload_ids
            if not timeline_changed or new_payload_ids:
                break
        else:
            blockers.append(
                Blocker(
                    code="V21_LATE_FINAL_TARGET_RECONCILE_MAX_PASSES_EXCEEDED",
                    message="late final-target semantic reconciliation exceeded max passes before validation",
                    layer="decision",
                    severity="write_blocker",
                    context={"max_passes": max_passes},
                )
            )
        return final_timeline, captions, blockers

    def _quality_mutation_report_fields(self, quality_mutations: list[dict[str, Any]]) -> dict[str, Any]:
        rejected = [row for row in quality_mutations if not bool(row.get("accepted"))]
        introduced_codes = sorted(
            {
                str(code)
                for row in quality_mutations
                for code in list(row.get("introduced_blocker_codes") or [])
                if str(code)
            }
        )
        cleared_codes = sorted(
            {
                str(code)
                for row in quality_mutations
                for code in list(row.get("cleared_blocker_codes") or [])
                if str(code)
            }
        )
        return {
            "quality_mutation_count": len(quality_mutations),
            "quality_mutations": quality_mutations,
            "quality_mutation_rejected_count": len(rejected),
            "quality_mutation_rejection_reasons": sorted(
                {
                    str(row.get("rejection_reason") or "")
                    for row in rejected
                    if str(row.get("rejection_reason") or "")
                }
            ),
            "quality_mutation_introduced_blocker_codes": introduced_codes,
            "quality_mutation_cleared_blocker_codes": cleared_codes,
        }

    def _final_visible_state_signature(self, final_timeline, captions) -> tuple[Any, ...]:
        segment_signature = self._final_timeline_state_signature(final_timeline)
        caption_signature = tuple(
            (
                str(caption.caption_id),
                tuple(str(segment_id) for segment_id in list(caption.timeline_segment_ids or [])),
                normalize_text(str(caption.text or "")),
                int(caption.target_start_us),
                int(caption.target_end_us),
            )
            for caption in list(captions or [])
        )
        return segment_signature, caption_signature

    def _final_timeline_state_signature(self, final_timeline) -> tuple[Any, ...]:
        return tuple(
            (
                str(segment.segment_id),
                tuple(str(word_id) for word_id in list(segment.word_ids or [])),
                normalize_text(str(segment.text or "")),
                int(segment.source_start_us),
                int(segment.source_end_us),
                int(segment.target_start_us),
                int(segment.target_end_us),
                segment.spoken_source_start_us,
                segment.spoken_source_end_us,
                segment.clip_source_start_us,
                segment.clip_source_end_us,
                int(segment.lead_handle_us or 0),
                int(segment.tail_handle_us or 0),
            )
            for segment in list(final_timeline or [])
        )

    def _drop_final_target_aborted_caption_restarts(
        self,
        final_timeline,
        captions,
        source_graph,
        decision_plan,
    ):
        caption_rows = [
            {
                "fragment_id": caption.caption_id,
                "fragment_text": caption.text,
                "text": caption.text,
                "word_ids": list(caption.word_ids),
                "target_start_us": int(caption.target_start_us),
                "target_duration_us": int(caption.target_end_us) - int(caption.target_start_us),
                "source_subtitle_uids": list(caption.source_subtitle_uids),
            }
            for caption in captions
        ]
        repeat_report = build_final_repeat_gate_report({"issues": []}, caption_rows)
        caption_by_id = {str(caption.caption_id): caption for caption in captions}
        drop_caption_ids: set[str] = set()
        trace_rows: list[dict[str, Any]] = []
        for candidate in list(repeat_report.get("final_target_repeat_candidates") or []):
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("cluster_type") or "") != "semantic_containment_take":
                continue
            aborted_takes = [
                row for row in list(candidate.get("candidates") or [])
                if isinstance(row, dict) and bool(row.get("is_aborted_start"))
            ]
            completed_takes = [
                row for row in list(candidate.get("candidates") or [])
                if isinstance(row, dict) and not bool(row.get("is_aborted_start"))
            ]
            if len(aborted_takes) != 1 or not completed_takes:
                continue
            aborted = aborted_takes[0]
            completed_start = min(int(row.get("source_start_us") or 0) for row in completed_takes)
            if int(aborted.get("source_end_us") or 0) > completed_start:
                continue
            matched_caption_ids = self._caption_ids_matching_final_target_take(candidate, aborted, caption_by_id)
            if not matched_caption_ids:
                continue
            drop_caption_ids.update(matched_caption_ids)
            cluster_id = self._normalized_final_target_cluster_id(str(candidate.get("cluster_id") or ""))
            trace_rows.append(
                {
                    "route": "final_target_repeat",
                    "cluster_id": cluster_id,
                    "decision": "drop_aborted_caption_restart",
                    "applied": True,
                    "reason": "deterministic final-target cleanup drops aborted start before completed restart",
                    "dropped_caption_ids": sorted(matched_caption_ids),
                    "dropped_segment_indices": [
                        int(item.get("subtitle_index") or 0)
                        for item in list(candidate.get("items") or [])
                        if str(item.get("subtitle_uid") or "") in matched_caption_ids
                    ],
                }
            )
        if not drop_caption_ids:
            return final_timeline
        drop_word_ids = {
            str(word_id)
            for caption_id in drop_caption_ids
            for word_id in list(caption_by_id.get(caption_id).word_ids if caption_by_id.get(caption_id) else [])
        }
        if not drop_word_ids:
            return final_timeline
        word_lookup = {word.word_id: word for word in source_graph.words}
        kept_bounds_by_segment = self._caption_bounds_by_segment(captions, drop_caption_ids)
        cleaned = []
        for segment in final_timeline:
            new_word_ids = [str(word_id) for word_id in list(segment.word_ids or []) if str(word_id) not in drop_word_ids]
            if len(new_word_ids) == len(list(segment.word_ids or [])):
                cleaned.append(segment)
                continue
            if not new_word_ids:
                continue
            words = [word_lookup[word_id] for word_id in new_word_ids if word_id in word_lookup]
            if not words:
                continue
            target_bounds = kept_bounds_by_segment.get(str(segment.segment_id))
            target_start = int(target_bounds[0]) if target_bounds else int(segment.target_start_us)
            target_end = int(target_bounds[1]) if target_bounds else int(segment.target_end_us)
            cleaned.append(
                replace(
                    segment,
                    word_ids=new_word_ids,
                    text="".join(str(word.text or "") for word in words),
                    source_start_us=min(int(word.source_start_us) for word in words),
                    source_end_us=max(int(word.source_end_us) for word in words),
                    target_start_us=target_start,
                    target_end_us=max(target_start, target_end),
                    decision_ids=sorted(set([*segment.decision_ids, "final_target_aborted_caption_restart_drop"])),
                )
            )
        decision_plan.decision_trace.extend(trace_rows)
        return self._repack_final_timeline(cleaned)

    def _caption_ids_matching_final_target_take(self, candidate: dict[str, Any], take: dict[str, Any], caption_by_id: dict[str, Any]) -> set[str]:
        take_text = normalize_text(str(take.get("text") or ""))
        take_start = int(take.get("source_start_us") or 0)
        take_end = int(take.get("source_end_us") or 0)
        matches: set[str] = set()
        for item in list(candidate.get("items") or []):
            if not isinstance(item, dict):
                continue
            caption_id = str(item.get("subtitle_uid") or "")
            caption = caption_by_id.get(caption_id)
            if caption is None:
                continue
            if normalize_text(str(item.get("text") or "")) != take_text:
                continue
            if int(item.get("start_us") or 0) != take_start or int(item.get("end_us") or 0) != take_end:
                continue
            matches.add(caption_id)
        return matches

    def _caption_bounds_by_segment(self, captions, drop_caption_ids: set[str]) -> dict[str, tuple[int, int]]:
        bounds: dict[str, tuple[int, int]] = {}
        for caption in captions:
            if str(caption.caption_id) in drop_caption_ids:
                continue
            for segment_id in list(caption.timeline_segment_ids or []):
                key = str(segment_id)
                current = bounds.get(key)
                start = int(caption.target_start_us)
                end = int(caption.target_end_us)
                if current is None:
                    bounds[key] = (start, end)
                    continue
                bounds[key] = (min(current[0], start), max(current[1], end))
        return bounds

    def _normalized_final_target_cluster_id(self, cluster_id: str) -> str:
        raw = str(cluster_id or "")
        return raw if raw.startswith("final_target_repeat_") else f"final_target_repeat_{raw}"

    def _sync_semantic_gate_with_final_output(
        self,
        decision_plan,
        final_timeline,
        captions,
    ) -> None:
        if not decision_plan.semantic_request_payloads:
            return
        final_text = normalize_text("".join(str(segment.text or "") for segment in final_timeline))
        captions_text = normalize_text("".join(str(caption.text or "") for caption in captions))
        final_target_resolver = FinalTargetRepeatResolver()
        active_final_target_pairs = {
            pair
            for pair in (final_target_resolver._cluster_text_pair(cluster) for cluster in final_target_resolver._clusters(final_timeline))
            if pair is not None
        }
        resolved_cluster_ids: set[str] = set()
        remaining_payloads: list[dict[str, Any]] = []
        for payload in decision_plan.semantic_request_payloads:
            cluster_id = str(payload.get("cluster_id") or "")
            if final_target_resolver._is_final_target_repeat_payload(payload):
                payload_pair = final_target_resolver._payload_text_pair(payload)
                if payload_pair is not None and payload_pair not in active_final_target_pairs:
                    resolved_cluster_ids.add(cluster_id)
                    decision_plan.decision_trace.append(
                        {
                            "route": "semantic_gate",
                            "cluster_id": cluster_id,
                            "decision": "resolved_by_final_target_cluster_absence",
                            "reason": "final-target request text pair no longer appears as an active final repeat cluster",
                            "requires_semantic_decision": False,
                        }
                    )
                    continue
            comparable_texts = self._semantic_payload_comparable_texts(payload)
            if comparable_texts and all(text not in final_text and text not in captions_text for text in comparable_texts):
                resolved_cluster_ids.add(cluster_id)
                decision_plan.decision_trace.append(
                    {
                        "route": "semantic_gate",
                        "cluster_id": cluster_id,
                        "decision": "resolved_by_final_timeline",
                        "reason": "cluster source text no longer appears in final_timeline/captions",
                        "requires_semantic_decision": False,
                    }
                )
                continue
            remaining_payloads.append(payload)
        if not resolved_cluster_ids:
            return

        decision_plan.semantic_request_payloads[:] = remaining_payloads
        decision_plan.blockers[:] = [
            blocker
            for blocker in decision_plan.blockers
            if not (
                blocker.severity == "write_blocker"
                and str(blocker.context.get("cluster_id") or "") in resolved_cluster_ids
                and blocker.code
                in {
                    "SEMANTIC_DECISION_NOT_PROVIDED",
                    "DEEPSEEK_SEMANTIC_PLANNER_NOT_CONFIGURED",
                    "FINAL_TARGET_REPEAT_SEMANTIC_DECISION_REQUIRED",
                    "FINAL_TARGET_REPEAT_REQUIRES_HUMAN_REVIEW",
                    "FINAL_MODIFIER_REDUNDANCY_SEMANTIC_DECISION_REQUIRED",
                    "FINAL_MODIFIER_REDUNDANCY_REQUIRES_HUMAN_REVIEW",
                    "V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
                    "V21_SEMANTIC_ADJUDICATION_PROVIDER_MISSING",
                    "V21_SEMANTIC_BATCH_PROVIDER_MISSING",
                    SEMANTIC_BATCH_PROVIDER_FAILED_CODE,
                    SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
                    SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE,
                }
            )
        ]
        if decision_plan.final_target_repeat_unresolved_cluster_ids:
            decision_plan.final_target_repeat_unresolved_cluster_ids[:] = [
                cluster_id for cluster_id in decision_plan.final_target_repeat_unresolved_cluster_ids if cluster_id not in resolved_cluster_ids
            ]
        if decision_plan.modifier_redundancy_unresolved_cluster_ids:
            decision_plan.modifier_redundancy_unresolved_cluster_ids[:] = [
                cluster_id for cluster_id in decision_plan.modifier_redundancy_unresolved_cluster_ids if cluster_id not in resolved_cluster_ids
            ]
        remaining_unresolved_ids = {str(payload.get("cluster_id") or "") for payload in decision_plan.semantic_request_payloads}
        remaining_write_blockers = [blocker for blocker in decision_plan.blockers if blocker.severity == "write_blocker"]
        fatal_blockers = [blocker for blocker in decision_plan.blockers if blocker.severity == "fatal"]
        human_review_decisions = [decision for decision in decision_plan.decisions if decision.requires_human_review]
        object.__setattr__(decision_plan, "semantic_unresolved_count", len(remaining_unresolved_ids))
        object.__setattr__(decision_plan, "requires_human_review", bool(remaining_unresolved_ids or remaining_write_blockers or human_review_decisions))
        object.__setattr__(decision_plan, "write_allowed", not remaining_unresolved_ids and not remaining_write_blockers and not fatal_blockers and not human_review_decisions)
        object.__setattr__(decision_plan, "dry_run_continued_for_discovery", bool(remaining_unresolved_ids))

    def _refresh_semantic_adjudication_report(self, decision_plan) -> None:
        report = dict(getattr(decision_plan, "semantic_adjudication_report", {}) or {})
        existing_requests = [row for row in report.get("requests") or [] if isinstance(row, dict)]
        requests_by_id = {str(row.get("issue_id") or row.get("cluster_id") or ""): dict(row) for row in existing_requests}
        for payload in decision_plan.semantic_request_payloads:
            if not isinstance(payload, dict):
                continue
            issue_id = str(payload.get("issue_id") or payload.get("cluster_id") or "")
            if not issue_id:
                continue
            if issue_id not in requests_by_id:
                requests_by_id[issue_id] = dict(payload)
        unresolved_ids = {
            str(payload.get("issue_id") or payload.get("cluster_id") or "")
            for payload in decision_plan.semantic_request_payloads
            if isinstance(payload, dict) and str(payload.get("issue_id") or payload.get("cluster_id") or "")
            and not bool(payload.get("warning_only"))
        }
        provider_required_ids = {
            str(row.get("issue_id") or "")
            for row in report.get("routing_decisions") or []
            if isinstance(row, dict) and bool(row.get("requires_provider")) and str(row.get("issue_id") or "")
        }
        provider_required_ids.update(
            str(payload.get("issue_id") or payload.get("cluster_id") or "")
            for payload in decision_plan.semantic_request_payloads
            if isinstance(payload, dict)
            and bool(payload.get("provider_required"))
            and str(payload.get("issue_id") or payload.get("cluster_id") or "")
        )
        provider_required_unresolved_ids = sorted(provider_required_ids & unresolved_ids)
        if (
            provider_required_unresolved_ids
            and bool(report.get("deepseek_provider_configured"))
            and int(report.get("deepseek_provider_called_count") or 0) == 0
        ):
            if not any(blocker.code == AUTO_PROVIDER_ROUTING_SKIPPED_CODE for blocker in decision_plan.blockers):
                decision_plan.blockers.append(
                    Blocker(
                        code=AUTO_PROVIDER_ROUTING_SKIPPED_CODE,
                        message="auto semantic provider routing skipped provider-required unresolved request",
                        layer="decision",
                        severity="write_blocker",
                        context={
                            "cluster_id": provider_required_unresolved_ids[0],
                            "unresolved_issue_ids": provider_required_unresolved_ids,
                            "deepseek_provider_configured": True,
                            "deepseek_provider_called_count": 0,
                        },
                    )
                )
        semantic_blocker_codes = {
            "V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
            "V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED",
            "V21_SEMANTIC_ADJUDICATION_PROVIDER_MISSING",
            "V21_SEMANTIC_BATCH_PROVIDER_MISSING",
            SEMANTIC_BATCH_PROVIDER_FAILED_CODE,
            SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
            SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE,
            FINAL_TARGET_PROVIDER_FAILURE_CODE,
            AUTO_PROVIDER_ROUTING_SKIPPED_CODE,
            "FINAL_TARGET_REPEAT_SEMANTIC_DECISION_REQUIRED",
            "FINAL_TARGET_REPEAT_REQUIRES_HUMAN_REVIEW",
            "FINAL_TARGET_REPEAT_HIGH_FATAL_KEEP_ALL_REJECTED",
            "SEMANTIC_DECISION_NOT_PROVIDED",
            "DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS",
            "SEMANTIC_DECISION_SCHEMA_INVALID",
        }
        blocker_codes = [
            blocker.code
            for blocker in decision_plan.blockers
            if blocker.severity in {"fatal", "write_blocker"} and blocker.code in semantic_blocker_codes
        ]
        fatal_semantic_issue_count = 0
        for issue_id in unresolved_ids:
            row = requests_by_id.get(issue_id, {})
            severity = str(row.get("severity") or "")
            issue_type = str(row.get("issue_type") or row.get("repeat_type") or "")
            if severity in {"high", "fatal"} or issue_type in {"modifier_redundancy", "self_repair_aborted_phrase"}:
                fatal_semantic_issue_count += 1
        if any(code in {"V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED", "V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED"} for code in blocker_codes):
            fatal_semantic_issue_count = max(fatal_semantic_issue_count, len(unresolved_ids) or 1)
        semantic_request_count = max(int(report.get("semantic_request_count") or 0), len(requests_by_id))
        semantic_request_unresolved_count = len(unresolved_ids)
        final_visible_repeat_advisory_report = self._final_visible_repeat_advisory_report(decision_plan, report)
        gate_passed = (
            semantic_request_unresolved_count == 0
            and fatal_semantic_issue_count == 0
            and not blocker_codes
            and not str(report.get("deepseek_provider_error") or "")
            and not str(report.get("deepseek_batch_error") or "")
            and bool(decision_plan.write_allowed)
        )
        report.update(
            {
                "semantic_adjudication_gate_passed": gate_passed,
                "semantic_request_count": semantic_request_count,
                "semantic_request_unresolved_count": semantic_request_unresolved_count,
                "fatal_semantic_issue_count": fatal_semantic_issue_count,
                "deepseek_provider_configured": bool(report.get("deepseek_provider_configured")),
                "deepseek_provider_called_count": int(report.get("deepseek_provider_called_count") or 0),
                "deepseek_provider_error": str(report.get("deepseek_provider_error") or ""),
                "deepseek_batch_enabled": bool(report.get("deepseek_batch_enabled")),
                "deepseek_batch_request_count": int(report.get("deepseek_batch_request_count") or 0),
                "deepseek_batch_attempt_count": int(report.get("deepseek_batch_attempt_count") or 0),
                "deepseek_batch_retry_count": int(report.get("deepseek_batch_retry_count") or 0),
                "deepseek_batch_issue_count": int(report.get("deepseek_batch_issue_count") or 0),
                "deepseek_batch_resolved_count": int(report.get("deepseek_batch_resolved_count") or 0),
                "deepseek_batch_unresolved_count": int(report.get("deepseek_batch_unresolved_count") or 0),
                "deepseek_batch_missing_issue_ids": list(report.get("deepseek_batch_missing_issue_ids") or []),
                "deepseek_batch_error": str(report.get("deepseek_batch_error") or ""),
                "deepseek_batch_chunk_count": int(report.get("deepseek_batch_chunk_count") or 0),
                "deepseek_batch_chunk_sizes": list(report.get("deepseek_batch_chunk_sizes") or []),
                "deepseek_provider_skipped_count": int(report.get("deepseek_provider_skipped_count") or 0),
                "deepseek_provider_skipped_reasons": dict(report.get("deepseek_provider_skipped_reasons") or {}),
                "semantic_decision_cache_used": bool(report.get("semantic_decision_cache_used")),
                "commit_reused_semantic_cache": bool(report.get("commit_reused_semantic_cache")),
                "semantic_cache_input_hash": str(report.get("semantic_cache_input_hash") or ""),
                "semantic_cache_issue_count": int(report.get("semantic_cache_issue_count") or 0),
                "semantic_cache_resolved_count": int(report.get("semantic_cache_resolved_count") or 0),
                "semantic_cache_unresolved_count": int(report.get("semantic_cache_unresolved_count") or 0),
                "semantic_auto_route_count": int(report.get("semantic_auto_route_count") or 0),
                "semantic_local_decision_count": int(report.get("semantic_local_decision_count") or 0),
                "semantic_provider_required_count": int(report.get("semantic_provider_required_count") or 0),
                "deterministic_baseline_refused_count": int(report.get("deterministic_baseline_refused_count") or 0),
                "unresolved_issue_ids": sorted(unresolved_ids),
                "blocker_codes": sorted(set(str(code) for code in blocker_codes if str(code))),
                "requests": [requests_by_id[key] for key in sorted(requests_by_id)],
                **final_visible_repeat_advisory_report,
            }
        )
        decision_plan.semantic_adjudication_report.clear()
        decision_plan.semantic_adjudication_report.update(report)

    def _final_visible_repeat_advisory_report(self, decision_plan, report: dict[str, Any]) -> dict[str, Any]:
        rows = [
            dict(row)
            for row in decision_plan.semantic_decision_rows
            if isinstance(row, dict) and str(row.get("_decision_kind") or "") == "advisory_final_visible_repeat"
        ]
        result_rows = [
            dict(row)
            for row in report.get("results") or []
            if isinstance(row, dict) and self._semantic_result_is_final_visible_repeat(row)
        ]
        decision_counts: dict[str, int] = {}
        for row in rows:
            decision = str(row.get("decision") or row.get("_semantic_json_decision") or "")
            if not decision:
                continue
            decision_counts[decision] = int(decision_counts.get(decision, 0)) + 1
        unresolved_count = 0
        review_count = 0
        for row in result_rows:
            blocker_code = str(row.get("blocker_code") or "")
            decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
            decision_value = str(decision.get("decision") or "")
            if blocker_code:
                unresolved_count += 1
            if decision_value in {SemanticAdjudicationDecisionType.NO_DECISION.value, SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW.value}:
                review_count += 1
        drop_candidate_count = sum(
            int(decision_counts.get(decision, 0))
            for decision in (
                SemanticAdjudicationDecisionType.DROP_LEFT.value,
                SemanticAdjudicationDecisionType.DROP_RIGHT.value,
                SemanticAdjudicationDecisionType.DROP_RECOMMENDED.value,
                SemanticAdjudicationDecisionType.DROP_ABORTED.value,
            )
        )
        applied_count = sum(1 for row in rows if bool(row.get("applied")))
        provider_called_count = int(report.get("final_visible_repeat_deepseek_provider_called_count") or 0)
        return {
            "final_visible_repeat_advisory_count": len(rows),
            "final_visible_repeat_advisory_result_count": len(result_rows),
            "final_visible_repeat_advisory_decision_counts": dict(sorted(decision_counts.items())),
            "final_visible_repeat_advisory_keep_count": int(decision_counts.get(SemanticAdjudicationDecisionType.KEEP_ALL.value, 0)),
            "final_visible_repeat_advisory_drop_candidate_count": drop_candidate_count,
            "final_visible_repeat_advisory_review_count": review_count,
            "final_visible_repeat_advisory_unresolved_count": unresolved_count,
            "final_visible_repeat_advisory_applied_count": applied_count,
            "final_visible_repeat_advisory_provider_called_count": provider_called_count,
            "final_visible_repeat_advisory_policy": "advisory_only_no_timeline_mutation",
            "final_visible_repeat_advisory_rows": rows,
        }

    def _semantic_result_is_final_visible_repeat(self, row: dict[str, Any]) -> bool:
        request = row.get("request") if isinstance(row.get("request"), dict) else {}
        local_context = request.get("local_context") if isinstance(request.get("local_context"), dict) else {}
        return str(local_context.get("cluster_type") or request.get("cluster_type") or "") == "final_visible_ambiguous_repeat"

    def _merge_final_visible_repeat_semantic_requests(self, decision_plan, validator_report: dict[str, Any]) -> bool:
        visible_gate = validator_report.get("final_caption_visible_repeat_gate") if isinstance(validator_report, dict) else {}
        if not isinstance(visible_gate, dict):
            return False
        payloads = [row for row in visible_gate.get("repeat_semantic_arbitration_request_payloads") or [] if isinstance(row, dict)]
        if not payloads:
            return False
        existing_ids = {
            str(payload.get("issue_id") or payload.get("cluster_id") or "")
            for payload in decision_plan.semantic_request_payloads
            if isinstance(payload, dict)
        }
        added = False
        for payload in payloads:
            issue_id = str(payload.get("issue_id") or payload.get("cluster_id") or "")
            if not issue_id or issue_id in existing_ids:
                continue
            row = dict(payload)
            row["issue_id"] = issue_id
            row["cluster_id"] = issue_id
            row["warning_only"] = True
            row["provider_required"] = False
            row["source"] = "final_caption_visible_repeat_gate"
            row["semantic_arbitration_mode"] = "request_only"
            decision_plan.semantic_request_payloads.append(row)
            decision_plan.decision_trace.append(
                {
                    "route": "semantic_warning",
                    "cluster_id": issue_id,
                    "decision": "final_visible_repeat_semantic_request_payload_emitted",
                    "reason": "final-visible repeat candidate is ambiguous but non-blocking; payload emitted for semantic arbitration",
                    "requires_semantic_decision": True,
                    "warning_only": True,
                }
            )
            existing_ids.add(issue_id)
            added = True
        return added

    def _route_final_visible_repeat_semantic_requests(self, decision_plan) -> bool:
        semantic_mode = self.decision_planner.semantic_mode
        if semantic_mode not in {SemanticAdjudicationMode.AUTO, SemanticAdjudicationMode.DEEPSEEK}:
            return False
        payloads = self._pending_final_visible_repeat_payloads(decision_plan)
        if not payloads:
            return False
        requests = [self._request_from_final_visible_repeat_payload(payload) for payload in payloads]
        routes = [
            self.decision_planner.issue_router.route_request(
                request,
                deterministic_action_available=False,
                local_action="final_visible_repeat_semantic_arbitration",
            )
            for request in requests
        ]
        self._merge_semantic_routes(decision_plan, routes)
        provider_requests = [
            request
            for request, route in zip(requests, routes)
            if route.requires_provider
        ]
        provider = self._semantic_adjudication_provider()
        if provider is None or not provider_requests:
            return False
        provider_called_before = int(getattr(provider, "provider_called_count", 0) or 0)
        try:
            decisions = list(provider.decide(provider_requests))
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
            self._record_final_visible_provider_warning_error(
                decision_plan,
                provider_requests,
                str(exc),
                provider_called_before=provider_called_before,
            )
            return True
        self._record_final_visible_provider_warning_metadata(
            decision_plan,
            provider,
            provider_called_before=provider_called_before,
        )
        self._inject_final_visible_provider_advisory_decisions(decision_plan, provider_requests, decisions)
        return True

    def _pending_final_visible_repeat_payloads(self, decision_plan) -> list[dict[str, Any]]:
        attempted_provider_ids = {
            str(
                row.get("issue_id")
                or row.get("cluster_id")
                or ((row.get("request") or {}).get("issue_id") if isinstance(row.get("request"), dict) else "")
                or ((row.get("request") or {}).get("cluster_id") if isinstance(row.get("request"), dict) else "")
                or ""
            )
            for row in (decision_plan.semantic_adjudication_report or {}).get("results") or []
            if isinstance(row, dict)
            and bool(row.get("provider_called"))
            and str(
                row.get("issue_id")
                or row.get("cluster_id")
                or ((row.get("request") or {}).get("issue_id") if isinstance(row.get("request"), dict) else "")
                or ((row.get("request") or {}).get("cluster_id") if isinstance(row.get("request"), dict) else "")
                or ""
            )
        }
        pending: list[dict[str, Any]] = []
        for payload in decision_plan.semantic_request_payloads:
            if not isinstance(payload, dict):
                continue
            issue_id = str(payload.get("issue_id") or payload.get("cluster_id") or "")
            if not issue_id or issue_id in attempted_provider_ids:
                continue
            if str(payload.get("cluster_type") or "") != "final_visible_ambiguous_repeat":
                continue
            if str(payload.get("source") or "") != "final_caption_visible_repeat_gate":
                continue
            if not bool(payload.get("warning_only")):
                continue
            pending.append(payload)
        return pending

    def _request_from_final_visible_repeat_payload(self, payload: dict[str, Any]) -> SemanticAdjudicationRequest:
        issue_id = str(payload.get("issue_id") or payload.get("cluster_id") or "")
        return SemanticAdjudicationRequest(
            issue_id=issue_id,
            issue_type=SemanticIssueType.AMBIGUOUS_REPEAT,
            severity=SemanticIssueSeverity.MEDIUM,
            candidate_segment_ids=[str(item) for item in payload.get("candidate_caption_ids") or [] if str(item)],
            candidate_caption_ids=[str(item) for item in payload.get("candidate_caption_ids") or [] if str(item)],
            word_ids=[str(item) for item in payload.get("word_ids") or [] if str(item)],
            target_start_us=int(payload.get("target_start_us") or 0),
            target_end_us=int(payload.get("target_end_us") or 0),
            text_before=str(payload.get("left_text") or payload.get("text") or ""),
            text_after=str(payload.get("right_text") or ""),
            local_context=dict(payload.get("local_context") or {}),
            recommended_action=str(payload.get("recommended_action") or "no_decision"),
            why_local_policy_cannot_decide=str(payload.get("why_local_policy_cannot_decide") or ""),
            allowed_decisions=[str(item) for item in payload.get("allowed_decisions") or [] if str(item)],
        )

    def _record_final_visible_provider_warning_metadata(
        self,
        decision_plan,
        provider: Any,
        *,
        provider_called_before: int,
    ) -> None:
        provider_called_after = int(getattr(provider, "provider_called_count", provider_called_before) or 0)
        delta = max(1, provider_called_after - int(provider_called_before or 0))
        self._increment_deepseek_provider_called_count(decision_plan, delta)
        report = decision_plan.semantic_adjudication_report
        report["final_visible_repeat_deepseek_provider_called_count"] = int(
            report.get("final_visible_repeat_deepseek_provider_called_count") or 0
        ) + delta
        report["final_visible_repeat_deepseek_provider_configured"] = True

    def _record_final_visible_provider_warning_error(
        self,
        decision_plan,
        requests: list[SemanticAdjudicationRequest],
        error: str,
        *,
        provider_called_before: int,
    ) -> None:
        self._record_final_visible_provider_warning_metadata(
            decision_plan,
            self._semantic_adjudication_provider(),
            provider_called_before=provider_called_before,
        )
        report = decision_plan.semantic_adjudication_report
        report["final_visible_repeat_deepseek_provider_error"] = error
        for request in requests:
            self._append_semantic_result(
                decision_plan,
                request,
                decision=None,
                resolved=False,
                blocker_code="FINAL_VISIBLE_REPEAT_PROVIDER_WARNING_FAILED",
                message=error or "final-visible repeat semantic provider failed in warning-only mode",
            )
            decision_plan.decision_trace.append(
                {
                    "route": "final_visible_repeat",
                    "cluster_id": request.issue_id,
                    "decision": "provider_error_warning_only",
                    "applied": False,
                    "source": "deepseek_semantic_planner",
                    "provider_called": True,
                    "warning_only": True,
                    "reason": error,
                }
            )

    def _inject_final_visible_provider_advisory_decisions(
        self,
        decision_plan,
        requests: list[SemanticAdjudicationRequest],
        decisions: list[SemanticAdjudicationDecision],
    ) -> None:
        decisions_by_issue = {str(decision.issue_id or ""): decision for decision in decisions}
        attempted_ids = {request.issue_id for request in requests}
        for request in requests:
            decision = decisions_by_issue.get(request.issue_id)
            if decision is None:
                self._append_semantic_result(
                    decision_plan,
                    request,
                    decision=None,
                    resolved=False,
                    blocker_code="FINAL_VISIBLE_REPEAT_PROVIDER_WARNING_PARTIAL_RESPONSE",
                    message="provider did not return a decision for this warning-only final-visible repeat request",
                )
                continue
            forbidden = self._forbidden_provider_fields(semantic_contract_to_dict(decision))
            if forbidden:
                self._append_semantic_result(
                    decision_plan,
                    request,
                    decision=decision,
                    resolved=False,
                    blocker_code="FINAL_VISIBLE_REPEAT_PROVIDER_FORBIDDEN_FIELDS",
                    message="provider returned forbidden physical fields for warning-only final-visible repeat request",
                )
                decision_plan.decision_trace.append(
                    {
                        "route": "final_visible_repeat",
                        "cluster_id": request.issue_id,
                        "decision": "provider_forbidden_fields_warning_only",
                        "applied": False,
                        "source": "deepseek_semantic_planner",
                        "provider_called": True,
                        "warning_only": True,
                        "forbidden_fields": forbidden,
                    }
                )
                continue
            resolved = decision.decision not in {
                SemanticAdjudicationDecisionType.NO_DECISION,
                SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW,
            } and not decision.requires_human_review
            if resolved:
                existing_ids = {
                    str(row.get("cluster_id") or row.get("issue_id") or "")
                    for row in decision_plan.semantic_decision_rows
                    if isinstance(row, dict)
                }
                if request.issue_id not in existing_ids:
                    decision_plan.semantic_decision_rows.append(
                        {
                            "decision_id": f"deepseek_final_visible_repeat_{request.issue_id}",
                            "cluster_id": request.issue_id,
                            "issue_type": request.issue_type.value,
                            "cluster_type": "final_visible_ambiguous_repeat",
                            "_decision_kind": "advisory_final_visible_repeat",
                            "decision": decision.decision.value,
                            "reason": decision.reason,
                            "confidence": float(decision.confidence or 0.0),
                            "requires_human_review": False,
                            "applied": False,
                            "_decision_source": "deepseek_semantic_planner",
                            "_semantic_json_decision": decision.decision.value,
                        }
                    )
            self._append_semantic_result(
                decision_plan,
                request,
                decision=decision,
                resolved=resolved,
                blocker_code="" if resolved else "FINAL_VISIBLE_REPEAT_PROVIDER_WARNING_NO_DECISION",
                message="" if resolved else (decision.reason or "provider returned no actionable warning-only decision"),
            )
            decision_plan.decision_trace.append(
                {
                    "route": "final_visible_repeat",
                    "cluster_id": request.issue_id,
                    "decision": decision.decision.value,
                    "applied": False,
                    "source": "deepseek_semantic_planner",
                    "provider_called": True,
                    "warning_only": True,
                    "reason": decision.reason,
                    "stage": "semantic_provider_advisory",
                }
            )
        decision_plan.semantic_request_payloads[:] = [
            payload
            for payload in decision_plan.semantic_request_payloads
            if str(payload.get("issue_id") or payload.get("cluster_id") or "") not in attempted_ids
        ]

    def _refresh_validator_semantic_gate_after_request_merge(self, validator_report: dict[str, Any], decision_plan) -> None:
        semantic = dict(decision_plan.semantic_adjudication_report or {})
        semantic_passed = bool(semantic.get("semantic_adjudication_gate_passed"))
        semantic.update(
            {
                "semantic_final_review_validator_passed": semantic_passed,
                "semantic_review_blocker_count": len([blocker for blocker in decision_plan.blockers if blocker.severity in {"fatal", "write_blocker"}]),
                "semantic_unresolved_count": int(decision_plan.semantic_unresolved_count or 0),
                "requires_human_review": bool(decision_plan.requires_human_review),
                "write_allowed": semantic_passed,
                "dry_run_continued_for_discovery": bool(decision_plan.dry_run_continued_for_discovery),
            }
        )
        validator_report["semantic_final_review_validator"] = semantic
        previous_quality = validator_report.get("quality_gate_report")
        if not isinstance(previous_quality, dict):
            return
        base_ok = bool(previous_quality.get("ready_for_user_manual_qc_preconditions_passed", validator_report.get("validator_report_ok")))
        quality_gate = build_quality_gate_report(
            effective_speed_gate=_normalize_effective_speed_prewrite_placeholder(previous_quality.get("effective_speed_gate")),
            final_repeat_convergence_gate=validator_report.get("final_repeat_convergence_gate"),
            final_caption_visible_repeat_gate=validator_report.get("final_caption_visible_repeat_gate"),
            semantic_adjudication_gate=semantic,
            visual_pacing_gate=validator_report.get("visual_pacing_gate"),
            caption_alignment_gate=validator_report.get("caption_alignment_gate"),
            final_timeline_quality_guard_gate=validator_report.get("final_timeline_quality_guard_report"),
            prewrite_projection_gate=validator_report.get("prewrite_projected_write_view"),
            ready_for_user_manual_qc_preconditions_passed=base_ok and semantic_passed,
        )
        validator_report["quality_gate_report"] = quality_gate
        validator_report["validator_report_ok"] = bool(validator_report.get("validator_report_ok")) and bool(quality_gate.get("gate_passed"))

    def _route_final_target_repeat_semantic_requests(self, decision_plan) -> bool:
        semantic_mode = self.decision_planner.semantic_mode
        if semantic_mode not in {SemanticAdjudicationMode.AUTO, SemanticAdjudicationMode.DEEPSEEK}:
            return False
        payloads = self._pending_final_target_repeat_payloads(decision_plan)
        if not payloads:
            return False
        requests = [request_from_final_target_payload(payload) for payload in payloads]
        routes = [
            self.decision_planner.issue_router.route_request(
                request,
                deterministic_action_available=False,
                local_action="final_target_repeat_semantic_review",
            )
            for request in requests
        ]
        self._merge_semantic_routes(decision_plan, routes)
        provider_requests = [
            request
            for request, route in zip(requests, routes)
            if route.requires_provider
        ]
        if not provider_requests:
            return False
        provider = self._semantic_adjudication_provider()
        if provider is None:
            return False
        provider_called_before = int(getattr(provider, "provider_called_count", 0) or 0)
        try:
            decisions = list(provider.decide(provider_requests))
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
            self._merge_deepseek_batch_metadata(decision_plan, provider, provider_called_before=provider_called_before)
            self._record_final_target_provider_error(decision_plan, provider_requests, str(exc))
            return False
        self._merge_deepseek_batch_metadata(decision_plan, provider, provider_called_before=provider_called_before)
        return self._inject_final_target_provider_decisions(decision_plan, provider_requests, decisions)

    def _pending_final_target_repeat_payloads(self, decision_plan) -> list[dict[str, Any]]:
        resolved_ids = {
            str(row.get("cluster_id") or row.get("issue_id") or "")
            for row in decision_plan.semantic_decision_rows
            if isinstance(row, dict) and str(row.get("cluster_id") or row.get("issue_id") or "")
        }
        attempted_provider_ids = {
            str(
                row.get("issue_id")
                or row.get("cluster_id")
                or ((row.get("request") or {}).get("issue_id") if isinstance(row.get("request"), dict) else "")
                or ((row.get("request") or {}).get("cluster_id") if isinstance(row.get("request"), dict) else "")
                or ""
            )
            for row in (decision_plan.semantic_adjudication_report or {}).get("results") or []
            if isinstance(row, dict)
            and bool(row.get("provider_called"))
            and str(
                row.get("issue_id")
                or row.get("cluster_id")
                or ((row.get("request") or {}).get("issue_id") if isinstance(row.get("request"), dict) else "")
                or ((row.get("request") or {}).get("cluster_id") if isinstance(row.get("request"), dict) else "")
                or ""
            )
        }
        payloads: list[dict[str, Any]] = []
        for payload in decision_plan.semantic_request_payloads:
            if not isinstance(payload, dict):
                continue
            cluster_id = str(payload.get("cluster_id") or payload.get("issue_id") or "")
            if not cluster_id or cluster_id in resolved_ids or cluster_id in attempted_provider_ids:
                continue
            if str(payload.get("type") or "") != "final_target_repeat":
                continue
            if not bool(payload.get("provider_required", True)):
                continue
            payloads.append(payload)
        return payloads

    def _semantic_adjudication_provider(self):
        if self.semantic_provider is not None:
            return self.semantic_provider
        planner = self.deepseek_planner or getattr(self.decision_planner, "deepseek_planner", None)
        provider = getattr(planner, "provider", None)
        return provider

    def _merge_semantic_routes(self, decision_plan, routes: list[SemanticRoutingDecision]) -> None:
        report = decision_plan.semantic_adjudication_report
        report.setdefault("semantic_mode", self.decision_planner.semantic_mode.value)
        report["deepseek_provider_configured"] = bool(report.get("deepseek_provider_configured") or self._semantic_adjudication_provider() is not None)
        existing = {
            str(row.get("issue_id") or "")
            for row in report.get("routing_decisions") or []
            if isinstance(row, dict)
        }
        routing_rows = list(report.get("routing_decisions") or [])
        skipped_reasons = dict(report.get("deepseek_provider_skipped_reasons") or {})
        provider_required_count = int(report.get("semantic_provider_required_count") or 0)
        skipped_count = int(report.get("deepseek_provider_skipped_count") or 0)
        for route in routes:
            if route.issue_id in existing:
                continue
            routing_rows.append(semantic_contract_to_dict(route))
            existing.add(route.issue_id)
            if route.requires_provider:
                provider_required_count += 1
            else:
                skipped_count += 1
                reason = route.provider_reason or route.local_action or "local_or_structural_issue"
                skipped_reasons[reason] = int(skipped_reasons.get(reason, 0)) + 1
        report["routing_decisions"] = routing_rows
        report["semantic_auto_route_count"] = len(routing_rows)
        report["semantic_provider_required_count"] = provider_required_count
        report["deepseek_provider_skipped_count"] = skipped_count
        report["deepseek_provider_skipped_reasons"] = dict(sorted(skipped_reasons.items()))

    def _increment_deepseek_provider_called_count(self, decision_plan, count: int) -> None:
        report = decision_plan.semantic_adjudication_report
        report["deepseek_provider_configured"] = True
        report["deepseek_provider_called_count"] = int(report.get("deepseek_provider_called_count") or 0) + int(count)
        report["deepseek_provider_error"] = str(report.get("deepseek_provider_error") or "")

    def _merge_deepseek_batch_metadata(self, decision_plan, provider: Any, *, provider_called_before: int = 0) -> None:
        report = decision_plan.semantic_adjudication_report
        provider_called_after = int(getattr(provider, "provider_called_count", provider_called_before) or 0)
        delta = max(0, provider_called_after - int(provider_called_before or 0))
        if delta == 0:
            delta = 1
        self._increment_deepseek_provider_called_count(decision_plan, delta)
        report["deepseek_batch_enabled"] = bool(getattr(provider, "deepseek_batch_enabled", report.get("deepseek_batch_enabled", False)))
        numeric_fields = {
            "deepseek_batch_request_count",
            "deepseek_batch_attempt_count",
            "deepseek_batch_retry_count",
            "deepseek_batch_issue_count",
            "deepseek_batch_resolved_count",
            "deepseek_batch_unresolved_count",
            "deepseek_batch_chunk_count",
        }
        list_fields = {
            "deepseek_batch_missing_issue_ids",
            "deepseek_batch_unknown_issue_ids",
            "deepseek_batch_chunk_sizes",
        }
        dict_fields = {
            "deepseek_batch_request",
            "deepseek_batch_response",
            "deepseek_batch_error_payload",
        }
        for field_name in BATCH_METADATA_FIELDS:
            value = getattr(provider, field_name, None)
            if value is None:
                continue
            if field_name in numeric_fields:
                report[field_name] = int(report.get(field_name) or 0) + int(value or 0)
            elif field_name in list_fields:
                existing = list(report.get(field_name) or [])
                if isinstance(value, list):
                    report[field_name] = existing + value
            elif field_name in dict_fields:
                if value:
                    existing_value = report.get(field_name)
                    if existing_value:
                        report[field_name] = {"previous": existing_value, "latest": value}
                    else:
                        report[field_name] = value
            elif field_name == "deepseek_batch_error":
                if value:
                    report[field_name] = str(value)

    def _record_final_target_provider_error(
        self,
        decision_plan,
        requests: list[SemanticAdjudicationRequest],
        error: str,
    ) -> None:
        message = error or "semantic provider failed while adjudicating final target repeat request"
        decision_plan.semantic_adjudication_report["deepseek_provider_error"] = message
        for request in requests:
            self._append_semantic_blocker_once(
                decision_plan,
                code=FINAL_TARGET_PROVIDER_FAILURE_CODE,
                message="semantic provider failed while adjudicating final target repeat request",
                cluster_id=request.issue_id,
                context={"issue_type": request.issue_type.value, "provider_error": message},
            )
            self._append_semantic_result(
                decision_plan,
                request,
                decision=None,
                resolved=False,
                blocker_code=FINAL_TARGET_PROVIDER_FAILURE_CODE,
                message=message,
            )
            decision_plan.decision_trace.append(
                {
                    "route": "final_target_repeat",
                    "cluster_id": request.issue_id,
                    "decision": "provider_error",
                    "applied": False,
                    "source": "deepseek_semantic_planner",
                    "provider_called": True,
                    "reason": message,
                }
            )

    def _inject_final_target_provider_decisions(
        self,
        decision_plan,
        requests: list[SemanticAdjudicationRequest],
        decisions: list[SemanticAdjudicationDecision],
    ) -> bool:
        decisions_by_issue = {str(decision.issue_id or ""): decision for decision in decisions}
        changed = False
        for request in requests:
            decision = decisions_by_issue.get(request.issue_id)
            if decision is None:
                self._append_semantic_blocker_once(
                    decision_plan,
                    code=SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
                    message="DeepSeek provider did not return a decision for this final target repeat request",
                    cluster_id=request.issue_id,
                    context={"issue_type": request.issue_type.value},
                )
                self._append_semantic_result(
                    decision_plan,
                    request,
                    decision=None,
                    resolved=False,
                    blocker_code=SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
                    message="DeepSeek provider did not return a decision for this final target repeat request",
                )
                continue
            forbidden = self._forbidden_provider_fields(semantic_contract_to_dict(decision))
            if forbidden:
                self._append_semantic_blocker_once(
                    decision_plan,
                    code="DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS",
                    message="DeepSeek provider returned forbidden physical timeline/material fields",
                    cluster_id=request.issue_id,
                    context={"forbidden_fields": forbidden},
                )
                self._append_semantic_result(
                    decision_plan,
                    request,
                    decision=decision,
                    resolved=False,
                    blocker_code="DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS",
                    message="DeepSeek provider returned forbidden physical timeline/material fields",
                )
                continue
            row = self._final_target_semantic_decision_row(request, decision)
            if row is None:
                blocker_code = SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE
                self._append_semantic_blocker_once(
                    decision_plan,
                    code=blocker_code,
                    message=decision.reason or "final target repeat decision requires human review",
                    cluster_id=request.issue_id,
                    context={"decision": decision.decision.value},
                )
                self._append_semantic_result(
                    decision_plan,
                    request,
                    decision=decision,
                    resolved=False,
                    blocker_code=blocker_code,
                    message=decision.reason or "final target repeat decision requires human review",
                )
                continue
            existing_ids = {
                str(existing.get("cluster_id") or existing.get("issue_id") or "")
                for existing in decision_plan.semantic_decision_rows
                if isinstance(existing, dict)
            }
            if request.issue_id not in existing_ids:
                decision_plan.semantic_decision_rows.append(row)
            self._append_semantic_result(
                decision_plan,
                request,
                decision=decision,
                resolved=True,
                blocker_code="",
                message="",
            )
            decision_plan.decision_trace.append(
                {
                    "route": "final_target_repeat",
                    "cluster_id": request.issue_id,
                    "decision": row["decision"],
                    "applied": False,
                    "source": "deepseek_semantic_planner",
                    "provider_called": True,
                    "reason": row["reason"],
                    "stage": "semantic_provider_adjudication",
                }
            )
            changed = True
        return changed

    def _final_target_semantic_decision_row(
        self,
        request: SemanticAdjudicationRequest,
        decision: SemanticAdjudicationDecision,
    ) -> dict[str, Any] | None:
        value = decision.decision.value
        if value in {SemanticAdjudicationDecisionType.NO_DECISION.value, SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW.value}:
            no_row: dict[str, Any] | None = None
            return no_row
        if decision.requires_human_review:
            no_row: dict[str, Any] | None = None
            return no_row
        allowed = set(request.allowed_decisions)
        if value not in allowed:
            no_row: dict[str, Any] | None = None
            return no_row
        if value == SemanticAdjudicationDecisionType.DROP_RECOMMENDED.value:
            no_row: dict[str, Any] | None = None
            return no_row
        return {
            "decision_id": f"deepseek_final_target_{request.issue_id}",
            "cluster_id": request.issue_id,
            "decision": value,
            "reason": decision.reason,
            "confidence": float(decision.confidence or 0.0),
            "requires_human_review": False,
            "_decision_source": "deepseek_semantic_planner",
            "_semantic_json_decision": value,
        }

    def _append_semantic_blocker_once(
        self,
        decision_plan,
        *,
        code: str,
        message: str,
        cluster_id: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        if any(
            blocker.code == code and str(blocker.context.get("cluster_id") or "") == cluster_id
            for blocker in decision_plan.blockers
        ):
            return
        merged_context = {"cluster_id": cluster_id, "allows_dry_run_discovery": True, "write_allowed": False}
        merged_context.update(context or {})
        decision_plan.blockers.append(
            Blocker(
                code=code,
                message=message,
                layer="decision",
                severity="write_blocker",
                context=merged_context,
            )
        )

    def _append_semantic_result(
        self,
        decision_plan,
        request: SemanticAdjudicationRequest,
        *,
        decision: SemanticAdjudicationDecision | None,
        resolved: bool,
        blocker_code: str,
        message: str,
    ) -> None:
        report = decision_plan.semantic_adjudication_report
        request_rows = [row for row in report.get("requests") or [] if isinstance(row, dict)]
        if request.issue_id not in {str(row.get("issue_id") or row.get("cluster_id") or "") for row in request_rows}:
            request_rows.append(semantic_contract_to_dict(request))
        report["requests"] = request_rows
        if decision is not None:
            decisions = [row for row in report.get("decisions") or [] if isinstance(row, dict)]
            decisions.append(semantic_contract_to_dict(decision))
            report["decisions"] = decisions
        results = [row for row in report.get("results") or [] if isinstance(row, dict)]
        results.append(
            semantic_contract_to_dict(
                SemanticAdjudicationResult(
                    request=request,
                    decision=decision,
                    resolved=resolved,
                    provider_configured=True,
                    provider_called=True,
                    blocker_code=blocker_code,
                    message=message,
                )
            )
        )
        report["results"] = results

    def _forbidden_provider_fields(self, value: Any) -> list[str]:
        found: set[str] = set()

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    if str(key) in FORBIDDEN_SEMANTIC_PROVIDER_FIELDS:
                        found.add(str(key))
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
        return sorted(found)

    def _harden_modifier_redundancy_semantic_requests(self, decision_plan) -> None:
        if not decision_plan.semantic_request_payloads:
            return
        changed = False
        unresolved_cluster_ids = set(decision_plan.modifier_redundancy_unresolved_cluster_ids)
        existing_blocker_ids = {
            str(blocker.context.get("cluster_id") or "")
            for blocker in decision_plan.blockers
            if blocker.code == "V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED"
        }
        for payload in decision_plan.semantic_request_payloads:
            if str(payload.get("repeat_type") or "") != "modifier_redundancy":
                continue
            if str(payload.get("type") or "") != "single_variant_modifier_redundancy":
                continue
            allowed = [str(item) for item in payload.get("allowed_decisions") or [] if str(item)]
            hardened = [item for item in allowed if item != "keep_all"]
            if allowed and hardened != allowed:
                payload["allowed_decisions"] = hardened
                changed = True
            payload.setdefault("suggested_for_rough_cut", "drop_redundant_modifier")
            payload["fatal_modifier_redundancy_keep_all_allowed"] = False
            schema = payload.get("required_decision_schema")
            if isinstance(schema, dict):
                schema["decision"] = "drop_redundant_modifier | requires_human_review"
            cluster_id = str(payload.get("cluster_id") or "")
            if cluster_id:
                unresolved_cluster_ids.add(cluster_id)
                if cluster_id not in existing_blocker_ids:
                    decision_plan.blockers.append(
                        Blocker(
                            code="V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
                            message="fatal modifier redundancy must be repaired or explicitly blocked; keep_all is not allowed",
                            layer="decision",
                            severity="write_blocker",
                            context={
                                "cluster_id": cluster_id,
                                "repeat_type": "modifier_redundancy",
                                "type": "single_variant_modifier_redundancy",
                            },
                        )
                    )
                    existing_blocker_ids.add(cluster_id)
                    changed = True
        if unresolved_cluster_ids:
            decision_plan.modifier_redundancy_unresolved_cluster_ids[:] = sorted(unresolved_cluster_ids)
            object.__setattr__(decision_plan, "semantic_unresolved_count", max(int(decision_plan.semantic_unresolved_count), len(unresolved_cluster_ids)))
            object.__setattr__(decision_plan, "requires_human_review", True)
            object.__setattr__(decision_plan, "write_allowed", False)
            object.__setattr__(decision_plan, "dry_run_continued_for_discovery", True)
        if changed:
            decision_plan.decision_trace.append(
                {
                    "route": "semantic_gate",
                    "decision": "harden_modifier_redundancy_request",
                    "applied": True,
                    "reason": "fatal modifier redundancy semantic requests cannot allow keep_all",
                }
            )

    def _block_final_modifier_keep_all(self, decision_plan) -> None:
        existing = {
            str(blocker.context.get("cluster_id") or "")
            for blocker in decision_plan.blockers
            if blocker.code == "V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED"
        }
        for row in decision_plan.decision_trace:
            if not isinstance(row, dict):
                continue
            if str(row.get("route") or "") != "final_modifier_redundancy":
                continue
            if str(row.get("decision") or "") != "keep_all":
                continue
            cluster_id = str(row.get("cluster_id") or "")
            if cluster_id in existing:
                continue
            decision_plan.blockers.append(
                Blocker(
                    code="V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
                    message="fatal modifier redundancy reached final timeline through keep_all",
                    layer="decision",
                    severity="fatal",
                    context={"cluster_id": cluster_id, "repeat_type": "modifier_redundancy"},
                )
            )
            existing.add(cluster_id)
            object.__setattr__(decision_plan, "write_allowed", False)
            object.__setattr__(decision_plan, "requires_human_review", True)

    def _drop_deterministic_self_repair_aborted_segments(self, final_timeline, decision_plan):
        if len(final_timeline) < 2:
            return final_timeline
        unresolved_pairs = {
            (normalize_text(str(payload.get("left_text") or payload.get("text_before") or "")), normalize_text(str(payload.get("right_text") or payload.get("text_after") or "")))
            for payload in decision_plan.semantic_request_payloads
            if isinstance(payload, dict)
            and str(payload.get("issue_type") or payload.get("repeat_type") or "") == "self_repair_aborted_phrase"
        }
        kept = []
        dropped = 0
        index = 0
        while index < len(final_timeline):
            current = final_timeline[index]
            next_segment = final_timeline[index + 1] if index + 1 < len(final_timeline) else None
            candidate = (
                self_repair_aborted_phrase_candidate(str(current.text or ""), str(next_segment.text or ""))
                if next_segment is not None
                else None
            )
            pair = (normalize_text(str(current.text or "")), normalize_text(str(next_segment.text or ""))) if next_segment is not None else ("", "")
            if pair in unresolved_pairs:
                kept.append(current)
                index += 1
                continue
            if candidate and bool(candidate.get("deterministic_drop_left")) and self._safe_self_repair_drop(current, next_segment):
                decision_plan.decision_trace.append(
                    {
                        "route": "self_repair_aborted_phrase",
                        "decision": "drop_left_keep_right",
                        "applied": True,
                        "dropped_segment_id": current.segment_id,
                        "kept_segment_id": next_segment.segment_id,
                        "common_prefix": str(candidate.get("common_prefix") or ""),
                        "similarity": float(candidate.get("similarity") or 0.0),
                        "reason": "deterministic self-repair restart drops incomplete aborted phrase segment",
                    }
                )
                dropped += 1
                index += 1
                continue
            kept.append(current)
            index += 1
        if not dropped:
            return final_timeline
        decision_plan.decision_trace.append(
            {
                "route": "self_repair_aborted_phrase",
                "decision": "repack_after_drop",
                "applied": True,
                "dropped_segment_count": dropped,
            }
        )
        return self._repack_final_timeline(kept)

    def _safe_self_repair_drop(self, left, right) -> bool:
        if right is None:
            return False
        if str(left.source_material_id or "") != str(right.source_material_id or ""):
            return False
        left_source_segment_id = str(left.source_segment_id or "")
        right_source_segment_id = str(right.source_segment_id or "")
        if left_source_segment_id and right_source_segment_id and left_source_segment_id != right_source_segment_id:
            return False
        gap_us = int(right.source_start_us) - int(left.source_end_us)
        if gap_us < -80_000 or gap_us > 1_500_000:
            return False
        if int(left.target_end_us) <= int(left.target_start_us) or int(right.target_end_us) <= int(right.target_start_us):
            return False
        return int(left.target_start_us) <= int(right.target_start_us)

    def _repack_final_timeline(self, final_timeline):
        repacked = []
        cursor = 0
        for segment in final_timeline:
            duration = max(0, int(segment.target_end_us) - int(segment.target_start_us))
            repacked.append(replace(segment, target_start_us=cursor, target_end_us=cursor + duration))
            cursor += duration
        return repacked

def _bind_engine_validation_helpers() -> None:
    engine_validation_helpers.configure_engine_validation_dependencies(globals())
    ArollEngine._semantic_payload_comparable_texts = engine_validation_helpers._semantic_payload_comparable_texts  # type: ignore[method-assign]
    ArollEngine._semantic_request_consistency_blockers = engine_validation_helpers._semantic_request_consistency_blockers  # type: ignore[method-assign]
    ArollEngine._final_repeat_validator_missing_request_blockers = engine_validation_helpers._final_repeat_validator_missing_request_blockers  # type: ignore[method-assign]
    ArollEngine._blocked = engine_validation_helpers._blocked  # type: ignore[method-assign]
    ArollEngine._validator_blockers = engine_validation_helpers._validator_blockers  # type: ignore[method-assign]
    ArollEngine._attach_final_caption_visible_repeat_gate = engine_validation_helpers._attach_final_caption_visible_repeat_gate  # type: ignore[method-assign]


_bind_engine_validation_helpers()
