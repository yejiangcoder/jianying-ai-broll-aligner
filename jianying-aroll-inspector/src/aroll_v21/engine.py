from __future__ import annotations

import json
import gzip
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

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
    SemanticRoutingDecision,
    semantic_contract_to_dict,
)
from aroll_v21.evidence import CandidateEvidenceBuilder
from aroll_v21.ingest import DraftIngest
from aroll_v21.ir.models import Blocker, BlockerReport, RunReport, dataclass_to_dict
from aroll_v21.quality import VisualPacingNormalizer, build_visual_pacing_report
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
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
        if fatal_ingest_blocker or not source_graph.invariant_report.single_source_graph_ok:
            return self._blocked(
                source_graph=source_graph,
                blockers=blockers,
                summary={"stage": "ingest", "ingest_metadata": inputs.ingest_metadata},
            )

        repeat_clusters = self.evidence_builder.build(source_graph)
        decision_plan = self.decision_planner.plan(repeat_clusters)
        self._harden_modifier_redundancy_semantic_requests(decision_plan)
        self._refresh_semantic_adjudication_report(decision_plan)
        if decision_plan.blocked:
            consistency_blockers = self._semantic_request_consistency_blockers(decision_plan, {})
            if consistency_blockers:
                decision_plan.blockers.extend(consistency_blockers)
            blockers.extend(decision_plan.blockers)
            return self._blocked(
                source_graph=source_graph,
                repeat_clusters=repeat_clusters,
                decision_plan=decision_plan,
                blockers=blockers,
                summary={"stage": "decision"},
            )

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
        if compiler_blockers or any(blocker.severity == "fatal" for blocker in new_decision_blockers):
            blockers.extend(decision_plan.blockers)
            return self._blocked(
                source_graph=source_graph,
                repeat_clusters=repeat_clusters,
                decision_plan=decision_plan,
                final_timeline=final_timeline,
                blockers=blockers,
                summary={"stage": "compiler"},
            )

        final_timeline = self._drop_deterministic_self_repair_aborted_segments(final_timeline, decision_plan)
        final_timeline, visual_pacing_report = self.visual_pacing.normalize(final_timeline, source_graph)
        captions = self.renderer.render(final_timeline, source_graph)
        final_visible_repair = repair_final_visible_caption_issues(
            final_timeline=final_timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda timeline: self.renderer.render(timeline, source_graph),
        )
        final_timeline = final_visible_repair.final_timeline
        captions = final_visible_repair.captions
        if int(final_visible_repair.report.get("final_visible_repair_action_count") or 0) > 0:
            visual_pacing_report = build_visual_pacing_report(
                final_timeline=final_timeline,
                captions=captions,
                executed=True,
                source_graph=source_graph,
                merge_report={
                    **dict(visual_pacing_report or {}),
                    "final_visible_repair_action_count": int(final_visible_repair.report.get("final_visible_repair_action_count") or 0),
                },
            )
        self._sync_semantic_gate_with_final_output(decision_plan, final_timeline, captions)
        self._refresh_semantic_adjudication_report(decision_plan)
        blockers.extend(decision_plan.blockers)
        material_write_plan, writer_blockers = self.writer.build_write_plan(source_graph, captions)
        blockers.extend(writer_blockers)
        if writer_blockers:
            return self._blocked(
                source_graph=source_graph,
                repeat_clusters=repeat_clusters,
                decision_plan=decision_plan,
                final_timeline=final_timeline,
                captions=captions,
                material_write_plan=material_write_plan,
                blockers=blockers,
                summary={"stage": "writer"},
            )

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
        validator_report["final_visible_caption_repair_report"] = final_visible_repair.report
        validator_report = self._attach_final_caption_visible_repeat_gate(validator_report, captions)
        consistency_blockers = self._semantic_request_consistency_blockers(decision_plan, validator_report)
        if consistency_blockers:
            blockers.extend(consistency_blockers)
            decision_plan.blockers.extend(consistency_blockers)
        validator_blockers: list[Blocker] = []
        if not validator_report.get("validator_report_ok"):
            validator_blockers = self._validator_blockers(validator_report)
            blockers.extend(validator_blockers)

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
        resolved_cluster_ids: set[str] = set()
        remaining_payloads: list[dict[str, Any]] = []
        for payload in decision_plan.semantic_request_payloads:
            cluster_id = str(payload.get("cluster_id") or "")
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
            }
        )
        decision_plan.semantic_adjudication_report.clear()
        decision_plan.semantic_adjudication_report.update(report)

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
        payloads: list[dict[str, Any]] = []
        for payload in decision_plan.semantic_request_payloads:
            if not isinstance(payload, dict):
                continue
            cluster_id = str(payload.get("cluster_id") or payload.get("issue_id") or "")
            if not cluster_id or cluster_id in resolved_ids:
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

    def _semantic_payload_comparable_texts(self, payload: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for key in ("text", "raw_phrase", "phrase"):
            texts.append(str(payload.get(key) or ""))
        for variant in payload.get("variants") or []:
            if isinstance(variant, dict):
                texts.append(str(variant.get("text") or ""))
        for evidence in payload.get("local_evidence") or []:
            if not isinstance(evidence, dict):
                continue
            metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), dict) else {}
            candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
            for key in ("raw_phrase", "phrase", "text"):
                texts.append(str(candidate.get(key) or ""))
        normalized: list[str] = []
        seen: set[str] = set()
        for text in texts:
            value = normalize_text(text)
            if len(value) < 2 or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _semantic_request_consistency_blockers(self, decision_plan, validator_report: dict[str, Any]) -> list[Blocker]:
        blockers: list[Blocker] = []
        payload_cluster_ids = {str(payload.get("cluster_id") or "") for payload in decision_plan.semantic_request_payloads}
        for blocker in decision_plan.blockers:
            if blocker.code == "SEMANTIC_DECISION_NOT_PROVIDED":
                cluster_id = str(blocker.context.get("cluster_id") or "")
                if cluster_id and cluster_id not in payload_cluster_ids:
                    blockers.append(
                        Blocker(
                            code="INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_DECISION_NOT_PROVIDED",
                            message="semantic decision missing blocker did not emit a matching semantic request payload",
                            layer="engine",
                            context={
                                "cluster_id": cluster_id,
                                "missing_request_for": "SEMANTIC_DECISION_NOT_PROVIDED",
                            },
                        )
                    )
                continue
            if blocker.code == "UNIT_SPLIT_REQUIRES_HUMAN_REVIEW":
                cluster_id = str(blocker.context.get("cluster_id") or "")
                if cluster_id and cluster_id not in payload_cluster_ids:
                    blockers.append(
                        Blocker(
                            code="INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_UNIT_SPLIT",
                            message="unit split human review blocker did not emit a matching semantic request payload",
                            layer="engine",
                            context={
                                "cluster_id": cluster_id,
                                "missing_request_for": "UNIT_SPLIT_REQUIRES_HUMAN_REVIEW",
                            },
                        )
                    )
        payload_texts = set()
        for payload in decision_plan.semantic_request_payloads:
            if str(payload.get("repeat_type") or "") != "modifier_redundancy":
                continue
            payload_texts.update(self._semantic_payload_comparable_texts(payload))
        for section_name in ("final_repeat_validator", "hidden_audio_repeat_validator"):
            section = validator_report.get(section_name) or {}
            for issue in section.get("blocking_issues") or []:
                if not isinstance(issue, dict):
                    continue
                if str(issue.get("type") or issue.get("issue_type") or "") != "adjacent_modifier_semantic_redundancy":
                    continue
                issue_texts = {
                    normalize_text(str(issue.get("text") or "")),
                    normalize_text(str(issue.get("phrase") or "")),
                    normalize_text(str(issue.get("fragment_text") or "")),
                }
                issue_texts = {text for text in issue_texts if len(text) >= 2}
                if issue_texts and payload_texts & issue_texts:
                    continue
                blockers.append(
                    Blocker(
                        code="INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_FATAL_REPEAT",
                        message="validator found semantic repeat fatal but no matching semantic request payload was emitted",
                        layer="engine",
                        context={
                            "validator_section": section_name,
                            "repeat_type": "modifier_redundancy",
                            "issue_type": "adjacent_modifier_semantic_redundancy",
                            "issue_text": str(issue.get("text") or issue.get("phrase") or ""),
                        },
                    )
                )
        if not payload_cluster_ids:
            blockers.extend(self._final_repeat_validator_missing_request_blockers(validator_report))
        return blockers

    def _final_repeat_validator_missing_request_blockers(self, validator_report: dict[str, Any]) -> list[Blocker]:
        blockers: list[Blocker] = []
        sections = (
            ("final_repeat_validator", "final_repeat_gate_passed"),
            ("hidden_audio_repeat_validator", "hidden_audio_repeat_gate_passed"),
        )
        for section_name, pass_key in sections:
            section = validator_report.get(section_name) or {}
            if section.get(pass_key, True):
                continue
            issues = [row for row in (section.get("blocking_issues") or []) if isinstance(row, dict)]
            candidates = [row for row in (section.get("final_target_repeat_candidates") or []) if isinstance(row, dict)]
            for issue in issues + candidates:
                issue_type = str(issue.get("type") or issue.get("issue_type") or issue.get("cluster_type") or "")
                if issue_type == "adjacent_modifier_semantic_redundancy":
                    continue
                blockers.append(
                    Blocker(
                        code="INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_FINAL_REPEAT_VALIDATOR",
                        message="final repeat validator found a fatal repeat but no semantic request payload was emitted",
                        layer="engine",
                        context={
                            "validator_section": section_name,
                            "candidate_type": issue_type,
                            "left_text": str(issue.get("left_text") or issue.get("prev_text") or ""),
                            "right_text": str(issue.get("right_text") or issue.get("next_text") or ""),
                            "overlap": str(issue.get("overlap") or issue.get("phrase") or ""),
                            "repeated_phrase": str(issue.get("phrase") or issue.get("text") or ""),
                            "row_index": int(issue.get("row_index") or issue.get("left_index") or issue.get("subtitle_index") or 0),
                            "next_row_index": int(issue.get("next_row_index") or issue.get("right_index") or 0),
                            "severity": str(issue.get("severity") or issue.get("confidence") or ""),
                            "reason": str(issue.get("reason") or ""),
                        },
                    )
                )
                break
        return blockers

    def _blocked(
        self,
        *,
        source_graph=None,
        repeat_clusters=None,
        decision_plan=None,
        final_timeline=None,
        captions=None,
        material_write_plan=None,
        blockers: list[Blocker],
        summary: dict[str, Any],
    ) -> RunReport:
        return RunReport(
            status="blocked",
            source_graph=source_graph,
            repeat_clusters=repeat_clusters or [],
            decision_plan=decision_plan,
            final_timeline=final_timeline or [],
            captions=captions or [],
            material_write_plan=material_write_plan or {},
            validator_report={},
            postwrite_report={},
            blocker_report=BlockerReport(blocked=True, blockers=blockers, summary=summary),
            decision_trace=decision_plan.decision_trace if decision_plan else [],
        )

    def _validator_blockers(self, report: dict[str, Any]) -> list[Blocker]:
        blockers: list[Blocker] = []
        emitted_codes: set[str] = set()
        mapping = {
            "final_repeat_validator": ("FINAL_REPEAT_VALIDATOR_FAILED", "final repeat validator failed"),
            "hidden_audio_repeat_validator": ("HIDDEN_AUDIO_REPEAT_VALIDATOR_FAILED", "hidden audio repeat validator failed"),
            "safe_cut_validator": ("SAFE_CUT_VALIDATOR_FAILED", "safe cut validator failed"),
            "subtitle_coverage_validator": ("SUBTITLE_COVERAGE_VALIDATOR_FAILED", "subtitle coverage validator failed"),
            "caption_alignment_gate": ("V21_CAPTION_SPOKEN_SPAN_ALIGNMENT_VALIDATOR", "caption spoken-span alignment validator failed"),
            "final_caption_visible_repeat_gate": ("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED", "final visible caption repeat gate failed"),
            "subtitle_style_validator": ("SUBTITLE_STYLE_VALIDATOR_FAILED", "subtitle style validator failed"),
            "rough_cut_quality_validator": ("ROUGH_CUT_QUALITY_VALIDATOR_FAILED", "rough cut quality validator failed"),
            "postwrite_material_validator": ("POSTWRITE_MATERIAL_VALIDATOR_FAILED", "postwrite material validator failed"),
            "semantic_final_review_validator": ("SEMANTIC_FINAL_REVIEW_VALIDATOR_FAILED", "semantic final review validator failed"),
            "quality_gate_report": ("V21_QUALITY_GATE_FAILED", "quality gate failed"),
        }
        pass_keys = {
            "final_repeat_validator": "final_repeat_gate_passed",
            "hidden_audio_repeat_validator": "hidden_audio_repeat_gate_passed",
            "safe_cut_validator": "safe_cut_boundary_gate_passed",
            "subtitle_coverage_validator": "subtitle_coverage_gate_passed",
            "caption_alignment_gate": "gate_passed",
            "final_caption_visible_repeat_gate": "gate_passed",
            "subtitle_style_validator": "prewrite_style_gate_ok",
            "rough_cut_quality_validator": "rough_cut_quality_gate_passed",
            "postwrite_material_validator": "postwrite_material_gate_ok",
            "semantic_final_review_validator": "semantic_final_review_validator_passed",
            "quality_gate_report": "gate_passed",
        }
        for section, (code, message) in mapping.items():
            payload = report.get(section) or {}
            if not payload.get(pass_keys[section], False):
                blockers.append(Blocker(code=code, message=message, layer="validate", context={"section": section, "report": payload}))
                emitted_codes.add(code)
                for detail_code in payload.get("blocker_codes") or []:
                    detail = str(detail_code or "")
                    if not detail or detail in emitted_codes:
                        continue
                    detail_context = {"section": section}
                    unresolved_ids = [str(item) for item in payload.get("unresolved_issue_ids") or [] if str(item)]
                    if not unresolved_ids:
                        semantic_payload = report.get("semantic_final_review_validator") or {}
                        unresolved_ids = [str(item) for item in semantic_payload.get("unresolved_issue_ids") or [] if str(item)]
                    if unresolved_ids:
                        detail_context["cluster_id"] = unresolved_ids[0]
                    blockers.append(
                        Blocker(
                            code=detail,
                            message="validator subgate failed",
                            layer="validate",
                            context=detail_context,
                        )
                    )
                    emitted_codes.add(detail)
        if not report.get("validators_read_only"):
            blockers.append(Blocker("VALIDATOR_MUTATED_INPUTS", "validator changed compiler/render/writer objects", "validate"))
        return blockers

    def _attach_final_caption_visible_repeat_gate(
        self,
        validator_report: dict[str, Any],
        captions,
    ) -> dict[str, Any]:
        report = dict(validator_report)
        visible_repeat_gate = build_final_caption_visible_repeat_gate(list(captions))
        repair_report = report.get("final_visible_caption_repair_report")
        if isinstance(repair_report, dict) and bool(repair_report.get("final_visible_repair_attempted")):
            visible_repeat_gate = dict(visible_repeat_gate)
            visible_repeat_gate["final_visible_repair_success"] = bool(repair_report.get("final_visible_repair_success"))
            visible_repeat_gate["final_visible_repair_unresolved_count"] = len(list(repair_report.get("final_visible_repair_unresolved") or []))
            visible_repeat_gate["final_visible_repair_final_timeline_counts"] = dict(repair_report.get("final_visible_repair_final_timeline_counts") or {})
            visible_repeat_gate["final_visible_effective_caption_count"] = int(repair_report.get("final_visible_effective_caption_count") or 0)
            visible_repeat_gate["caption_only_materialized_merge_count"] = int(repair_report.get("caption_only_materialized_merge_count") or 0)
            visible_repeat_gate["caption_only_consumed_caption_ids"] = list(repair_report.get("caption_only_consumed_caption_ids") or [])
            visible_repeat_gate["caption_only_materialized_merges"] = list(repair_report.get("caption_only_materialized_merges") or [])
        if isinstance(repair_report, dict) and bool(repair_report.get("final_visible_repair_attempted")) and not bool(repair_report.get("final_visible_repair_success")):
            blocker_codes = list(visible_repeat_gate.get("blocker_codes") or [])
            for code in ("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED", "V21_FINAL_VISIBLE_REPAIR_UNRESOLVED"):
                if code not in blocker_codes:
                    blocker_codes.append(code)
            visible_repeat_gate["gate_passed"] = False
            visible_repeat_gate["blocker_codes"] = blocker_codes
        report["final_caption_visible_repeat_gate"] = visible_repeat_gate
        previous_quality = report.get("quality_gate_report")
        quality_ok = True
        if isinstance(previous_quality, dict):
            base_ok = bool(previous_quality.get("ready_for_user_manual_qc_preconditions_passed", report.get("validator_report_ok")))
            quality_gate = build_quality_gate_report(
                effective_speed_gate=_normalize_effective_speed_prewrite_placeholder(previous_quality.get("effective_speed_gate")),
                final_repeat_convergence_gate=report.get("final_repeat_convergence_gate"),
                final_caption_visible_repeat_gate=visible_repeat_gate,
                semantic_adjudication_gate=report.get("semantic_final_review_validator"),
                visual_pacing_gate=report.get("visual_pacing_gate"),
                caption_alignment_gate=report.get("caption_alignment_gate"),
                ready_for_user_manual_qc_preconditions_passed=base_ok and bool(visible_repeat_gate.get("gate_passed")),
            )
            report["quality_gate_report"] = quality_gate
            quality_ok = bool(quality_gate.get("gate_passed"))
        report["validator_report_ok"] = bool(report.get("validator_report_ok")) and bool(visible_repeat_gate.get("gate_passed")) and quality_ok
        return report


def write_run_artifacts(run_report: RunReport, output_dir: Path, *, report_profile: str = "standard") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = str(report_profile or "standard").strip().lower()
    if profile not in {"minimal", "standard", "debug"}:
        profile = "standard"
    effective_profile = "debug" if profile == "standard" and run_report.status != "ok" else profile
    blocked_by_stage = str((run_report.blocker_report.summary or {}).get("stage") or "")
    blocked_by_codes = [blocker.code for blocker in (run_report.blocker_report.blockers if run_report.blocker_report else [])]

    def not_reached(stage: str, blocked_by: str | None = None) -> dict[str, Any]:
        return {
            "stage": stage,
            "status": "not_reached",
            "blocked_by_stage": blocked_by or blocked_by_stage,
            "blocked_by_codes": blocked_by_codes,
            "items": [],
        }

    def reached_after(stage: str) -> bool:
        if run_report.status != "blocked":
            return True
        order = {
            "ingest": 0,
            "decision": 1,
            "compiler": 2,
            "renderer": 3,
            "writer": 4,
            "validate": 5,
        }
        blocked_order = order.get(blocked_by_stage, 99)
        return order[stage] <= blocked_order

    final_timeline_payload: Any = run_report.final_timeline
    final_edl_payload: Any = [
        {
            "clip_id": segment.segment_id,
            "source_material_id": segment.source_material_id,
            "source_segment_id": segment.source_segment_id,
            "source_start_us": segment.source_start_us,
            "source_end_us": segment.source_end_us,
            "target_start_us": segment.target_start_us,
            "target_duration_us": segment.target_end_us - segment.target_start_us,
            "word_ids": segment.word_ids,
            "text": segment.text,
            "decision_ids": segment.decision_ids,
        }
        for segment in run_report.final_timeline
    ]
    if not reached_after("compiler") or (run_report.status == "blocked" and blocked_by_stage == "decision"):
        blocked_by = "SemanticDecisionPlanner" if blocked_by_stage == "decision" else blocked_by_stage
        final_timeline_payload = not_reached("FinalTimelineCompiler", blocked_by)
        final_edl_payload = not_reached("FinalTimelineCompiler", blocked_by)

    captions_payload: Any = run_report.captions
    if not reached_after("renderer") or (run_report.status == "blocked" and blocked_by_stage in {"decision", "compiler"}):
        captions_payload = not_reached("SubtitleRenderer")

    canonical_template_payload: Any = (run_report.material_write_plan or {}).get("canonical_caption_template") or {}
    material_write_plan_payload: Any = run_report.material_write_plan
    if not reached_after("writer") or (run_report.status == "blocked" and blocked_by_stage in {"decision", "compiler", "writer"} and not run_report.material_write_plan):
        canonical_template_payload = not_reached("CaptionMaterialWriter")
        material_write_plan_payload = not_reached("CaptionMaterialWriter")

    validator_payload: Any = run_report.validator_report
    postwrite_payload: Any = run_report.postwrite_report
    if not run_report.validator_report:
        validator_payload = not_reached("ReadOnlyValidators")
        postwrite_payload = not_reached("PostwriteVerification")
    if effective_profile != "debug":
        postwrite_payload = _compact_runtime_report_payload(postwrite_payload)

    decision_plan = run_report.decision_plan
    local_policy_decisions = []
    deepseek_decisions = []
    if decision_plan is not None:
        local_policy_decisions = [
            dataclass_to_dict(item)
            for item in [*decision_plan.decisions, *decision_plan.split_decisions]
            if str(getattr(item, "source", "")) == "local_policy"
        ]
        deepseek_decisions = [
            dataclass_to_dict(item)
            for item in [*decision_plan.decisions, *decision_plan.split_decisions]
            if str(getattr(item, "source", "")) == "deepseek_semantic_planner"
        ]
    resolved_semantic_rows = _resolved_semantic_decision_rows(decision_plan)
    semantic_report_payload = run_report.decision_plan.semantic_adjudication_report if run_report.decision_plan else {}
    artifacts = {
        "source_graph.json": run_report.source_graph,
        "edit_units.json": run_report.source_graph.edit_units if run_report.source_graph else [],
        "repeat_clusters.json": run_report.repeat_clusters,
        "decision_plan.json": run_report.decision_plan,
        "semantic_request_payloads.json": (run_report.decision_plan.semantic_request_payloads if run_report.decision_plan else []),
        "semantic_decisions.json": (run_report.decision_plan.semantic_decision_rows if run_report.decision_plan else []),
        "semantic_decisions.resolved.json": resolved_semantic_rows,
        "semantic_decision_cache.json": resolved_semantic_rows,
        "semantic_adjudication_report.json": semantic_report_payload,
        "deepseek_batch_request.json": semantic_report_payload.get("deepseek_batch_request") or {},
        "deepseek_batch_response.json": semantic_report_payload.get("deepseek_batch_response") or {},
        "deepseek_batch_error.json": semantic_report_payload.get("deepseek_batch_error_payload") or (
            {"error": semantic_report_payload.get("deepseek_batch_error")}
            if semantic_report_payload.get("deepseek_batch_error")
            else {}
        ),
        "final_timeline.json": final_timeline_payload,
        "final_edl.json": final_edl_payload,
        "captions.json": captions_payload,
        "canonical_caption_template.json": canonical_template_payload,
        "material_write_plan.json": material_write_plan_payload,
        "validator_report.json": validator_payload,
        "postwrite_report.json": postwrite_payload,
        "final_caption_visible_repeat_gate.json": (validator_payload or {}).get("final_caption_visible_repeat_gate") if isinstance(validator_payload, dict) else not_reached("FinalCaptionVisibleRepeatGate"),
        "final_visible_caption_repair_report.json": (validator_payload or {}).get("final_visible_caption_repair_report") if isinstance(validator_payload, dict) else not_reached("FinalVisibleCaptionRepair"),
        "quality_gate_report.json": (validator_payload or {}).get("quality_gate_report") if isinstance(validator_payload, dict) else not_reached("QualityGate"),
        "blocker_report.json": run_report.blocker_report,
        "decision_trace.json": run_report.decision_trace,
        "local_policy_decisions.json": local_policy_decisions,
        "deepseek_decisions.json": deepseek_decisions,
        "run_summary.json": build_run_summary(run_report),
        "run_report.json": run_report,
    }
    compressed_artifacts: dict[str, Any] = {}
    if effective_profile == "minimal":
        artifacts = {
            "blocker_report.json": run_report.blocker_report,
            "writeback_report.json": postwrite_payload,
        }
    elif effective_profile == "standard":
        compressed_artifacts = {
            "source_graph.json.gz": run_report.source_graph,
            "validator_report.json.gz": validator_payload,
        }
        for debug_name in (
            "run_report.json",
            "source_graph.json",
            "validator_report.json",
            "deepseek_batch_request.json",
            "deepseek_batch_response.json",
        ):
            artifacts.pop(debug_name, None)
        if not semantic_report_payload.get("deepseek_batch_error"):
            artifacts.pop("deepseek_batch_error.json", None)
    for name, payload in artifacts.items():
        (output_dir / name).write_text(json.dumps(dataclass_to_dict(payload), ensure_ascii=False, indent=2), "utf-8")
    for name, payload in compressed_artifacts.items():
        with gzip.open(output_dir / name, "wt", encoding="utf-8") as f:
            json.dump(dataclass_to_dict(payload), f, ensure_ascii=False, separators=(",", ":"))


COMPACT_RUNTIME_REPORT_DROP_KEYS = {
    "post_write_actual_draft_audit",
    "staged_post_write_actual_draft_audit",
    "postwrite_actual_draft_audit",
    "actual_draft_data",
    "draft_data",
}


def _compact_runtime_report_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    compact: dict[str, Any] = {}
    omitted: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if key in COMPACT_RUNTIME_REPORT_DROP_KEYS:
            omitted[key] = {
                "omitted": True,
                "reason": "debug_payload_available_only_in_debug_report_profile",
                "approx_json_bytes": len(json.dumps(dataclass_to_dict(value), ensure_ascii=False)),
            }
            continue
        compact[key] = value
    if omitted:
        compact["compact_report_omitted_debug_payloads"] = omitted
    return compact


def _resolved_semantic_decision_rows(decision_plan) -> list[dict[str, Any]]:
    if decision_plan is None:
        empty_rows: list[dict[str, Any]] = []
        return empty_rows
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in list(getattr(decision_plan, "semantic_decision_rows", []) or []):
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "")
        if not cluster_id or cluster_id in seen:
            continue
        if str(row.get("_blocker_code") or ""):
            continue
        rows.append(dict(row))
        seen.add(cluster_id)
    return rows


def build_run_summary(run_report: RunReport, *, commit_performed: bool = False, write_status: str = "") -> dict[str, Any]:
    validator = run_report.validator_report or {}
    style = validator.get("subtitle_style_validator") or {}
    coverage = validator.get("subtitle_coverage_validator") or {}
    final_repeat = validator.get("final_repeat_validator") or {}
    final_repeat_convergence = validator.get("final_repeat_convergence_gate") or {}
    final_caption_visible_repeat = validator.get("final_caption_visible_repeat_gate") or {}
    final_visible_caption_repair = validator.get("final_visible_caption_repair_report") or {}
    hidden = validator.get("hidden_audio_repeat_validator") or {}
    safe_cut = validator.get("safe_cut_validator") or {}
    rough_cut = validator.get("rough_cut_quality_validator") or {}
    visual_pacing = validator.get("visual_pacing_gate") or {}
    caption_alignment = validator.get("caption_alignment_gate") or {}
    quality_gate = validator.get("quality_gate_report") or {}
    semantic_gate = validator.get("semantic_final_review_validator") or {}
    effective_speed_gate = quality_gate.get("effective_speed_gate") if isinstance(quality_gate.get("effective_speed_gate"), dict) else {}
    postwrite = run_report.postwrite_report or {}
    blocker_summary = run_report.blocker_report.summary if run_report.blocker_report else {}
    ingest_metadata = blocker_summary.get("ingest_metadata") if isinstance(blocker_summary.get("ingest_metadata"), dict) else {}
    blockers = run_report.blocker_report.blockers if run_report.blocker_report else []
    mode = str((run_report.blocker_report.summary or {}).get("mode") or "")
    semantic_unresolved_count = int(run_report.decision_plan.semantic_unresolved_count if run_report.decision_plan else 0)
    semantic_report = run_report.decision_plan.semantic_adjudication_report if run_report.decision_plan else {}
    semantic_mode = str(
        postwrite.get("semantic_mode")
        or semantic_report.get("semantic_mode")
        or quality_gate.get("semantic_mode")
        or semantic_gate.get("semantic_mode")
        or "default"
    )
    semantic_request_count = int(
        quality_gate.get(
            "semantic_request_count",
            semantic_gate.get("semantic_request_count", semantic_report.get("semantic_request_count") or 0),
        )
        or 0
    )
    semantic_request_unresolved_count = int(
        quality_gate.get(
            "semantic_request_unresolved_count",
            semantic_gate.get("semantic_request_unresolved_count", semantic_report.get("semantic_request_unresolved_count") or 0),
        )
        or 0
    )
    fatal_semantic_issue_count = int(
        quality_gate.get(
            "fatal_semantic_issue_count",
            semantic_gate.get("fatal_semantic_issue_count", semantic_report.get("fatal_semantic_issue_count") or 0),
        )
        or 0
    )
    deepseek_provider_configured = bool(
        quality_gate.get(
            "deepseek_provider_configured",
            semantic_gate.get("deepseek_provider_configured", semantic_report.get("deepseek_provider_configured")),
        )
    )
    deepseek_provider_called_count = int(
        quality_gate.get(
            "deepseek_provider_called_count",
            semantic_gate.get("deepseek_provider_called_count", semantic_report.get("deepseek_provider_called_count") or 0),
        )
        or 0
    )
    deepseek_provider_error = str(
        quality_gate.get(
            "deepseek_provider_error",
            semantic_gate.get("deepseek_provider_error", semantic_report.get("deepseek_provider_error") or ""),
        )
        or ""
    )

    def semantic_int(key: str) -> int:
        return int(
            semantic_report.get(key)
            or semantic_gate.get(key)
            or quality_gate.get(key)
            or 0
        )

    def semantic_bool(key: str) -> bool:
        return bool(
            semantic_report.get(key)
            if key in semantic_report
            else semantic_gate.get(key)
            if key in semantic_gate
            else quality_gate.get(key)
        )

    def semantic_list(key: str) -> list[Any]:
        value = semantic_report.get(key) or semantic_gate.get(key) or quality_gate.get(key) or []
        return list(value) if isinstance(value, list) else []

    def semantic_str(key: str) -> str:
        return str(semantic_report.get(key) or semantic_gate.get(key) or quality_gate.get(key) or "")

    deepseek_provider_skipped_count = int(
        semantic_report.get("deepseek_provider_skipped_count")
        or semantic_gate.get("deepseek_provider_skipped_count")
        or quality_gate.get("deepseek_provider_skipped_count")
        or 0
    )
    deepseek_provider_skipped_reasons = dict(
        semantic_report.get("deepseek_provider_skipped_reasons")
        or semantic_gate.get("deepseek_provider_skipped_reasons")
        or quality_gate.get("deepseek_provider_skipped_reasons")
        or {}
    )
    semantic_decision_cache_used = bool(
        semantic_report.get("semantic_decision_cache_used")
        or semantic_gate.get("semantic_decision_cache_used")
        or quality_gate.get("semantic_decision_cache_used")
    )
    semantic_auto_route_count = int(
        max(
            int(quality_gate.get("semantic_auto_route_count") or 0),
            int(semantic_gate.get("semantic_auto_route_count") or 0),
            int(semantic_report.get("semantic_auto_route_count") or 0),
        )
    )
    semantic_local_decision_count = int(
        max(
            int(quality_gate.get("semantic_local_decision_count") or 0),
            int(semantic_gate.get("semantic_local_decision_count") or 0),
            int(semantic_report.get("semantic_local_decision_count") or 0),
        )
    )
    semantic_provider_required_count = int(
        max(
            int(quality_gate.get("semantic_provider_required_count") or 0),
            int(semantic_gate.get("semantic_provider_required_count") or 0),
            int(semantic_report.get("semantic_provider_required_count") or 0),
        )
    )
    deterministic_baseline_refused_count = int(
        max(
            int(quality_gate.get("deterministic_baseline_refused_count") or 0),
            int(semantic_gate.get("deterministic_baseline_refused_count") or 0),
            int(semantic_report.get("deterministic_baseline_refused_count") or 0),
        )
    )
    write_blocker_count = sum(1 for blocker in blockers if blocker.severity == "write_blocker")
    semantic_write_allowed = bool(
        run_report.decision_plan
        and run_report.decision_plan.semantic_unresolved_count == 0
        and run_report.decision_plan.write_allowed
    )
    validator_write_allowed = bool(validator.get("validator_report_ok"))
    writer_fallback_count = int((run_report.material_write_plan or {}).get("writer_fallback_count") or 0)
    ready_for_write = bool(run_report.status == "ok" and semantic_write_allowed and validator_write_allowed and writer_fallback_count == 0)
    sacrificial_postwrite_skip = bool(postwrite.get("postwrite_decrypt_skipped_for_sacrificial_draft"))
    source_stage_blocked = any(blocker.layer in {"operator", "ingest"} for blocker in blockers)
    audit = postwrite.get("post_write_actual_draft_audit") if isinstance(postwrite.get("post_write_actual_draft_audit"), dict) else {}
    post_write_audit_required_on_commit = True
    post_write_audit_commit_required = bool(commit_performed or postwrite.get("commit_performed") or postwrite.get("WRITE_SUCCESS") or postwrite.get("writeback_success"))
    post_write_audit_executed = bool(postwrite.get("post_write_actual_draft_audit_executed") or audit.get("executed"))
    post_write_audit_gate_passed = bool(postwrite.get("post_write_actual_draft_audit_gate_passed") or audit.get("gate_passed"))
    post_write_audit_ready = (not post_write_audit_commit_required) or (post_write_audit_executed and post_write_audit_gate_passed)
    ready_for_user_manual_qc = bool(postwrite.get("ready_for_user_manual_qc")) and post_write_audit_ready
    writeback_success = bool(postwrite.get("writeback_success"))
    effective_speed_min = postwrite.get("effective_speed_min", effective_speed_gate.get("effective_speed_min"))
    effective_speed_max = postwrite.get("effective_speed_max", effective_speed_gate.get("effective_speed_max"))
    effective_speed_not_applicable = bool(effective_speed_gate.get("not_applicable")) or (
        bool(effective_speed_gate.get("prewrite_pending"))
        and effective_speed_gate.get("effective_speed_min") is None
        and effective_speed_gate.get("effective_speed_max") is None
    )
    effective_speed_passed_raw = bool(effective_speed_gate.get("gate_passed") or postwrite.get("effective_speed_gate_passed"))
    effective_speed_has_bounds = effective_speed_min is not None and effective_speed_max is not None

    def postwrite_bool(flat_key: str, audit_key: str) -> bool:
        if flat_key in postwrite:
            return bool(postwrite.get(flat_key))
        return bool(audit.get(audit_key))

    def postwrite_int(flat_key: str, audit_key: str) -> int:
        if flat_key in postwrite:
            return int(postwrite.get(flat_key) or 0)
        return int(audit.get(audit_key) or 0)

    summary = {
        "status": run_report.status,
        "write_status": write_status or ("blocked" if run_report.status == "blocked" else "not_requested"),
        "commit_performed": bool(commit_performed),
        "single_source_graph_ok": bool(
            run_report.source_graph and run_report.source_graph.invariant_report.single_source_graph_ok and not source_stage_blocked
        ),
        "all_final_segments_have_word_ids": bool(run_report.final_timeline)
        and all(bool(segment.word_ids) for segment in run_report.final_timeline),
        "all_captions_derived_from_final_timeline": bool(coverage.get("all_captions_derived_from_final_timeline")),
        "all_materials_from_canonical_template": bool((run_report.material_write_plan or {}).get("canonical_caption_template_id")),
        "no_writer_fallback": bool((run_report.material_write_plan or {}).get("no_writer_fallback")),
        "writer_fallback_count": writer_fallback_count,
        "validators_readonly": bool(validator.get("validators_read_only")),
        "final_repeat_count": int(final_repeat.get("final_text_repeat_high_count") or 0)
        + int(final_repeat.get("final_text_repeat_medium_count") or 0)
        + int(final_repeat.get("final_cjk_short_repeat_fatal_count") or 0)
        + int(final_repeat.get("adjacent_modifier_semantic_redundancy_fatal_count") or 0)
        + int(final_repeat.get("final_target_repeat_high_count") or 0)
        + int(final_repeat.get("final_target_repeat_medium_count") or 0),
        "hidden_audio_repeat_count": int(hidden.get("word_timeline_hidden_repeat_count") or 0)
        + int(hidden.get("word_timeline_repeated_island_count") or 0)
        + int(hidden.get("final_spoken_text_short_repeat_fatal_count") or 0)
        + int(hidden.get("adjacent_modifier_semantic_redundancy_fatal_count") or 0),
        "cut_inside_word_count": int(safe_cut.get("cut_inside_word_count") or 0),
        "partial_multichar_cut_count": int(safe_cut.get("partial_multichar_cut_count") or 0),
        "giant_subtitle_count": int(style.get("giant_subtitle_count") or 0),
        "template_fingerprint_mismatch_count": int(style.get("template_fingerprint_mismatch_count") or 0),
        "content_schema_error_count": int(postwrite.get("content_schema_error_count") or 0),
        "caption_coverage_gap_count": len(coverage.get("missing_caption_segment_ids") or []),
        "rough_cut_quality_gate_passed": bool(rough_cut.get("rough_cut_quality_gate_passed")),
        "quality_gate_passed": bool(quality_gate.get("gate_passed")),
        "quality_gate_blocker_codes": list(quality_gate.get("blocker_codes") or []),
        "semantic_adjudication_gate_passed": bool(
            quality_gate.get("semantic_adjudication_gate_passed", semantic_gate.get("semantic_adjudication_gate_passed"))
        ),
        "semantic_request_count": semantic_request_count,
        "semantic_request_unresolved_count": semantic_request_unresolved_count,
        "fatal_semantic_issue_count": fatal_semantic_issue_count,
        "deepseek_provider_configured": deepseek_provider_configured,
        "deepseek_provider_called_count": deepseek_provider_called_count,
        "deepseek_provider_error": deepseek_provider_error,
        "deepseek_batch_enabled": semantic_bool("deepseek_batch_enabled"),
        "deepseek_batch_request_count": semantic_int("deepseek_batch_request_count"),
        "deepseek_batch_attempt_count": semantic_int("deepseek_batch_attempt_count"),
        "deepseek_batch_retry_count": semantic_int("deepseek_batch_retry_count"),
        "deepseek_batch_issue_count": semantic_int("deepseek_batch_issue_count"),
        "deepseek_batch_resolved_count": semantic_int("deepseek_batch_resolved_count"),
        "deepseek_batch_unresolved_count": semantic_int("deepseek_batch_unresolved_count"),
        "deepseek_batch_missing_issue_ids": semantic_list("deepseek_batch_missing_issue_ids"),
        "deepseek_batch_error": semantic_str("deepseek_batch_error"),
        "commit_reused_semantic_cache": semantic_bool("commit_reused_semantic_cache"),
        "semantic_cache_input_hash": semantic_str("semantic_cache_input_hash"),
        "semantic_cache_issue_count": semantic_int("semantic_cache_issue_count"),
        "semantic_cache_resolved_count": semantic_int("semantic_cache_resolved_count"),
        "semantic_cache_unresolved_count": semantic_int("semantic_cache_unresolved_count"),
        "deepseek_provider_not_called_reason": _deepseek_provider_not_called_reason(
            semantic_mode=semantic_mode,
            provider_configured=deepseek_provider_configured,
            provider_called_count=deepseek_provider_called_count,
            semantic_request_count=semantic_request_count,
            semantic_decision_cache_used=semantic_decision_cache_used,
        ),
        "deepseek_provider_skipped_count": deepseek_provider_skipped_count,
        "deepseek_provider_skipped_reasons": deepseek_provider_skipped_reasons,
        "semantic_decision_cache_used": semantic_decision_cache_used,
        "semantic_auto_route_count": semantic_auto_route_count,
        "semantic_local_decision_count": semantic_local_decision_count,
        "semantic_provider_required_count": semantic_provider_required_count,
        "deterministic_baseline_refused_count": deterministic_baseline_refused_count,
        "effective_speed_gate_passed": bool(effective_speed_passed_raw and effective_speed_has_bounds and not effective_speed_not_applicable),
        "effective_speed_not_applicable": effective_speed_not_applicable,
        "effective_speed_not_applicable_reason": str(
            effective_speed_gate.get("not_applicable_reason")
            or ("prewrite_source_binding_pending" if effective_speed_not_applicable else "")
        ),
        "effective_speed_min": effective_speed_min,
        "effective_speed_max": effective_speed_max,
        "effective_speed_drift_count": int(postwrite.get("effective_speed_drift_count") or (quality_gate.get("effective_speed_gate") or {}).get("effective_speed_drift_count") or 0),
        "safe_handle_policy_enabled": bool(postwrite.get("safe_handle_policy_enabled", effective_speed_gate.get("safe_handle_policy_enabled"))),
        "lead_handle_requested_count": int(postwrite.get("lead_handle_requested_count", effective_speed_gate.get("lead_handle_requested_count") or 0) or 0),
        "tail_handle_requested_count": int(postwrite.get("tail_handle_requested_count", effective_speed_gate.get("tail_handle_requested_count") or 0) or 0),
        "lead_handle_applied_count": int(postwrite.get("lead_handle_applied_count", effective_speed_gate.get("lead_handle_applied_count") or 0) or 0),
        "tail_handle_applied_count": int(postwrite.get("tail_handle_applied_count", effective_speed_gate.get("tail_handle_applied_count") or 0) or 0),
        "segments_with_no_lead_handle": int(postwrite.get("segments_with_no_lead_handle", effective_speed_gate.get("segments_with_no_lead_handle") or 0) or 0),
        "segments_with_no_tail_handle": int(postwrite.get("segments_with_no_tail_handle", effective_speed_gate.get("segments_with_no_tail_handle") or 0) or 0),
        "handle_blocked_count": int(postwrite.get("handle_blocked_count", effective_speed_gate.get("handle_blocked_count") or 0) or 0),
        "handle_blocked_reasons": dict(postwrite.get("handle_blocked_reasons", effective_speed_gate.get("handle_blocked_reasons") or {}) or {}),
        "final_repeat_convergence_gate_passed": bool(final_repeat_convergence.get("gate_passed")),
        "final_repeat_high_count_after_convergence": int(final_repeat_convergence.get("final_repeat_high_count_after") or 0),
        "final_repeat_dropped_segment_count": int(
            final_repeat_convergence.get("dropped_segment_count")
            or final_repeat_convergence.get("final_repeat_dropped_segment_count")
            or len(final_repeat_convergence.get("dropped_segment_indices") or [])
        ),
        "dropped_cluster_count": int(final_repeat_convergence.get("dropped_cluster_count") or len(final_repeat_convergence.get("dropped_cluster_ids") or [])),
        "dropped_segment_count": int(final_repeat_convergence.get("dropped_segment_count") or len(final_repeat_convergence.get("dropped_segment_indices") or [])),
        "dropped_cluster_ids": list(final_repeat_convergence.get("dropped_cluster_ids") or []),
        "dropped_segment_indices": list(final_repeat_convergence.get("dropped_segment_indices") or []),
        "final_caption_visible_repeat_gate_passed": bool(final_caption_visible_repeat.get("gate_passed")),
        "visible_repeat_candidate_count": int(final_caption_visible_repeat.get("visible_repeat_candidate_count") or 0),
        "containment_repeat_count": int(final_caption_visible_repeat.get("containment_repeat_count") or 0),
        "prefix_suffix_overlap_count": int(final_caption_visible_repeat.get("prefix_suffix_overlap_count") or 0),
        "ngram_repeat_count": int(final_caption_visible_repeat.get("ngram_repeat_count") or 0),
        "near_duplicate_visible_caption_count": int(final_caption_visible_repeat.get("near_duplicate_visible_caption_count") or 0),
        "modifier_redundancy_residual_count": int(final_caption_visible_repeat.get("modifier_redundancy_residual_count") or 0),
        "self_repair_aborted_phrase_count": int(final_caption_visible_repeat.get("self_repair_aborted_phrase_count") or 0),
        "dangling_prefix_suffix_count": int(final_caption_visible_repeat.get("dangling_prefix_suffix_count") or 0),
        "semantic_garbage_or_asr_suspect_count": int(final_caption_visible_repeat.get("semantic_garbage_or_asr_suspect_count") or 0),
        "cross_caption_semantic_containment_count": int(final_caption_visible_repeat.get("cross_caption_semantic_containment_count") or 0),
        "restart_repeat_visible_count": int(final_caption_visible_repeat.get("restart_repeat_visible_count") or 0),
        "final_visible_repair_attempted": bool(final_visible_caption_repair.get("final_visible_repair_attempted")),
        "final_visible_repair_success": bool(final_visible_caption_repair.get("final_visible_repair_success")),
        "final_visible_repair_action_count": int(final_visible_caption_repair.get("final_visible_repair_action_count") or 0),
        "final_visible_repair_initial_counts": dict(final_visible_caption_repair.get("final_visible_repair_initial_counts") or {}),
        "final_visible_repair_final_counts": dict(final_visible_caption_repair.get("final_visible_repair_final_counts") or {}),
        "final_visible_repair_final_timeline_counts": dict(final_visible_caption_repair.get("final_visible_repair_final_timeline_counts") or {}),
        "final_visible_effective_caption_count": int(final_visible_caption_repair.get("final_visible_effective_caption_count") or 0),
        "caption_only_materialized_merge_count": int(final_visible_caption_repair.get("caption_only_materialized_merge_count") or 0),
        "caption_only_consumed_caption_ids": list(final_visible_caption_repair.get("caption_only_consumed_caption_ids") or []),
        "final_visible_repair_unresolved": list(final_visible_caption_repair.get("final_visible_repair_unresolved") or []),
        "final_caption_visible_repeat_blocker_codes": list(final_caption_visible_repeat.get("blocker_codes") or []),
        "final_caption_visible_repeat_candidates": list(final_caption_visible_repeat.get("visible_repeat_candidates") or []),
        "modifier_redundancy_residual_candidates": list(final_caption_visible_repeat.get("modifier_redundancy_residual_candidates") or []),
        "self_repair_aborted_phrase_candidates": list(final_caption_visible_repeat.get("self_repair_aborted_phrase_candidates") or []),
        "dangling_prefix_suffix_candidates": list(final_caption_visible_repeat.get("dangling_prefix_suffix_candidates") or []),
        "semantic_garbage_or_asr_suspect_candidates": list(final_caption_visible_repeat.get("semantic_garbage_or_asr_suspect_candidates") or []),
        "cross_caption_semantic_containment_candidates": list(final_caption_visible_repeat.get("cross_caption_semantic_containment_candidates") or []),
        "restart_repeat_visible_candidates": list(final_caption_visible_repeat.get("restart_repeat_visible_candidates") or []),
        "visual_pacing_gate_passed": bool(visual_pacing.get("gate_passed")),
        "visual_pacing_executed": bool(visual_pacing.get("visual_pacing_executed")),
        "visual_pacing_merge_attempted_count": int(visual_pacing.get("visual_pacing_merge_attempted_count") or 0),
        "visual_pacing_merged_count": int(visual_pacing.get("visual_pacing_merged_count") or 0),
        "visual_merge_safety_gate_passed": bool(visual_pacing.get("visual_merge_safety_gate_passed")),
        "unsafe_merge_group_count": int(visual_pacing.get("unsafe_merge_group_count") or 0),
        "dropped_content_reintroduced_count": int(visual_pacing.get("dropped_content_reintroduced_count") or 0),
        "max_bridged_gap_us": int(visual_pacing.get("max_bridged_gap_us") or 0),
        "total_bridged_gap_us": int(visual_pacing.get("total_bridged_gap_us") or 0),
        "unspoken_bridge_ratio": float(visual_pacing.get("unspoken_bridge_ratio") or 0.0),
        "final_video_segment_count": int(visual_pacing.get("final_video_segment_count") or len(run_report.final_timeline)),
        "caption_count": int(visual_pacing.get("caption_count") or len(run_report.captions)),
        "visual_short_segment_count_lt_1200ms": int(visual_pacing.get("visual_short_segment_count_lt_1200ms") or 0),
        "visual_short_segment_count_lt_1200ms_before": int(visual_pacing.get("visual_short_segment_count_lt_1200ms_before") or 0),
        "visual_short_segment_count_lt_1200ms_after": int(visual_pacing.get("visual_short_segment_count_lt_1200ms_after") or 0),
        "visual_short_segment_count_lt_1200ms_after_blocking": int(
            visual_pacing.get("visual_short_segment_count_lt_1200ms_after_blocking") or 0
        ),
        "semantic_bridge_short_segment_count": int(visual_pacing.get("semantic_bridge_short_segment_count") or 0),
        "semantic_bridge_short_segment_details": list(visual_pacing.get("semantic_bridge_short_segment_details") or []),
        "semantic_bridge_reason_counts": dict(visual_pacing.get("semantic_bridge_reason_counts") or {}),
        "semantic_bridge_cap": int(visual_pacing.get("semantic_bridge_cap") or 0),
        "semantic_bridge_safe_merge_candidate_count": int(visual_pacing.get("semantic_bridge_safe_merge_candidate_count") or 0),
        "semantic_bridge_safe_merge_candidates": list(visual_pacing.get("semantic_bridge_safe_merge_candidates") or []),
        "cuts_per_minute": float(visual_pacing.get("cuts_per_minute") or 0.0),
        "max_cuts_in_5s": int(visual_pacing.get("max_cuts_in_5s") or 0),
        "burst_cut_count": int(visual_pacing.get("burst_cut_count") or 0),
        "cut_density_gate_enabled": bool(visual_pacing.get("cut_density_gate_enabled")),
        "cut_density_gate_passed": bool(visual_pacing.get("cut_density_gate_passed")),
        "cut_density_thresholds": dict(visual_pacing.get("cut_density_thresholds") or {}),
        "visual_pacing_allowed_short_segment_threshold": int(visual_pacing.get("visual_pacing_allowed_short_segment_threshold") or 0),
        "visual_pacing_blocker_codes": list(visual_pacing.get("visual_pacing_blocker_codes") or []),
        "residual_visual_short_segments": list(visual_pacing.get("residual_visual_short_segments") or []),
        "hidden_repeat_cleanup_dropped_word_count": int(visual_pacing.get("visual_pacing_hidden_repeat_dropped_word_count") or 0),
        "boundary_overlap_cleanup_dropped_word_count": int(visual_pacing.get("visual_pacing_boundary_overlap_dropped_word_count") or 0),
        "median_segment_duration_us": int(visual_pacing.get("median_segment_duration_us") or 0),
        "p10_segment_duration_us": int(visual_pacing.get("p10_segment_duration_us") or 0),
        "caption_per_video_segment_ratio": float(visual_pacing.get("caption_per_video_segment_ratio") or 0.0),
        "caption_alignment_gate_passed": bool(caption_alignment.get("gate_passed")),
        "caption_gui_track_gate_passed": bool(caption_alignment.get("caption_gui_track_gate_passed")),
        "subtitle_readability_gate_passed": bool(caption_alignment.get("subtitle_readability_gate_passed")),
        "visible_caption_track_count": int(caption_alignment.get("visible_caption_track_count") or 0),
        "caption_lane_count": int(caption_alignment.get("caption_lane_count") or 0),
        "orphan_caption_count": int(caption_alignment.get("orphan_caption_count") or 0),
        "floating_caption_count": int(caption_alignment.get("floating_caption_count") or 0),
        "caption_render_order_stable": bool(caption_alignment.get("caption_render_order_stable")),
        "caption_outside_video_count": int(caption_alignment.get("caption_outside_video_count") or 0),
        "caption_overlap_count": int(caption_alignment.get("caption_overlap_count") or 0),
        "caption_too_short_count": int(caption_alignment.get("caption_too_short_count") or 0),
        "one_char_caption_count": int(caption_alignment.get("one_char_caption_count") or 0),
        "residual_too_short_captions": list(caption_alignment.get("residual_too_short_captions") or []),
        "residual_one_char_captions": list(caption_alignment.get("residual_one_char_captions") or []),
        "caption_without_video_container_count": int(caption_alignment.get("caption_without_video_container_count") or 0),
        "caption_without_container_count": int(caption_alignment.get("caption_without_video_container_count") or 0),
        "caption_cross_primary_window_count": int(caption_alignment.get("caption_cross_primary_window_count") or 0),
        "captions_le_3_chars": int(caption_alignment.get("captions_le_3_chars") or 0),
        "captions_le_3_chars_cap": int(caption_alignment.get("captions_le_3_chars_cap") or 0),
        "subtitle_interval_too_short_count": int(caption_alignment.get("subtitle_interval_too_short_count") or 0),
        "subtitle_interval_too_long_count": int(caption_alignment.get("subtitle_interval_too_long_count") or 0),
        "subtitle_hard_max_char_count": int(caption_alignment.get("subtitle_hard_max_char_count") or 0),
        "caption_density_per_minute": float(caption_alignment.get("caption_density_per_minute") or 0.0),
        "max_captions_in_5s": int(caption_alignment.get("max_captions_in_5s") or 0),
        "caption_burst_density_count": int(caption_alignment.get("caption_burst_density_count") or 0),
        "prewrite_style_gate_ok": bool(style.get("prewrite_style_gate_ok")),
        "postwrite_style_gate_ok": bool(postwrite.get("postwrite_material_gate_ok")),
        "postwrite_decrypt_ok": bool(postwrite.get("postwrite_decrypt_ok")),
        "postwrite_mode": str(postwrite.get("postwrite_mode") or ""),
        "sacrificial_write_override_used": bool(postwrite.get("sacrificial_write_override_used")),
        "postwrite_decrypt_skipped_for_sacrificial_draft": sacrificial_postwrite_skip,
        "postwrite_decrypt_skip_reason": str(postwrite.get("postwrite_decrypt_skip_reason") or ""),
        "ready_for_user_manual_qc": ready_for_user_manual_qc,
        "writeback_success": writeback_success,
        "WRITE_SUCCESS": bool(postwrite.get("WRITE_SUCCESS")),
        "ENCRYPT_SUCCESS": bool(postwrite.get("ENCRYPT_SUCCESS")),
        "post_write_actual_draft_audit_required_on_commit": post_write_audit_required_on_commit,
        "post_write_actual_draft_audit_executed": post_write_audit_executed,
        "post_write_actual_draft_audit_gate_passed": post_write_audit_gate_passed,
        "post_write_actual_draft_audit_blocker_codes": list(
            postwrite.get("post_write_actual_draft_audit_blocker_codes")
            or audit.get("blocker_codes")
            or []
        ),
        "post_write_actual_draft_audit_failure_reasons": list(
            postwrite.get("post_write_actual_draft_audit_failure_reasons")
            or audit.get("failure_reasons")
            or []
        ),
        "post_write_actual_draft_loaded": bool(postwrite.get("post_write_actual_draft_loaded") or audit.get("actual_draft_loaded")),
        "post_write_actual_video_rows_match_plan": bool(postwrite.get("post_write_actual_video_rows_match_plan") or audit.get("actual_video_rows_match_plan")),
        "post_write_actual_caption_rows_match_plan": bool(postwrite.get("post_write_actual_caption_rows_match_plan") or audit.get("actual_caption_rows_match_plan")),
        "post_write_expected_caption_rows_present": postwrite_bool("post_write_expected_caption_rows_present", "expected_caption_rows_present"),
        "post_write_actual_has_no_extra_caption_like_text_segments": postwrite_bool(
            "post_write_actual_has_no_extra_caption_like_text_segments",
            "actual_has_no_extra_caption_like_text_segments",
        ),
        "post_write_actual_caption_rows_exact_match_plan": postwrite_bool(
            "post_write_actual_caption_rows_exact_match_plan",
            "actual_caption_rows_exact_match_plan",
        ),
        "post_write_actual_text_residue_gate_passed": postwrite_bool("post_write_actual_text_residue_gate_passed", "actual_text_residue_gate_passed"),
        "post_write_actual_audio_coverage_gate_passed": postwrite_bool("post_write_actual_audio_coverage_gate_passed", "actual_audio_coverage_gate_passed"),
        "post_write_actual_visible_text_repeat_gate_passed": postwrite_bool(
            "post_write_actual_visible_text_repeat_gate_passed",
            "actual_visible_text_repeat_gate_passed",
        ),
        "post_write_actual_text_segment_count": postwrite_int("post_write_actual_text_segment_count", "actual_text_segment_count"),
        "post_write_generated_caption_segment_count": postwrite_int("post_write_generated_caption_segment_count", "generated_caption_segment_count"),
        "post_write_preserved_non_subtitle_count": postwrite_int("post_write_preserved_non_subtitle_count", "preserved_non_subtitle_count"),
        "post_write_old_subtitle_residue_count": postwrite_int("post_write_old_subtitle_residue_count", "old_subtitle_residue_count"),
        "post_write_orphan_text_segment_count": postwrite_int("post_write_orphan_text_segment_count", "orphan_text_segment_count"),
        "post_write_text_after_final_video_end_count": postwrite_int("post_write_text_after_final_video_end_count", "text_after_final_video_end_count"),
        "post_write_floating_caption_count": postwrite_int("post_write_floating_caption_count", "floating_caption_count"),
        "post_write_audio_coverage_failure_count": postwrite_int("post_write_audio_coverage_failure_count", "audio_coverage_failure_count"),
        "post_write_heard_but_uncaptioned_word_count": postwrite_int("post_write_heard_but_uncaptioned_word_count", "heard_but_uncaptioned_word_count"),
        "post_write_dropped_but_reintroduced_word_count": postwrite_int(
            "post_write_dropped_but_reintroduced_word_count",
            "dropped_but_reintroduced_word_count",
        ),
        "post_write_actual_visible_repeat_candidate_count": postwrite_int(
            "post_write_actual_visible_repeat_candidate_count",
            "actual_visible_repeat_candidate_count",
        ),
        "jianying_canonical_timeline_sync_gate_passed": postwrite_bool(
            "jianying_canonical_timeline_sync_gate_passed",
            "jianying_canonical_timeline_sync_gate_passed",
        ),
        "final_video_end_us": postwrite_int("final_video_end_us", "final_video_end_us"),
        "max_caption_end_us": postwrite_int("max_caption_end_us", "max_caption_end_us"),
        "captions_after_final_video_end_count": postwrite_int(
            "captions_after_final_video_end_count",
            "captions_after_final_video_end_count",
        ),
        "post_write_video_target_gap_count_gt_300ms": postwrite_int(
            "post_write_video_target_gap_count_gt_300ms",
            "post_write_video_target_gap_count_gt_300ms",
        ),
        "post_write_total_video_target_gap_us": postwrite_int(
            "post_write_total_video_target_gap_us",
            "post_write_total_video_target_gap_us",
        ),
        "caption_video_drift_count": postwrite_int("caption_video_drift_count", "caption_video_drift_count"),
        "max_caption_video_drift_us": postwrite_int("max_caption_video_drift_us", "max_caption_video_drift_us"),
        "split_caption_container_mismatch_count": postwrite_int(
            "split_caption_container_mismatch_count",
            "split_caption_container_mismatch_count",
        ),
        "caption_crosses_video_split_gap_count": postwrite_int(
            "caption_crosses_video_split_gap_count",
            "caption_crosses_video_split_gap_count",
        ),
        "caption_words_not_covered_by_actual_video_count": postwrite_int(
            "caption_words_not_covered_by_actual_video_count",
            "caption_words_not_covered_by_actual_video_count",
        ),
        "actual_has_no_extra_caption_like_text_segments": postwrite_bool(
            "post_write_actual_has_no_extra_caption_like_text_segments",
            "actual_has_no_extra_caption_like_text_segments",
        ),
        "actual_caption_rows_exact_match_plan": postwrite_bool("post_write_actual_caption_rows_exact_match_plan", "actual_caption_rows_exact_match_plan"),
        "actual_text_residue_gate_passed": postwrite_bool("post_write_actual_text_residue_gate_passed", "actual_text_residue_gate_passed"),
        "actual_audio_coverage_gate_passed": postwrite_bool("post_write_actual_audio_coverage_gate_passed", "actual_audio_coverage_gate_passed"),
        "actual_visible_text_repeat_gate_passed": postwrite_bool(
            "post_write_actual_visible_text_repeat_gate_passed",
            "actual_visible_text_repeat_gate_passed",
        ),
        "actual_audio_coverage_failure_count": postwrite_int("post_write_audio_coverage_failure_count", "audio_coverage_failure_count"),
        "heard_but_uncaptioned_word_count": postwrite_int("post_write_heard_but_uncaptioned_word_count", "heard_but_uncaptioned_word_count"),
        "dropped_but_reintroduced_word_count": postwrite_int(
            "post_write_dropped_but_reintroduced_word_count",
            "dropped_but_reintroduced_word_count",
        ),
        "old_subtitle_residue_count": postwrite_int("post_write_old_subtitle_residue_count", "old_subtitle_residue_count"),
        "orphan_text_segment_count": postwrite_int("post_write_orphan_text_segment_count", "orphan_text_segment_count"),
        "text_after_final_video_end_count": postwrite_int("post_write_text_after_final_video_end_count", "text_after_final_video_end_count"),
        "post_write_actual_effective_speed_gate_passed": bool(postwrite.get("post_write_actual_effective_speed_gate_passed") or audit.get("actual_effective_speed_gate_passed")),
        "post_write_actual_visual_pacing_gate_passed": bool(postwrite.get("post_write_actual_visual_pacing_gate_passed") or audit.get("actual_visual_pacing_gate_passed")),
        "post_write_actual_caption_gui_readability_gate_passed": bool(
            postwrite.get("post_write_actual_caption_gui_readability_gate_passed")
            or audit.get("actual_caption_gui_readability_gate_passed")
        ),
        "post_write_actual_final_caption_visible_repeat_gate_passed": bool(
            postwrite.get("post_write_actual_final_caption_visible_repeat_gate_passed")
            or audit.get("actual_final_caption_visible_repeat_gate_passed")
        ),
        "post_write_actual_caption_alignment_gate_passed": bool(postwrite.get("post_write_actual_caption_alignment_gate_passed") or audit.get("actual_caption_alignment_gate_passed")),
        "draft_dir": str(postwrite.get("draft_dir") or ""),
        "jy_draftc_path": str(postwrite.get("jy_draftc_path") or ""),
        "jy_install_dir": str(postwrite.get("jy_install_dir") or ""),
        "postwrite_decrypt_cwd": str(postwrite.get("postwrite_decrypt_cwd") or ""),
        "draft_content_path": str(postwrite.get("draft_content_path") or ""),
        "only_specified_draft_written": bool(postwrite.get("only_specified_draft_written")),
        "source_segment_template_exact_match_count": int(postwrite.get("source_segment_template_exact_match_count") or 0),
        "source_segment_template_rebind_count": int(postwrite.get("source_segment_template_rebind_count") or 0),
        "source_segment_template_missing_count": int(postwrite.get("source_segment_template_missing_count") or 0),
        "source_segment_template_ambiguous_count": int(postwrite.get("source_segment_template_ambiguous_count") or 0),
        "resolved_template_map_count": int(postwrite.get("resolved_template_map_count") or len(run_report.resolved_template_map or {})),
        "current_draft_video_track_count": int(postwrite.get("current_draft_video_track_count") or 0),
        "current_draft_video_segment_count": int(postwrite.get("current_draft_video_segment_count") or 0),
        "current_draft_video_material_count": int(postwrite.get("current_draft_video_material_count") or 0),
        "current_source_template_candidate_count": int(postwrite.get("current_source_template_candidate_count") or 0),
        "speech_timeline_provider": str(blocker_summary.get("speech_timeline_provider") or ingest_metadata.get("speech_timeline_provider") or ""),
        "speech_timeline_granularity": str(blocker_summary.get("speech_timeline_granularity") or ingest_metadata.get("speech_timeline_granularity") or ""),
        "speech_timeline_precision": str(blocker_summary.get("speech_timeline_precision") or ingest_metadata.get("speech_timeline_precision") or ""),
        "speech_timeline_can_cut_inside_caption": bool(
            blocker_summary.get("speech_timeline_can_cut_inside_caption")
            or ingest_metadata.get("speech_timeline_can_cut_inside_caption")
        ),
        "word_timeline_count": int(ingest_metadata.get("word_timeline_count") or (len(run_report.source_graph.words) if run_report.source_graph else 0)),
        "word_timeline_count_source": "ingest_metadata" if ingest_metadata.get("word_timeline_count") is not None else "source_graph",
        "semantic_unresolved_count": semantic_unresolved_count,
        "semantic_mode": semantic_mode,
        "semantic_decisions_generated_from_current_draft": bool(postwrite.get("semantic_decisions_generated_from_current_draft")),
        "semantic_decisions_reused_from_old_draft": bool(postwrite.get("semantic_decisions_reused_from_old_draft")),
        "semantic_review_blocker_count": semantic_unresolved_count,
        "write_blocker_count": write_blocker_count,
        "requires_human_review": bool(semantic_unresolved_count),
        "semantic_write_allowed": semantic_write_allowed,
        "validator_write_allowed": validator_write_allowed,
        "validator_report_ok": validator_write_allowed,
        "write_allowed": ready_for_write,
        "ready_for_write": ready_for_write,
        "READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT": ready_for_write,
        "dry_run_continued_for_discovery": bool(mode == "dry-run" and semantic_unresolved_count > 0 and run_report.final_timeline),
        "commit_only_after_all_validators": bool(
            (not commit_performed)
            or (
                validator.get("validator_report_ok")
                and postwrite.get("postwrite_material_gate_ok")
                and postwrite.get("writeback_success")
                and (postwrite.get("postwrite_decrypt_ok") or sacrificial_postwrite_skip)
                and post_write_audit_ready
            )
        ),
        "blocker_count": len(blockers),
        "blocker_codes": [blocker.code for blocker in blockers],
        "fatal_blocker": blockers[0].code if blockers else None,
        "rough_cut_quality": postwrite.get("rough_cut_quality") or rough_cut,
    }
    return summary


def _deepseek_provider_not_called_reason(
    *,
    semantic_mode: str,
    provider_configured: bool,
    provider_called_count: int,
    semantic_request_count: int,
    semantic_decision_cache_used: bool = False,
) -> str:
    if provider_called_count > 0:
        return ""
    if semantic_decision_cache_used:
        return "semantic_decision_cache_used"
    if semantic_mode not in {"auto", "deepseek"}:
        return f"semantic_mode={semantic_mode}"
    if not provider_configured:
        if semantic_mode == "auto" and semantic_request_count == 0:
            return "no_provider_required"
        return "deepseek_provider_not_configured"
    if semantic_request_count == 0:
        return "no_provider_required" if semantic_mode == "auto" else "no_semantic_requests"
    return "deepseek_provider_not_called_with_pending_semantic_requests"


def _normalize_effective_speed_prewrite_placeholder(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "gate_passed": False,
            "blocker_codes": [],
            "prewrite_pending": True,
            "not_applicable": True,
            "not_applicable_reason": "prewrite_source_binding_pending",
        }
    normalized = dict(payload)
    if (
        bool(normalized.get("prewrite_pending"))
        and normalized.get("effective_speed_min") is None
        and normalized.get("effective_speed_max") is None
    ):
        normalized["gate_passed"] = False
        normalized["not_applicable"] = True
        normalized.setdefault("not_applicable_reason", "prewrite_source_binding_pending")
    return normalized
