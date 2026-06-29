from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V21_SRC = ROOT / "src" / "aroll_v21"
V21_ENTRY = ROOT / "run_aroll_v21_operator.ps1"


def v21_text() -> str:
    return "\n".join([*(path.read_text("utf-8") for path in V21_SRC.rglob("*.py")), V21_ENTRY.read_text("utf-8")])


class ArollV21NoArchitectureDriftTests(unittest.TestCase):
    def test_no_v20_patch_modules_or_patch_entrypoints(self) -> None:
        text = v21_text()
        for token in (
            "aroll_phase4e_full_aroll",
            "aroll_uat_full",
            "aroll_downstream_repair_pipeline",
            "aroll_repair_applier",
            "aroll_safe_cut_boundary_resolver",
            "material_text_rows",
            "run_downstream_repair_pipeline",
            "apply_repair_proposals",
            "resolve_safe_cut_boundaries",
        ):
            self.assertNotIn(token, text)

    def test_no_fixup_or_downstream_style_logic_in_v21(self) -> None:
        text = v21_text().lower()
        for token in ("auto_fix", "fixup", "downstream_repair", "safe_cut_expand", "validator repair", "writer fallback"):
            self.assertNotIn(token, text)

    def test_fallback_mentions_are_only_no_writer_contract_fields_or_block_messages(self) -> None:
        lines = [line.strip() for line in v21_text().splitlines() if "fallback" in line.lower()]
        for line in lines:
            lowered = line.lower()
            self.assertTrue(
                "no_writer_fallback" in lowered
                or "writer_fallback_count" in lowered
                or "without fallback" in lowered
                or "fallback is forbidden" in lowered
                or "fallback_policy" in lowered
                or "fail-closed fallback" in lowered,
                line,
            )

    def test_final_caption_visible_repeat_has_explicit_quality_layers(self) -> None:
        root = V21_SRC / "quality" / "final_caption_visible"
        for name in ("detector.py", "classifier.py", "policy.py", "repair_signal.py", "gate.py"):
            self.assertTrue((root / name).exists(), name)
        entry = (V21_SRC / "quality" / "final_caption_visible_repeat.py").read_text("utf-8")
        for token in (
            "detect_final_caption_visible_evidence",
            "classify_final_caption_visible_evidence",
            "build_final_caption_visible_policy",
            "build_final_caption_visible_repair_signal",
            "build_final_caption_visible_gate_report",
        ):
            self.assertIn(token, entry)
        policy = (root / "policy.py").read_text("utf-8")
        for verdict in ("BLOCKER_FATAL", "REPAIRABLE_FATAL", "WARNING", "ALLOW", "HUMAN_REVIEW"):
            self.assertIn(verdict, policy)

    def test_final_visible_repair_has_transaction_pipeline_seed(self) -> None:
        root = V21_SRC / "quality" / "final_visible_repair"
        context = (root / "context.py").read_text("utf-8")
        pipeline = (root / "pipeline.py").read_text("utf-8")
        proposal_apply = (root / "proposal_apply.py").read_text("utf-8")
        registry = (root / "registry.py").read_text("utf-8")
        loop_state = (root / "loop_state.py").read_text("utf-8")
        entry = (V21_SRC / "quality" / "final_visible_caption_repair.py").read_text("utf-8")
        leading = (root / "rules" / "leading_filler.py").read_text("utf-8")
        word_span = (root / "rules" / "word_span_edit.py").read_text("utf-8")
        caption_fragment = (root / "rules" / "caption_fragment.py").read_text("utf-8")
        final_repeat_caption = (root / "rules" / "final_repeat_caption.py").read_text("utf-8")
        connector = (root / "rules" / "connector_intrusion.py").read_text("utf-8")
        short_residual = (root / "rules" / "short_residual.py").read_text("utf-8")
        pre_visible = (root / "rules" / "pre_visible_semantic_junk.py").read_text("utf-8")
        source_boundary = (root / "rules" / "source_boundary_prefix.py").read_text("utf-8")
        restart_repeat = (root / "rules" / "restart_repeat.py").read_text("utf-8")
        caption_only_merge = (root / "rules" / "caption_only_merge.py").read_text("utf-8")
        for token in (
            "source_graph",
            "render_captions",
            "repair_state_signature",
        ):
            self.assertIn(token, context)
        for token in (
            "FinalVisibleRepairTransaction",
            "FinalVisibleRepairRuleOutcome",
            "FinalVisibleRepairPipelineResult",
            "ProposalRepairRule",
            "StepRepairRule",
            "run_final_visible_repair_pipeline_once",
            "unresolved_rule_name",
        ):
            self.assertIn(token, pipeline)
        for token in (
            "FinalVisibleRepairLoopState",
            "consume_pipeline_result",
        ):
            self.assertIn(token, loop_state)
            self.assertIn(token, entry)
        for token in (
            "FinalVisibleRepairRuleCallbacks",
            "build_final_visible_repair_rule_registry",
            "build_gate_candidate_repair_rules",
        ):
            self.assertIn(token, registry)
            self.assertIn(token, entry)
        self.assertIn("FinalVisibleRepairRuleRegistry", registry)
        for token in (
            "CaptionOnlyFinalizerRule",
            "FinalVisibleRepairRuleOutcome",
        ):
            self.assertIn(token, caption_only_merge)
        self.assertIn("CaptionOnlyFinalizerRule", registry)
        self.assertIn("run_final_visible_repair_pipeline_once", entry)
        self.assertIn("rule_registry.proposal_transaction_rules", entry)
        self.assertIn("rule_registry.open_tail_transaction_rules", entry)
        self.assertIn("rule_registry.caption_only_finalizer_rules", entry)
        for token in (
            "RenderCallbackAdapter",
            "apply_timeline_repair_proposal_as_step",
            "build_caption_span_drop_proposal",
            "repair_boundary_restart_with_proposal",
            "repair_repeated_island_with_proposal",
            "proposal_unresolved",
            "proposal_action",
        ):
            self.assertIn(token, proposal_apply)
        for token in (
            "repair_contained_short_fragment_with_proposal",
            "repair_self_repair_aborted_phrase_with_proposal",
            "repair_short_aborted_prefix_caption_with_proposal",
            "repair_open_tail_short_caption_with_next",
            "repair_fatal_tiny_caption_with_proposal",
            "short_aborted_prefix_candidate",
            "open_tail_short_caption_should_merge",
            "COMMON_CLOSED_DE_PHRASES",
        ):
            self.assertIn(token, caption_fragment)
        self.assertIn("_caption_fragment_rules.repair_contained_short_fragment_with_proposal", entry)
        self.assertIn("_caption_fragment_rules.repair_self_repair_aborted_phrase_with_proposal", entry)
        self.assertIn("_caption_fragment_rules.repair_short_aborted_prefix_caption_with_proposal", entry)
        self.assertIn("_caption_fragment_rules.repair_open_tail_short_caption_with_next", entry)
        self.assertIn("_caption_fragment_rules.repair_fatal_tiny_caption_with_proposal", entry)
        self.assertIn("partial(", entry)
        for token in (
            "repair_caption_level_final_repeat_aborted_containment",
            "final_repeat_caption_rows",
            "caption_level_final_repeat_aborted_drop_caption_id",
            "caption_level_containment_match",
            "relaxed_containment_text",
            "final_target_repeat_caption_containment",
        ):
            self.assertIn(token, final_repeat_caption)
        self.assertIn(
            "_final_repeat_caption_rules.repair_caption_level_final_repeat_aborted_containment",
            entry,
        )
        self.assertIn("LeadingFillerGapRule", registry)
        for token in (
            "ConnectorSingleWordIntrusionRule",
            "ConnectorFillerRestartRule",
            "RepeatedObjectHeadTailRule",
            "SubjectPrefixCompletedPredicateRestartRule",
        ):
            self.assertIn(token, connector)
            self.assertIn(token, registry)
        self.assertIn("ShortRepairResidualRule", short_residual)
        self.assertIn("ShortRepairResidualRule", registry)
        for token in (
            "PreVisibleSemanticJunkCandidateRule",
            "IsolatedSemanticJunkCaptionRule",
        ):
            self.assertIn(token, pre_visible)
            self.assertIn(token, registry)
        for token in (
            "OmittedLegalReduplicationRule",
            "SourceBoundaryPrefixGapRule",
            "SourceBoundaryCompoundSuffixRule",
            "SourceBoundaryTruncatedCompoundTailRule",
        ):
            self.assertIn(token, source_boundary)
            self.assertIn(token, registry)
        self.assertIn("GateCandidateRepairRule", restart_repeat)
        self.assertIn("GateCandidateRepairRule", registry)
        self.assertNotIn("_configure_final_visible_rule_modules", entry)
        self.assertNotIn(".configure_rule_dependencies", entry)
        for rule_path in sorted((root / "rules").glob("*.py")):
            rule_text = rule_path.read_text("utf-8")
            self.assertNotIn("configure_rule_dependencies", rule_text, rule_path.name)
            self.assertNotIn("globals().update", rule_text, rule_path.name)

        for token in (
            "_segment_with_word_ids",
            "_drop_contiguous_word_ids_from_timeline",
            "_safe_merge_segments",
        ):
            self.assertIn(token, word_span)
        for token in (
            "class _RenderCallbackAdapter",
            "def _repair_boundary_restart_with_proposal",
            "def _repair_repeated_island_with_proposal",
            "def _caption_span_drop_proposal",
            "def _apply_caption_span_drop_proposal",
            "def _repair_contained_short_fragment_with_proposal",
            "def _repair_self_repair_aborted_phrase_with_proposal",
            "def _repair_short_aborted_prefix_caption_with_proposal",
            "def _repair_open_tail_short_caption_with_next",
            "def _repair_fatal_tiny_caption_with_proposal",
            "def _short_aborted_prefix_candidate",
            "def _open_tail_short_caption_should_merge",
            "TimelineRepairProposal(",
            "build_tiny_caption_classification_report",
            "build_final_repeat_gate_report",
            "def _repair_caption_level_final_repeat_aborted_containment",
            "def _final_repeat_caption_rows",
            "def _caption_level_final_repeat_aborted_drop_caption_id",
            "def _caption_level_containment_match",
            "def _relaxed_containment_text",
            "CONTAINED_SHORT_FRAGMENT_OPEN_TAIL_CHARS",
            "OPEN_TAIL_SHORT_CAPTION_MAX_CHARS",
            "SHORT_ABORTED_PREFIX_MAX_CHARS",
            "COMMON_CLOSED_DE_PHRASES",
            "connector_intrusion_step =",
            "connector_restart_step =",
            "repeated_object_head_step =",
            "subject_prefix_restart_step =",
            "residual_step = _repair_short_repair_residual_segments",
            "pre_semantic_junk_step =",
            "omitted_reduplication_step =",
            "source_prefix_step =",
            "compound_step =",
            "truncated_tail_step =",
            "junk_step =",
            "step = _repair_next_issue(",
            "open_tail_merge_step =",
            "actions.append(",
            "repeated_island_step,",
            "boundary_restart_step,",
            "containment_fragment_step,",
            "self_repair_step,",
            "short_aborted_prefix_step,",
            "tiny_residual_step,",
            "final_caption_only_actions =",
            "subject_prefix_caption_actions =",
            "same_subtitle_short_tail_actions =",
        ):
            self.assertNotIn(token, entry)

    def test_engine_quality_stage_delegates_to_quality_pipeline(self) -> None:
        engine = (V21_SRC / "engine.py").read_text("utf-8")
        quality_pipeline = (V21_SRC / "quality" / "pipeline.py").read_text("utf-8")

        for token in (
            "class QualityPipelineHooks",
            "class QualityPipelineResult",
            "class QualityPipeline",
            "def run(",
            "repair_final_visible_caption_issues",
            "recompute_final_timeline_safe_handles",
        ):
            self.assertIn(token, quality_pipeline)

        for token in (
            "QualityPipeline(",
            "QualityPipelineHooks(",
            "repair_final_visible_caption_issues=repair_final_visible_caption_issues",
        ):
            self.assertIn(token, engine)

        for token in (
            "max_final_visible_repair_cycles = 8",
            "repair_cycle_state_repeated",
            "visual_pacing_reintroduced_seen_repair_state",
            "post_cleanup_visual_pacing_reintroduced_seen_repair_state",
        ):
            self.assertNotIn(token, engine)

    def test_final_timeline_quality_guard_uses_unified_fact_intent_apply_gate_layers(self) -> None:
        quality_root = V21_SRC / "quality"
        fact = (quality_root / "final_timeline_quality_guard.py").read_text("utf-8")
        intent = (quality_root / "final_timeline_repair_intent.py").read_text("utf-8")
        apply = (quality_root / "final_timeline_repair_apply.py").read_text("utf-8")
        gate = (quality_root / "quality_gate.py").read_text("utf-8")
        validators = (V21_SRC / "validate" / "validators.py").read_text("utf-8")
        engine_validation = (V21_SRC / "engine_validation.py").read_text("utf-8")
        repair_entry = (quality_root / "final_visible_caption_repair.py").read_text("utf-8")
        repair_registry = (quality_root / "final_visible_repair" / "registry.py").read_text("utf-8")
        artifacts = (V21_SRC / "artifact_manifest.py").read_text("utf-8")

        for token in (
            "build_final_timeline_quality_guard_report",
            "PHYSICAL_BLOCKING_CANDIDATE_TYPES",
            "caption_video_word_text_mismatch",
            "blocking_candidates",
        ):
            self.assertIn(token, fact)
        for token in (
            "build_final_timeline_repair_intent_report",
            "source_words_are_authoritative",
            "drop_restart_residue_segment",
            "trim_dangling_words_before_connector",
            "recompute_missing_lead_handle",
        ):
            self.assertIn(token, intent)
        for token in (
            "apply_next_final_timeline_repair_intent",
            "_recompute_safe_handles",
            "render_captions",
            "is_visual_gap_split",
        ):
            self.assertIn(token, apply)
        for token in (
            "final_timeline_quality_guard_gate",
            "V21_FINAL_TIMELINE_QUALITY_GUARD_FAILED",
        ):
            self.assertIn(token, gate)
            self.assertIn(token, engine_validation)
        self.assertIn("final_timeline_quality_guard.get(\"gate_passed\")", validators)
        self.assertIn("final_timeline_quality_intent.apply_next", repair_registry)
        self.assertIn("apply_next_final_timeline_repair_intent", repair_entry)
        self.assertIn("final_timeline_quality_guard_report.json", artifacts)
        self.assertIn("final_timeline_quality_guard_report.json", (V21_SRC / "engine_artifacts.py").read_text("utf-8"))

        self.assertLess(
            repair_registry.index("final_timeline_quality_intent.apply_next"),
            repair_registry.index("LeadingFillerGapRule"),
        )


if __name__ == "__main__":
    unittest.main()
