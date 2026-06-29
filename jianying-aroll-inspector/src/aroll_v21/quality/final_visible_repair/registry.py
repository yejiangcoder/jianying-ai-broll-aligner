from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from aroll_v21.quality.final_visible_repair.pipeline import ProposalRepairRule, RepairRule, StepRepairRule
from aroll_v21.quality.final_visible_repair.rules import (
    caption_only_merge as _caption_only_merge_rules,
    connector_intrusion as _connector_intrusion_rules,
    leading_filler as _leading_filler_rules,
    pre_visible_semantic_junk as _pre_visible_semantic_junk_rules,
    restart_repeat as _restart_repeat_rules,
    short_residual as _short_residual_rules,
    source_boundary_prefix as _source_boundary_prefix_rules,
)


@dataclass(frozen=True)
class FinalVisibleRepairRuleCallbacks:
    repair_final_timeline_quality_intent: Callable[..., Any]
    repair_leading_filler_gap: Callable[..., Any]
    repair_connector_single_word_intrusion: Callable[..., Any]
    repair_connector_filler_restart: Callable[..., Any]
    repair_repeated_object_head_tail: Callable[..., Any]
    repair_subject_prefix_completed_predicate_restart: Callable[..., Any]
    repair_pre_visible_semantic_junk_candidate: Callable[..., Any]
    repair_caption_level_final_repeat_aborted_containment: Callable[..., Any]
    repair_omitted_legal_reduplication_word: Callable[..., Any]
    repair_source_boundary_prefix_gap: Callable[..., Any]
    repair_source_boundary_compound_suffix_gap: Callable[..., Any]
    repair_source_boundary_truncated_compound_tail: Callable[..., Any]
    repair_isolated_semantic_junk_caption: Callable[..., Any]
    repair_short_repair_residual_segments: Callable[..., Any]
    repair_repeated_island_with_proposal: Callable[..., Any]
    repair_boundary_restart_with_proposal: Callable[..., Any]
    repair_contained_short_fragment_with_proposal: Callable[..., Any]
    repair_self_repair_aborted_phrase_with_proposal: Callable[..., Any]
    repair_short_aborted_prefix_caption_with_proposal: Callable[..., Any]
    repair_open_tail_short_caption_with_next: Callable[..., Any]
    repair_fatal_tiny_caption_with_proposal: Callable[..., Any]
    finalize_caption_only_dangling_merges: Callable[..., Any]
    finalize_subject_prefix_completed_predicate_caption_merges: Callable[..., Any]
    finalize_same_subtitle_short_tail_caption_merges: Callable[..., Any]
    repair_next_issue: Callable[..., Any]


@dataclass(frozen=True)
class FinalVisibleRepairRuleRegistry:
    transaction_rules: list[RepairRule]
    residual_transaction_rules: list[RepairRule]
    proposal_transaction_rules: list[RepairRule]
    open_tail_transaction_rules: list[RepairRule]
    tail_proposal_transaction_rules: list[RepairRule]
    caption_only_finalizer_rules: list[RepairRule]


def build_final_visible_repair_rule_registry(callbacks: FinalVisibleRepairRuleCallbacks) -> FinalVisibleRepairRuleRegistry:
    transaction_rules: list[RepairRule] = [
        StepRepairRule(
            name="final_timeline_quality_intent.apply_next",
            repair_step=callbacks.repair_final_timeline_quality_intent,
            include_current_captions=True,
            include_render_captions=True,
        ),
        _leading_filler_rules.LeadingFillerGapRule(repair_leading_filler_gap=callbacks.repair_leading_filler_gap),
        _connector_intrusion_rules.ConnectorSingleWordIntrusionRule(
            repair_connector_single_word_intrusion=callbacks.repair_connector_single_word_intrusion,
        ),
        _connector_intrusion_rules.ConnectorFillerRestartRule(
            repair_connector_filler_restart=callbacks.repair_connector_filler_restart,
        ),
        _connector_intrusion_rules.RepeatedObjectHeadTailRule(
            repair_repeated_object_head_tail=callbacks.repair_repeated_object_head_tail,
        ),
        _connector_intrusion_rules.SubjectPrefixCompletedPredicateRestartRule(
            repair_subject_prefix_completed_predicate_restart=callbacks.repair_subject_prefix_completed_predicate_restart,
        ),
        _pre_visible_semantic_junk_rules.PreVisibleSemanticJunkCandidateRule(
            repair_pre_visible_semantic_junk_candidate=callbacks.repair_pre_visible_semantic_junk_candidate,
        ),
        StepRepairRule(
            name="final_repeat.caption_aborted_containment",
            repair_step=callbacks.repair_caption_level_final_repeat_aborted_containment,
            include_current_captions=True,
        ),
        _source_boundary_prefix_rules.OmittedLegalReduplicationRule(
            repair_omitted_legal_reduplication_word=callbacks.repair_omitted_legal_reduplication_word,
        ),
        _source_boundary_prefix_rules.SourceBoundaryPrefixGapRule(
            repair_source_boundary_prefix_gap=callbacks.repair_source_boundary_prefix_gap,
        ),
        _source_boundary_prefix_rules.SourceBoundaryCompoundSuffixRule(
            repair_source_boundary_compound_suffix_gap=callbacks.repair_source_boundary_compound_suffix_gap,
        ),
        _source_boundary_prefix_rules.SourceBoundaryTruncatedCompoundTailRule(
            repair_source_boundary_truncated_compound_tail=callbacks.repair_source_boundary_truncated_compound_tail,
        ),
        _pre_visible_semantic_junk_rules.IsolatedSemanticJunkCaptionRule(
            repair_isolated_semantic_junk_caption=callbacks.repair_isolated_semantic_junk_caption,
        ),
    ]
    residual_transaction_rules: list[RepairRule] = [
        _short_residual_rules.ShortRepairResidualRule(
            repair_short_repair_residual_segments=callbacks.repair_short_repair_residual_segments,
        )
    ]
    proposal_transaction_rules: list[RepairRule] = [
        ProposalRepairRule(
            name="proposal.repeated_island",
            repair_with_proposal=callbacks.repair_repeated_island_with_proposal,
        ),
        ProposalRepairRule(
            name="proposal.boundary_restart",
            repair_with_proposal=callbacks.repair_boundary_restart_with_proposal,
        ),
        ProposalRepairRule(
            name="proposal.contained_short_fragment",
            repair_with_proposal=callbacks.repair_contained_short_fragment_with_proposal,
            include_current_captions=True,
        ),
        ProposalRepairRule(
            name="proposal.self_repair_aborted_phrase",
            repair_with_proposal=callbacks.repair_self_repair_aborted_phrase_with_proposal,
            include_current_captions=True,
        ),
        ProposalRepairRule(
            name="proposal.short_aborted_prefix_caption",
            repair_with_proposal=callbacks.repair_short_aborted_prefix_caption_with_proposal,
            include_current_captions=True,
        ),
    ]
    open_tail_transaction_rules: list[RepairRule] = [
        StepRepairRule(
            name="open_tail_short_caption",
            repair_step=callbacks.repair_open_tail_short_caption_with_next,
            include_current_captions=True,
            include_render_captions=True,
        )
    ]
    tail_proposal_transaction_rules: list[RepairRule] = [
        ProposalRepairRule(
            name="proposal.fatal_tiny_caption",
            repair_with_proposal=callbacks.repair_fatal_tiny_caption_with_proposal,
            include_current_captions=True,
        )
    ]
    caption_only_finalizer_rules: list[RepairRule] = [
        _caption_only_merge_rules.CaptionOnlyFinalizerRule(
            name="caption_only_finalizer.dangling_merges",
            finalize_captions=callbacks.finalize_caption_only_dangling_merges,
        ),
        _caption_only_merge_rules.CaptionOnlyFinalizerRule(
            name="caption_only_finalizer.subject_prefix_completed_predicate",
            finalize_captions=callbacks.finalize_subject_prefix_completed_predicate_caption_merges,
            include_final_timeline=True,
        ),
        _caption_only_merge_rules.CaptionOnlyFinalizerRule(
            name="caption_only_finalizer.same_subtitle_short_tail",
            finalize_captions=callbacks.finalize_same_subtitle_short_tail_caption_merges,
        ),
    ]
    return FinalVisibleRepairRuleRegistry(
        transaction_rules=transaction_rules,
        residual_transaction_rules=residual_transaction_rules,
        proposal_transaction_rules=proposal_transaction_rules,
        open_tail_transaction_rules=open_tail_transaction_rules,
        tail_proposal_transaction_rules=tail_proposal_transaction_rules,
        caption_only_finalizer_rules=caption_only_finalizer_rules,
    )


def build_gate_candidate_repair_rules(
    *,
    repair_next_issue: Callable[..., Any],
    rendered_gate: dict[str, Any],
    timeline_gate: dict[str, Any],
    current_captions: list[Any],
    effective_timeline_captions: list[Any],
) -> list[RepairRule]:
    return [
        _restart_repeat_rules.GateCandidateRepairRule(
            name="gate_candidate.rendered_dangling_prefix",
            repair_next_issue=repair_next_issue,
            gate=rendered_gate,
            candidate_captions=current_captions,
            issue_types={"dangling_prefix_suffix"},
        ),
        _restart_repeat_rules.GateCandidateRepairRule(
            name="gate_candidate.timeline_gate",
            repair_next_issue=repair_next_issue,
            gate=timeline_gate,
            candidate_captions=effective_timeline_captions,
        ),
        _restart_repeat_rules.GateCandidateRepairRule(
            name="gate_candidate.rendered_gate",
            repair_next_issue=repair_next_issue,
            gate=rendered_gate,
            candidate_captions=current_captions,
        ),
    ]
