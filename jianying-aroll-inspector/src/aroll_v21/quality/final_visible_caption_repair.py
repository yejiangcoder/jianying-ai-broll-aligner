from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any, Callable

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.final_visible_repair.convergence import (
    _caption_only_state_signature as _caption_only_state_signature_impl,
    _repair_state_signature as _repair_state_signature_impl,
)
from aroll_v21.quality.final_visible_repair.context import FinalVisibleRepairContext
from aroll_v21.quality.final_visible_repair.loop_state import FinalVisibleRepairLoopState
from aroll_v21.quality.final_visible_repair.pipeline import run_final_visible_repair_pipeline_once
from aroll_v21.quality.final_visible_repair.registry import (
    FinalVisibleRepairRuleCallbacks,
    build_final_visible_repair_rule_registry,
    build_gate_candidate_repair_rules,
)
from aroll_v21.quality.final_visible_repair.report import (
    FINAL_VISIBLE_REPAIR_COUNT_KEYS,
    _action,
    _is_prefix,
    _is_suffix,
    _repair_counts,
    _unique,
)
from aroll_v21.quality.final_visible_repair.rules import (
    caption_fragment as _caption_fragment_rules,
    caption_only_merge as _caption_only_merge_rules,
    connector_intrusion as _connector_intrusion_rules,
    de_shi_bridge as _de_shi_bridge_rules,
    final_repeat_caption as _final_repeat_caption_rules,
    leading_filler as _leading_filler_rules,
    pre_visible_semantic_junk as _pre_visible_semantic_junk_rules,
    repeated_island as _repeated_island_rules,
    restart_repeat as _restart_repeat_rules,
    short_residual as _short_residual_rules,
    source_boundary_prefix as _source_boundary_prefix_rules,
    word_span_edit as _word_span_edit_rules,
)
from aroll_v21.quality.final_visible_repair.result import (
    FinalVisibleCaptionRepairResult,
    _RepairStep,
    _SourceBoundaryCompoundCandidate,
    _SourceBoundaryPrefixCandidate,
)
from aroll_v21.quality.final_visible_repair.proposal_apply import (
    repair_boundary_restart_with_proposal as _repair_boundary_restart_with_proposal,
    repair_repeated_island_with_proposal as _repair_repeated_island_with_proposal,
)
from aroll_v21.quality.final_timeline_repair_apply import apply_next_final_timeline_repair_intent, recompute_final_timeline_safe_handles
from aroll_v21.quality.final_visible_repair.text_boundary import (
    DE_SHI_BOUNDARY_NORMALIZE_AFTER,
    de_shi_boundary_should_drop_de as _de_shi_boundary_should_drop_de,
    drop_leading_de_from_de_shi_text as _drop_leading_de_from_de_shi_text,
    join_visible_boundary_text as _join_visible_boundary_text,
    join_visible_caption_sequence_text as _join_visible_caption_sequence_text,
    normalized_prefix_before_suffix as _normalized_prefix_before_suffix,
    right_boundary_text_for_join as _right_boundary_text_for_join,
    right_boundary_text_options_after_non_de_left as _right_boundary_text_options_after_non_de_left,
    text_before_suffix as _text_before_suffix,
)
from aroll_v21.quality.final_visible_repair.timeline_utils import (
    caption_by_id as _caption_by_id,
    caption_index as _caption_index,
    caption_segment_ids as _caption_segment_ids,
    ordered_captions as _ordered_captions,
    ordered_segments as _ordered_segments,
    renumber_captions as _renumber_captions,
    repack_timeline as _repack_timeline,
    segment_duration_us as _segment_duration_us,
    text_from_word_ids as _text_from_word_ids,
)
from aroll_v21.quality.final_caption_visible_repeat import (
    FINAL_VISIBLE_RECHECK_DECISIONS,
    build_final_caption_visible_repeat_gate,
    _dangling_pronoun_modal_suffix,
)
from aroll_v21.quality.final_semantic_integrity import dangling_discourse_connector_suffix
from aroll_v21.quality.pre_visible_semantic_junk_candidate_detector import (
    MIN_HIGH_CONFIDENCE as PRE_VISIBLE_SEMANTIC_JUNK_MIN_HIGH_CONFIDENCE,
    build_pre_visible_semantic_junk_candidate_report,
)
from aroll_v21.quality.subtitle_readability import HARD_MAX_CHARS
from aroll_v21.quality.tiny_segment_classifier import classify_tiny_segment


MAX_FINAL_VISIBLE_REPAIR_PASSES = 128































def repair_final_visible_caption_issues(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    max_passes: int = MAX_FINAL_VISIBLE_REPAIR_PASSES,
) -> FinalVisibleCaptionRepairResult:
    current_timeline = list(final_timeline)
    current_captions = _renumber_captions(list(captions))
    initial_gate = build_final_caption_visible_repeat_gate(current_captions)
    initial_timeline_gate = build_final_caption_visible_repeat_gate(_timeline_caption_units(current_timeline, source_graph))
    initial_semantic_junk_report = build_pre_visible_semantic_junk_candidate_report(current_captions, source_graph)
    initial_repeated_island_candidates = [
        candidate.to_evidence()
        for candidate in _repeated_island_rules.detect_repeated_island_candidates(current_timeline, source_graph)
    ]
    max_pass_limit = max(1, int(max_passes))
    current_signature = _repair_state_signature(current_timeline, current_captions)
    loop_state = FinalVisibleRepairLoopState(
        current_timeline=current_timeline,
        current_captions=current_captions,
        current_signature=current_signature,
        seen_signatures={current_signature},
    )
    passes_executed = 0
    repair_context = FinalVisibleRepairContext(
        source_graph=source_graph,
        render_captions=render_captions,
        repack_timeline=_repack_timeline,
        renumber_captions=_renumber_captions,
        render_captions_preserving_caption_only_materializations=_render_captions_preserving_caption_only_materializations,
        repair_state_signature=_repair_state_signature,
    )
    rule_registry = build_final_visible_repair_rule_registry(
        FinalVisibleRepairRuleCallbacks(
            repair_final_timeline_quality_intent=_repair_final_timeline_quality_intent,
            repair_leading_filler_gap=_repair_leading_filler_gap,
            repair_connector_single_word_intrusion=_repair_connector_single_word_intrusion,
            repair_connector_filler_restart=_repair_connector_filler_restart,
            repair_repeated_object_head_tail=_repair_repeated_object_head_tail,
            repair_subject_prefix_completed_predicate_restart=_repair_subject_prefix_completed_predicate_restart,
            repair_pre_visible_semantic_junk_candidate=_repair_pre_visible_semantic_junk_candidate,
            repair_caption_level_final_repeat_aborted_containment=(
                _final_repeat_caption_rules.repair_caption_level_final_repeat_aborted_containment
            ),
            repair_omitted_legal_reduplication_word=_repair_omitted_legal_reduplication_word,
            repair_source_boundary_prefix_gap=_repair_source_boundary_prefix_gap,
            repair_source_boundary_compound_suffix_gap=_repair_source_boundary_compound_suffix_gap,
            repair_source_boundary_truncated_compound_tail=_repair_source_boundary_truncated_compound_tail,
            repair_isolated_semantic_junk_caption=_repair_isolated_semantic_junk_caption,
            repair_short_repair_residual_segments=_repair_short_repair_residual_segments,
            repair_repeated_island_with_proposal=_repair_repeated_island_with_proposal,
            repair_boundary_restart_with_proposal=_repair_boundary_restart_with_proposal,
            repair_contained_short_fragment_with_proposal=(
                _caption_fragment_rules.repair_contained_short_fragment_with_proposal
            ),
            repair_self_repair_aborted_phrase_with_proposal=(
                _caption_fragment_rules.repair_self_repair_aborted_phrase_with_proposal
            ),
            repair_short_aborted_prefix_caption_with_proposal=(
                _caption_fragment_rules.repair_short_aborted_prefix_caption_with_proposal
            ),
            repair_open_tail_short_caption_with_next=partial(
                _caption_fragment_rules.repair_open_tail_short_caption_with_next,
                render_captions_preserving_caption_only_materializations=(
                    _render_captions_preserving_caption_only_materializations
                ),
            ),
            repair_fatal_tiny_caption_with_proposal=(
                _caption_fragment_rules.repair_fatal_tiny_caption_with_proposal
            ),
            finalize_caption_only_dangling_merges=_finalize_caption_only_dangling_merges,
            finalize_subject_prefix_completed_predicate_caption_merges=_finalize_subject_prefix_completed_predicate_caption_merges,
            finalize_same_subtitle_short_tail_caption_merges=_finalize_same_subtitle_short_tail_caption_merges,
            repair_next_issue=_repair_next_issue,
        )
    )

    for pass_index in range(max_pass_limit):
        passes_executed = pass_index + 1
        transaction_result = run_final_visible_repair_pipeline_once(
            context=repair_context,
            final_timeline=loop_state.current_timeline,
            captions=loop_state.current_captions,
            pass_index=pass_index + 1,
            current_signature=loop_state.current_signature,
            seen_signatures=loop_state.seen_signatures,
            rules=rule_registry.transaction_rules,
        )
        transaction_status = loop_state.consume_pipeline_result(transaction_result, pass_index=pass_index + 1)
        if transaction_status == "stop":
            break
        if transaction_status == "accepted":
            continue

        proposal_result = run_final_visible_repair_pipeline_once(
            context=repair_context,
            final_timeline=loop_state.current_timeline,
            captions=loop_state.current_captions,
            pass_index=pass_index + 1,
            current_signature=loop_state.current_signature,
            seen_signatures=loop_state.seen_signatures,
            rules=rule_registry.proposal_transaction_rules,
        )
        proposal_status = loop_state.consume_pipeline_result(proposal_result, pass_index=pass_index + 1)
        if proposal_status == "stop":
            break
        if proposal_status == "accepted":
            continue

        open_tail_result = run_final_visible_repair_pipeline_once(
            context=repair_context,
            final_timeline=loop_state.current_timeline,
            captions=loop_state.current_captions,
            pass_index=pass_index + 1,
            current_signature=loop_state.current_signature,
            seen_signatures=loop_state.seen_signatures,
            rules=rule_registry.open_tail_transaction_rules,
        )
        open_tail_status = loop_state.consume_pipeline_result(open_tail_result, pass_index=pass_index + 1)
        if open_tail_status == "stop":
            break
        if open_tail_status == "accepted":
            continue

        tail_proposal_result = run_final_visible_repair_pipeline_once(
            context=repair_context,
            final_timeline=loop_state.current_timeline,
            captions=loop_state.current_captions,
            pass_index=pass_index + 1,
            current_signature=loop_state.current_signature,
            seen_signatures=loop_state.seen_signatures,
            rules=rule_registry.tail_proposal_transaction_rules,
        )
        tail_proposal_status = loop_state.consume_pipeline_result(tail_proposal_result, pass_index=pass_index + 1)
        if tail_proposal_status == "stop":
            break
        if tail_proposal_status == "accepted":
            continue

        rendered_gate = build_final_caption_visible_repeat_gate(loop_state.current_captions)
        timeline_captions = _timeline_caption_units(loop_state.current_timeline, source_graph)
        effective_timeline_captions, timeline_materializations = _effective_timeline_caption_units(
            timeline_captions,
            loop_state.current_captions,
        )
        timeline_gate = _timeline_gate(effective_timeline_captions, timeline_materializations)
        rendered_counts = _repair_counts(rendered_gate)
        timeline_counts = _repair_counts(timeline_gate)
        if not any(rendered_counts.values()) and not any(timeline_counts.values()):
            loop_state.stop_reason = "converged"
            break
        gate_candidate_result = run_final_visible_repair_pipeline_once(
            context=repair_context,
            final_timeline=loop_state.current_timeline,
            captions=loop_state.current_captions,
            pass_index=pass_index + 1,
            current_signature=loop_state.current_signature,
            seen_signatures=loop_state.seen_signatures,
            rules=build_gate_candidate_repair_rules(
                repair_next_issue=_repair_next_issue,
                rendered_gate=rendered_gate,
                timeline_gate=timeline_gate,
                current_captions=loop_state.current_captions,
                effective_timeline_captions=effective_timeline_captions,
            ),
        )
        if gate_candidate_result.transaction is None:
            loop_state.unresolved.append(
                {
                    "pass_index": pass_index + 1,
                    "counts": rendered_counts,
                    "timeline_counts": timeline_counts,
                    "blocker_codes": list(rendered_gate.get("blocker_codes") or []),
                    "timeline_blocker_codes": list(timeline_gate.get("blocker_codes") or []),
                    "reason": "no_safe_deterministic_repair_available",
                }
            )
            loop_state.stop_reason = "no_safe_deterministic_repair_available"
            break
        gate_candidate_status = loop_state.consume_pipeline_result(gate_candidate_result, pass_index=pass_index + 1)
        if gate_candidate_status == "stop":
            break
        continue

    residual_result = run_final_visible_repair_pipeline_once(
        context=repair_context,
        final_timeline=loop_state.current_timeline,
        captions=loop_state.current_captions,
        pass_index=len(loop_state.actions) + 1,
        current_signature=loop_state.current_signature,
        seen_signatures=loop_state.seen_signatures,
        rules=rule_registry.residual_transaction_rules,
    )
    loop_state.consume_pipeline_result(residual_result, pass_index=len(loop_state.actions) + 1)

    for caption_only_finalizer_rule in rule_registry.caption_only_finalizer_rules:
        caption_only_result = run_final_visible_repair_pipeline_once(
            context=repair_context,
            final_timeline=loop_state.current_timeline,
            captions=loop_state.current_captions,
            pass_index=len(loop_state.actions) + 1,
            current_signature=loop_state.current_signature,
            seen_signatures=loop_state.seen_signatures,
            rules=[caption_only_finalizer_rule],
        )
        caption_only_status = loop_state.consume_pipeline_result(
            caption_only_result,
            pass_index=len(loop_state.actions) + 1,
        )
        if caption_only_status == "stop":
            break

    final_safe_handle_result = recompute_final_timeline_safe_handles(
        final_timeline=loop_state.current_timeline,
        captions=loop_state.current_captions,
        source_graph=source_graph,
        render_captions=render_captions,
        pass_index=len(loop_state.actions) + 1,
    )
    if final_safe_handle_result is not None:
        loop_state.current_timeline = final_safe_handle_result.final_timeline
        loop_state.current_captions = final_safe_handle_result.captions
        loop_state.actions.extend([final_safe_handle_result.action])
        loop_state.current_signature = _repair_state_signature(loop_state.current_timeline, loop_state.current_captions)
        loop_state.seen_signatures.add(loop_state.current_signature)

    final_gate = build_final_caption_visible_repeat_gate(loop_state.current_captions)
    final_semantic_junk_report = build_pre_visible_semantic_junk_candidate_report(loop_state.current_captions, source_graph)
    semantic_junk_actions = [
        action
        for action in loop_state.actions
        if str(action.get("issue_type") or "") == "pre_visible_semantic_junk_candidate"
    ]
    final_semantic_junk_report = {
        **final_semantic_junk_report,
        "pre_visible_semantic_junk_audit_only": False,
        "pre_visible_semantic_junk_candidate_detector_audit_only": True,
        "pre_visible_semantic_junk_timeline_mutation_allowed": True,
        "pre_visible_semantic_junk_deterministic_apply_enabled": True,
        "pre_visible_semantic_junk_deterministic_apply_policy": "local_high_confidence_drop_fragment_only",
        "pre_visible_semantic_junk_deterministic_apply_min_confidence": PRE_VISIBLE_SEMANTIC_JUNK_MIN_HIGH_CONFIDENCE,
        "pre_visible_semantic_junk_repair_action_count": len(semantic_junk_actions),
        "pre_visible_semantic_junk_repair_actions": semantic_junk_actions,
    }
    final_effective_timeline_captions, final_materializations = _effective_timeline_caption_units(
        _timeline_caption_units(loop_state.current_timeline, source_graph),
        loop_state.current_captions,
    )
    final_timeline_gate = _timeline_gate(final_effective_timeline_captions, final_materializations)
    final_repeated_island_candidates = [
        candidate.to_evidence()
        for candidate in _repeated_island_rules.detect_repeated_island_candidates(loop_state.current_timeline, source_graph)
    ]
    boundary_restart_actions = [
        action
        for action in loop_state.actions
        if str(action.get("issue_type") or "") == "boundary_restart"
    ]
    repeated_island_actions = [
        action
        for action in loop_state.actions
        if str(action.get("issue_type") or "") == "repeated_island"
    ]
    timeline_repair_proposal_actions = [
        action
        for action in loop_state.actions
        if str(action.get("proposal_id") or "")
    ]
    final_timeline_intent_actions = [
        action
        for action in loop_state.actions
        if str(action.get("issue_type") or "") == "final_timeline_repair_intent"
    ]
    transaction_actions = [
        action
        for action in loop_state.actions
        if str(action.get("repair_transaction_rule_name") or "")
    ]
    final_counts = _repair_counts(final_gate)
    final_timeline_counts = _repair_counts(final_timeline_gate)
    repair_success = not any(final_counts.values()) and not any(final_timeline_counts.values())
    if not repair_success and not loop_state.unresolved:
        reason = "max_repair_passes_exhausted" if len(loop_state.actions) >= max_pass_limit else "unresolved_after_repair"
        loop_state.stop_reason = reason
        loop_state.unresolved.append(
            {
                "pass_index": len(loop_state.actions) + 1,
                "counts": final_counts,
                "timeline_counts": final_timeline_counts,
                "blocker_codes": list(final_gate.get("blocker_codes") or []),
                "timeline_blocker_codes": list(final_timeline_gate.get("blocker_codes") or []),
                "reason": reason,
            }
        )
    if repair_success and not loop_state.stop_reason:
        loop_state.stop_reason = "converged"

    report = {
        "final_visible_repair_enabled": True,
        "final_visible_repair_attempted": bool(loop_state.actions) or any(_repair_counts(initial_gate).values()) or any(_repair_counts(initial_timeline_gate).values()),
        "final_visible_repair_success": repair_success,
        "final_visible_repair_max_passes": max_pass_limit,
        "final_visible_repair_passes_executed": passes_executed,
        "final_visible_repair_stop_reason": loop_state.stop_reason,
        "final_visible_repair_no_progress_detected": loop_state.stop_reason == "no_progress_detected",
        "final_visible_repair_max_pass_exhausted": any(
            str(row.get("reason") or "") == "max_repair_passes_exhausted"
            for row in loop_state.unresolved
        ),
        "final_visible_repair_progress_state_count": len(loop_state.seen_signatures),
        "final_visible_repair_action_count": len(loop_state.actions),
        "final_visible_repair_actions": loop_state.actions,
        "final_visible_repair_transaction_count": len(transaction_actions),
        "final_visible_repair_transaction_rule_names": _unique(
            [str(action.get("repair_transaction_rule_name") or "") for action in transaction_actions]
        ),
        "final_visible_repair_pipeline_rule_names": [rule.name for rule in rule_registry.transaction_rules],
        "final_visible_repair_unresolved": loop_state.unresolved,
        "final_visible_repair_initial_counts": _repair_counts(initial_gate),
        "final_visible_repair_initial_timeline_counts": _repair_counts(initial_timeline_gate),
        "final_visible_repair_final_counts": final_counts,
        "final_visible_repair_final_timeline_counts": final_timeline_counts,
        "pre_visible_semantic_junk_initial_report": initial_semantic_junk_report,
        "pre_visible_semantic_junk_report": final_semantic_junk_report,
        "pre_visible_semantic_junk_initial_candidate_count": int(initial_semantic_junk_report.get("pre_visible_semantic_junk_candidate_count") or 0),
        "pre_visible_semantic_junk_final_candidate_count": int(final_semantic_junk_report.get("pre_visible_semantic_junk_candidate_count") or 0),
        "pre_visible_semantic_junk_repair_action_count": len(semantic_junk_actions),
        "pre_visible_semantic_junk_repair_actions": semantic_junk_actions,
        "pre_visible_semantic_junk_audit_only": False,
        "pre_visible_semantic_junk_candidate_detector_audit_only": True,
        "pre_visible_semantic_junk_timeline_mutation_allowed": True,
        "pre_visible_semantic_junk_deterministic_apply_enabled": True,
        "pre_visible_semantic_junk_deterministic_apply_policy": "local_high_confidence_drop_fragment_only",
        "repeated_island_initial_candidate_count": len(initial_repeated_island_candidates),
        "repeated_island_initial_candidates": initial_repeated_island_candidates,
        "repeated_island_candidate_count": len(final_repeated_island_candidates),
        "repeated_island_high_confidence_count": _repeated_island_confidence_count(
            final_repeated_island_candidates,
            "high",
        ),
        "repeated_island_medium_confidence_count": _repeated_island_confidence_count(
            final_repeated_island_candidates,
            "medium",
        ),
        "repeated_island_low_confidence_count": _repeated_island_confidence_count(
            final_repeated_island_candidates,
            "low",
        ),
        "repeated_island_warning_count": _repeated_island_confidence_count(
            final_repeated_island_candidates,
            "medium",
        ),
        "repeated_island_candidates": final_repeated_island_candidates,
        "repeated_island_repair_action_count": len(repeated_island_actions),
        "repeated_island_repair_actions": repeated_island_actions,
        "boundary_restart_repair_action_count": len(boundary_restart_actions),
        "boundary_restart_repair_actions": boundary_restart_actions,
        "timeline_repair_proposal_action_count": len(timeline_repair_proposal_actions),
        "timeline_repair_proposal_actions": timeline_repair_proposal_actions,
        "final_timeline_repair_intent_action_count": len(final_timeline_intent_actions),
        "final_timeline_repair_intent_actions": final_timeline_intent_actions,
        "final_visible_effective_caption_count": len(final_effective_timeline_captions),
        "caption_only_materialized_merge_count": len(final_materializations),
        "caption_only_materialized_merges": final_materializations,
        "caption_only_consumed_caption_ids": [
            caption_id
            for row in final_materializations
            for caption_id in list(row.get("consumed_caption_ids") or [])
        ],
        "source_boundary_prefix_repair_count": sum(
            1
            for action in loop_state.actions
            if str(action.get("decision") or "") == "prepend_source_boundary_prefix"
        ),
        "final_visible_repair_initial_blocker_codes": list(initial_gate.get("blocker_codes") or []),
        "final_visible_repair_initial_timeline_blocker_codes": list(initial_timeline_gate.get("blocker_codes") or []),
        "final_visible_repair_final_blocker_codes": list(final_gate.get("blocker_codes") or []),
        "final_visible_repair_final_timeline_blocker_codes": list(final_timeline_gate.get("blocker_codes") or []),
        "final_visible_recheck_allowed_decisions": list(FINAL_VISIBLE_RECHECK_DECISIONS),
        "final_visible_recheck_required_count": max(
            int(final_counts.get("semantic_garbage_or_asr_suspect_count") or 0),
            int(final_timeline_counts.get("semantic_garbage_or_asr_suspect_count") or 0),
        ),
    }
    return FinalVisibleCaptionRepairResult(
        final_timeline=loop_state.current_timeline,
        captions=loop_state.current_captions,
        report=report,
    )


def _repair_final_timeline_quality_intent(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> _RepairStep | None:
    result = apply_next_final_timeline_repair_intent(
        final_timeline=final_timeline,
        captions=captions,
        source_graph=source_graph,
        render_captions=render_captions,
        pass_index=pass_index,
    )
    if result is None:
        no_step: _RepairStep | None = None
        return no_step
    return _RepairStep(
        final_timeline=result.final_timeline,
        captions=result.captions,
        action=result.action,
        timeline_changed=result.timeline_changed,
    )


def _repeated_island_confidence_count(
    candidates: list[dict[str, Any]],
    confidence: str,
) -> int:
    return sum(1 for candidate in candidates if str(candidate.get("confidence") or "") == confidence)


def _repair_next_issue(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    gate: dict[str, Any],
    pass_index: int,
    issue_types: set[str] | None = None,
) -> _RepairStep | None:
    if issue_types is None or "dangling_prefix_suffix" in issue_types:
        for candidate in list(gate.get("dangling_prefix_suffix_candidates") or []):
            step = _repair_dangling_prefix_suffix(final_timeline, captions, source_graph, candidate, pass_index)
            if step is not None:
                return step
    if issue_types is None or "cross_caption_semantic_containment" in issue_types:
        for candidate in list(gate.get("cross_caption_semantic_containment_candidates") or []):
            step = _drop_repeated_caption_span(final_timeline, captions, source_graph, candidate, "cross_caption_semantic_containment", pass_index)
            if step is not None:
                return step
    if issue_types is None or "restart_repeat_visible" in issue_types:
        for candidate in list(gate.get("restart_repeat_visible_candidates") or []):
            step = _drop_restart_repeat_word_span(final_timeline, captions, source_graph, candidate, pass_index)
            if step is None:
                step = _trim_restart_repeat_visible_prefix(final_timeline, captions, source_graph, candidate, pass_index)
            if step is None:
                step = _drop_repeated_caption_span(final_timeline, captions, source_graph, candidate, "restart_repeat_visible", pass_index)
            if step is not None:
                return step
    if issue_types is None or "semantic_garbage_or_asr_suspect" in issue_types:
        for candidate in list(gate.get("semantic_garbage_or_asr_suspect_candidates") or []):
            step = _trim_asr_restart_prefix(final_timeline, captions, source_graph, candidate, pass_index)
            if step is not None:
                return step
    if issue_types is None or "semantic_integrity" in issue_types:
        for candidate in list(gate.get("semantic_integrity_candidates") or []):
            step = _repair_semantic_integrity_issue(final_timeline, captions, source_graph, candidate, pass_index)
            if step is not None:
                return step
    no_step: _RepairStep | None = None
    return no_step


def _repair_semantic_integrity_issue(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    reason = str(candidate.get("reason") or "")
    caption = _caption_by_id(captions, str(candidate.get("caption_id") or ""))
    if caption is None:
        no_step: _RepairStep | None = None
        return no_step
    if reason in {
        "opening_vocalization_residual",
        "non_primary_device_prompt_residual",
        "short_abandoned_open_clause",
        "previous_complete_prefix_retry",
    }:
        dropped = _drop_or_trim_caption_words(final_timeline, captions, source_graph, caption)
        if dropped is None:
            no_step: _RepairStep | None = None
            return no_step
        repaired_timeline, dropped_segment_ids, trimmed_segment_ids = dropped
        return _RepairStep(
            final_timeline=repaired_timeline,
            captions=captions,
            timeline_changed=True,
            action=_action(
                "semantic_integrity",
                "drop_semantic_integrity_residual",
                pass_index,
                candidate,
                affected_caption_ids=[caption.caption_id],
                dropped_segment_ids=dropped_segment_ids,
                trimmed_segment_ids=trimmed_segment_ids,
                dropped_word_ids=list(caption.word_ids),
                dropped_text=str(caption.text or ""),
            ),
        )
    if reason == "repeated_interjection_residual":
        repeat_text = normalize_text(str((candidate.get("evidence") or {}).get("repeat_text") or ""))
        caption_text = normalize_text(caption.text)
        if len(repeat_text) < 2 or not caption_text.endswith(repeat_text):
            no_step: _RepairStep | None = None
            return no_step
        drop_text = repeat_text[1:]
        drop_word_ids = _trailing_word_ids_for_text(caption.word_ids, source_graph, drop_text)
        if not drop_word_ids:
            fused_word_ids = _trailing_word_ids_for_text(caption.word_ids, source_graph, repeat_text)
            if fused_word_ids:
                drop_text = repeat_text
                drop_word_ids = fused_word_ids
        if not drop_word_ids or len(drop_word_ids) >= len(caption.word_ids):
            no_step: _RepairStep | None = None
            return no_step
        repaired_timeline = _trim_word_ids_from_timeline(final_timeline, source_graph, drop_word_ids)
        if repaired_timeline is None:
            no_step: _RepairStep | None = None
            return no_step
        return _RepairStep(
            final_timeline=repaired_timeline,
            captions=captions,
            timeline_changed=True,
            action=_action(
                "semantic_integrity",
                "trim_duplicate_interjection_tail",
                pass_index,
                candidate,
                affected_caption_ids=[caption.caption_id],
                dropped_word_ids=drop_word_ids,
                drop_text=drop_text,
            ),
        )
    if reason in {
        "open_coordination_tail",
        "single_char_false_start_tail",
        "truncated_nominal_prefix_tail",
        "dangling_discourse_connector_tail",
        "dangling_discourse_pronoun_tail",
        "incomplete_lexical_tail",
        "local_recurrence_with_open_tail",
    }:
        drop_text = str(candidate.get("overlap_text") or "")
        dangling_connector = dangling_discourse_connector_suffix(str(caption.text or ""))
        if reason in {"dangling_discourse_connector_tail", "incomplete_lexical_tail", "local_recurrence_with_open_tail"}:
            if not dangling_connector:
                no_step: _RepairStep | None = None
                return no_step
            drop_text = dangling_connector
        if reason == "single_char_false_start_tail":
            drop_text = normalize_text(str(caption.text or ""))[-1:]
        if reason == "truncated_nominal_prefix_tail":
            caption_text = normalize_text(caption.text)
            if len(caption_text) >= 2 and caption_text[-2] in {"的", "地", "得"}:
                drop_text = caption_text[-2:]
        drop_word_ids = _trailing_word_ids_for_text(caption.word_ids, source_graph, drop_text)
        if not drop_word_ids or len(drop_word_ids) >= len(caption.word_ids):
            no_step: _RepairStep | None = None
            return no_step
        repaired_timeline = _trim_word_ids_from_timeline(final_timeline, source_graph, drop_word_ids)
        if repaired_timeline is None:
            no_step: _RepairStep | None = None
            return no_step
        return _RepairStep(
            final_timeline=repaired_timeline,
            captions=captions,
            timeline_changed=True,
            action=_action(
                "semantic_integrity",
                "trim_open_semantic_tail",
                pass_index,
                candidate,
                affected_caption_ids=[caption.caption_id],
                dropped_word_ids=drop_word_ids,
                drop_text=drop_text,
            ),
        )
    no_step: _RepairStep | None = None
    return no_step


def _render_captions_preserving_caption_only_materializations(
    final_timeline: list[FinalTimelineSegment],
    previous_captions: list[CaptionRenderUnit],
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
) -> list[CaptionRenderUnit]:
    rendered = _renumber_captions(render_captions(final_timeline))
    effective, materializations = _effective_timeline_caption_units(rendered, previous_captions)
    if not materializations:
        return rendered
    return _renumber_captions(effective)


def _repair_dangling_prefix_suffix(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    ordered = _ordered_captions(captions)
    index = _caption_index(ordered, str(candidate.get("caption_id") or ""))
    if index is None:
        no_step: _RepairStep | None = None
        return no_step
    current = ordered[index]
    tail_suffix_step = _repair_dangling_pronoun_modal_suffix(
        final_timeline,
        current,
        source_graph,
        candidate,
        pass_index,
    )
    if tail_suffix_step is not None:
        return tail_suffix_step
    same_segment_de_duplicate = _repair_same_segment_de_duplicate_prefix(
        final_timeline,
        current,
        source_graph,
        candidate,
        pass_index,
    )
    if same_segment_de_duplicate is not None:
        return same_segment_de_duplicate
    if str(candidate.get("reason") or "") == "dangling_weak_pronoun_fragment":
        dropped = _drop_or_trim_caption_words(final_timeline, captions, source_graph, current)
        if dropped is None:
            no_step: _RepairStep | None = None
            return no_step
        repaired_timeline, dropped_segment_ids, trimmed_segment_ids = dropped
        return _RepairStep(
            final_timeline=repaired_timeline,
            captions=[],
            timeline_changed=True,
            action=_action(
                "dangling_prefix_suffix",
                "drop_dangling_weak_pronoun_fragment",
                pass_index,
                candidate,
                affected_caption_ids=[current.caption_id],
                dropped_segment_ids=dropped_segment_ids,
                trimmed_segment_ids=trimmed_segment_ids,
            ),
        )
    if index == 0:
        no_step: _RepairStep | None = None
        return no_step
    previous = ordered[index - 1]
    combined_text = f"{previous.text}{current.text}"
    if len(normalize_text(combined_text)) > HARD_MAX_CHARS:
        prefix_transfer_step = _transfer_leading_function_prefix_to_previous_caption(
            final_timeline=final_timeline,
            captions=ordered,
            previous_index=index - 1,
            current_index=index,
            source_graph=source_graph,
            candidate=candidate,
            pass_index=pass_index,
        )
        if prefix_transfer_step is not None:
            return prefix_transfer_step
        no_step: _RepairStep | None = None
        return no_step

    de_shi_bridge = _repair_de_shi_duplicate_bridge(
        final_timeline,
        previous,
        current,
        source_graph,
        candidate,
        pass_index,
    )
    if de_shi_bridge is not None:
        return de_shi_bridge

    merged_timeline = _merge_adjacent_caption_segments(final_timeline, previous, current, source_graph)
    if merged_timeline is not None:
        return _RepairStep(
            final_timeline=merged_timeline,
            captions=captions,
            timeline_changed=True,
            action=_action(
                "dangling_prefix_suffix",
                "merge_with_previous_segment",
                pass_index,
                candidate,
                affected_caption_ids=[previous.caption_id, current.caption_id],
            ),
        )

    merged_caption_result = _merge_adjacent_captions(previous, current)
    if merged_caption_result is None:
        no_step: _RepairStep | None = None
        return no_step
    merged_caption, merge_decision = merged_caption_result
    rows = list(ordered)
    rows[index - 1] = merged_caption
    repaired = [*rows[:index], *rows[index + 1 :]]
    caption_only_merge = merge_decision == "caption_only_merge_with_previous"
    return _RepairStep(
        final_timeline=final_timeline,
        captions=repaired,
        timeline_changed=False,
        action=_action(
            "dangling_prefix_suffix",
            merge_decision,
            pass_index,
            candidate,
            affected_caption_ids=[previous.caption_id, current.caption_id],
            target_gap_us=int(current.target_start_us) - int(previous.target_end_us),
            video_segment_merged=False,
            caption_only_merge_materialized=caption_only_merge,
            merged_into_caption_id=previous.caption_id if caption_only_merge else "",
            consumed_caption_id=current.caption_id if caption_only_merge else "",
            consumed_caption_state="consumed_by_caption_only_merge" if caption_only_merge else "",
            merged_caption_text=merged_caption.text if caption_only_merge else "",
            merged_caption_timeline_segment_ids=list(merged_caption.timeline_segment_ids) if caption_only_merge else [],
            merged_caption_target_start_us=int(merged_caption.target_start_us) if caption_only_merge else 0,
            merged_caption_target_end_us=int(merged_caption.target_end_us) if caption_only_merge else 0,
        ),
    )


def _timeline_caption_units(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> list[CaptionRenderUnit]:
    captions: list[CaptionRenderUnit] = []
    words_by_id = {word.word_id: word for word in source_graph.words}
    for index, segment in enumerate(_ordered_segments(final_timeline), start=1):
        words = [words_by_id[word_id] for word_id in segment.word_ids if word_id in words_by_id]
        text = "".join(word.text for word in words) or str(segment.text or "")
        if not normalize_text(text):
            continue
        source_subtitle_uids = _unique([str(word.subtitle_uid or "") for word in words])
        spoken_start_us = min((int(word.source_start_us) for word in words), default=int(segment.source_start_us))
        spoken_end_us = max((int(word.source_end_us) for word in words), default=int(segment.source_end_us))
        captions.append(
            CaptionRenderUnit(
                caption_id=f"v21_timeline_cap_{index:06d}",
                timeline_segment_ids=[segment.segment_id],
                word_ids=list(segment.word_ids),
                text=text,
                target_start_us=int(segment.target_start_us),
                target_end_us=int(segment.target_end_us),
                source_subtitle_uids=source_subtitle_uids,
                style_template_id="final_visible_timeline_detection",
                spoken_source_start_us=spoken_start_us,
                spoken_source_end_us=spoken_end_us,
                containing_video_segment_id=segment.segment_id,
            )
        )
    return captions


def _timeline_gate(captions: list[CaptionRenderUnit], materializations: list[dict[str, Any]]) -> dict[str, Any]:
    gate = build_final_caption_visible_repeat_gate(captions)
    gate["effective_visible_caption_count"] = len(captions)
    gate["caption_only_materialized_merge_count"] = len(materializations)
    gate["caption_only_materialized_merges"] = materializations
    gate["caption_only_consumed_caption_ids"] = [
        caption_id
        for row in materializations
        for caption_id in list(row.get("consumed_caption_ids") or [])
    ]
    return gate


def _effective_timeline_caption_units(
    timeline_captions: list[CaptionRenderUnit],
    visible_captions: list[CaptionRenderUnit],
) -> tuple[list[CaptionRenderUnit], list[dict[str, Any]]]:
    ordered = _ordered_captions(timeline_captions)
    if not ordered:
        return [], []
    materialized_by_first_index: dict[int, list[CaptionRenderUnit]] = {}
    consumed_indices: set[int] = set()
    materializations: list[dict[str, Any]] = []
    for visible in _ordered_captions(visible_captions):
        match = _caption_only_materialization_for_visible_caption(visible, ordered, consumed_indices)
        if match is None:
            continue
        first_index, indices, replacements, row = match
        materialized_by_first_index[first_index] = replacements
        consumed_indices.update(indices)
        materializations.append(row)
    effective: list[CaptionRenderUnit] = []
    for index, caption in enumerate(ordered):
        if index in materialized_by_first_index:
            effective.extend(materialized_by_first_index[index])
            continue
        if index in consumed_indices:
            continue
        effective.append(caption)
    return effective, materializations


def _repair_state_signature(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
) -> tuple[Any, ...]:
    return _repair_state_signature_impl(
        final_timeline,
        captions,
        ordered_segments=_ordered_segments,
        ordered_captions=_ordered_captions,
        caption_segment_ids=_caption_segment_ids,
    )

LEADING_FILLER_WORDS = _leading_filler_rules.LEADING_FILLER_WORDS
MIN_LEADING_FILLER_GAP_US = _leading_filler_rules.MIN_LEADING_FILLER_GAP_US
MAX_LEADING_FILLER_DURATION_US = _leading_filler_rules.MAX_LEADING_FILLER_DURATION_US
MIN_LEADING_FILLER_REMAINING_CHARS = _leading_filler_rules.MIN_LEADING_FILLER_REMAINING_CHARS
CONNECTOR_INTRUSION_NEXT_WORDS = _connector_intrusion_rules.CONNECTOR_INTRUSION_NEXT_WORDS
MIN_CONNECTOR_INTRUSION_SIDE_GAP_US = _connector_intrusion_rules.MIN_CONNECTOR_INTRUSION_SIDE_GAP_US
MAX_CONNECTOR_INTRUSION_WORD_DURATION_US = _connector_intrusion_rules.MAX_CONNECTOR_INTRUSION_WORD_DURATION_US
MIN_CONNECTOR_INTRUSION_REMAINING_CHARS = _connector_intrusion_rules.MIN_CONNECTOR_INTRUSION_REMAINING_CHARS
CONNECTOR_RESTART_WORDS = _connector_intrusion_rules.CONNECTOR_RESTART_WORDS
CONNECTOR_RESTART_INTRUSION_WORDS = _connector_intrusion_rules.CONNECTOR_RESTART_INTRUSION_WORDS
MAX_CONNECTOR_RESTART_INTRUSION_DURATION_US = _connector_intrusion_rules.MAX_CONNECTOR_RESTART_INTRUSION_DURATION_US
MIN_CONNECTOR_RESTART_REMAINING_CHARS = _connector_intrusion_rules.MIN_CONNECTOR_RESTART_REMAINING_CHARS
MIN_REPEATED_OBJECT_HEAD_GAP_US = _connector_intrusion_rules.MIN_REPEATED_OBJECT_HEAD_GAP_US
MIN_REPEATED_OBJECT_REMAINING_CHARS = _connector_intrusion_rules.MIN_REPEATED_OBJECT_REMAINING_CHARS
MAX_ISOLATED_SHORT_FRAGMENT_CHARS = _pre_visible_semantic_junk_rules.MAX_ISOLATED_SHORT_FRAGMENT_CHARS
MAX_ISOLATED_SHORT_FRAGMENT_DURATION_US = _pre_visible_semantic_junk_rules.MAX_ISOLATED_SHORT_FRAGMENT_DURATION_US
MIN_ISOLATED_SHORT_FRAGMENT_SOURCE_GAP_US = _pre_visible_semantic_junk_rules.MIN_ISOLATED_SHORT_FRAGMENT_SOURCE_GAP_US
MIN_ISOLATED_SHORT_FRAGMENT_NEIGHBOR_CHARS = _pre_visible_semantic_junk_rules.MIN_ISOLATED_SHORT_FRAGMENT_NEIGHBOR_CHARS
MAX_SOURCE_BOUNDARY_PREFIX_GAP_US = _source_boundary_prefix_rules.MAX_SOURCE_BOUNDARY_PREFIX_GAP_US
MAX_SOURCE_BOUNDARY_COMPOUND_GAP_US = _source_boundary_prefix_rules.MAX_SOURCE_BOUNDARY_COMPOUND_GAP_US
SOURCE_BOUNDARY_FUNCTION_PREFIXES = _source_boundary_prefix_rules.SOURCE_BOUNDARY_FUNCTION_PREFIXES
SOURCE_BOUNDARY_PREFIX_DEPENDENT_STARTS = _source_boundary_prefix_rules.SOURCE_BOUNDARY_PREFIX_DEPENDENT_STARTS
SOURCE_BOUNDARY_COMPOUND_SUFFIXES = _source_boundary_prefix_rules.SOURCE_BOUNDARY_COMPOUND_SUFFIXES
MIN_TRANSFERRED_PREFIX_TARGET_US = _source_boundary_prefix_rules.MIN_TRANSFERRED_PREFIX_TARGET_US
MAX_TRANSFERRED_PREFIX_TARGET_US = _source_boundary_prefix_rules.MAX_TRANSFERRED_PREFIX_TARGET_US
MAX_CAPTION_ONLY_TARGET_GAP_US = _caption_only_merge_rules.MAX_CAPTION_ONLY_TARGET_GAP_US
MAX_SAME_SUBTITLE_SHORT_TAIL_CHARS = _caption_only_merge_rules.MAX_SAME_SUBTITLE_SHORT_TAIL_CHARS
MAX_SAME_SUBTITLE_SHORT_TAIL_SOURCE_GAP_US = _caption_only_merge_rules.MAX_SAME_SUBTITLE_SHORT_TAIL_SOURCE_GAP_US
MIN_REPAIRED_SEGMENT_DURATION_US = _short_residual_rules.MIN_REPAIRED_SEGMENT_DURATION_US
MAX_REPAIRED_RESIDUAL_DROP_DURATION_US = _short_residual_rules.MAX_REPAIRED_RESIDUAL_DROP_DURATION_US
MAX_REPAIRED_RESIDUAL_DROP_CHARS = _short_residual_rules.MAX_REPAIRED_RESIDUAL_DROP_CHARS
MIN_REBALANCED_CAPTION_DURATION_US = _short_residual_rules.MIN_REBALANCED_CAPTION_DURATION_US
_repair_leading_filler_gap = _leading_filler_rules._repair_leading_filler_gap
_repair_connector_single_word_intrusion = _connector_intrusion_rules._repair_connector_single_word_intrusion
_repair_connector_filler_restart = _connector_intrusion_rules._repair_connector_filler_restart
_repair_repeated_object_head_tail = _connector_intrusion_rules._repair_repeated_object_head_tail
_repair_subject_prefix_completed_predicate_restart = _connector_intrusion_rules._repair_subject_prefix_completed_predicate_restart
_repair_pre_visible_semantic_junk_candidate = _pre_visible_semantic_junk_rules._repair_pre_visible_semantic_junk_candidate
_is_deterministic_pre_visible_semantic_junk_drop = _pre_visible_semantic_junk_rules._is_deterministic_pre_visible_semantic_junk_drop
_repair_isolated_semantic_junk_caption = _pre_visible_semantic_junk_rules._repair_isolated_semantic_junk_caption
_is_isolated_short_source_gap_fragment = _pre_visible_semantic_junk_rules._is_isolated_short_source_gap_fragment
_transfer_leading_function_prefix_to_previous_caption = _source_boundary_prefix_rules._transfer_leading_function_prefix_to_previous_caption
_target_boundary_after_leading_word = _source_boundary_prefix_rules._target_boundary_after_leading_word
_repair_source_boundary_prefix_gap = _source_boundary_prefix_rules._repair_source_boundary_prefix_gap
_repair_omitted_legal_reduplication_word = _source_boundary_prefix_rules._repair_omitted_legal_reduplication_word
_source_boundary_prefix_candidate = _source_boundary_prefix_rules._source_boundary_prefix_candidate
_repair_source_boundary_compound_suffix_gap = _source_boundary_prefix_rules._repair_source_boundary_compound_suffix_gap
_repair_source_boundary_truncated_compound_tail = _source_boundary_prefix_rules._repair_source_boundary_truncated_compound_tail
_source_boundary_compound_candidate = _source_boundary_prefix_rules._source_boundary_compound_candidate
_source_boundary_compound_words_match = _source_boundary_prefix_rules._source_boundary_compound_words_match
_merge_source_boundary_compound_segments = _source_boundary_prefix_rules._merge_source_boundary_compound_segments
_source_boundary_prefix_dependent_start = _source_boundary_prefix_rules._source_boundary_prefix_dependent_start
_apply_source_boundary_prefix_candidate = _source_boundary_prefix_rules._apply_source_boundary_prefix_candidate
_finalize_caption_only_dangling_merges = _caption_only_merge_rules._finalize_caption_only_dangling_merges
_finalize_subject_prefix_completed_predicate_caption_merges = _caption_only_merge_rules._finalize_subject_prefix_completed_predicate_caption_merges
_finalize_same_subtitle_short_tail_caption_merges = _caption_only_merge_rules._finalize_same_subtitle_short_tail_caption_merges
_repair_dangling_prefix_suffix_caption_only = _caption_only_merge_rules._repair_dangling_prefix_suffix_caption_only
_merge_adjacent_caption_segments = _caption_only_merge_rules._merge_adjacent_caption_segments
_merge_adjacent_captions = _caption_only_merge_rules._merge_adjacent_captions
_caption_only_merge_allowed = _caption_only_merge_rules._caption_only_merge_allowed
_same_subtitle_short_tail_should_merge = _caption_only_merge_rules._same_subtitle_short_tail_should_merge
_source_subtitle_texts_by_uid = _caption_only_merge_rules._source_subtitle_texts_by_uid
_caption_only_materialization_for_visible_caption = _caption_only_merge_rules._caption_only_materialization_for_visible_caption
_caption_only_source_windows = _caption_only_merge_rules._caption_only_source_windows
_caption_only_window_gaps_are_safe = _caption_only_merge_rules._caption_only_window_gaps_are_safe
_visible_target_range_covers_materialization = _caption_only_merge_rules._visible_target_range_covers_materialization
_caption_only_replacements = _caption_only_merge_rules._caption_only_replacements
_caption_only_state_signature = _caption_only_merge_rules._caption_only_state_signature
_caption_ids_with_dangling_boundary_candidates = _pre_visible_semantic_junk_rules._caption_ids_with_dangling_boundary_candidates
_caption_source_range = _pre_visible_semantic_junk_rules._caption_source_range
_merge_short_repaired_segments = _short_residual_rules._merge_short_repaired_segments
_repair_short_repair_residual_segments = _short_residual_rules._repair_short_repair_residual_segments
_cleanup_short_repair_residual_segments = _short_residual_rules._cleanup_short_repair_residual_segments
_next_short_repair_residual_action = _short_residual_rules._next_short_repair_residual_action
_is_short_repair_residual_segment = _short_residual_rules._is_short_repair_residual_segment
_can_merge_short_repair_residual = _short_residual_rules._can_merge_short_repair_residual
_can_drop_short_repair_residual = _short_residual_rules._can_drop_short_repair_residual
_merge_timeline_segment_pair_at = _short_residual_rules._merge_timeline_segment_pair_at
_repair_same_segment_de_duplicate_prefix = _restart_repeat_rules._repair_same_segment_de_duplicate_prefix
_leading_duplicate_word_count = _restart_repeat_rules._leading_duplicate_word_count
_repair_dangling_pronoun_modal_suffix = _restart_repeat_rules._repair_dangling_pronoun_modal_suffix
_trim_asr_restart_prefix = _restart_repeat_rules._trim_asr_restart_prefix
_trim_restart_repeat_visible_prefix = _restart_repeat_rules._trim_restart_repeat_visible_prefix
_drop_restart_repeat_word_span = _restart_repeat_rules._drop_restart_repeat_word_span
_candidate_window_captions = _restart_repeat_rules._candidate_window_captions
_partial_previous_tail_match = _restart_repeat_rules._partial_previous_tail_match
_partial_tail_visible_text_match = _restart_repeat_rules._partial_tail_visible_text_match
_repair_de_shi_duplicate_bridge = _de_shi_bridge_rules._repair_de_shi_duplicate_bridge
_drop_repeated_caption_span = _de_shi_bridge_rules._drop_repeated_caption_span
_drop_or_trim_caption_words = _word_span_edit_rules._drop_or_trim_caption_words
_trim_word_ids_from_timeline = _word_span_edit_rules._trim_word_ids_from_timeline
_drop_contiguous_word_ids_from_timeline = _word_span_edit_rules._drop_contiguous_word_ids_from_timeline
_contains_contiguous_subsequence = _word_span_edit_rules._contains_contiguous_subsequence
_leading_word_ids_for_text = _word_span_edit_rules._leading_word_ids_for_text
_trailing_word_ids_for_text = _word_span_edit_rules._trailing_word_ids_for_text
_contiguous_word_ids_for_text = _word_span_edit_rules._contiguous_word_ids_for_text
_segment_with_word_ids = _word_span_edit_rules._segment_with_word_ids
_segment_with_word_ids_preserving_effective_speed = _word_span_edit_rules._segment_with_word_ids_preserving_effective_speed
_segments_with_word_ids_preserving_effective_speed = _word_span_edit_rules._segments_with_word_ids_preserving_effective_speed
_source_range_has_unselected_words = _word_span_edit_rules._source_range_has_unselected_words
_source_bounds_for_word_ids = _word_span_edit_rules._source_bounds_for_word_ids
_merged_segment_pair_preserving_effective_speed = _word_span_edit_rules._merged_segment_pair_preserving_effective_speed
_target_duration_preserving_effective_speed = _word_span_edit_rules._target_duration_preserving_effective_speed
_unique_split_segment_id = _word_span_edit_rules._unique_split_segment_id
_caption_segments_exclusive = _word_span_edit_rules._caption_segments_exclusive
_safe_merge_segments = _word_span_edit_rules._safe_merge_segments
_source_gap_has_unselected_words = _word_span_edit_rules._source_gap_has_unselected_words
