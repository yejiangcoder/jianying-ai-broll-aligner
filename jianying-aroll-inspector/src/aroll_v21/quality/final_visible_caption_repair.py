from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.final_visible_repair.convergence import (
    _caption_only_state_signature as _caption_only_state_signature_impl,
    _repair_state_signature as _repair_state_signature_impl,
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
    boundary_restart as _boundary_restart_rules,
    caption_only_merge as _caption_only_merge_rules,
    connector_intrusion as _connector_intrusion_rules,
    de_shi_bridge as _de_shi_bridge_rules,
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
from aroll_v21.quality.final_visible_repair.proposal import TimelineRepairProposal
from aroll_v21.quality.final_visible_repair.timeline_materializer import (
    apply_timeline_repair_proposal as _apply_timeline_repair_proposal,
)
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
from aroll_v21.quality.subtitle_readability import HARD_MAX_CHARS, HARD_MAX_DURATION_US
from aroll_v21.quality.tiny_caption_classification import build_tiny_caption_classification_report
from aroll_v21.quality.tiny_segment_classifier import classify_tiny_segment


MAX_FINAL_VISIBLE_REPAIR_PASSES = 128
CONTAINED_SHORT_FRAGMENT_OPEN_TAIL_CHARS = set("\u7684\u5f97\u5730\u4e4b\u5728\u4ece\u5bf9\u628a\u88ab\u5c06\u8ba9\u4f7f\u8ddf\u548c\u4e0e\u6216\u53ca\u4ee5\u4e3a\u4e8e\u5230")
OPEN_TAIL_SHORT_CAPTION_MAX_CHARS = 5
OPEN_TAIL_SHORT_CAPTION_MAX_GAP_US = 120_000
OPEN_TAIL_SHORT_CAPTION_MERGE_TAILS = set("\u7684\u5f97\u5730\u4e4b")
SHORT_ABORTED_PREFIX_MAX_CHARS = 5
SHORT_ABORTED_PREFIX_MAX_GAP_US = 300_000
COMMON_CLOSED_DE_PHRASES = {
    "可以的",
    "不会的",
    "不是的",
    "对的",
    "好的",
    "真的",
    "假的",
    "是的",
}































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
    actions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    max_pass_limit = max(1, int(max_passes))
    current_signature = _repair_state_signature(current_timeline, current_captions)
    seen_signatures: set[tuple[Any, ...]] = {current_signature}
    stop_reason = ""
    passes_executed = 0

    for pass_index in range(max_pass_limit):
        passes_executed = pass_index + 1
        leading_filler_step = _repair_leading_filler_gap(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if leading_filler_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(leading_filler_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(leading_filler_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": leading_filler_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        connector_intrusion_step = _repair_connector_single_word_intrusion(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if connector_intrusion_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(connector_intrusion_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(connector_intrusion_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": connector_intrusion_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        connector_restart_step = _repair_connector_filler_restart(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if connector_restart_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(connector_restart_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(connector_restart_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": connector_restart_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        repeated_object_head_step = _repair_repeated_object_head_tail(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if repeated_object_head_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(repeated_object_head_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(repeated_object_head_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": repeated_object_head_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        subject_prefix_restart_step = _repair_subject_prefix_completed_predicate_restart(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if subject_prefix_restart_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(subject_prefix_restart_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(subject_prefix_restart_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": subject_prefix_restart_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        pre_semantic_junk_step = _repair_pre_visible_semantic_junk_candidate(
            final_timeline=current_timeline,
            captions=current_captions,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if pre_semantic_junk_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(pre_semantic_junk_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(pre_semantic_junk_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": pre_semantic_junk_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        omitted_reduplication_step = _repair_omitted_legal_reduplication_word(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if omitted_reduplication_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(omitted_reduplication_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(omitted_reduplication_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": omitted_reduplication_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        source_prefix_step = _repair_source_boundary_prefix_gap(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if source_prefix_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(source_prefix_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(source_prefix_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": source_prefix_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        compound_step = _repair_source_boundary_compound_suffix_gap(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if compound_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(compound_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(compound_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": compound_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        truncated_tail_step = _repair_source_boundary_truncated_compound_tail(
            final_timeline=current_timeline,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if truncated_tail_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(truncated_tail_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(truncated_tail_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": truncated_tail_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        junk_step = _repair_isolated_semantic_junk_caption(
            final_timeline=current_timeline,
            captions=current_captions,
            source_graph=source_graph,
            pass_index=pass_index + 1,
        )
        if junk_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(junk_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(junk_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": junk_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        repeated_island_step, repeated_island_unresolved = _repair_repeated_island_with_proposal(
            final_timeline=current_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index + 1,
        )
        if repeated_island_unresolved is not None:
            unresolved.append(repeated_island_unresolved)
            stop_reason = str(repeated_island_unresolved.get("reason") or "repeated_island_proposal_failed")
            break
        if repeated_island_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(repeated_island_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(repeated_island_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": repeated_island_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        boundary_restart_step, boundary_restart_unresolved = _repair_boundary_restart_with_proposal(
            final_timeline=current_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index + 1,
        )
        if boundary_restart_unresolved is not None:
            unresolved.append(boundary_restart_unresolved)
            stop_reason = str(boundary_restart_unresolved.get("reason") or "boundary_restart_proposal_failed")
            break
        if boundary_restart_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(boundary_restart_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(boundary_restart_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": boundary_restart_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        containment_fragment_step, containment_fragment_unresolved = _repair_contained_short_fragment_with_proposal(
            final_timeline=current_timeline,
            captions=current_captions,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index + 1,
        )
        if containment_fragment_unresolved is not None:
            unresolved.append(containment_fragment_unresolved)
            stop_reason = str(containment_fragment_unresolved.get("reason") or "contained_short_fragment_proposal_failed")
            break
        if containment_fragment_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(containment_fragment_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(containment_fragment_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": containment_fragment_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        self_repair_step, self_repair_unresolved = _repair_self_repair_aborted_phrase_with_proposal(
            final_timeline=current_timeline,
            captions=current_captions,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index + 1,
        )
        if self_repair_unresolved is not None:
            unresolved.append(self_repair_unresolved)
            stop_reason = str(self_repair_unresolved.get("reason") or "self_repair_aborted_phrase_proposal_failed")
            break
        if self_repair_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(self_repair_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(self_repair_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": self_repair_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        short_aborted_prefix_step, short_aborted_prefix_unresolved = _repair_short_aborted_prefix_caption_with_proposal(
            final_timeline=current_timeline,
            captions=current_captions,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index + 1,
        )
        if short_aborted_prefix_unresolved is not None:
            unresolved.append(short_aborted_prefix_unresolved)
            stop_reason = str(short_aborted_prefix_unresolved.get("reason") or "short_aborted_prefix_caption_proposal_failed")
            break
        if short_aborted_prefix_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(short_aborted_prefix_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(short_aborted_prefix_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": short_aborted_prefix_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        open_tail_merge_step = _repair_open_tail_short_caption_with_next(
            final_timeline=current_timeline,
            captions=current_captions,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index + 1,
        )
        if open_tail_merge_step is not None:
            current_timeline = _repack_timeline(open_tail_merge_step.final_timeline)
            current_captions = open_tail_merge_step.captions
            actions.append(open_tail_merge_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": open_tail_merge_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        tiny_residual_step, tiny_residual_unresolved = _repair_fatal_tiny_caption_with_proposal(
            final_timeline=current_timeline,
            captions=current_captions,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index + 1,
        )
        if tiny_residual_unresolved is not None:
            unresolved.append(tiny_residual_unresolved)
            stop_reason = str(tiny_residual_unresolved.get("reason") or "tiny_caption_residual_proposal_failed")
            break
        if tiny_residual_step is not None:
            previous_captions = current_captions
            current_timeline = _repack_timeline(tiny_residual_step.final_timeline)
            current_captions = _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            actions.append(tiny_residual_step.action)
            next_signature = _repair_state_signature(current_timeline, current_captions)
            if next_signature == current_signature or next_signature in seen_signatures:
                stop_reason = "no_progress_detected"
                unresolved.append(
                    {
                        "pass_index": pass_index + 1,
                        "reason": stop_reason,
                        "last_action": tiny_residual_step.action,
                    }
                )
                break
            seen_signatures.add(next_signature)
            current_signature = next_signature
            continue

        rendered_gate = build_final_caption_visible_repeat_gate(current_captions)
        timeline_captions = _timeline_caption_units(current_timeline, source_graph)
        effective_timeline_captions, timeline_materializations = _effective_timeline_caption_units(timeline_captions, current_captions)
        timeline_gate = _timeline_gate(effective_timeline_captions, timeline_materializations)
        rendered_counts = _repair_counts(rendered_gate)
        timeline_counts = _repair_counts(timeline_gate)
        if not any(rendered_counts.values()) and not any(timeline_counts.values()):
            stop_reason = "converged"
            break
        step = _repair_next_issue(
            final_timeline=current_timeline,
            captions=current_captions,
            source_graph=source_graph,
            gate=rendered_gate,
            pass_index=pass_index + 1,
            issue_types={"dangling_prefix_suffix"},
        )
        if step is None:
            step = _repair_next_issue(
                final_timeline=current_timeline,
                captions=effective_timeline_captions,
                source_graph=source_graph,
                gate=timeline_gate,
                pass_index=pass_index + 1,
            )
        if step is None:
            step = _repair_next_issue(
                final_timeline=current_timeline,
                captions=current_captions,
                source_graph=source_graph,
                gate=rendered_gate,
                pass_index=pass_index + 1,
            )
        if step is None:
            unresolved.append(
                {
                    "pass_index": pass_index + 1,
                    "counts": rendered_counts,
                    "timeline_counts": timeline_counts,
                    "blocker_codes": list(rendered_gate.get("blocker_codes") or []),
                    "timeline_blocker_codes": list(timeline_gate.get("blocker_codes") or []),
                    "reason": "no_safe_deterministic_repair_available",
                }
            )
            stop_reason = "no_safe_deterministic_repair_available"
            break
        previous_captions = current_captions
        current_timeline = _repack_timeline(step.final_timeline)
        current_captions = (
            _render_captions_preserving_caption_only_materializations(
                current_timeline,
                previous_captions,
                render_captions,
            )
            if step.timeline_changed
            else _renumber_captions(step.captions)
        )
        actions.append(step.action)
        next_signature = _repair_state_signature(current_timeline, current_captions)
        if next_signature == current_signature or next_signature in seen_signatures:
            stop_reason = "no_progress_detected"
            unresolved.append(
                {
                    "pass_index": pass_index + 1,
                    "reason": stop_reason,
                    "last_action": step.action,
                }
            )
            break
        seen_signatures.add(next_signature)
        current_signature = next_signature

    residual_step = _repair_short_repair_residual_segments(
        final_timeline=current_timeline,
        source_graph=source_graph,
        pass_index=len(actions) + 1,
    )
    if residual_step is not None:
        previous_captions = current_captions
        current_timeline = _repack_timeline(residual_step.final_timeline)
        current_captions = _render_captions_preserving_caption_only_materializations(
            current_timeline,
            previous_captions,
            render_captions,
        )
        actions.append(residual_step.action)

    current_captions, final_caption_only_actions = _finalize_caption_only_dangling_merges(
        current_captions,
        source_graph=source_graph,
        pass_index_start=len(actions) + 1,
    )
    actions.extend(final_caption_only_actions)
    current_captions, subject_prefix_caption_actions = _finalize_subject_prefix_completed_predicate_caption_merges(
        current_captions,
        final_timeline=current_timeline,
        source_graph=source_graph,
        pass_index_start=len(actions) + 1,
    )
    actions.extend(subject_prefix_caption_actions)
    current_captions, same_subtitle_short_tail_actions = _finalize_same_subtitle_short_tail_caption_merges(
        current_captions,
        source_graph=source_graph,
        pass_index_start=len(actions) + 1,
    )
    actions.extend(same_subtitle_short_tail_actions)

    final_gate = build_final_caption_visible_repeat_gate(current_captions)
    final_semantic_junk_report = build_pre_visible_semantic_junk_candidate_report(current_captions, source_graph)
    semantic_junk_actions = [
        action
        for action in actions
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
        _timeline_caption_units(current_timeline, source_graph),
        current_captions,
    )
    final_timeline_gate = _timeline_gate(final_effective_timeline_captions, final_materializations)
    final_repeated_island_candidates = [
        candidate.to_evidence()
        for candidate in _repeated_island_rules.detect_repeated_island_candidates(current_timeline, source_graph)
    ]
    boundary_restart_actions = [
        action
        for action in actions
        if str(action.get("issue_type") or "") == "boundary_restart"
    ]
    repeated_island_actions = [
        action
        for action in actions
        if str(action.get("issue_type") or "") == "repeated_island"
    ]
    timeline_repair_proposal_actions = [
        action
        for action in actions
        if str(action.get("proposal_id") or "")
    ]
    final_counts = _repair_counts(final_gate)
    final_timeline_counts = _repair_counts(final_timeline_gate)
    repair_success = not any(final_counts.values()) and not any(final_timeline_counts.values())
    if not repair_success and not unresolved:
        reason = "max_repair_passes_exhausted" if len(actions) >= max_pass_limit else "unresolved_after_repair"
        stop_reason = reason
        unresolved.append(
            {
                "pass_index": len(actions) + 1,
                "counts": final_counts,
                "timeline_counts": final_timeline_counts,
                "blocker_codes": list(final_gate.get("blocker_codes") or []),
                "timeline_blocker_codes": list(final_timeline_gate.get("blocker_codes") or []),
                "reason": reason,
            }
        )
    if repair_success and not stop_reason:
        stop_reason = "converged"

    report = {
        "final_visible_repair_enabled": True,
        "final_visible_repair_attempted": bool(actions) or any(_repair_counts(initial_gate).values()) or any(_repair_counts(initial_timeline_gate).values()),
        "final_visible_repair_success": repair_success,
        "final_visible_repair_max_passes": max_pass_limit,
        "final_visible_repair_passes_executed": passes_executed,
        "final_visible_repair_stop_reason": stop_reason,
        "final_visible_repair_no_progress_detected": stop_reason == "no_progress_detected",
        "final_visible_repair_max_pass_exhausted": any(
            str(row.get("reason") or "") == "max_repair_passes_exhausted"
            for row in unresolved
        ),
        "final_visible_repair_progress_state_count": len(seen_signatures),
        "final_visible_repair_action_count": len(actions),
        "final_visible_repair_actions": actions,
        "final_visible_repair_unresolved": unresolved,
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
            for action in actions
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
        final_timeline=current_timeline,
        captions=current_captions,
        report=report,
    )


class _RenderCallbackAdapter:
    def __init__(self, render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]]) -> None:
        self._render_captions = render_captions

    def render(
        self,
        final_timeline: list[FinalTimelineSegment],
        _source_graph: CanonicalSourceGraph,
    ) -> list[CaptionRenderUnit]:
        return self._render_captions(final_timeline)


def _repair_boundary_restart_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    proposals = _boundary_restart_rules.build_boundary_restart_proposals(final_timeline, source_graph)
    if not proposals:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    proposal = proposals[0]
    materialized = _apply_timeline_repair_proposal(
        proposal,
        final_timeline,
        source_graph,
        renderer=_RenderCallbackAdapter(render_captions),
    )
    if not materialized.applied:
        unresolved = {
            "pass_index": pass_index,
            "issue_type": proposal.issue_type,
            "proposal_id": proposal.proposal_id,
            "reason": materialized.reason,
            "blocker_code": materialized.blocker_code,
            "target_segment_id": proposal.target_segment_id,
            "target_word_ids": list(proposal.target_word_ids),
            "evidence": dict(proposal.evidence),
        }
        no_step: _RepairStep | None = None
        return no_step, unresolved
    action = _action(
        "boundary_restart",
        "suffix_trim",
        pass_index,
        dict(proposal.evidence),
        proposal_id=proposal.proposal_id,
        repair_action=proposal.repair_action,
        confidence=float(proposal.confidence),
        target_segment_id=proposal.target_segment_id,
        target_word_ids=list(proposal.target_word_ids),
        target_text=proposal.target_text,
        risk_tags=list(proposal.risk_tags),
        evidence=dict(proposal.evidence),
        coverage_report={
            "missing_final_timeline_caption_word_count": int(
                materialized.coverage_report.get("missing_final_timeline_caption_word_count") or 0
            ),
            "prewrite_uncaptioned_spoken_word_count": int(
                materialized.coverage_report.get("prewrite_uncaptioned_spoken_word_count") or 0
            ),
        },
    )
    return (
        _RepairStep(
            final_timeline=materialized.final_timeline,
            captions=materialized.captions,
            timeline_changed=True,
            action=action,
        ),
        None,
    )


def _repair_repeated_island_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    proposals = _repeated_island_rules.build_repeated_island_proposals(final_timeline, source_graph)
    if not proposals:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    proposal = proposals[0]
    materialized = _apply_timeline_repair_proposal(
        proposal,
        final_timeline,
        source_graph,
        renderer=_RenderCallbackAdapter(render_captions),
    )
    if not materialized.applied:
        unresolved = {
            "pass_index": pass_index,
            "issue_type": proposal.issue_type,
            "proposal_id": proposal.proposal_id,
            "reason": materialized.reason,
            "blocker_code": materialized.blocker_code,
            "target_segment_id": proposal.target_segment_id,
            "target_word_ids": list(proposal.target_word_ids),
            "evidence": dict(proposal.evidence),
        }
        no_step: _RepairStep | None = None
        return no_step, unresolved
    action = _action(
        "repeated_island",
        "internal_drop",
        pass_index,
        dict(proposal.evidence),
        proposal_id=proposal.proposal_id,
        repair_action=proposal.repair_action,
        confidence=float(proposal.confidence),
        target_segment_id=proposal.target_segment_id,
        target_word_ids=list(proposal.target_word_ids),
        target_text=proposal.target_text,
        risk_tags=list(proposal.risk_tags),
        evidence=dict(proposal.evidence),
        coverage_report={
            "missing_final_timeline_caption_word_count": int(
                materialized.coverage_report.get("missing_final_timeline_caption_word_count") or 0
            ),
            "prewrite_uncaptioned_spoken_word_count": int(
                materialized.coverage_report.get("prewrite_uncaptioned_spoken_word_count") or 0
            ),
        },
    )
    return (
        _RepairStep(
            final_timeline=materialized.final_timeline,
            captions=materialized.captions,
            timeline_changed=True,
            action=action,
        ),
        None,
    )


def _repair_contained_short_fragment_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    gate = build_final_caption_visible_repeat_gate(captions)
    candidates = [
        row
        for row in list(gate.get("containment_repeat_candidates") or [])
        if str(row.get("severity") or "") in {"fatal", "high"}
        and str(row.get("classification") or "") == "local_containment_restart"
        and str(row.get("distance_kind") or "") in {"adjacent", "near"}
    ]
    if not candidates:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    captions_by_id = {caption.caption_id: caption for caption in captions}
    for candidate in candidates:
        left = captions_by_id.get(str(candidate.get("caption_id") or ""))
        right = captions_by_id.get(str(candidate.get("related_caption_id") or ""))
        drop_caption, kept_caption = _contained_short_fragment_drop_caption(left, right)
        if drop_caption is None or kept_caption is None:
            continue
        proposal = _caption_span_drop_proposal(
            proposal_id=f"contained_short_fragment_{pass_index:06d}_{drop_caption.caption_id}",
            issue_type="contained_short_caption_fragment",
            confidence=0.94,
            repair_action="span_drop",
            caption=drop_caption,
            final_timeline=final_timeline,
            source_graph=source_graph,
            risk_tags=["local_containment_restart", "contained_short_fragment"],
            evidence={
                "candidate": dict(candidate),
                "dropped_caption_id": drop_caption.caption_id,
                "kept_caption_id": kept_caption.caption_id,
                "dropped_text": drop_caption.text,
                "kept_text": kept_caption.text,
                "policy": "drop_open_tail_short_fragment_that_is_prefix_of_adjacent_complete_caption",
            },
        )
        if proposal is None:
            continue
        step, unresolved = _apply_caption_span_drop_proposal(
            proposal=proposal,
            final_timeline=final_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index,
            decision="span_drop",
        )
        if step is not None or unresolved is not None:
            return step, unresolved
    no_step: _RepairStep | None = None
    no_unresolved: dict[str, Any] | None = None
    return no_step, no_unresolved


def _repair_self_repair_aborted_phrase_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    gate = build_final_caption_visible_repeat_gate(captions)
    candidates = [
        row
        for row in list(gate.get("self_repair_aborted_phrase_candidates") or [])
        if bool(row.get("deterministic_drop_left"))
    ]
    if not candidates:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    captions_by_id = {caption.caption_id: caption for caption in captions}
    for candidate in candidates:
        drop_caption = captions_by_id.get(str(candidate.get("caption_id") or ""))
        kept_caption = captions_by_id.get(str(candidate.get("related_caption_id") or ""))
        if drop_caption is None or kept_caption is None:
            continue
        proposal = _caption_span_drop_proposal(
            proposal_id=f"self_repair_aborted_phrase_{pass_index:06d}_{drop_caption.caption_id}",
            issue_type="self_repair_aborted_phrase",
            confidence=float(candidate.get("similarity") or 0.9),
            repair_action="span_drop",
            caption=drop_caption,
            final_timeline=final_timeline,
            source_graph=source_graph,
            risk_tags=["self_repair_aborted_phrase", "drop_left_keep_right"],
            evidence={
                "candidate": dict(candidate),
                "dropped_caption_id": drop_caption.caption_id,
                "kept_caption_id": kept_caption.caption_id,
                "dropped_text": drop_caption.text,
                "kept_text": kept_caption.text,
                "policy": "drop_left_aborted_phrase_keep_completed_restart",
            },
        )
        if proposal is None:
            continue
        step, unresolved = _apply_caption_span_drop_proposal(
            proposal=proposal,
            final_timeline=final_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index,
            decision="drop_left_keep_right",
        )
        if step is not None or unresolved is not None:
            return step, unresolved
    no_step: _RepairStep | None = None
    no_unresolved: dict[str, Any] | None = None
    return no_step, no_unresolved


def _repair_short_aborted_prefix_caption_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    ordered = _ordered_captions(captions)
    for left, right in zip(ordered, ordered[1:]):
        row = _short_aborted_prefix_candidate(left, right)
        if not row:
            continue
        proposal = _caption_span_drop_proposal(
            proposal_id=f"short_aborted_prefix_caption_{pass_index:06d}_{left.caption_id}",
            issue_type="short_aborted_prefix_caption",
            confidence=float(row.get("confidence") or 0.92),
            repair_action="span_drop",
            caption=left,
            final_timeline=final_timeline,
            source_graph=source_graph,
            risk_tags=["short_aborted_prefix_caption", "single_char_asr_tail"],
            evidence={
                **row,
                "dropped_caption_id": left.caption_id,
                "kept_caption_id": right.caption_id,
                "dropped_text": left.text,
                "kept_text": right.text,
                "policy": "drop_short_caption_restarted_by_adjacent_longer_caption_with_single_char_tail_mismatch",
            },
        )
        if proposal is None:
            continue
        step, unresolved = _apply_caption_span_drop_proposal(
            proposal=proposal,
            final_timeline=final_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index,
            decision="span_drop",
        )
        if step is not None or unresolved is not None:
            return step, unresolved
    no_step: _RepairStep | None = None
    no_unresolved: dict[str, Any] | None = None
    return no_step, no_unresolved


def _repair_open_tail_short_caption_with_next(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> _RepairStep | None:
    ordered = _ordered_captions(captions)
    for index in range(len(ordered) - 1):
        current = ordered[index]
        next_caption = ordered[index + 1]
        if not _open_tail_short_caption_should_merge(current, next_caption):
            continue
        merged_timeline = _merge_adjacent_caption_segments(final_timeline, current, next_caption, source_graph)
        if merged_timeline is not None:
            rendered_captions = _render_captions_preserving_caption_only_materializations(
                merged_timeline,
                captions,
                render_captions,
            )
            return _RepairStep(
                final_timeline=merged_timeline,
                captions=rendered_captions,
                timeline_changed=True,
                action=_action(
                    "open_tail_short_caption",
                    "merge_with_next_segment",
                    pass_index,
                    {
                        "caption_id": current.caption_id,
                        "related_caption_id": next_caption.caption_id,
                        "text": current.text,
                        "related_text": next_caption.text,
                    },
                    affected_caption_ids=[current.caption_id, next_caption.caption_id],
                    target_gap_us=int(next_caption.target_start_us) - int(current.target_end_us),
                ),
            )
    no_step: _RepairStep | None = None
    return no_step


def _short_aborted_prefix_candidate(
    left: CaptionRenderUnit,
    right: CaptionRenderUnit,
) -> dict[str, Any] | None:
    no_candidate: dict[str, Any] | None = None
    left_text = normalize_text(left.text)
    right_text = normalize_text(right.text)
    if not left_text or not right_text or left_text == right_text:
        return no_candidate
    gap_us = int(right.target_start_us) - int(left.target_end_us)
    if gap_us < -80_000 or gap_us > SHORT_ABORTED_PREFIX_MAX_GAP_US:
        return no_candidate
    if len(left_text) > SHORT_ABORTED_PREFIX_MAX_CHARS or len(right_text) < len(left_text) + 2:
        return no_candidate
    prefix_len = _common_prefix_len(left_text, right_text)
    left_tail = left_text[prefix_len:]
    right_tail = right_text[prefix_len:]
    if prefix_len < 2 or len(left_tail) != 1 or len(right_tail) < 2:
        return no_candidate
    if right_text.startswith(left_text):
        return no_candidate
    return {
        "reason": "short caption is reopened by the next caption with a longer continuation and a single-character tail mismatch",
        "shared_prefix": left_text[:prefix_len],
        "left_tail": left_tail,
        "right_tail": right_tail,
        "gap_us": gap_us,
        "confidence": 0.92,
    }


def _open_tail_short_caption_should_merge(
    current: CaptionRenderUnit,
    next_caption: CaptionRenderUnit,
) -> bool:
    text = normalize_text(current.text)
    next_text = normalize_text(next_caption.text)
    if not text or not next_text:
        return False
    if len(text) > OPEN_TAIL_SHORT_CAPTION_MAX_CHARS:
        return False
    if text in COMMON_CLOSED_DE_PHRASES:
        return False
    if text.startswith("是") and text.endswith("的"):
        return False
    if next_text.startswith(("的", "的是")):
        return False
    if text[-1] not in OPEN_TAIL_SHORT_CAPTION_MERGE_TAILS:
        return False
    gap_us = int(next_caption.target_start_us) - int(current.target_end_us)
    if gap_us < -80_000 or gap_us > OPEN_TAIL_SHORT_CAPTION_MAX_GAP_US:
        return False
    combined = normalize_text(f"{current.text}{next_caption.text}")
    return bool(combined) and len(combined) <= HARD_MAX_CHARS


def _common_prefix_len(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def _contained_short_fragment_drop_caption(
    left: CaptionRenderUnit | None,
    right: CaptionRenderUnit | None,
) -> tuple[CaptionRenderUnit | None, CaptionRenderUnit | None]:
    if left is None or right is None:
        return None, None
    left_text = normalize_text(left.text)
    right_text = normalize_text(right.text)
    if not left_text or not right_text or left_text == right_text:
        return None, None
    if right_text.startswith(left_text) and len(right_text) > len(left_text) and _safe_contained_short_fragment(left_text):
        return left, right
    if left_text.startswith(right_text) and len(left_text) > len(right_text) and _safe_contained_short_fragment(right_text):
        return right, left
    return None, None


def _safe_contained_short_fragment(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized or len(normalized) > 8:
        return False
    return normalized[-1] in CONTAINED_SHORT_FRAGMENT_OPEN_TAIL_CHARS


def _caption_span_drop_proposal(
    *,
    proposal_id: str,
    issue_type: str,
    confidence: float,
    repair_action: str,
    caption: CaptionRenderUnit,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    risk_tags: list[str],
    evidence: dict[str, Any],
) -> TimelineRepairProposal | None:
    target_segment_id = _target_segment_id_for_caption(caption, final_timeline)
    if not target_segment_id:
        no_proposal: TimelineRepairProposal | None = None
        return no_proposal
    target_word_ids = [str(word_id) for word_id in caption.word_ids if str(word_id)]
    if not target_word_ids:
        no_proposal: TimelineRepairProposal | None = None
        return no_proposal
    segments_by_id = {segment.segment_id: segment for segment in final_timeline}
    target_segment = segments_by_id.get(target_segment_id)
    if target_segment is None:
        no_proposal: TimelineRepairProposal | None = None
        return no_proposal
    return TimelineRepairProposal(
        proposal_id=proposal_id,
        issue_type=issue_type,
        confidence=confidence,
        target_segment_id=target_segment_id,
        target_word_ids=target_word_ids,
        target_source_start_us=int(caption.spoken_source_start_us or _word_source_start_us(target_word_ids, source_graph) or target_segment.source_start_us),
        target_source_end_us=int(caption.spoken_source_end_us or _word_source_end_us(target_word_ids, source_graph) or target_segment.source_end_us),
        target_text=str(caption.text or ""),
        repair_action=repair_action,
        risk_tags=risk_tags,
        evidence={
            **evidence,
            "target_segment_id": target_segment_id,
            "target_word_ids": target_word_ids,
        },
    )


def _target_segment_id_for_caption(
    caption: CaptionRenderUnit,
    final_timeline: list[FinalTimelineSegment],
) -> str:
    if caption.containing_video_segment_id:
        return str(caption.containing_video_segment_id)
    if len(caption.timeline_segment_ids) == 1:
        return str(caption.timeline_segment_ids[0])
    caption_word_ids = {str(word_id) for word_id in caption.word_ids if str(word_id)}
    if not caption_word_ids:
        return ""
    for segment in final_timeline:
        segment_word_ids = {str(word_id) for word_id in segment.word_ids if str(word_id)}
        if caption_word_ids <= segment_word_ids:
            return str(segment.segment_id)
    return ""


def _word_source_start_us(word_ids: list[str], source_graph: CanonicalSourceGraph) -> int:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for word_id in word_ids:
        word = words_by_id.get(word_id)
        if word is not None:
            return int(word.source_start_us)
    return 0


def _word_source_end_us(word_ids: list[str], source_graph: CanonicalSourceGraph) -> int:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for word_id in reversed(word_ids):
        word = words_by_id.get(word_id)
        if word is not None:
            return int(word.source_end_us)
    return 0


def _apply_caption_span_drop_proposal(
    *,
    proposal: TimelineRepairProposal,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
    decision: str,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    materialized = _apply_timeline_repair_proposal(
        proposal,
        final_timeline,
        source_graph,
        renderer=_RenderCallbackAdapter(render_captions),
    )
    if not materialized.applied:
        unresolved = {
            "pass_index": pass_index,
            "issue_type": proposal.issue_type,
            "proposal_id": proposal.proposal_id,
            "reason": materialized.reason,
            "blocker_code": materialized.blocker_code,
            "target_segment_id": proposal.target_segment_id,
            "target_word_ids": list(proposal.target_word_ids),
            "evidence": dict(proposal.evidence),
        }
        no_step: _RepairStep | None = None
        return no_step, unresolved
    action = _action(
        proposal.issue_type,
        decision,
        pass_index,
        dict(proposal.evidence),
        proposal_id=proposal.proposal_id,
        repair_action=proposal.repair_action,
        confidence=float(proposal.confidence),
        target_segment_id=proposal.target_segment_id,
        target_word_ids=list(proposal.target_word_ids),
        target_text=proposal.target_text,
        risk_tags=list(proposal.risk_tags),
        evidence=dict(proposal.evidence),
        coverage_report={
            "missing_final_timeline_caption_word_count": int(
                materialized.coverage_report.get("missing_final_timeline_caption_word_count") or 0
            ),
            "prewrite_uncaptioned_spoken_word_count": int(
                materialized.coverage_report.get("prewrite_uncaptioned_spoken_word_count") or 0
            ),
        },
    )
    return (
        _RepairStep(
            final_timeline=materialized.final_timeline,
            captions=materialized.captions,
            timeline_changed=True,
            action=action,
        ),
        None,
    )


def _repair_fatal_tiny_caption_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    tiny_report = build_tiny_caption_classification_report(captions)
    fatal_rows = [
        row
        for row in list(tiny_report.get("tiny_caption_classifications") or [])
        if str(row.get("severity") or "") == "fatal"
    ]
    if not fatal_rows:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    captions_by_id = {caption.caption_id: caption for caption in captions}
    segments_by_id = {segment.segment_id: segment for segment in final_timeline}
    for row in fatal_rows:
        caption = captions_by_id.get(str(row.get("caption_id") or ""))
        if caption is None:
            continue
        target_segment_id = str(caption.containing_video_segment_id or "")
        if not target_segment_id and len(caption.timeline_segment_ids) == 1:
            target_segment_id = str(caption.timeline_segment_ids[0])
        if target_segment_id not in segments_by_id:
            continue
        target_word_ids = [str(word_id) for word_id in caption.word_ids if str(word_id)]
        if not target_word_ids:
            continue
        proposal = TimelineRepairProposal(
            proposal_id=f"tiny_caption_residual_{pass_index:06d}_{target_segment_id}",
            issue_type="tiny_caption_residual",
            confidence=0.95,
            target_segment_id=target_segment_id,
            target_word_ids=target_word_ids,
            target_source_start_us=int(caption.spoken_source_start_us or segments_by_id[target_segment_id].source_start_us),
            target_source_end_us=int(caption.spoken_source_end_us or segments_by_id[target_segment_id].source_end_us),
            target_text=str(caption.text or row.get("caption_text") or ""),
            repair_action="span_drop",
            risk_tags=[*list(row.get("risk_tags") or []), "tiny_caption_residual"],
            evidence={
                "caption_id": caption.caption_id,
                "classification": str(row.get("classification") or ""),
                "classification_reason": str(row.get("classification_reason") or ""),
                "caption_text": str(row.get("caption_text") or caption.text or ""),
                "word_ids": target_word_ids,
            },
        )
        materialized = _apply_timeline_repair_proposal(
            proposal,
            final_timeline,
            source_graph,
            renderer=_RenderCallbackAdapter(render_captions),
        )
        if not materialized.applied:
            unresolved = {
                "pass_index": pass_index,
                "issue_type": proposal.issue_type,
                "proposal_id": proposal.proposal_id,
                "reason": materialized.reason,
                "blocker_code": materialized.blocker_code,
                "target_segment_id": proposal.target_segment_id,
                "target_word_ids": list(proposal.target_word_ids),
                "evidence": dict(proposal.evidence),
            }
            no_step: _RepairStep | None = None
            return no_step, unresolved
        action = _action(
            "tiny_caption_residual",
            "span_drop",
            pass_index,
            dict(proposal.evidence),
            proposal_id=proposal.proposal_id,
            repair_action=proposal.repair_action,
            confidence=float(proposal.confidence),
            target_segment_id=proposal.target_segment_id,
            target_word_ids=list(proposal.target_word_ids),
            target_text=proposal.target_text,
            risk_tags=list(proposal.risk_tags),
            evidence=dict(proposal.evidence),
            coverage_report={
                "missing_final_timeline_caption_word_count": int(
                    materialized.coverage_report.get("missing_final_timeline_caption_word_count") or 0
                ),
                "prewrite_uncaptioned_spoken_word_count": int(
                    materialized.coverage_report.get("prewrite_uncaptioned_spoken_word_count") or 0
                ),
            },
        )
        return (
            _RepairStep(
                final_timeline=materialized.final_timeline,
                captions=materialized.captions,
                timeline_changed=True,
                action=action,
            ),
            None,
        )
    no_step: _RepairStep | None = None
    no_unresolved: dict[str, Any] | None = None
    return no_step, no_unresolved


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


def _caption_ids_with_dangling_boundary_candidates(captions: list[CaptionRenderUnit]) -> set[str]:
    gate = build_final_caption_visible_repeat_gate(captions)
    ids: set[str] = set()
    for candidate in list(gate.get("dangling_prefix_suffix_candidates") or []):
        for key in ("caption_id", "related_caption_id"):
            value = str(candidate.get(key) or "")
            if value:
                ids.add(value)
        for value in list(candidate.get("affected_caption_ids") or []):
            if value:
                ids.add(str(value))
    return ids


def _safe_merge_segments(left: FinalTimelineSegment, right: FinalTimelineSegment, source_graph: CanonicalSourceGraph) -> bool:
    if str(left.source_material_id or "") and str(right.source_material_id or "") and str(left.source_material_id) != str(right.source_material_id):
        return False
    if str(left.source_segment_id or "") and str(right.source_segment_id or "") and str(left.source_segment_id) != str(right.source_segment_id):
        return False
    if int(left.target_end_us) <= int(left.target_start_us) or int(right.target_end_us) <= int(right.target_start_us):
        return False
    if int(right.target_start_us) < int(left.target_start_us):
        return False
    source_gap_us = int(right.source_start_us) - int(left.source_end_us)
    if not -80_000 <= source_gap_us <= 1_500_000:
        return False
    return not _source_gap_has_unselected_words(left, right, source_graph)


def _source_gap_has_unselected_words(
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
) -> bool:
    return _source_range_has_unselected_words(
        source_graph=source_graph,
        start_us=int(left.source_end_us),
        end_us=int(right.source_start_us),
        selected_word_ids=set(left.word_ids) | set(right.word_ids),
    )


def _caption_source_range(
    caption: CaptionRenderUnit,
    source_graph: CanonicalSourceGraph,
) -> tuple[int, int] | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    words = [words_by_id[word_id] for word_id in caption.word_ids if word_id in words_by_id]
    if not words or len(words) != len(caption.word_ids):
        no_range: tuple[int, int] | None = None
        return no_range
    start_us = min(int(getattr(word, "source_start_us", 0) or 0) for word in words)
    end_us = max(int(getattr(word, "source_end_us", 0) or 0) for word in words)
    if end_us <= start_us:
        no_range: tuple[int, int] | None = None
        return no_range
    return start_us, end_us


def _segment_with_word_ids(
    segment: FinalTimelineSegment,
    word_ids: list[str],
    source_graph: CanonicalSourceGraph,
) -> FinalTimelineSegment | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    words = [words_by_id[word_id] for word_id in word_ids if word_id in words_by_id]
    if len(words) != len(word_ids):
        no_segment: FinalTimelineSegment | None = None
        return no_segment
    source_start_us = min(int(word.source_start_us) for word in words)
    source_end_us = max(int(word.source_end_us) for word in words)
    duration_us = max(0, source_end_us - source_start_us)
    if duration_us <= 0:
        no_segment: FinalTimelineSegment | None = None
        return no_segment
    return replace(
        segment,
        source_start_us=source_start_us,
        source_end_us=source_end_us,
        target_end_us=int(segment.target_start_us) + duration_us,
        word_ids=list(word_ids),
        text="".join(word.text for word in words),
        spoken_source_start_us=source_start_us,
        spoken_source_end_us=source_end_us,
        clip_source_start_us=source_start_us if segment.clip_source_start_us is not None else segment.clip_source_start_us,
        clip_source_end_us=source_end_us if segment.clip_source_end_us is not None else segment.clip_source_end_us,
        debug_hints={**dict(segment.debug_hints or {}), "final_visible_repair": "trim_repeated_caption_words"},
    )


def _segment_with_word_ids_preserving_effective_speed(
    segment: FinalTimelineSegment,
    word_ids: list[str],
    source_graph: CanonicalSourceGraph,
    repair_reason: str,
) -> FinalTimelineSegment | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    words = [words_by_id[word_id] for word_id in word_ids if word_id in words_by_id]
    if len(words) != len(word_ids):
        no_segment: FinalTimelineSegment | None = None
        return no_segment
    source_start_us = min(int(word.source_start_us) for word in words)
    source_end_us = max(int(word.source_end_us) for word in words)
    if source_end_us <= source_start_us:
        no_segment: FinalTimelineSegment | None = None
        return no_segment
    target_duration_us = _target_duration_preserving_effective_speed(segment, source_start_us, source_end_us)
    return replace(
        segment,
        source_start_us=source_start_us,
        source_end_us=source_end_us,
        target_end_us=int(segment.target_start_us) + target_duration_us,
        word_ids=list(word_ids),
        text="".join(word.text for word in words),
        spoken_source_start_us=source_start_us,
        spoken_source_end_us=source_end_us,
        clip_source_start_us=source_start_us if segment.clip_source_start_us is not None else segment.clip_source_start_us,
        clip_source_end_us=source_end_us if segment.clip_source_end_us is not None else segment.clip_source_end_us,
        debug_hints={**dict(segment.debug_hints or {}), "final_visible_repair": repair_reason},
    )


def _segments_with_word_ids_preserving_effective_speed(
    segment: FinalTimelineSegment,
    word_ids: list[str],
    source_graph: CanonicalSourceGraph,
    repair_reason: str,
    *,
    existing_segment_ids: set[str],
) -> list[FinalTimelineSegment] | None:
    word_runs = _split_word_ids_on_unselected_source_words(word_ids, source_graph)
    if not word_runs:
        no_segments: list[FinalTimelineSegment] | None = None
        return no_segments
    if len(word_runs) == 1:
        repaired = _segment_with_word_ids_preserving_effective_speed(segment, word_runs[0], source_graph, repair_reason)
        if repaired is None:
            no_segments: list[FinalTimelineSegment] | None = None
            return no_segments
        return [repaired]
    repaired_segments: list[FinalTimelineSegment] = []
    used_ids = set(existing_segment_ids)
    for index, run_word_ids in enumerate(word_runs):
        base = segment if index == 0 else replace(segment, segment_id=_unique_split_segment_id(segment.segment_id, used_ids))
        used_ids.add(base.segment_id)
        repaired = _segment_with_word_ids_preserving_effective_speed(base, run_word_ids, source_graph, repair_reason)
        if repaired is None:
            no_segments: list[FinalTimelineSegment] | None = None
            return no_segments
        repaired_segments.append(repaired)
    return repaired_segments


def _split_word_ids_on_unselected_source_words(
    word_ids: list[str],
    source_graph: CanonicalSourceGraph,
) -> list[list[str]]:
    words_by_id = {word.word_id: word for word in source_graph.words}
    words = [words_by_id[word_id] for word_id in word_ids if word_id in words_by_id]
    if len(words) != len(word_ids):
        empty: list[list[str]] = []
        return empty
    selected_ids = set(word_ids)
    runs: list[list[str]] = []
    current: list[str] = []
    previous_word: Any | None = None
    for word in words:
        if previous_word is not None and _source_gap_has_unselected_words_between_words(
            previous_word,
            word,
            source_graph,
            selected_ids,
        ):
            if current:
                runs.append(current)
            current = []
        current.append(str(word.word_id))
        previous_word = word
    if current:
        runs.append(current)
    return runs


def _source_gap_has_unselected_words_between_words(
    left_word: Any,
    right_word: Any,
    source_graph: CanonicalSourceGraph,
    selected_word_ids: set[str],
) -> bool:
    return _source_range_has_unselected_words(
        source_graph=source_graph,
        start_us=int(getattr(left_word, "source_end_us", 0) or 0),
        end_us=int(getattr(right_word, "source_start_us", 0) or 0),
        selected_word_ids=selected_word_ids,
    )


def _source_range_has_unselected_words(
    *,
    source_graph: CanonicalSourceGraph,
    start_us: int,
    end_us: int,
    selected_word_ids: set[str],
) -> bool:
    if int(end_us) <= int(start_us):
        return False
    for word in source_graph.words:
        word_id = str(getattr(word, "word_id", "") or "")
        if not word_id or word_id in selected_word_ids:
            continue
        word_start_us = int(getattr(word, "source_start_us", 0) or 0)
        word_end_us = int(getattr(word, "source_end_us", 0) or 0)
        if word_end_us <= int(start_us) + 20_000 or word_start_us >= int(end_us) - 20_000:
            continue
        return True
    return False


def _source_bounds_for_word_ids(
    word_ids: list[str],
    source_graph: CanonicalSourceGraph,
) -> tuple[int, int] | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    words = [words_by_id[word_id] for word_id in word_ids if word_id in words_by_id]
    if not words or len(words) != len(word_ids):
        no_bounds: tuple[int, int] | None = None
        return no_bounds
    source_start_us = min(int(word.source_start_us) for word in words)
    source_end_us = max(int(word.source_end_us) for word in words)
    if source_end_us <= source_start_us:
        no_bounds: tuple[int, int] | None = None
        return no_bounds
    return source_start_us, source_end_us


def _merged_segment_pair_preserving_effective_speed(
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    repair_reason: str,
) -> FinalTimelineSegment:
    merged_word_ids = [*left.word_ids, *right.word_ids]
    text = _text_from_word_ids(merged_word_ids, source_graph) or f"{left.text}{right.text}"
    bounds = _source_bounds_for_word_ids(merged_word_ids, source_graph)
    if bounds is None:
        source_start_us = int(left.source_start_us)
        source_end_us = int(right.source_end_us)
    else:
        source_start_us, source_end_us = bounds
    target_duration_us = _target_duration_preserving_effective_speed(left, source_start_us, source_end_us)
    has_clip_bounds = any(
        value is not None
        for value in (
            left.clip_source_start_us,
            left.clip_source_end_us,
            right.clip_source_start_us,
            right.clip_source_end_us,
        )
    )
    clip_start_us = None
    clip_end_us = None
    if has_clip_bounds:
        clip_start_values = [
            int(value)
            for value in (left.clip_source_start_us, right.clip_source_start_us, source_start_us)
            if value is not None
        ]
        clip_end_values = [
            int(value)
            for value in (left.clip_source_end_us, right.clip_source_end_us, source_end_us)
            if value is not None
        ]
        clip_start_us = min(clip_start_values) if clip_start_values else source_start_us
        clip_end_us = max(clip_end_values) if clip_end_values else source_end_us
    return replace(
        left,
        source_start_us=source_start_us,
        source_end_us=source_end_us,
        target_end_us=int(left.target_start_us) + target_duration_us,
        word_ids=merged_word_ids,
        text=text,
        decision_ids=_unique([*left.decision_ids, *right.decision_ids]),
        spoken_source_start_us=source_start_us,
        spoken_source_end_us=source_end_us,
        clip_source_start_us=clip_start_us,
        clip_source_end_us=clip_end_us,
        tail_handle_us=max(int(left.tail_handle_us), int(right.tail_handle_us)),
        debug_hints={
            **dict(left.debug_hints or {}),
            "final_visible_repair": repair_reason,
            "merged_segment_ids": [left.segment_id, right.segment_id],
        },
    )


def _target_duration_preserving_effective_speed(
    segment: FinalTimelineSegment,
    source_start_us: int,
    source_end_us: int,
) -> int:
    new_source_duration_us = max(1, int(source_end_us) - int(source_start_us))
    return new_source_duration_us


def _candidate_window_captions(
    captions: list[CaptionRenderUnit],
    candidate: dict[str, Any],
) -> list[CaptionRenderUnit]:
    ordered = _ordered_captions(captions)
    ids = [str(value) for value in list(candidate.get("window_caption_ids") or []) if str(value)]
    if ids:
        by_id = {caption.caption_id: caption for caption in ordered}
        rows = [by_id[caption_id] for caption_id in ids if caption_id in by_id]
        if len(rows) == len(ids):
            return rows
    caption_id = str(candidate.get("caption_id") or "")
    related_caption_id = str(candidate.get("related_caption_id") or caption_id)
    start = _caption_index(ordered, caption_id)
    end = _caption_index(ordered, related_caption_id)
    if start is None or end is None:
        empty: list[CaptionRenderUnit] = []
        return empty
    if end < start:
        start, end = end, start
    return ordered[start : end + 1]


def _unique_split_segment_id(base_segment_id: str, existing_segment_ids: set[str]) -> str:
    for index in range(1, 1000):
        candidate = f"{base_segment_id}_split_{index:03d}"
        if candidate not in existing_segment_ids:
            return candidate
    return f"{base_segment_id}_split"


def _caption_segments_exclusive(
    caption: CaptionRenderUnit,
    captions: list[CaptionRenderUnit],
    segment_ids: list[str],
) -> bool:
    target = set(segment_ids)
    for other in captions:
        if other.caption_id == caption.caption_id:
            continue
        if target.intersection(_caption_segment_ids(other)):
            return False
    return True


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

def _configure_final_visible_rule_modules() -> None:
    dependencies = globals()
    _leading_filler_rules.configure_rule_dependencies(dependencies)
    _connector_intrusion_rules.configure_rule_dependencies(dependencies)
    _pre_visible_semantic_junk_rules.configure_rule_dependencies(dependencies)
    _source_boundary_prefix_rules.configure_rule_dependencies(dependencies)
    _caption_only_merge_rules.configure_rule_dependencies(dependencies)
    _short_residual_rules.configure_rule_dependencies(dependencies)
    _restart_repeat_rules.configure_rule_dependencies(dependencies)
    _de_shi_bridge_rules.configure_rule_dependencies(dependencies)
    _word_span_edit_rules.configure_rule_dependencies(dependencies)

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

_configure_final_visible_rule_modules()
