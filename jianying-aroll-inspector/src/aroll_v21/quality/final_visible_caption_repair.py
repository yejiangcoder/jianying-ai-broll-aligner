from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.final_caption_visible_repeat import (
    FINAL_VISIBLE_RECHECK_DECISIONS,
    build_final_caption_visible_repeat_gate,
)
from aroll_v21.quality.subtitle_readability import HARD_MAX_CHARS, HARD_MAX_DURATION_US
from aroll_v21.quality.tiny_segment_classifier import classify_tiny_segment


FINAL_VISIBLE_REPAIR_COUNT_KEYS = (
    "dangling_prefix_suffix_count",
    "semantic_garbage_or_asr_suspect_count",
    "cross_caption_semantic_containment_count",
    "restart_repeat_visible_count",
)
MAX_FINAL_VISIBLE_REPAIR_PASSES = 32
MAX_CAPTION_ONLY_TARGET_GAP_US = 120_000
MAX_SOURCE_BOUNDARY_PREFIX_GAP_US = 600_000
MAX_SOURCE_BOUNDARY_COMPOUND_GAP_US = 120_000
SOURCE_BOUNDARY_FUNCTION_PREFIXES = ("就", "也", "还", "才", "又", "再", "都", "只", "却", "仍", "便")
SOURCE_BOUNDARY_PREFIX_DEPENDENT_STARTS = ("有", "能", "敢", "会", "要", "把", "让", "给", "对", "在", "被", "将", "成", "可以")
SOURCE_BOUNDARY_COMPOUND_SUFFIXES = (
    "区",
    "圈",
    "群",
    "场",
    "端",
    "口",
    "线",
    "面",
    "点",
    "处",
    "侧",
    "边",
)
DE_SHI_BOUNDARY_NORMALIZE_AFTER = ("对", "在", "把", "给", "向", "从", "跟", "为", "为了", "因为", "被", "让", "将")
MAX_ISOLATED_SHORT_FRAGMENT_CHARS = 4
MAX_ISOLATED_SHORT_FRAGMENT_DURATION_US = 900_000
MIN_ISOLATED_SHORT_FRAGMENT_SOURCE_GAP_US = 300_000
MIN_ISOLATED_SHORT_FRAGMENT_NEIGHBOR_CHARS = 6
MIN_REPAIRED_SEGMENT_DURATION_US = 1_200_000
MAX_REPAIRED_RESIDUAL_DROP_DURATION_US = 500_000
MAX_REPAIRED_RESIDUAL_DROP_CHARS = 2


@dataclass(frozen=True)
class FinalVisibleCaptionRepairResult:
    final_timeline: list[FinalTimelineSegment]
    captions: list[CaptionRenderUnit]
    report: dict[str, Any]


@dataclass(frozen=True)
class _RepairStep:
    final_timeline: list[FinalTimelineSegment]
    captions: list[CaptionRenderUnit]
    action: dict[str, Any]
    timeline_changed: bool = False


@dataclass(frozen=True)
class _SourceBoundaryPrefixCandidate:
    word: Any
    transfer_from_segment_id: str = ""


@dataclass(frozen=True)
class _SourceBoundaryCompoundCandidate:
    left_segment: FinalTimelineSegment
    right_segment: FinalTimelineSegment
    left_word: Any
    right_word: Any


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
    actions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    max_pass_limit = max(1, int(max_passes))
    current_signature = _repair_state_signature(current_timeline, current_captions)
    seen_signatures: set[tuple[Any, ...]] = {current_signature}
    stop_reason = ""
    passes_executed = 0

    for pass_index in range(max_pass_limit):
        passes_executed = pass_index + 1
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
        pass_index_start=len(actions) + 1,
    )
    actions.extend(final_caption_only_actions)

    final_gate = build_final_caption_visible_repeat_gate(current_captions)
    final_effective_timeline_captions, final_materializations = _effective_timeline_caption_units(
        _timeline_caption_units(current_timeline, source_graph),
        current_captions,
    )
    final_timeline_gate = _timeline_gate(final_effective_timeline_captions, final_materializations)
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
    same_segment_de_duplicate = _repair_same_segment_de_duplicate_prefix(
        final_timeline,
        current,
        source_graph,
        candidate,
        pass_index,
    )
    if same_segment_de_duplicate is not None:
        return same_segment_de_duplicate
    if index == 0:
        no_step: _RepairStep | None = None
        return no_step
    previous = ordered[index - 1]
    combined_text = f"{previous.text}{current.text}"
    if len(normalize_text(combined_text)) > HARD_MAX_CHARS:
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


def _finalize_caption_only_dangling_merges(
    captions: list[CaptionRenderUnit],
    *,
    pass_index_start: int,
) -> tuple[list[CaptionRenderUnit], list[dict[str, Any]]]:
    current = _renumber_captions(list(captions))
    actions: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = {_caption_only_state_signature(current)}
    pass_index = max(1, pass_index_start)
    for _ in range(MAX_FINAL_VISIBLE_REPAIR_PASSES):
        gate = build_final_caption_visible_repeat_gate(current)
        step: _RepairStep | None = None
        for candidate in list(gate.get("dangling_prefix_suffix_candidates") or []):
            step = _repair_dangling_prefix_suffix_caption_only(current, candidate, pass_index)
            if step is not None:
                break
        if step is None:
            return current, actions
        repaired = _renumber_captions(step.captions)
        signature = _caption_only_state_signature(repaired)
        if signature in seen:
            return current, actions
        seen.add(signature)
        current = repaired
        actions.append(step.action)
        pass_index += 1
    return current, actions


def _repair_dangling_prefix_suffix_caption_only(
    captions: list[CaptionRenderUnit],
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    ordered = _ordered_captions(captions)
    index = _caption_index(ordered, str(candidate.get("caption_id") or ""))
    if index is None or index == 0:
        no_step: _RepairStep | None = None
        return no_step
    current = ordered[index]
    previous = ordered[index - 1]
    combined_text = f"{previous.text}{current.text}"
    if len(normalize_text(combined_text)) > HARD_MAX_CHARS:
        no_step: _RepairStep | None = None
        return no_step
    merged_caption_result = _merge_adjacent_captions(previous, current)
    if merged_caption_result is None:
        no_step: _RepairStep | None = None
        return no_step
    merged_caption, merge_decision = merged_caption_result
    rows = list(ordered)
    rows[index - 1] = merged_caption
    repaired = [*rows[:index], *rows[index + 1 :]]
    return _RepairStep(
        final_timeline=[],
        captions=repaired,
        timeline_changed=False,
        action=_action(
            "dangling_prefix_suffix",
            "finalize_caption_only_dangling_merge",
            pass_index,
            candidate,
            affected_caption_ids=[previous.caption_id, current.caption_id],
            target_gap_us=int(current.target_start_us) - int(previous.target_end_us),
            video_segment_merged=False,
            caption_only_merge_materialized=True,
            caption_only_merge_decision=merge_decision,
            merged_into_caption_id=previous.caption_id,
            consumed_caption_id=current.caption_id,
            consumed_caption_state="consumed_by_final_caption_only_merge",
            merged_caption_text=merged_caption.text,
            merged_caption_timeline_segment_ids=list(merged_caption.timeline_segment_ids),
            merged_caption_target_start_us=int(merged_caption.target_start_us),
            merged_caption_target_end_us=int(merged_caption.target_end_us),
        ),
    )


def _repair_isolated_semantic_junk_caption(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    ordered = _ordered_captions(captions)
    if len(ordered) < 3:
        no_step: _RepairStep | None = None
        return no_step
    for index, caption in enumerate(ordered):
        if index == 0 or index == len(ordered) - 1:
            continue
        if not _is_isolated_short_source_gap_fragment(ordered, index, source_graph):
            continue
        text = normalize_text(str(caption.text or ""))
        dropped = _drop_or_trim_caption_words(final_timeline, captions, source_graph, caption)
        if dropped is None:
            continue
        repaired_timeline, dropped_segment_ids, trimmed_segment_ids = dropped
        decision = "drop_isolated_junk_segment" if dropped_segment_ids else "trim_isolated_junk_words"
        return _RepairStep(
            final_timeline=repaired_timeline,
            captions=captions,
            timeline_changed=True,
            action=_action(
                "isolated_semantic_junk_caption",
                decision,
                pass_index,
                {
                    "caption_id": caption.caption_id,
                    "related_caption_id": caption.caption_id,
                    "reason": "isolated_semantic_junk_caption",
                    "overlap_text": text,
                },
                affected_caption_ids=[caption.caption_id],
                dropped_segment_ids=dropped_segment_ids,
                trimmed_segment_ids=trimmed_segment_ids,
                dropped_word_ids=list(caption.word_ids),
                junk_text=str(caption.text or ""),
            ),
        )
    no_step: _RepairStep | None = None
    return no_step


def _is_isolated_short_source_gap_fragment(
    ordered: list[CaptionRenderUnit],
    index: int,
    source_graph: CanonicalSourceGraph,
) -> bool:
    caption = ordered[index]
    text = normalize_text(str(caption.text or ""))
    if not (2 <= len(text) <= MAX_ISOLATED_SHORT_FRAGMENT_CHARS):
        return False
    if not text or not all("\u4e00" <= char <= "\u9fff" for char in text):
        return False
    duration_us = int(caption.target_end_us) - int(caption.target_start_us)
    if duration_us <= 0 or duration_us > MAX_ISOLATED_SHORT_FRAGMENT_DURATION_US:
        return False
    previous = ordered[index - 1]
    next_caption = ordered[index + 1]
    previous_text = normalize_text(str(previous.text or ""))
    next_text = normalize_text(str(next_caption.text or ""))
    if len(previous_text) < MIN_ISOLATED_SHORT_FRAGMENT_NEIGHBOR_CHARS:
        return False
    if len(next_text) < MIN_ISOLATED_SHORT_FRAGMENT_NEIGHBOR_CHARS:
        return False
    if text in previous_text or text in next_text or previous_text.endswith(text) or next_text.startswith(text):
        return False
    previous_range = _caption_source_range(previous, source_graph)
    current_range = _caption_source_range(caption, source_graph)
    next_range = _caption_source_range(next_caption, source_graph)
    if previous_range is None or current_range is None or next_range is None:
        return False
    previous_gap_us = current_range[0] - previous_range[1]
    next_gap_us = next_range[0] - current_range[1]
    return (
        previous_gap_us >= MIN_ISOLATED_SHORT_FRAGMENT_SOURCE_GAP_US
        and next_gap_us >= MIN_ISOLATED_SHORT_FRAGMENT_SOURCE_GAP_US
    )


def _repair_same_segment_de_duplicate_prefix(
    final_timeline: list[FinalTimelineSegment],
    current: CaptionRenderUnit,
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    if normalize_text(str(candidate.get("reason") or "")) != "dangling_de_prefix":
        no_step: _RepairStep | None = None
        return no_step
    current_ids = _caption_segment_ids(current)
    if len(current_ids) != 1:
        no_step: _RepairStep | None = None
        return no_step
    segment = next((row for row in final_timeline if row.segment_id == current_ids[0]), None)
    if segment is None:
        no_step: _RepairStep | None = None
        return no_step
    segment_word_ids = list(segment.word_ids)
    caption_word_ids = list(current.word_ids)
    if not caption_word_ids or not _is_suffix(segment_word_ids, caption_word_ids):
        no_step: _RepairStep | None = None
        return no_step
    prefix_word_ids = segment_word_ids[: len(segment_word_ids) - len(caption_word_ids)]
    if not prefix_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    words_by_id = {word.word_id: word for word in source_graph.words}
    first_word = words_by_id.get(caption_word_ids[0])
    if normalize_text(str(getattr(first_word, "text", "") or "")) != "的":
        no_step: _RepairStep | None = None
        return no_step
    after_de_word_ids = caption_word_ids[1:]
    duplicate_len = _leading_duplicate_word_count(prefix_word_ids, after_de_word_ids, source_graph)
    if duplicate_len <= 0:
        no_step: _RepairStep | None = None
        return no_step
    remaining_caption_word_ids = after_de_word_ids[duplicate_len:]
    if not remaining_caption_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    repaired_word_ids = [*prefix_word_ids, *remaining_caption_word_ids]
    repaired_segment = _segment_with_word_ids_preserving_effective_speed(
        segment,
        repaired_word_ids,
        source_graph,
        "same_segment_de_duplicate_prefix_trim",
    )
    if repaired_segment is None:
        no_step: _RepairStep | None = None
        return no_step
    repaired = [repaired_segment if row.segment_id == segment.segment_id else row for row in final_timeline]
    dropped_word_ids = [caption_word_ids[0], *after_de_word_ids[:duplicate_len]]
    return _RepairStep(
        final_timeline=repaired,
        captions=[],
        timeline_changed=True,
        action=_action(
            "dangling_prefix_suffix",
            "trim_same_segment_de_duplicate_prefix",
            pass_index,
            candidate,
            affected_caption_ids=[current.caption_id],
            trimmed_segment_id=segment.segment_id,
            dropped_word_ids=dropped_word_ids,
            duplicate_prefix_text=_text_from_word_ids(after_de_word_ids[:duplicate_len], source_graph),
            remaining_word_ids=repaired_word_ids,
        ),
    )


def _leading_duplicate_word_count(
    prefix_word_ids: list[str],
    after_de_word_ids: list[str],
    source_graph: CanonicalSourceGraph,
) -> int:
    max_len = min(len(prefix_word_ids), len(after_de_word_ids))
    for count in range(max_len, 0, -1):
        left_text = normalize_text(_text_from_word_ids(prefix_word_ids[-count:], source_graph))
        right_text = normalize_text(_text_from_word_ids(after_de_word_ids[:count], source_graph))
        if left_text and left_text == right_text:
            return count
    return 0


def _repair_de_shi_duplicate_bridge(
    final_timeline: list[FinalTimelineSegment],
    previous: CaptionRenderUnit,
    current: CaptionRenderUnit,
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    if normalize_text(str(candidate.get("reason") or "")) != "dangling_de_shi_prefix":
        no_step: _RepairStep | None = None
        return no_step
    previous_ids = _caption_segment_ids(previous)
    current_ids = _caption_segment_ids(current)
    if len(previous_ids) != 1 or len(current_ids) != 1:
        no_step: _RepairStep | None = None
        return no_step
    index_by_id = {segment.segment_id: index for index, segment in enumerate(final_timeline)}
    previous_index = index_by_id.get(previous_ids[0])
    current_index = index_by_id.get(current_ids[0])
    if previous_index is None or current_index is None or current_index != previous_index + 1:
        no_step: _RepairStep | None = None
        return no_step
    left = final_timeline[previous_index]
    right = final_timeline[current_index]
    if not left.word_ids or not right.word_ids:
        no_step: _RepairStep | None = None
        return no_step
    words_by_id = {word.word_id: word for word in source_graph.words}
    ordered_words = list(source_graph.words)
    index_by_word_id = {word.word_id: index for index, word in enumerate(ordered_words)}
    left_last_index = index_by_word_id.get(left.word_ids[-1])
    right_first_index = index_by_word_id.get(right.word_ids[0])
    if left_last_index is None or right_first_index is None or right_first_index <= left_last_index + 1:
        no_step: _RepairStep | None = None
        return no_step
    bridge_words = ordered_words[left_last_index + 1 : right_first_index]
    selected_word_ids = {word_id for segment in final_timeline for word_id in segment.word_ids}
    bridge_word_ids = [str(getattr(word, "word_id", "") or "") for word in bridge_words]
    if not bridge_word_ids or any(word_id in selected_word_ids for word_id in bridge_word_ids):
        no_step: _RepairStep | None = None
        return no_step
    bridge_text = "".join(str(getattr(word, "text", "") or "") for word in bridge_words)
    if not normalize_text(bridge_text):
        no_step: _RepairStep | None = None
        return no_step
    first_right_word = words_by_id.get(right.word_ids[0])
    if normalize_text(str(getattr(first_right_word, "text", "") or "")) != "的":
        no_step: _RepairStep | None = None
        return no_step
    duplicate_bridge_ids = _leading_word_ids_for_text(list(right.word_ids[1:]), source_graph, bridge_text)
    if not duplicate_bridge_ids:
        no_step: _RepairStep | None = None
        return no_step
    drop_count = 1 + len(duplicate_bridge_ids)
    remaining_right_word_ids = list(right.word_ids[drop_count:])
    if not remaining_right_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    extended_left = _segment_with_word_ids_preserving_effective_speed(
        left,
        [*left.word_ids, *bridge_word_ids],
        source_graph,
        "de_shi_duplicate_bridge_extend_previous",
    )
    trimmed_right = _segment_with_word_ids_preserving_effective_speed(
        right,
        remaining_right_word_ids,
        source_graph,
        "de_shi_duplicate_bridge_trim_current",
    )
    if extended_left is None or trimmed_right is None:
        no_step: _RepairStep | None = None
        return no_step
    repaired = list(final_timeline)
    repaired[previous_index] = extended_left
    repaired[current_index] = trimmed_right
    return _RepairStep(
        final_timeline=repaired,
        captions=[],
        timeline_changed=True,
        action=_action(
            "dangling_prefix_suffix",
            "bridge_omitted_source_tail_and_trim_de_shi_duplicate",
            pass_index,
            candidate,
            affected_caption_ids=[previous.caption_id, current.caption_id],
            extended_segment_id=left.segment_id,
            trimmed_segment_id=right.segment_id,
            bridge_word_ids=bridge_word_ids,
            bridge_text=bridge_text,
            dropped_word_ids=[right.word_ids[0], *duplicate_bridge_ids],
            remaining_right_word_ids=remaining_right_word_ids,
        ),
    )


def _drop_repeated_caption_span(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    issue_type: str,
    pass_index: int,
) -> _RepairStep | None:
    caption = _caption_by_id(captions, str(candidate.get("caption_id") or ""))
    if caption is None:
        no_step: _RepairStep | None = None
        return no_step
    dropped = _drop_or_trim_caption_words(final_timeline, captions, source_graph, caption)
    if dropped is None:
        no_step: _RepairStep | None = None
        return no_step
    repaired_timeline, dropped_segment_ids, trimmed_segment_ids = dropped
    decision = "drop_shorter_repeated_segment" if dropped_segment_ids else "trim_shorter_repeated_words"
    return _RepairStep(
        final_timeline=repaired_timeline,
        captions=captions,
        timeline_changed=True,
        action=_action(
            issue_type,
            decision,
            pass_index,
            candidate,
            affected_caption_ids=[caption.caption_id],
            dropped_segment_ids=dropped_segment_ids,
            trimmed_segment_ids=trimmed_segment_ids,
        ),
    )


def _trim_asr_restart_prefix(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    caption = _caption_by_id(captions, str(candidate.get("caption_id") or ""))
    if caption is None:
        no_step: _RepairStep | None = None
        return no_step
    repeated_prefix = normalize_text(str(candidate.get("repeated_prefix") or ""))
    if not repeated_prefix:
        no_step: _RepairStep | None = None
        return no_step
    drop_text = f"{repeated_prefix}就"
    drop_word_ids = _leading_word_ids_for_text(caption.word_ids, source_graph, drop_text)
    if not drop_word_ids:
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
            "semantic_garbage_or_asr_suspect",
            "trim_repeated_prefix",
            pass_index,
            candidate,
            affected_caption_ids=[caption.caption_id],
            dropped_word_ids=drop_word_ids,
            recheck_decision="trim_repeated_prefix",
        ),
    )


def _trim_restart_repeat_visible_prefix(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    drop_text = normalize_text(str(candidate.get("drop_text") or ""))
    if not drop_text:
        no_step: _RepairStep | None = None
        return no_step
    caption = _caption_by_id(captions, str(candidate.get("caption_id") or ""))
    if caption is None:
        no_step: _RepairStep | None = None
        return no_step
    drop_word_ids = _leading_word_ids_for_text(caption.word_ids, source_graph, drop_text)
    if not drop_word_ids:
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
            "restart_repeat_visible",
            "trim_restart_prefix",
            pass_index,
            candidate,
            affected_caption_ids=[caption.caption_id],
            dropped_word_ids=drop_word_ids,
            drop_text=drop_text,
        ),
    )


def _drop_restart_repeat_word_span(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    pattern = str(candidate.get("pattern") or "")
    repair_reason_by_pattern = {
        "negative_predicate_restart": "negative_predicate_restart_repair",
        "partial_phrase_restart": "partial_phrase_restart_repair",
    }
    repair_reason = repair_reason_by_pattern.get(pattern)
    if repair_reason is None:
        no_step: _RepairStep | None = None
        return no_step
    drop_text = normalize_text(str(candidate.get("drop_text") or ""))
    if not drop_text:
        no_step: _RepairStep | None = None
        return no_step
    window_captions = _candidate_window_captions(captions, candidate)
    if not window_captions:
        no_step: _RepairStep | None = None
        return no_step
    word_ids = [word_id for caption in window_captions for word_id in caption.word_ids]
    drop_word_ids = _contiguous_word_ids_for_text(word_ids, source_graph, drop_text)
    if not drop_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    repaired_timeline = _drop_contiguous_word_ids_from_timeline(
        final_timeline,
        source_graph,
        drop_word_ids,
        repair_reason,
    )
    if repaired_timeline is None:
        no_step: _RepairStep | None = None
        return no_step
    return _RepairStep(
        final_timeline=repaired_timeline,
        captions=captions,
        timeline_changed=True,
        action=_action(
            "restart_repeat_visible",
            "drop_partial_phrase_restart_span" if pattern == "partial_phrase_restart" else "drop_negative_predicate_restart_span",
            pass_index,
            candidate,
            affected_caption_ids=[caption.caption_id for caption in window_captions],
            dropped_word_ids=drop_word_ids,
            drop_text=drop_text,
        ),
    )


def _repair_source_boundary_prefix_gap(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    ordered_words = list(source_graph.words)
    index_by_word_id = {word.word_id: index for index, word in enumerate(ordered_words)}
    for segment in _ordered_segments(final_timeline):
        prefix_candidate = _source_boundary_prefix_candidate(
            segment,
            final_timeline,
            words_by_id,
            ordered_words,
            index_by_word_id,
        )
        if prefix_candidate is None:
            continue
        repaired = _apply_source_boundary_prefix_candidate(final_timeline, segment, prefix_candidate, source_graph)
        if repaired is None:
            continue
        prefix_word = prefix_candidate.word
        return _RepairStep(
            final_timeline=repaired,
            captions=[],
            timeline_changed=True,
            action=_action(
                "source_boundary_prefix_gap",
                "prepend_source_boundary_prefix",
                pass_index,
                {
                    "caption_id": "",
                    "related_caption_id": "",
                    "reason": "source-aware boundary prefix was omitted before a dependent visible caption start",
                    "overlap_text": normalize_text(str(getattr(prefix_word, "text", "") or "")),
                },
                affected_segment_id=segment.segment_id,
                prepended_word_id=prefix_word.word_id,
                prepended_text=prefix_word.text,
                transferred_from_segment_id=prefix_candidate.transfer_from_segment_id,
            ),
        )
    no_step: _RepairStep | None = None
    return no_step


def _source_boundary_prefix_candidate(
    segment: FinalTimelineSegment,
    final_timeline: list[FinalTimelineSegment],
    words_by_id: dict[str, Any],
    ordered_words: list[Any],
    index_by_word_id: dict[str, int],
) -> _SourceBoundaryPrefixCandidate | None:
    if not segment.word_ids:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    first_word_id = segment.word_ids[0]
    first_word = words_by_id.get(first_word_id)
    first_index = index_by_word_id.get(first_word_id)
    if first_word is None or first_index is None or first_index <= 0:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    current_text = normalize_text(str(segment.text or ""))
    if not _source_boundary_prefix_dependent_start(current_text):
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    prefix_word = ordered_words[first_index - 1]
    prefix_word_id = str(getattr(prefix_word, "word_id", "") or "")
    prefix_text = normalize_text(str(getattr(prefix_word, "text", "") or ""))
    if prefix_text not in SOURCE_BOUNDARY_FUNCTION_PREFIXES:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    prefix_material_id = str(getattr(prefix_word, "source_material_id", "") or "")
    segment_material_id = str(segment.source_material_id or "")
    if prefix_material_id and segment_material_id and prefix_material_id != segment_material_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    prefix_segment_id = str(getattr(prefix_word, "source_segment_id", "") or "")
    segment_source_id = str(segment.source_segment_id or "")
    if prefix_segment_id and segment_source_id and prefix_segment_id != segment_source_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    first_material_id = str(getattr(first_word, "source_material_id", "") or "")
    if first_material_id and segment_material_id and first_material_id != segment_material_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    first_segment_id = str(getattr(first_word, "source_segment_id", "") or "")
    if first_segment_id and segment_source_id and first_segment_id != segment_source_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    source_gap_us = int(getattr(first_word, "source_start_us", 0)) - int(getattr(prefix_word, "source_end_us", 0))
    if source_gap_us < -80_000 or source_gap_us > MAX_SOURCE_BOUNDARY_PREFIX_GAP_US:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    if abs(int(segment.source_start_us) - int(getattr(first_word, "source_start_us", segment.source_start_us))) > 80_000:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    containing_segments = [row for row in final_timeline if prefix_word_id in list(row.word_ids)]
    if not containing_segments:
        return _SourceBoundaryPrefixCandidate(word=prefix_word)
    if len(containing_segments) != 1:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    transfer_segment = containing_segments[0]
    if transfer_segment.segment_id == segment.segment_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    if not _is_suffix(list(transfer_segment.word_ids), [prefix_word_id]):
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    if int(transfer_segment.target_end_us) > int(segment.target_start_us) + 80_000:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    return _SourceBoundaryPrefixCandidate(word=prefix_word, transfer_from_segment_id=transfer_segment.segment_id)


def _repair_source_boundary_compound_suffix_gap(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    candidate = _source_boundary_compound_candidate(final_timeline, source_graph)
    if candidate is None:
        no_step: _RepairStep | None = None
        return no_step
    repaired = _merge_source_boundary_compound_segments(final_timeline, candidate, source_graph)
    if repaired is None:
        no_step: _RepairStep | None = None
        return no_step
    return _RepairStep(
        final_timeline=repaired,
        captions=[],
        timeline_changed=True,
        action=_action(
            "source_boundary_compound_suffix",
            "merge_source_boundary_compound_suffix",
            pass_index,
            {
                "caption_id": "",
                "related_caption_id": "",
                "reason": "source-aware lexical suffix belongs with the previous visible word",
                "overlap_text": f"{getattr(candidate.left_word, 'text', '')}{getattr(candidate.right_word, 'text', '')}",
            },
            affected_segment_ids=[candidate.left_segment.segment_id, candidate.right_segment.segment_id],
            suffix_word_id=str(getattr(candidate.right_word, "word_id", "") or ""),
            suffix_text=str(getattr(candidate.right_word, "text", "") or ""),
        ),
    )


def _source_boundary_compound_candidate(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> _SourceBoundaryCompoundCandidate | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    ordered = _ordered_segments(final_timeline)
    for left, right in zip(ordered, ordered[1:]):
        if not left.word_ids or not right.word_ids:
            continue
        if len(normalize_text(f"{left.text}{right.text}")) > HARD_MAX_CHARS:
            continue
        if int(right.target_end_us) - int(left.target_start_us) > HARD_MAX_DURATION_US:
            continue
        left_word = words_by_id.get(left.word_ids[-1])
        right_word = words_by_id.get(right.word_ids[0])
        if left_word is None or right_word is None:
            continue
        if not _source_boundary_compound_words_match(left_word, right_word):
            continue
        if not _safe_merge_segments(left, right, source_graph):
            continue
        return _SourceBoundaryCompoundCandidate(
            left_segment=left,
            right_segment=right,
            left_word=left_word,
            right_word=right_word,
        )
    no_candidate: _SourceBoundaryCompoundCandidate | None = None
    return no_candidate


def _source_boundary_compound_words_match(left_word: Any, right_word: Any) -> bool:
    left_text = normalize_text(str(getattr(left_word, "text", "") or ""))
    right_text = normalize_text(str(getattr(right_word, "text", "") or ""))
    if len(left_text) < 2 or right_text not in SOURCE_BOUNDARY_COMPOUND_SUFFIXES:
        return False
    source_gap_us = int(getattr(right_word, "source_start_us", 0)) - int(getattr(left_word, "source_end_us", 0))
    if source_gap_us < -80_000 or source_gap_us > MAX_SOURCE_BOUNDARY_COMPOUND_GAP_US:
        return False
    left_material = str(getattr(left_word, "source_material_id", "") or "")
    right_material = str(getattr(right_word, "source_material_id", "") or "")
    if left_material and right_material and left_material != right_material:
        return False
    left_segment = str(getattr(left_word, "source_segment_id", "") or "")
    right_segment = str(getattr(right_word, "source_segment_id", "") or "")
    if left_segment and right_segment and left_segment != right_segment:
        return False
    return True


def _merge_source_boundary_compound_segments(
    final_timeline: list[FinalTimelineSegment],
    candidate: _SourceBoundaryCompoundCandidate,
    source_graph: CanonicalSourceGraph,
) -> list[FinalTimelineSegment] | None:
    left = candidate.left_segment
    right = candidate.right_segment
    index_by_id = {segment.segment_id: index for index, segment in enumerate(final_timeline)}
    left_index = index_by_id.get(left.segment_id)
    right_index = index_by_id.get(right.segment_id)
    if left_index is None or right_index is None or right_index != left_index + 1:
        no_repair: list[FinalTimelineSegment] | None = None
        return no_repair
    merged_word_ids = [*left.word_ids, *right.word_ids]
    text = _text_from_word_ids(merged_word_ids, source_graph) or f"{left.text}{right.text}"
    source_start_us = int(left.source_start_us)
    source_end_us = int(right.source_end_us)
    target_duration_us = max(1, source_end_us - source_start_us)
    merged = replace(
        left,
        source_end_us=source_end_us,
        target_end_us=int(left.target_start_us) + target_duration_us,
        word_ids=merged_word_ids,
        text=text,
        decision_ids=_unique([*left.decision_ids, *right.decision_ids]),
        spoken_source_end_us=right.spoken_source_end_us if right.spoken_source_end_us is not None else left.spoken_source_end_us,
        clip_source_end_us=right.clip_source_end_us if right.clip_source_end_us is not None else left.clip_source_end_us,
        tail_handle_us=max(int(left.tail_handle_us), int(right.tail_handle_us)),
        debug_hints={
            **dict(left.debug_hints or {}),
            "final_visible_repair": "source_boundary_compound_suffix_merge",
            "merged_segment_ids": [left.segment_id, right.segment_id],
        },
    )
    return [*final_timeline[:left_index], merged, *final_timeline[right_index + 1 :]]


def _source_boundary_prefix_dependent_start(text: str) -> bool:
    if not text:
        return False
    return any(text.startswith(prefix) for prefix in SOURCE_BOUNDARY_PREFIX_DEPENDENT_STARTS)


def _apply_source_boundary_prefix_candidate(
    final_timeline: list[FinalTimelineSegment],
    segment: FinalTimelineSegment,
    candidate: _SourceBoundaryPrefixCandidate,
    source_graph: CanonicalSourceGraph,
) -> list[FinalTimelineSegment] | None:
    prefix_word = candidate.word
    prefix_word_id = str(getattr(prefix_word, "word_id", "") or "")
    if not prefix_word_id:
        no_repair: list[FinalTimelineSegment] | None = None
        return no_repair
    repaired: list[FinalTimelineSegment] = []
    changed = False
    for row in final_timeline:
        if candidate.transfer_from_segment_id and row.segment_id == candidate.transfer_from_segment_id:
            remaining_word_ids = [word_id for word_id in row.word_ids if word_id != prefix_word_id]
            if not remaining_word_ids:
                no_repair: list[FinalTimelineSegment] | None = None
                return no_repair
            trimmed = _segment_with_word_ids_preserving_effective_speed(row, remaining_word_ids, source_graph, "source_boundary_prefix_transfer")
            if trimmed is None:
                no_repair: list[FinalTimelineSegment] | None = None
                return no_repair
            repaired.append(trimmed)
            changed = True
            continue
        if row.segment_id != segment.segment_id:
            repaired.append(row)
            continue
        word_ids = [prefix_word_id, *row.word_ids]
        text = _text_from_word_ids(word_ids, source_graph)
        if not normalize_text(text):
            no_repair: list[FinalTimelineSegment] | None = None
            return no_repair
        source_start_us = int(getattr(prefix_word, "source_start_us", row.source_start_us))
        source_end_us = int(row.source_end_us)
        target_duration_us = _target_duration_preserving_effective_speed(row, source_start_us, source_end_us)
        repaired.append(
            replace(
                row,
                source_start_us=source_start_us,
                target_end_us=int(row.target_start_us) + target_duration_us,
                word_ids=word_ids,
                text=text,
                spoken_source_start_us=source_start_us,
                clip_source_start_us=source_start_us
                if row.clip_source_start_us is not None
                else row.clip_source_start_us,
                debug_hints={
                    **dict(row.debug_hints or {}),
                    "final_visible_repair": "source_boundary_prefix_prepend",
                    "prepended_word_id": str(getattr(prefix_word, "word_id", "") or ""),
                },
            )
        )
        changed = True
    if changed:
        return repaired
    no_repair: list[FinalTimelineSegment] | None = None
    return no_repair


def _drop_or_trim_caption_words(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    caption: CaptionRenderUnit,
) -> tuple[list[FinalTimelineSegment], list[str], list[str]] | None:
    segment_ids = _caption_segment_ids(caption)
    if segment_ids and _caption_segments_exclusive(caption, captions, segment_ids):
        segment_id_set = set(segment_ids)
        kept = [segment for segment in final_timeline if segment.segment_id not in segment_id_set]
        if len(kept) < len(final_timeline):
            return kept, segment_ids, []
    repaired = _trim_word_ids_from_timeline(final_timeline, source_graph, list(caption.word_ids))
    if repaired is None:
        no_drop: tuple[list[FinalTimelineSegment], list[str], list[str]] | None = None
        return no_drop
    trimmed_ids = [
        before.segment_id
        for before, after in zip(final_timeline, repaired)
        if before.segment_id == after.segment_id and list(before.word_ids) != list(after.word_ids)
    ]
    return repaired, [], trimmed_ids


def _trim_word_ids_from_timeline(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    drop_word_ids: list[str],
) -> list[FinalTimelineSegment] | None:
    if not drop_word_ids:
        no_trim: list[FinalTimelineSegment] | None = None
        return no_trim
    drop_set = set(drop_word_ids)
    repaired: list[FinalTimelineSegment] = []
    changed = False
    for segment in final_timeline:
        word_ids = list(segment.word_ids)
        if not drop_set.intersection(word_ids):
            repaired.append(segment)
            continue
        if _is_prefix(word_ids, drop_word_ids):
            remaining = word_ids[len(drop_word_ids) :]
        elif _is_suffix(word_ids, drop_word_ids):
            remaining = word_ids[: len(word_ids) - len(drop_word_ids)]
        elif set(word_ids) == drop_set:
            remaining = []
        else:
            no_trim: list[FinalTimelineSegment] | None = None
            return no_trim
        changed = True
        if not remaining:
            continue
        adjusted = _segment_with_word_ids(segment, remaining, source_graph)
        if adjusted is None:
            no_trim: list[FinalTimelineSegment] | None = None
            return no_trim
        repaired.append(adjusted)
    if not changed:
        no_trim: list[FinalTimelineSegment] | None = None
        return no_trim
    return repaired


def _merge_short_repaired_segments(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    repair_reason: str,
) -> list[FinalTimelineSegment]:
    current = list(segments)
    while True:
        merge_index: int | None = None
        for index, segment in enumerate(current):
            if str((segment.debug_hints or {}).get("final_visible_repair") or "") != repair_reason:
                continue
            if _segment_duration_us(segment) >= MIN_REPAIRED_SEGMENT_DURATION_US:
                continue
            candidates: list[int] = []
            if index + 1 < len(current):
                candidates.append(index)
            if index > 0:
                candidates.append(index - 1)
            for candidate_index in candidates:
                left = current[candidate_index]
                right = current[candidate_index + 1]
                if len(normalize_text(f"{left.text}{right.text}")) > HARD_MAX_CHARS:
                    continue
                if int(right.target_end_us) - int(left.target_start_us) > HARD_MAX_DURATION_US:
                    continue
                if not _safe_merge_segments(left, right, source_graph):
                    continue
                merge_index = candidate_index
                break
            if merge_index is not None:
                break
        if merge_index is None:
            return current
        current = _merge_timeline_segment_pair_at(current, merge_index, source_graph, "merge_short_repaired_segment")


def _repair_short_repair_residual_segments(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    repaired, rows = _cleanup_short_repair_residual_segments(final_timeline, source_graph)
    if not rows:
        no_step: _RepairStep | None = None
        return no_step
    affected_ids = _unique(
        [
            segment_id
            for row in rows
            for segment_id in [
                str(row.get("segment_id") or ""),
                *[str(value) for value in list(row.get("merged_segment_ids") or [])],
            ]
            if segment_id
        ]
    )
    return _RepairStep(
        final_timeline=repaired,
        captions=[],
        timeline_changed=True,
        action=_action(
            "repair_short_residual",
            "cleanup_short_repair_residual_segments",
            pass_index,
            {
                "caption_id": "",
                "related_caption_id": "",
                "reason": "final visible repair left blocking short residual segments",
                "overlap_text": "",
            },
            affected_segment_ids=affected_ids,
            residual_cleanup_actions=rows,
        ),
    )


def _cleanup_short_repair_residual_segments(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> tuple[list[FinalTimelineSegment], list[dict[str, Any]]]:
    current = list(segments)
    actions: list[dict[str, Any]] = []
    while True:
        action = _next_short_repair_residual_action(current, source_graph)
        if not action:
            return current, actions
        kind = str(action.get("action") or "")
        if kind == "merge":
            merge_index = int(action.get("merge_index") or 0)
            left = current[merge_index]
            right = current[merge_index + 1]
            actions.append(
                {
                    "action": "merge",
                    "segment_id": str(action.get("segment_id") or ""),
                    "text": str(action.get("text") or ""),
                    "duration_us": int(action.get("duration_us") or 0),
                    "merged_segment_ids": [left.segment_id, right.segment_id],
                    "repair_reason": str(action.get("repair_reason") or ""),
                }
            )
            current = _merge_timeline_segment_pair_at(current, merge_index, source_graph, "merge_short_repaired_segment")
            continue
        if kind == "drop":
            index = int(action.get("index") or 0)
            segment = current[index]
            actions.append(
                {
                    "action": "drop",
                    "segment_id": segment.segment_id,
                    "text": segment.text,
                    "word_ids": list(segment.word_ids),
                    "duration_us": _segment_duration_us(segment),
                    "repair_reason": str((segment.debug_hints or {}).get("final_visible_repair") or ""),
                }
            )
            current = [*current[:index], *current[index + 1 :]]
            continue
        return current, actions


def _next_short_repair_residual_action(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> dict[str, Any]:
    for index, segment in enumerate(segments):
        if not _is_short_repair_residual_segment(segment):
            continue
        candidates: list[int] = []
        if index + 1 < len(segments):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        for merge_index in candidates:
            left = segments[merge_index]
            right = segments[merge_index + 1]
            if not _can_merge_short_repair_residual(left, right, source_graph):
                continue
            return {
                "action": "merge",
                "merge_index": merge_index,
                "segment_id": segment.segment_id,
                "text": segment.text,
                "duration_us": _segment_duration_us(segment),
                "repair_reason": str((segment.debug_hints or {}).get("final_visible_repair") or ""),
            }
        if _can_drop_short_repair_residual(segment, timeline_segment_count=len(segments)):
            return {"action": "drop", "index": index}
    no_action: dict[str, Any] = {}
    return no_action


def _is_short_repair_residual_segment(segment: FinalTimelineSegment) -> bool:
    if not str((segment.debug_hints or {}).get("final_visible_repair") or ""):
        return False
    duration_us = _segment_duration_us(segment)
    if duration_us <= 0 or duration_us >= MIN_REPAIRED_SEGMENT_DURATION_US:
        return False
    classification = classify_tiny_segment(segment)
    return not classification.semantic_bridge


def _can_merge_short_repair_residual(
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
) -> bool:
    if len(normalize_text(f"{left.text}{right.text}")) > HARD_MAX_CHARS:
        return False
    if int(right.target_end_us) - int(left.target_start_us) > HARD_MAX_DURATION_US:
        return False
    return _safe_merge_segments(left, right, source_graph)


def _can_drop_short_repair_residual(segment: FinalTimelineSegment, *, timeline_segment_count: int) -> bool:
    if timeline_segment_count <= 1:
        return False
    duration_us = _segment_duration_us(segment)
    if duration_us <= 0 or duration_us > MAX_REPAIRED_RESIDUAL_DROP_DURATION_US:
        return False
    text = normalize_text(segment.text)
    if not text or len(text) > MAX_REPAIRED_RESIDUAL_DROP_CHARS:
        return False
    classification = classify_tiny_segment(segment)
    return not classification.semantic_bridge


def _merge_timeline_segment_pair_at(
    segments: list[FinalTimelineSegment],
    index: int,
    source_graph: CanonicalSourceGraph,
    repair_reason: str,
) -> list[FinalTimelineSegment]:
    left = segments[index]
    right = segments[index + 1]
    merged_word_ids = [*left.word_ids, *right.word_ids]
    text = _text_from_word_ids(merged_word_ids, source_graph) or f"{left.text}{right.text}"
    source_start_us = int(left.source_start_us)
    source_end_us = int(right.source_end_us)
    target_duration_us = max(1, source_end_us - source_start_us)
    merged = replace(
        left,
        source_end_us=source_end_us,
        target_end_us=int(left.target_start_us) + target_duration_us,
        word_ids=merged_word_ids,
        text=text,
        decision_ids=_unique([*left.decision_ids, *right.decision_ids]),
        spoken_source_end_us=right.spoken_source_end_us if right.spoken_source_end_us is not None else left.spoken_source_end_us,
        clip_source_end_us=right.clip_source_end_us if right.clip_source_end_us is not None else left.clip_source_end_us,
        tail_handle_us=max(int(left.tail_handle_us), int(right.tail_handle_us)),
        debug_hints={
            **dict(left.debug_hints or {}),
            "final_visible_repair": repair_reason,
            "merged_segment_ids": [left.segment_id, right.segment_id],
        },
    )
    return [*segments[:index], merged, *segments[index + 2 :]]


def _drop_contiguous_word_ids_from_timeline(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    drop_word_ids: list[str],
    repair_reason: str,
) -> list[FinalTimelineSegment] | None:
    if not drop_word_ids:
        no_trim: list[FinalTimelineSegment] | None = None
        return no_trim
    flattened_word_ids = [word_id for segment in final_timeline for word_id in segment.word_ids]
    if not _contains_contiguous_subsequence(flattened_word_ids, drop_word_ids):
        no_trim: list[FinalTimelineSegment] | None = None
        return no_trim
    drop_set = set(drop_word_ids)
    repaired: list[FinalTimelineSegment] = []
    changed = False
    existing_segment_ids = {segment.segment_id for segment in final_timeline}
    for segment in final_timeline:
        word_ids = list(segment.word_ids)
        positions = [index for index, word_id in enumerate(word_ids) if word_id in drop_set]
        if not positions:
            repaired.append(segment)
            continue
        start = positions[0]
        end = positions[-1] + 1
        if positions != list(range(start, end)):
            no_trim: list[FinalTimelineSegment] | None = None
            return no_trim
        prefix_word_ids = word_ids[:start]
        suffix_word_ids = word_ids[end:]
        if not prefix_word_ids and not suffix_word_ids:
            changed = True
            continue
        cursor = int(segment.target_start_us)
        if prefix_word_ids:
            prefix_segment = _segment_with_word_ids_preserving_effective_speed(
                replace(segment, target_start_us=cursor),
                prefix_word_ids,
                source_graph,
                repair_reason,
            )
            if prefix_segment is None:
                no_trim: list[FinalTimelineSegment] | None = None
                return no_trim
            repaired.append(prefix_segment)
            cursor = int(prefix_segment.target_end_us)
        if suffix_word_ids:
            suffix_id = segment.segment_id if not prefix_word_ids else _unique_split_segment_id(segment.segment_id, existing_segment_ids)
            existing_segment_ids.add(suffix_id)
            suffix_segment = _segment_with_word_ids_preserving_effective_speed(
                replace(segment, segment_id=suffix_id, target_start_us=cursor),
                suffix_word_ids,
                source_graph,
                repair_reason,
            )
            if suffix_segment is None:
                no_trim: list[FinalTimelineSegment] | None = None
                return no_trim
            repaired.append(suffix_segment)
        changed = True
    if not changed:
        no_trim: list[FinalTimelineSegment] | None = None
        return no_trim
    return _merge_short_repaired_segments(repaired, source_graph, repair_reason)


def _contains_contiguous_subsequence(values: list[str], subsequence: list[str]) -> bool:
    if not subsequence or len(subsequence) > len(values):
        return False
    width = len(subsequence)
    return any(values[index : index + width] == subsequence for index in range(0, len(values) - width + 1))


def _merge_adjacent_caption_segments(
    final_timeline: list[FinalTimelineSegment],
    previous: CaptionRenderUnit,
    current: CaptionRenderUnit,
    source_graph: CanonicalSourceGraph,
) -> list[FinalTimelineSegment] | None:
    previous_ids = _caption_segment_ids(previous)
    current_ids = _caption_segment_ids(current)
    if len(previous_ids) != 1 or len(current_ids) != 1:
        no_merge: list[FinalTimelineSegment] | None = None
        return no_merge
    index_by_id = {segment.segment_id: index for index, segment in enumerate(final_timeline)}
    previous_index = index_by_id.get(previous_ids[0])
    current_index = index_by_id.get(current_ids[0])
    if previous_index is None or current_index is None or current_index != previous_index + 1:
        no_merge: list[FinalTimelineSegment] | None = None
        return no_merge
    left = final_timeline[previous_index]
    right = final_timeline[current_index]
    if not _safe_merge_segments(left, right, source_graph):
        no_merge: list[FinalTimelineSegment] | None = None
        return no_merge
    merged_word_ids = [*left.word_ids, *right.word_ids]
    text = _text_from_word_ids(merged_word_ids, source_graph) or f"{left.text}{right.text}"
    merged = replace(
        left,
        source_end_us=int(right.source_end_us),
        target_end_us=int(right.target_end_us),
        word_ids=merged_word_ids,
        text=text,
        decision_ids=_unique([*left.decision_ids, *right.decision_ids]),
        spoken_source_end_us=right.spoken_source_end_us if right.spoken_source_end_us is not None else left.spoken_source_end_us,
        clip_source_end_us=right.clip_source_end_us if right.clip_source_end_us is not None else left.clip_source_end_us,
        tail_handle_us=max(int(left.tail_handle_us), int(right.tail_handle_us)),
        debug_hints={
            **dict(left.debug_hints or {}),
            "final_visible_repair": "merge_dangling_prefix_suffix",
            "merged_segment_ids": [left.segment_id, right.segment_id],
        },
    )
    return [*final_timeline[:previous_index], merged, *final_timeline[current_index + 1 :]]


def _merge_adjacent_captions(left: CaptionRenderUnit, right: CaptionRenderUnit) -> tuple[CaptionRenderUnit, str] | None:
    if int(right.target_start_us) < int(left.target_end_us):
        no_merge: tuple[CaptionRenderUnit, str] | None = None
        return no_merge
    text = _join_visible_boundary_text(str(left.text or ""), str(right.text or ""))
    duration_us = int(right.target_end_us) - int(left.target_start_us)
    if len(normalize_text(text)) > HARD_MAX_CHARS or duration_us > HARD_MAX_DURATION_US:
        no_merge: tuple[CaptionRenderUnit, str] | None = None
        return no_merge
    same_container = str(left.containing_video_segment_id or "") == str(right.containing_video_segment_id or "")
    if not same_container and not _caption_only_merge_allowed(left, right):
        no_merge: tuple[CaptionRenderUnit, str] | None = None
        return no_merge
    containing_video_segment_id = left.containing_video_segment_id if same_container else None
    return replace(
        left,
        timeline_segment_ids=_unique([*left.timeline_segment_ids, *right.timeline_segment_ids]),
        word_ids=[*left.word_ids, *right.word_ids],
        text=text,
        target_end_us=int(right.target_end_us),
        source_subtitle_uids=_unique([*left.source_subtitle_uids, *right.source_subtitle_uids]),
        spoken_source_start_us=left.spoken_source_start_us,
        spoken_source_end_us=right.spoken_source_end_us,
        containing_video_segment_id=containing_video_segment_id,
    ), ("merge_with_previous_caption" if same_container else "caption_only_merge_with_previous")


def _caption_only_merge_allowed(left: CaptionRenderUnit, right: CaptionRenderUnit) -> bool:
    target_gap_us = int(right.target_start_us) - int(left.target_end_us)
    if target_gap_us < 0 or target_gap_us > MAX_CAPTION_ONLY_TARGET_GAP_US:
        return False
    if not _caption_segment_ids(left) or not _caption_segment_ids(right):
        return False
    text = _join_visible_boundary_text(str(left.text or ""), str(right.text or ""))
    merged = CaptionRenderUnit(
        caption_id="caption_only_merge_probe",
        timeline_segment_ids=_unique([*left.timeline_segment_ids, *right.timeline_segment_ids]),
        word_ids=[*left.word_ids, *right.word_ids],
        text=text,
        target_start_us=int(left.target_start_us),
        target_end_us=int(right.target_end_us),
        source_subtitle_uids=_unique([*left.source_subtitle_uids, *right.source_subtitle_uids]),
        style_template_id=left.style_template_id,
    )
    gate = build_final_caption_visible_repeat_gate([merged])
    return bool(gate.get("gate_passed"))


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
    if int(right.source_start_us) <= int(left.source_end_us):
        return False
    selected = set(left.word_ids) | set(right.word_ids)
    gap_start_us = int(left.source_end_us)
    gap_end_us = int(right.source_start_us)
    for word in source_graph.words:
        word_id = str(getattr(word, "word_id", "") or "")
        if not word_id or word_id in selected:
            continue
        word_start_us = int(getattr(word, "source_start_us", 0))
        word_end_us = int(getattr(word, "source_end_us", 0))
        if word_end_us <= gap_start_us + 20_000 or word_start_us >= gap_end_us - 20_000:
            continue
        return True
    return False


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


def _target_duration_preserving_effective_speed(
    segment: FinalTimelineSegment,
    source_start_us: int,
    source_end_us: int,
) -> int:
    new_source_duration_us = max(1, int(source_end_us) - int(source_start_us))
    return new_source_duration_us


def _leading_word_ids_for_text(
    word_ids: list[str],
    source_graph: CanonicalSourceGraph,
    text: str,
) -> list[str]:
    target = normalize_text(text)
    if not target:
        empty: list[str] = []
        return empty
    words_by_id = {word.word_id: word for word in source_graph.words}
    selected: list[str] = []
    joined = ""
    for word_id in word_ids:
        word = words_by_id.get(word_id)
        if word is None:
            empty: list[str] = []
            return empty
        selected.append(word_id)
        joined += normalize_text(word.text)
        if joined == target:
            return selected
        if not target.startswith(joined):
            empty: list[str] = []
            return empty
    empty: list[str] = []
    return empty


def _contiguous_word_ids_for_text(
    word_ids: list[str],
    source_graph: CanonicalSourceGraph,
    text: str,
) -> list[str]:
    target = normalize_text(text)
    if not target:
        empty: list[str] = []
        return empty
    words_by_id = {word.word_id: word for word in source_graph.words}
    for start in range(0, len(word_ids)):
        selected: list[str] = []
        joined = ""
        for word_id in word_ids[start:]:
            word = words_by_id.get(word_id)
            if word is None:
                break
            selected.append(word_id)
            joined += normalize_text(word.text)
            if joined == target:
                return selected
            if not target.startswith(joined):
                break
    empty: list[str] = []
    return empty


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


def _caption_segment_ids(caption: CaptionRenderUnit) -> list[str]:
    values = list(caption.timeline_segment_ids or [])
    if caption.containing_video_segment_id:
        values.append(str(caption.containing_video_segment_id))
    return _unique(values)


def _text_from_word_ids(word_ids: list[str], source_graph: CanonicalSourceGraph) -> str:
    words_by_id = {word.word_id: word for word in source_graph.words}
    return "".join(words_by_id[word_id].text for word_id in word_ids if word_id in words_by_id)


def _segment_duration_us(segment: FinalTimelineSegment) -> int:
    return max(0, int(segment.target_end_us) - int(segment.target_start_us))


def _ordered_captions(captions: list[CaptionRenderUnit]) -> list[CaptionRenderUnit]:
    return sorted(captions, key=lambda row: (int(row.target_start_us), int(row.target_end_us), str(row.caption_id)))


def _ordered_segments(segments: list[FinalTimelineSegment]) -> list[FinalTimelineSegment]:
    return sorted(segments, key=lambda row: (int(row.target_start_us), int(row.target_end_us), str(row.segment_id)))


def _caption_by_id(captions: list[CaptionRenderUnit], caption_id: str) -> CaptionRenderUnit | None:
    for caption in captions:
        if caption.caption_id == caption_id:
            return caption
    no_caption: CaptionRenderUnit | None = None
    return no_caption


def _caption_index(captions: list[CaptionRenderUnit], caption_id: str) -> int | None:
    for index, caption in enumerate(captions):
        if caption.caption_id == caption_id:
            return index
    no_index: int | None = None
    return no_index


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


def _caption_only_materialization_for_visible_caption(
    visible: CaptionRenderUnit,
    timeline_captions: list[CaptionRenderUnit],
    consumed_indices: set[int],
) -> tuple[int, list[int], list[CaptionRenderUnit], dict[str, Any]] | None:
    if not bool(build_final_caption_visible_repeat_gate([visible]).get("gate_passed")):
        no_match: tuple[int, list[int], list[CaptionRenderUnit], dict[str, Any]] | None = None
        return no_match
    for indices, source_captions in _caption_only_source_windows(visible, timeline_captions, consumed_indices):
        replacements, materialization_type, partial_row = _caption_only_replacements(visible, source_captions)
        if replacements is None:
            continue
        if not _visible_target_range_covers_materialization(visible, source_captions, materialization_type):
            continue
        if not _caption_only_window_gaps_are_safe(source_captions):
            continue
        row = {
            "merged_caption_id": visible.caption_id,
            "merged_caption_text": visible.text,
            "merged_caption_timeline_segment_ids": list(visible.timeline_segment_ids),
            "source_caption_ids": [caption.caption_id for caption in source_captions],
            "consumed_caption_ids": [caption.caption_id for caption in source_captions[1:]],
            "consumed_timeline_segment_ids": [
                segment_id
                for caption in source_captions[1:]
                for segment_id in _caption_segment_ids(caption)
            ],
            "merged_into_caption_id": source_captions[0].caption_id,
            "state": "materialized_caption_only_merge",
            "materialization_type": materialization_type,
            **partial_row,
        }
        return indices[0], indices, replacements, row
    no_match: tuple[int, list[int], list[CaptionRenderUnit], dict[str, Any]] | None = None
    return no_match


def _caption_only_source_windows(
    visible: CaptionRenderUnit,
    timeline_captions: list[CaptionRenderUnit],
    consumed_indices: set[int],
) -> list[tuple[list[int], list[CaptionRenderUnit]]]:
    visible_segment_ids = set(_caption_segment_ids(visible))
    if not visible_segment_ids:
        empty_windows: list[tuple[list[int], list[CaptionRenderUnit]]] = []
        return empty_windows
    candidate_indices = [
        index
        for index, caption in enumerate(timeline_captions)
        if index not in consumed_indices and visible_segment_ids.intersection(_caption_segment_ids(caption))
    ]
    windows: list[tuple[list[int], list[CaptionRenderUnit]]] = []
    for start_offset in range(len(candidate_indices)):
        for end_offset in range(start_offset + 1, len(candidate_indices)):
            indices = candidate_indices[start_offset : end_offset + 1]
            if indices != list(range(indices[0], indices[-1] + 1)):
                continue
            source_captions = [timeline_captions[index] for index in indices]
            source_segment_ids = {
                segment_id
                for caption in source_captions
                for segment_id in _caption_segment_ids(caption)
            }
            if not visible_segment_ids.issubset(source_segment_ids):
                continue
            windows.append((indices, source_captions))
    windows.sort(key=lambda row: (len(row[0]), row[0][0]))
    return windows


def _caption_only_window_gaps_are_safe(source_captions: list[CaptionRenderUnit]) -> bool:
    for left, right in zip(source_captions, source_captions[1:]):
        gap_us = int(right.target_start_us) - int(left.target_end_us)
        if gap_us < 0 or gap_us > MAX_CAPTION_ONLY_TARGET_GAP_US:
            return False
    return True


def _visible_target_range_covers_materialization(
    visible: CaptionRenderUnit,
    source_captions: list[CaptionRenderUnit],
    materialization_type: str,
) -> bool:
    if int(visible.target_end_us) < int(source_captions[-1].target_end_us):
        return False
    if materialization_type == "partial_previous_segment_tail":
        first = source_captions[0]
        return int(first.target_start_us) <= int(visible.target_start_us) <= int(first.target_end_us)
    return int(visible.target_start_us) <= int(source_captions[0].target_start_us)


def _caption_only_replacements(
    visible: CaptionRenderUnit,
    source_captions: list[CaptionRenderUnit],
) -> tuple[list[CaptionRenderUnit] | None, str, dict[str, Any]]:
    expected_text = normalize_text(_join_visible_caption_sequence_text([str(caption.text or "") for caption in source_captions]))
    if normalize_text(visible.text) == expected_text:
        return [visible], "whole_segment_sequence", {}
    if len(source_captions) < 2:
        return None, "", {}
    first = source_captions[0]
    tail_match = _partial_previous_tail_match(visible, source_captions)
    if tail_match is None:
        return None, "", {}
    first_tail_word_ids, first_prefix_word_ids, first_tail_text, first_prefix_text = tail_match
    replacements: list[CaptionRenderUnit] = []
    if first_prefix_word_ids and normalize_text(first_prefix_text):
        prefix_end_us = min(int(first.target_end_us), int(visible.target_start_us))
        if prefix_end_us <= int(first.target_start_us):
            prefix_end_us = int(first.target_end_us)
        replacements.append(
            replace(
                first,
                caption_id=f"{first.caption_id}_prefix",
                word_ids=first_prefix_word_ids,
                text=first_prefix_text,
                target_end_us=prefix_end_us,
                containing_video_segment_id=first.containing_video_segment_id,
            )
        )
    replacements.append(visible)
    return replacements, "partial_previous_segment_tail", {
        "partial_previous_caption_id": first.caption_id,
        "covered_previous_tail_word_ids": first_tail_word_ids,
        "preserved_previous_prefix_word_ids": first_prefix_word_ids,
        "covered_previous_tail_text": first_tail_text,
        "preserved_previous_prefix_text": first_prefix_text,
    }


def _partial_previous_tail_match(
    visible: CaptionRenderUnit,
    source_captions: list[CaptionRenderUnit],
) -> tuple[list[str], list[str], str, str] | None:
    first = source_captions[0]
    later_word_ids = [word_id for caption in source_captions[1:] for word_id in caption.word_ids]
    visible_word_ids = list(visible.word_ids)
    if not later_word_ids or not _is_suffix(visible_word_ids, later_word_ids):
        no_match: tuple[list[str], list[str], str, str] | None = None
        return no_match
    first_tail_word_ids = visible_word_ids[: len(visible_word_ids) - len(later_word_ids)]
    if not first_tail_word_ids or not _is_suffix(list(first.word_ids), first_tail_word_ids):
        no_match: tuple[list[str], list[str], str, str] | None = None
        return no_match
    first_prefix_word_ids = list(first.word_ids)[: len(first.word_ids) - len(first_tail_word_ids)]
    later_text = "".join(caption.text for caption in source_captions[1:])
    visible_text = str(visible.text or "")
    text_match = _partial_tail_visible_text_match(visible_text, str(first.text or ""), later_text)
    if text_match is None:
        no_match: tuple[list[str], list[str], str, str] | None = None
        return no_match
    tail_text, prefix_text = text_match
    return first_tail_word_ids, first_prefix_word_ids, tail_text, prefix_text


def _partial_tail_visible_text_match(
    visible_text: str,
    first_text: str,
    later_text: str,
) -> tuple[str, str] | None:
    if not normalize_text(later_text):
        no_match: tuple[str, str] | None = None
        return no_match
    for later_visible_text in _right_boundary_text_options_after_non_de_left(later_text):
        tail_text = ""
        if visible_text.endswith(later_visible_text):
            tail_text = visible_text[: max(0, len(visible_text) - len(later_visible_text))]
        else:
            tail_text = _normalized_prefix_before_suffix(visible_text, later_visible_text)
        if not normalize_text(tail_text):
            continue
        prefix_text = _text_before_suffix(first_text, tail_text)
        if prefix_text is None:
            continue
        expected_visible = _join_visible_boundary_text(tail_text, later_text)
        if normalize_text(visible_text) == normalize_text(expected_visible):
            return tail_text, prefix_text
    no_match: tuple[str, str] | None = None
    return no_match


def _join_visible_caption_sequence_text(texts: list[str]) -> str:
    merged = ""
    for text in texts:
        merged = _join_visible_boundary_text(merged, text)
    return merged


def _join_visible_boundary_text(left_text: str, right_text: str) -> str:
    return f"{left_text}{_right_boundary_text_for_join(left_text, right_text)}"


def _right_boundary_text_for_join(left_text: str, right_text: str) -> str:
    normalized_left = normalize_text(left_text)
    normalized_right = normalize_text(right_text)
    if (
        normalized_left
        and not normalized_left.endswith("的")
        and normalized_right.startswith("的是")
        and _de_shi_boundary_should_drop_de(normalized_right)
    ):
        return _drop_leading_de_from_de_shi_text(right_text)
    return right_text


def _right_boundary_text_options_after_non_de_left(right_text: str) -> list[str]:
    options = [right_text]
    normalized_right = normalize_text(right_text)
    if normalized_right.startswith("的是") and _de_shi_boundary_should_drop_de(normalized_right):
        stripped = _drop_leading_de_from_de_shi_text(right_text)
        if stripped and normalize_text(stripped) != normalize_text(right_text):
            options.append(stripped)
    return options


def _de_shi_boundary_should_drop_de(normalized_right: str) -> bool:
    if not normalized_right.startswith("的是"):
        return False
    after_shi = normalized_right[2:]
    return any(after_shi.startswith(prefix) for prefix in DE_SHI_BOUNDARY_NORMALIZE_AFTER)


def _drop_leading_de_from_de_shi_text(text: str) -> str:
    if text.startswith("的是"):
        return text[1:]
    normalized = normalize_text(text)
    if normalized.startswith("的是"):
        return normalized[1:]
    return text


def _text_before_suffix(text: str, suffix: str) -> str | None:
    if text.endswith(suffix):
        return text[: len(text) - len(suffix)]
    normalized_text = normalize_text(text)
    normalized_suffix = normalize_text(suffix)
    if normalized_suffix and normalized_text.endswith(normalized_suffix):
        return normalized_text[: len(normalized_text) - len(normalized_suffix)]
    no_text: str | None = None
    return no_text


def _normalized_prefix_before_suffix(text: str, suffix: str) -> str:
    normalized_text = normalize_text(text)
    normalized_suffix = normalize_text(suffix)
    if normalized_suffix and normalized_text.endswith(normalized_suffix):
        return normalized_text[: len(normalized_text) - len(normalized_suffix)]
    return ""


def _repair_counts(gate: dict[str, Any]) -> dict[str, int]:
    return {key: int(gate.get(key) or 0) for key in FINAL_VISIBLE_REPAIR_COUNT_KEYS}


def _repair_state_signature(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
) -> tuple[Any, ...]:
    timeline_state = tuple(
        (
            segment.segment_id,
            tuple(segment.word_ids),
            normalize_text(segment.text),
            int(segment.source_start_us),
            int(segment.source_end_us),
            int(segment.target_start_us),
            int(segment.target_end_us),
        )
        for segment in _ordered_segments(final_timeline)
    )
    caption_state = tuple(
        (
            caption.caption_id,
            tuple(_caption_segment_ids(caption)),
            tuple(caption.word_ids),
            normalize_text(caption.text),
            int(caption.target_start_us),
            int(caption.target_end_us),
        )
        for caption in _ordered_captions(captions)
    )
    return timeline_state, caption_state


def _caption_only_state_signature(captions: list[CaptionRenderUnit]) -> tuple[Any, ...]:
    return tuple(
        (
            tuple(_caption_segment_ids(caption)),
            tuple(caption.word_ids),
            normalize_text(caption.text),
            int(caption.target_start_us),
            int(caption.target_end_us),
        )
        for caption in _ordered_captions(captions)
    )


def _repack_timeline(final_timeline: list[FinalTimelineSegment]) -> list[FinalTimelineSegment]:
    repacked: list[FinalTimelineSegment] = []
    cursor = 0
    for segment in final_timeline:
        duration = max(0, int(segment.target_end_us) - int(segment.target_start_us))
        repacked.append(replace(segment, target_start_us=cursor, target_end_us=cursor + duration))
        cursor += duration
    return repacked


def _renumber_captions(captions: list[CaptionRenderUnit]) -> list[CaptionRenderUnit]:
    return [
        replace(caption, caption_id=f"v21_cap_{index:06d}")
        for index, caption in enumerate(_ordered_captions(captions), start=1)
    ]


def _action(
    issue_type: str,
    decision: str,
    pass_index: int,
    candidate: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "pass_index": pass_index,
        "issue_type": issue_type,
        "decision": decision,
        "caption_id": str(candidate.get("caption_id") or ""),
        "related_caption_id": str(candidate.get("related_caption_id") or ""),
        "reason": str(candidate.get("reason") or ""),
        "overlap_text": str(candidate.get("overlap_text") or ""),
        **extra,
    }


def _is_prefix(values: list[str], prefix: list[str]) -> bool:
    return len(values) >= len(prefix) and values[: len(prefix)] == prefix


def _is_suffix(values: list[str], suffix: list[str]) -> bool:
    return len(values) >= len(suffix) and values[len(values) - len(suffix) :] == suffix


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
