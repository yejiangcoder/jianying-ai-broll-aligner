from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.final_timeline_quality_guard import build_final_timeline_quality_guard_report


DEFAULT_LEAD_HANDLE_US = 220_000
DEFAULT_TAIL_HANDLE_US = 220_000


@dataclass(frozen=True)
class FinalTimelineRepairApplyResult:
    final_timeline: list[FinalTimelineSegment]
    captions: list[CaptionRenderUnit]
    action: dict[str, Any]
    timeline_changed: bool


def apply_next_final_timeline_repair_intent(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> FinalTimelineRepairApplyResult | None:
    """Apply one deterministic final-timeline repair intent in memory."""

    guard_report = build_final_timeline_quality_guard_report(
        source_graph=source_graph,
        final_timeline=final_timeline,
        captions=captions,
    )
    intents = list((guard_report.get("repair_intent_report") or {}).get("repair_intents") or [])
    for intent in intents:
        if str(intent.get("safety_level") or "") != "deterministic_candidate":
            continue
        result = _apply_intent(
            intent=intent,
            final_timeline=final_timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index,
        )
        if result is not None:
            return result
    no_result: FinalTimelineRepairApplyResult | None = None
    return no_result


def recompute_final_timeline_safe_handles(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> FinalTimelineRepairApplyResult | None:
    """Run the final safe-cut handle pass after all source-word mutations settle."""

    if not _timeline_has_safe_handle_policy(final_timeline):
        no_result: FinalTimelineRepairApplyResult | None = None
        return no_result
    handled = _recompute_safe_handles(list(final_timeline), source_graph)
    if _timeline_physical_signature(handled) == _timeline_physical_signature(final_timeline):
        no_result: FinalTimelineRepairApplyResult | None = None
        return no_result
    intent = {
        "intent_id": f"final_timeline_safe_handle_recompute_{pass_index:06d}",
        "intent_type": "recompute_all_safe_cut_handles",
        "source_candidate_type": "missing_requested_lead_handle",
        "safe_cut_recompute_required": True,
        "reason": "final safe-cut recompute after all final timeline repairs",
    }
    return FinalTimelineRepairApplyResult(
        final_timeline=handled,
        captions=list(captions),
        action=_action_from_intent(
            intent,
            pass_index,
            decision="recompute_safe_cut_handles_after_final_timeline_repairs",
        ),
        timeline_changed=True,
    )


def _timeline_has_safe_handle_policy(segments: list[FinalTimelineSegment]) -> bool:
    for segment in segments:
        debug_hints = dict(segment.debug_hints or {})
        if (
            debug_hints.get("safe_handle_policy_enabled")
            or "safe_handle_requested_lead_us" in debug_hints
            or "safe_handle_requested_tail_us" in debug_hints
            or int(segment.lead_handle_us or 0) > 0
            or int(segment.tail_handle_us or 0) > 0
        ):
            return True
    return False


def _apply_intent(
    *,
    intent: dict[str, Any],
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> FinalTimelineRepairApplyResult | None:
    intent_type = str(intent.get("intent_type") or "")
    if intent_type == "drop_restart_residue_segment":
        return _apply_drop_segment_intent(intent, final_timeline, source_graph, render_captions, pass_index)
    if intent_type == "split_intra_sentence_pause_gap":
        return _apply_split_intra_sentence_pause_gap_intent(intent, final_timeline, source_graph, render_captions, pass_index)
    if intent_type == "trim_dangling_words_before_connector":
        return _apply_trim_leading_word_ids_intent(intent, final_timeline, source_graph, render_captions, pass_index)
    if intent_type == "refresh_segment_text_from_source_words":
        return _apply_refresh_segment_text_intent(intent, final_timeline, source_graph, render_captions, pass_index)
    if intent_type == "rerender_caption_from_source_words":
        rendered = render_captions(final_timeline)
        if _caption_signature(rendered) == _caption_signature(captions):
            no_result: FinalTimelineRepairApplyResult | None = None
            return no_result
        return FinalTimelineRepairApplyResult(
            final_timeline=list(final_timeline),
            captions=rendered,
            action=_action_from_intent(intent, pass_index, decision="rerender_captions_from_source_words"),
            timeline_changed=False,
        )
    if intent_type == "recompute_missing_lead_handle":
        handled = _recompute_safe_handles(list(final_timeline), source_graph)
        if _timeline_physical_signature(handled) == _timeline_physical_signature(final_timeline):
            no_result: FinalTimelineRepairApplyResult | None = None
            return no_result
        return FinalTimelineRepairApplyResult(
            final_timeline=handled,
            captions=list(captions),
            action=_action_from_intent(intent, pass_index, decision="recompute_safe_cut_handles"),
            timeline_changed=True,
        )
    no_result = None
    return no_result


def _apply_drop_segment_intent(
    intent: dict[str, Any],
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> FinalTimelineRepairApplyResult | None:
    segment_id = str(intent.get("segment_id") or "")
    drop_word_ids = _string_list(intent.get("drop_word_ids"))
    semantic_safe_drop = bool(intent.get("semantic_restart_residue_safe_drop"))
    if not bool(intent.get("is_visual_gap_split")) and not semantic_safe_drop:
        no_result: FinalTimelineRepairApplyResult | None = None
        return no_result
    if not drop_word_ids:
        no_result: FinalTimelineRepairApplyResult | None = None
        return no_result
    kept: list[FinalTimelineSegment] = []
    dropped: FinalTimelineSegment | None = None
    for segment in final_timeline:
        if _string_list(segment.word_ids) != drop_word_ids:
            kept.append(segment)
            continue
        if segment_id and str(segment.segment_id) != segment_id:
            segment_id = str(segment.segment_id)
        dropped = segment
    if dropped is None or not kept:
        no_result = None
        return no_result
    expected_preserved_text = str(intent.get("expected_preserved_text") or "")
    if expected_preserved_text and not _timeline_contains_source_text(kept, source_graph, expected_preserved_text):
        no_result = None
        return no_result
    repaired = _repack_and_recompute_handles(kept, source_graph)
    return FinalTimelineRepairApplyResult(
        final_timeline=repaired,
        captions=render_captions(repaired),
        action=_action_from_intent(
            intent,
            pass_index,
            decision="drop_restart_residue_segment",
            dropped_segment_id=segment_id,
            dropped_word_ids=drop_word_ids,
            semantic_restart_residue_safe_drop=semantic_safe_drop,
        ),
        timeline_changed=True,
    )


def _apply_trim_leading_word_ids_intent(
    intent: dict[str, Any],
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> FinalTimelineRepairApplyResult | None:
    segment_id = str(intent.get("segment_id") or "")
    drop_word_ids = _string_list(intent.get("drop_word_ids"))
    if not segment_id or not drop_word_ids:
        no_result: FinalTimelineRepairApplyResult | None = None
        return no_result
    repaired: list[FinalTimelineSegment] = []
    trimmed_segment: FinalTimelineSegment | None = None
    for segment in final_timeline:
        if str(segment.segment_id) != segment_id:
            repaired.append(segment)
            continue
        word_ids = _string_list(segment.word_ids)
        if word_ids[: len(drop_word_ids)] != drop_word_ids:
            no_result = None
            return no_result
        remaining_word_ids = word_ids[len(drop_word_ids) :]
        if not remaining_word_ids:
            no_result = None
            return no_result
        trimmed_segment = _segment_from_word_ids(
            segment,
            remaining_word_ids,
            source_graph,
            repair_reason="final_timeline_intent_trim_dangling_connector_prefix",
        )
        if trimmed_segment is None:
            no_result = None
            return no_result
        repaired.append(trimmed_segment)
    if trimmed_segment is None:
        no_result = None
        return no_result
    repaired = _repack_and_recompute_handles(repaired, source_graph)
    return FinalTimelineRepairApplyResult(
        final_timeline=repaired,
        captions=render_captions(repaired),
        action=_action_from_intent(
            intent,
            pass_index,
            decision="trim_dangling_words_before_connector",
            trimmed_segment_id=segment_id,
            dropped_word_ids=drop_word_ids,
            kept_word_ids=list(trimmed_segment.word_ids),
        ),
        timeline_changed=True,
    )


def _apply_split_intra_sentence_pause_gap_intent(
    intent: dict[str, Any],
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> FinalTimelineRepairApplyResult | None:
    segment_id = str(intent.get("segment_id") or "")
    left_word_ids = _string_list(intent.get("left_word_ids"))
    right_word_ids = _string_list(intent.get("right_word_ids"))
    if not segment_id or not left_word_ids or not right_word_ids:
        no_result: FinalTimelineRepairApplyResult | None = None
        return no_result
    repaired: list[FinalTimelineSegment] = []
    split_segment: FinalTimelineSegment | None = None
    split_left: FinalTimelineSegment | None = None
    split_right: FinalTimelineSegment | None = None
    for segment in final_timeline:
        if str(segment.segment_id) != segment_id:
            repaired.append(segment)
            continue
        word_ids = _string_list(segment.word_ids)
        if word_ids != [*left_word_ids, *right_word_ids]:
            no_result = None
            return no_result
        split_segment = segment
        left = _segment_from_word_ids(
            segment,
            left_word_ids,
            source_graph,
            repair_reason="final_timeline_intent_split_intra_sentence_pause_gap_left",
        )
        right = _segment_from_word_ids(
            segment,
            right_word_ids,
            source_graph,
            repair_reason="final_timeline_intent_split_intra_sentence_pause_gap_right",
        )
        if left is None or right is None:
            no_result = None
            return no_result
        split_left = _mark_pause_gap_split_segment(left, intent)
        split_right = _mark_pause_gap_split_segment(right, intent)
        repaired.extend([split_left, split_right])
    if split_segment is None or split_left is None or split_right is None:
        no_result = None
        return no_result
    repaired = _repack_and_recompute_handles(repaired, source_graph)
    return FinalTimelineRepairApplyResult(
        final_timeline=repaired,
        captions=render_captions(repaired),
        action=_action_from_intent(
            intent,
            pass_index,
            decision="split_intra_sentence_pause_gap",
            split_segment_id=segment_id,
            split_after_word_id=str(intent.get("split_after_word_id") or ""),
            split_before_word_id=str(intent.get("split_before_word_id") or ""),
            left_word_ids=left_word_ids,
            right_word_ids=right_word_ids,
            gap_us=int(intent.get("gap_us") or 0),
        ),
        timeline_changed=True,
    )


def _mark_pause_gap_split_segment(segment: FinalTimelineSegment, intent: dict[str, Any]) -> FinalTimelineSegment:
    debug_hints = dict(segment.debug_hints or {})
    debug_hints.update(
        {
            "safe_handle_policy_enabled": True,
            "final_timeline_intent_split_intra_sentence_pause_gap": True,
            "intra_sentence_pause_gap_us": int(intent.get("gap_us") or 0),
            "intra_sentence_pause_gap_threshold_us": int(intent.get("max_allowed_gap_us") or 0),
        }
    )
    return replace(
        segment,
        decision_ids=sorted(set([*segment.decision_ids, "final_timeline_intent_split_intra_sentence_pause_gap"])),
        debug_hints=debug_hints,
    )


def _apply_refresh_segment_text_intent(
    intent: dict[str, Any],
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> FinalTimelineRepairApplyResult | None:
    segment_id = str(intent.get("segment_id") or "")
    words_by_id = {str(word.word_id): word for word in source_graph.words}
    repaired: list[FinalTimelineSegment] = []
    changed = False
    for segment in final_timeline:
        if str(segment.segment_id) != segment_id:
            repaired.append(segment)
            continue
        source_text = "".join(str(words_by_id[word_id].text) for word_id in _string_list(segment.word_ids) if word_id in words_by_id)
        if not source_text or normalize_text(source_text) == normalize_text(segment.text):
            repaired.append(segment)
            continue
        repaired.append(
            replace(
                segment,
                text=source_text,
                debug_hints={**dict(segment.debug_hints or {}), "final_timeline_intent_metadata_refresh": True},
            )
        )
        changed = True
    if not changed:
        no_result: FinalTimelineRepairApplyResult | None = None
        return no_result
    return FinalTimelineRepairApplyResult(
        final_timeline=repaired,
        captions=render_captions(repaired),
        action=_action_from_intent(intent, pass_index, decision="refresh_segment_text_from_source_words"),
        timeline_changed=True,
    )


def _segment_from_word_ids(
    segment: FinalTimelineSegment,
    word_ids: list[str],
    source_graph: CanonicalSourceGraph,
    *,
    repair_reason: str,
) -> FinalTimelineSegment | None:
    words_by_id = {str(word.word_id): word for word in source_graph.words}
    words = [words_by_id[word_id] for word_id in word_ids if word_id in words_by_id]
    if len(words) != len(word_ids):
        no_segment: FinalTimelineSegment | None = None
        return no_segment
    source_start_us = min(int(word.source_start_us) for word in words)
    source_end_us = max(int(word.source_end_us) for word in words)
    if source_end_us <= source_start_us:
        no_segment = None
        return no_segment
    return replace(
        segment,
        source_start_us=source_start_us,
        source_end_us=source_end_us,
        target_end_us=int(segment.target_start_us) + (source_end_us - source_start_us),
        word_ids=list(word_ids),
        text="".join(str(word.text) for word in words),
        spoken_source_start_us=source_start_us,
        spoken_source_end_us=source_end_us,
        clip_source_start_us=None,
        clip_source_end_us=None,
        lead_handle_us=0,
        tail_handle_us=0,
        debug_hints={**dict(segment.debug_hints or {}), "final_timeline_intent_repair": repair_reason},
    )


def _repack_and_recompute_handles(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> list[FinalTimelineSegment]:
    repacked: list[FinalTimelineSegment] = []
    cursor = 0
    for index, segment in enumerate(segments, start=1):
        duration_us = max(0, int(segment.source_end_us) - int(segment.source_start_us))
        repacked.append(
            replace(
                segment,
                segment_id=f"v21_seg_{index:06d}",
                target_start_us=cursor,
                target_end_us=cursor + duration_us,
            )
        )
        cursor += duration_us
    return _recompute_safe_handles(repacked, source_graph)


def _recompute_safe_handles(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> list[FinalTimelineSegment]:
    if not segments:
        return list()
    lower, upper = _source_bounds(source_graph)
    windows = _source_windows(source_graph)
    handled: list[FinalTimelineSegment] = []
    for index, segment in enumerate(segments):
        spoken_start = int(segment.source_start_us)
        spoken_end = int(segment.source_end_us)
        window = _window_for_range(windows, spoken_start, spoken_end)
        window_lower, window_upper = window if window is not None else (lower, upper)
        debug_hints = dict(segment.debug_hints or {})
        policy_enabled = bool(
            debug_hints.get("safe_handle_policy_enabled")
            or "safe_handle_requested_lead_us" in debug_hints
            or "safe_handle_requested_tail_us" in debug_hints
        )
        requested_lead_us = _requested_handle_us(
            debug_hints,
            "safe_handle_requested_lead_us",
            default_us=DEFAULT_LEAD_HANDLE_US,
            existing_us=int(segment.lead_handle_us or 0),
            policy_enabled=policy_enabled,
        )
        requested_tail_us = _requested_handle_us(
            debug_hints,
            "safe_handle_requested_tail_us",
            default_us=DEFAULT_TAIL_HANDLE_US,
            existing_us=int(segment.tail_handle_us or 0),
            policy_enabled=policy_enabled,
        )
        clip_start = max(window_lower, spoken_start - requested_lead_us)
        clip_end = min(window_upper, spoken_end + requested_tail_us)
        if index > 0:
            previous = handled[-1]
            previous_clip_end = int(previous.clip_source_end_us if previous.clip_source_end_us is not None else previous.source_end_us)
            if clip_start < previous_clip_end:
                clip_start = min(spoken_start, previous_clip_end)
        if index + 1 < len(segments):
            next_segment = segments[index + 1]
            if clip_end > int(next_segment.source_start_us):
                clip_end = spoken_end
        lead = max(0, spoken_start - clip_start)
        tail = max(0, clip_end - spoken_end)
        debug_hints.update(
            {
                "safe_handle_policy_enabled": True,
                "safe_handle_source_window_start_us": int(window_lower),
                "safe_handle_source_window_end_us": int(window_upper),
                "safe_handle_requested_lead_us": requested_lead_us,
                "safe_handle_requested_tail_us": requested_tail_us,
                "safe_handle_recomputed_by_final_timeline_intent": True,
            }
        )
        handled.append(
            replace(
                segment,
                spoken_source_start_us=spoken_start,
                spoken_source_end_us=spoken_end,
                clip_source_start_us=clip_start,
                clip_source_end_us=clip_end,
                lead_handle_us=lead,
                tail_handle_us=tail,
                debug_hints=debug_hints,
            )
        )
    return handled


def _requested_handle_us(
    debug_hints: dict[str, Any],
    key: str,
    *,
    default_us: int,
    existing_us: int,
    policy_enabled: bool,
) -> int:
    if key in debug_hints:
        try:
            return max(0, int(debug_hints.get(key) or 0))
        except (TypeError, ValueError):
            return 0
    if policy_enabled:
        return default_us
    return max(0, int(existing_us or 0))


def _source_bounds(source_graph: CanonicalSourceGraph) -> tuple[int, int]:
    starts: list[int] = []
    ends: list[int] = []
    for row in source_graph.source_segments:
        start, end = _source_segment_bounds(row)
        if end > start:
            starts.append(start)
            ends.append(end)
    if starts and ends:
        return min(starts), max(ends)
    word_starts = [int(word.source_start_us) for word in source_graph.words]
    word_ends = [int(word.source_end_us) for word in source_graph.words]
    return min(word_starts, default=0), max(word_ends, default=0)


def _source_windows(source_graph: CanonicalSourceGraph) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for row in source_graph.source_segments:
        if "video" not in str(row.get("track_type") or row.get("type") or "").lower():
            continue
        start, end = _source_segment_bounds(row)
        if end > start:
            windows.append((start, end))
    return sorted(set(windows))


def _source_segment_bounds(row: dict[str, Any]) -> tuple[int, int]:
    for start_key, end_key in (
        ("canonical_source_start_us", "canonical_source_end_us"),
        ("target_start_us", "target_end_us"),
        ("source_start_us", "source_end_us"),
    ):
        start = _time_value(row.get(start_key))
        end = _time_value(row.get(end_key))
        if start is not None and end is not None and end > start:
            return start, end
    return 0, 0


def _timeline_contains_source_text(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    expected_text: str,
) -> bool:
    expected = normalize_text(expected_text)
    if not expected:
        return True
    words_by_id = {str(word.word_id): word for word in source_graph.words}
    for segment in segments:
        source_text = "".join(str(words_by_id[word_id].text) for word_id in _string_list(segment.word_ids) if word_id in words_by_id)
        if normalize_text(source_text) == expected:
            return True
    return False


def _time_value(value: Any) -> int | None:
    if value is None:
        missing: int | None = None
        return missing
    try:
        return int(value)
    except (TypeError, ValueError):
        invalid: int | None = None
        return invalid


def _window_for_range(windows: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    matches = [(window_start, window_end) for window_start, window_end in windows if window_start <= start and end <= window_end]
    return matches[0] if len(matches) == 1 else None


def _action_from_intent(
    intent: dict[str, Any],
    pass_index: int,
    *,
    decision: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "pass_index": pass_index,
        "issue_type": "final_timeline_repair_intent",
        "decision": decision,
        "intent_id": str(intent.get("intent_id") or ""),
        "intent_type": str(intent.get("intent_type") or ""),
        "source_candidate_type": str(intent.get("source_candidate_type") or ""),
        "segment_id": str(intent.get("segment_id") or ""),
        "caption_id": str(intent.get("caption_id") or ""),
        "safe_cut_recompute_required": bool(intent.get("safe_cut_recompute_required")),
        "reason": str(intent.get("reason") or ""),
        **extra,
    }


def _timeline_physical_signature(segments: list[FinalTimelineSegment]) -> tuple[Any, ...]:
    return tuple(
        (
            segment.segment_id,
            tuple(segment.word_ids),
            normalize_text(segment.text),
            int(segment.source_start_us),
            int(segment.source_end_us),
            segment.spoken_source_start_us,
            segment.spoken_source_end_us,
            segment.clip_source_start_us,
            segment.clip_source_end_us,
            int(segment.lead_handle_us or 0),
            int(segment.tail_handle_us or 0),
        )
        for segment in segments
    )


def _caption_signature(captions: list[CaptionRenderUnit]) -> tuple[Any, ...]:
    return tuple(
        (
            tuple(caption.timeline_segment_ids),
            tuple(caption.word_ids),
            normalize_text(caption.text),
            int(caption.target_start_us),
            int(caption.target_end_us),
        )
        for caption in captions
    )


def _string_list(values: Any) -> list[str]:
    return [str(value) for value in list(values or [])]
