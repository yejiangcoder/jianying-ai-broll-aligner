from __future__ import annotations

from typing import Any


def configure_rule_dependencies(dependencies: dict[str, Any]) -> None:
    globals().update(dependencies)


MIN_REPAIRED_SEGMENT_DURATION_US = 1_200_000


MAX_REPAIRED_RESIDUAL_DROP_DURATION_US = 500_000


MAX_REPAIRED_RESIDUAL_DROP_CHARS = 2


MIN_REBALANCED_CAPTION_DURATION_US = 300_000


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
    merged = _merged_segment_pair_preserving_effective_speed(left, right, source_graph, repair_reason)
    return [*segments[:index], merged, *segments[index + 2 :]]
