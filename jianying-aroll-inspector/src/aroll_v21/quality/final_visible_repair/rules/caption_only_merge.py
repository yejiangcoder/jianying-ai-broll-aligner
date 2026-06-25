from __future__ import annotations

from typing import Any


def configure_rule_dependencies(dependencies: dict[str, Any]) -> None:
    globals().update(dependencies)


MAX_CAPTION_ONLY_TARGET_GAP_US = 120_000

MAX_SAME_SUBTITLE_SHORT_TAIL_CHARS = 2
MAX_SAME_SUBTITLE_SHORT_TAIL_SOURCE_GAP_US = 800_000


SUBJECT_PREFIX_COMPLETED_PREDICATE_REPAIR_REASON = "subject_prefix_completed_predicate_restart"
SUBJECT_PREFIX_COMPLETED_PREDICATE_STARTS = ("全是", "都是", "全都是", "全部是", "尽是", "就是")


def _finalize_caption_only_dangling_merges(
    captions: list[CaptionRenderUnit],
    *,
    source_graph: CanonicalSourceGraph,
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
            if step is None:
                index = _caption_index(_ordered_captions(current), str(candidate.get("caption_id") or ""))
                if index is not None and index > 0:
                    step = _transfer_leading_function_prefix_to_previous_caption(
                        final_timeline=[],
                        captions=_ordered_captions(current),
                        previous_index=index - 1,
                        current_index=index,
                        source_graph=source_graph,
                        candidate=candidate,
                        pass_index=pass_index,
                    )
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


def _finalize_subject_prefix_completed_predicate_caption_merges(
    captions: list[CaptionRenderUnit],
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index_start: int,
) -> tuple[list[CaptionRenderUnit], list[dict[str, Any]]]:
    current = _renumber_captions(list(captions))
    actions: list[dict[str, Any]] = []
    pass_index = max(1, pass_index_start)
    segments_by_id = {segment.segment_id: segment for segment in final_timeline}
    while True:
        ordered = _ordered_captions(current)
        step: tuple[list[CaptionRenderUnit], dict[str, Any]] | None = None
        for index in range(len(ordered) - 1):
            left = ordered[index]
            right = ordered[index + 1]
            if not _caption_has_subject_prefix_restart_marker(left, segments_by_id):
                continue
            if not _caption_starts_with_completed_predicate(right):
                continue
            merged_result = _merge_adjacent_captions(left, right)
            if merged_result is None:
                continue
            merged_caption, merge_decision = merged_result
            if not bool(build_final_caption_visible_repeat_gate([merged_caption]).get("gate_passed")):
                continue
            repaired = [*ordered[:index], merged_caption, *ordered[index + 2 :]]
            step = (
                _renumber_captions(repaired),
                _action(
                    "subject_prefix_completed_predicate_restart",
                    "caption_only_merge_subject_prefix_with_completed_predicate",
                    pass_index,
                    {
                        "caption_id": left.caption_id,
                        "related_caption_id": right.caption_id,
                        "reason": "trimmed subject prefix caption should display with the completed predicate caption",
                        "overlap_text": normalize_text(str(right.text or "")),
                    },
                    affected_caption_ids=[left.caption_id, right.caption_id],
                    video_segment_merged=False,
                    caption_only_merge_materialized=True,
                    caption_only_merge_decision=merge_decision,
                    merged_into_caption_id=left.caption_id,
                    consumed_caption_id=right.caption_id,
                    consumed_caption_state="consumed_by_subject_prefix_completed_predicate_merge",
                    merged_caption_text=merged_caption.text,
                    merged_caption_timeline_segment_ids=list(merged_caption.timeline_segment_ids),
                    merged_caption_target_start_us=int(merged_caption.target_start_us),
                    merged_caption_target_end_us=int(merged_caption.target_end_us),
                ),
            )
            break
        if step is None:
            return current, actions
        repaired, action = step
        if _caption_only_state_signature(repaired) == _caption_only_state_signature(current):
            return current, actions
        current = repaired
        actions.append(action)
        pass_index += 1


def _finalize_same_subtitle_short_tail_caption_merges(
    captions: list[CaptionRenderUnit],
    *,
    source_graph: CanonicalSourceGraph,
    pass_index_start: int,
) -> tuple[list[CaptionRenderUnit], list[dict[str, Any]]]:
    current = _renumber_captions(list(captions))
    actions: list[dict[str, Any]] = []
    pass_index = max(1, pass_index_start)
    while True:
        ordered = _ordered_captions(current)
        step: tuple[list[CaptionRenderUnit], dict[str, Any]] | None = None
        for index in range(len(ordered) - 1):
            left = ordered[index]
            right = ordered[index + 1]
            if not _same_subtitle_short_tail_should_merge(left, right, source_graph):
                continue
            merged_result = _merge_adjacent_captions(left, right)
            if merged_result is None:
                continue
            merged_caption, merge_decision = merged_result
            if not bool(build_final_caption_visible_repeat_gate([merged_caption]).get("gate_passed")):
                continue
            repaired = [*ordered[:index], merged_caption, *ordered[index + 2 :]]
            step = (
                _renumber_captions(repaired),
                _action(
                    "same_subtitle_short_tail_caption",
                    "caption_only_merge_same_subtitle_short_tail",
                    pass_index,
                    {
                        "caption_id": right.caption_id,
                        "related_caption_id": left.caption_id,
                        "reason": "short tail caption belongs to the same source subtitle as the previous caption",
                        "overlap_text": normalize_text(str(right.text or "")),
                    },
                    affected_caption_ids=[left.caption_id, right.caption_id],
                    target_gap_us=int(right.target_start_us) - int(left.target_end_us),
                    video_segment_merged=False,
                    caption_only_merge_materialized=True,
                    caption_only_merge_decision=merge_decision,
                    merged_into_caption_id=left.caption_id,
                    consumed_caption_id=right.caption_id,
                    consumed_caption_state="consumed_by_same_subtitle_short_tail_merge",
                    merged_caption_text=merged_caption.text,
                    merged_caption_timeline_segment_ids=list(merged_caption.timeline_segment_ids),
                    merged_caption_target_start_us=int(merged_caption.target_start_us),
                    merged_caption_target_end_us=int(merged_caption.target_end_us),
                ),
            )
            break
        if step is None:
            return current, actions
        repaired, action = step
        if _caption_only_state_signature(repaired) == _caption_only_state_signature(current):
            return current, actions
        current = repaired
        actions.append(action)
        pass_index += 1


def _same_subtitle_short_tail_should_merge(
    left: CaptionRenderUnit,
    right: CaptionRenderUnit,
    source_graph: CanonicalSourceGraph,
) -> bool:
    right_text = normalize_text(str(right.text or ""))
    if not right_text or len(right_text) > MAX_SAME_SUBTITLE_SHORT_TAIL_CHARS:
        return False
    target_gap_us = int(right.target_start_us) - int(left.target_end_us)
    if target_gap_us < 0 or target_gap_us > MAX_CAPTION_ONLY_TARGET_GAP_US:
        return False
    source_gap_us = int(right.spoken_source_start_us or 0) - int(left.spoken_source_end_us or 0)
    if source_gap_us < 0 or source_gap_us > MAX_SAME_SUBTITLE_SHORT_TAIL_SOURCE_GAP_US:
        return False
    shared_uids = {
        uid
        for uid in set(str(value) for value in list(left.source_subtitle_uids or []))
        & set(str(value) for value in list(right.source_subtitle_uids or []))
        if uid
    }
    if not shared_uids:
        return False
    merged_text = normalize_text(_join_visible_boundary_text(str(left.text or ""), str(right.text or "")))
    if not merged_text:
        return False
    subtitle_texts = _source_subtitle_texts_by_uid(source_graph)
    if not any(merged_text in normalize_text(subtitle_texts.get(uid, "")) for uid in shared_uids):
        return False
    return _caption_only_merge_allowed(left, right)


def _source_subtitle_texts_by_uid(source_graph: CanonicalSourceGraph) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in list(source_graph.subtitle_rows or []):
        uid = str(row.get("subtitle_uid") or row.get("id") or "")
        if not uid:
            continue
        result[uid] = str(row.get("text") or row.get("subtitle_text") or row.get("recognize_text") or "")
    return result


def _caption_has_subject_prefix_restart_marker(
    caption: CaptionRenderUnit,
    segments_by_id: dict[str, FinalTimelineSegment],
) -> bool:
    for segment_id in _caption_segment_ids(caption):
        segment = segments_by_id.get(segment_id)
        if segment is None:
            continue
        if str((segment.debug_hints or {}).get("final_visible_repair") or "") == SUBJECT_PREFIX_COMPLETED_PREDICATE_REPAIR_REASON:
            return True
    return False


def _caption_starts_with_completed_predicate(caption: CaptionRenderUnit) -> bool:
    text = normalize_text(str(caption.text or ""))
    return bool(text) and any(text.startswith(prefix) for prefix in SUBJECT_PREFIX_COMPLETED_PREDICATE_STARTS)


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
    merged = _merged_segment_pair_preserving_effective_speed(
        left,
        right,
        source_graph,
        "merge_dangling_prefix_suffix",
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


def _caption_only_state_signature(captions: list[CaptionRenderUnit]) -> tuple[Any, ...]:
    return _caption_only_state_signature_impl(
        captions,
        ordered_captions=_ordered_captions,
        caption_segment_ids=_caption_segment_ids,
    )
