from __future__ import annotations

from dataclasses import replace
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.boundary_overlap import is_open_predicate_bridge
from aroll_v21.quality.final_visible_repair.report import _is_prefix, _is_suffix, _unique
from aroll_v21.quality.final_visible_repair.timeline_utils import (
    caption_segment_ids as _caption_segment_ids,
    segment_duration_us as _segment_duration_us,
    text_from_word_ids as _text_from_word_ids,
)
from aroll_v21.quality.subtitle_readability import HARD_MAX_CHARS, HARD_MAX_DURATION_US


MIN_REPAIRED_SEGMENT_DURATION_US = 1_200_000
MAX_PROTECTED_BRIDGE_CONTEXT_GAP_US = 800_000


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
    for segment_index, segment in enumerate(final_timeline):
        word_ids = list(segment.word_ids)
        if not drop_set.intersection(word_ids):
            repaired.append(segment)
            continue
        if _is_prefix(word_ids, drop_word_ids):
            dropped_from_segment = word_ids[: len(drop_word_ids)]
            remaining = word_ids[len(drop_word_ids) :]
        elif _is_suffix(word_ids, drop_word_ids):
            dropped_from_segment = word_ids[len(word_ids) - len(drop_word_ids) :]
            remaining = word_ids[: len(word_ids) - len(drop_word_ids)]
        elif set(word_ids) == drop_set:
            dropped_from_segment = word_ids
            remaining = []
        else:
            no_trim: list[FinalTimelineSegment] | None = None
            return no_trim
        if _would_drop_protected_open_predicate_bridge(
            final_timeline=final_timeline,
            segment_index=segment_index,
            segment_word_ids=word_ids,
            dropped_word_ids=dropped_from_segment,
            source_graph=source_graph,
        ):
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
    for segment_index, segment in enumerate(final_timeline):
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
        dropped_from_segment = word_ids[start:end]
        if _would_drop_protected_open_predicate_bridge(
            final_timeline=final_timeline,
            segment_index=segment_index,
            segment_word_ids=word_ids,
            dropped_word_ids=dropped_from_segment,
            source_graph=source_graph,
        ):
            no_trim: list[FinalTimelineSegment] | None = None
            return no_trim
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


def _would_drop_protected_open_predicate_bridge(
    *,
    final_timeline: list[FinalTimelineSegment],
    segment_index: int,
    segment_word_ids: list[str],
    dropped_word_ids: list[str],
    source_graph: CanonicalSourceGraph,
) -> bool:
    dropped_text = normalize_text(_text_from_word_ids(dropped_word_ids, source_graph))
    if not dropped_text or not is_open_predicate_bridge(dropped_text):
        return False
    span = _word_id_span(segment_word_ids, dropped_word_ids)
    if span is None:
        return False
    start, end = span
    words_by_id = {word.word_id: word for word in source_graph.words}
    first_dropped = words_by_id.get(dropped_word_ids[0])
    last_dropped = words_by_id.get(dropped_word_ids[-1])
    if first_dropped is None or last_dropped is None:
        return False
    previous_word_id = segment_word_ids[start - 1] if start > 0 else _adjacent_segment_word_id(final_timeline, segment_index, -1)
    next_word_id = segment_word_ids[end] if end < len(segment_word_ids) else _adjacent_segment_word_id(final_timeline, segment_index, 1)
    previous_word = words_by_id.get(previous_word_id) if previous_word_id else None
    next_word = words_by_id.get(next_word_id) if next_word_id else None
    return (
        previous_word is not None
        and _words_source_contiguous(previous_word, first_dropped)
    ) or (
        next_word is not None
        and _words_source_contiguous(last_dropped, next_word)
    )


def _word_id_span(values: list[str], subsequence: list[str]) -> tuple[int, int] | None:
    if not values or not subsequence or len(subsequence) > len(values):
        no_span: tuple[int, int] | None = None
        return no_span
    width = len(subsequence)
    for index in range(0, len(values) - width + 1):
        if values[index : index + width] == subsequence:
            return index, index + width
    no_span: tuple[int, int] | None = None
    return no_span


def _adjacent_segment_word_id(
    final_timeline: list[FinalTimelineSegment],
    segment_index: int,
    direction: int,
) -> str:
    adjacent_index = int(segment_index) + int(direction)
    if adjacent_index < 0 or adjacent_index >= len(final_timeline):
        return ""
    word_ids = list(final_timeline[adjacent_index].word_ids)
    if not word_ids:
        return ""
    return word_ids[-1] if direction < 0 else word_ids[0]


def _words_source_contiguous(left: Any, right: Any) -> bool:
    if str(getattr(left, "source_material_id", "") or "") and str(getattr(right, "source_material_id", "") or ""):
        if str(getattr(left, "source_material_id", "") or "") != str(getattr(right, "source_material_id", "") or ""):
            return False
    if str(getattr(left, "source_segment_id", "") or "") and str(getattr(right, "source_segment_id", "") or ""):
        if str(getattr(left, "source_segment_id", "") or "") != str(getattr(right, "source_segment_id", "") or ""):
            return False
    gap_us = int(getattr(right, "source_start_us", 0) or 0) - int(getattr(left, "source_end_us", 0) or 0)
    return -80_000 <= gap_us <= MAX_PROTECTED_BRIDGE_CONTEXT_GAP_US


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


def _trailing_word_ids_for_text(
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
    for word_id in reversed(word_ids):
        word = words_by_id.get(word_id)
        if word is None:
            empty: list[str] = []
            return empty
        selected.append(word_id)
        joined = normalize_text(word.text) + joined
        if joined == target:
            return list(reversed(selected))
        if not target.endswith(joined):
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


def _safe_merge_segments(
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
) -> bool:
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
