from __future__ import annotations

from dataclasses import replace
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, FinalTimelineSegment
from aroll_v21.quality.boundary_overlap import (
    boundary_suffix_prefix_overlap,
    is_semantic_label_reuse_boundary,
)
from aroll_v21.quality.safe_boundary import trailing_word_ids_for_suffix_overlap
from aroll_v21.quality.repeated_suffix_island import is_coordinated_parallel_suffix_repetition
from aroll_v21.quality.visual_pacing.timeline_utils import _repack

def _drop_repeated_suffix_islands_by_subtitle(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> tuple[list[FinalTimelineSegment], int, int]:
    word_lookup = {word.word_id: word for word in source_graph.words}
    cleaned: list[FinalTimelineSegment] = []
    dropped_count = 0
    split_segment_count = 0
    for segment in segments:
        cleaned_segments, dropped_word_ids = _clean_segment_repeated_suffix_islands(segment, word_lookup)
        dropped_count += len(dropped_word_ids)
        if dropped_word_ids and len(cleaned_segments) != 1:
            split_segment_count += len(cleaned_segments)
        cleaned.extend(cleaned_segments)
    return cleaned, dropped_count, split_segment_count


def _clean_segment_repeated_suffix_islands(
    segment: FinalTimelineSegment,
    word_lookup: dict[str, Any],
) -> tuple[list[FinalTimelineSegment], list[str]]:
    words = [word_lookup[word_id] for word_id in segment.word_ids if word_id in word_lookup]
    if len(words) < 3:
        return [segment], []
    dropped_word_ids: set[str] = set()
    group: list[Any] = []
    group_key: object = object()
    for word in [*words, None]:
        key = (
            getattr(word, "subtitle_index", None),
            getattr(word, "subtitle_uid", None),
        ) if word is not None else object()
        if group and key != group_key:
            tokens = [normalize_text(str(getattr(item, "text", "") or "")) for item in group]
            drop_start = _repeated_suffix_island_start(tokens)
            if drop_start is not None:
                dropped_word_ids.update(str(getattr(item, "word_id")) for item in group[drop_start:])
            group = []
        if word is not None:
            group.append(word)
            group_key = key
    if not dropped_word_ids:
        return [segment], []
    kept_runs: list[list[Any]] = []
    current_run: list[Any] = []
    for word in words:
        word_id = str(getattr(word, "word_id"))
        if word_id in dropped_word_ids:
            if current_run:
                kept_runs.append(current_run)
                current_run = []
            continue
        current_run.append(word)
    if current_run:
        kept_runs.append(current_run)
    if not kept_runs:
        return [segment], []
    cleaned_segments = [
        replace(
            segment,
            source_start_us=int(getattr(run[0], "source_start_us")),
            source_end_us=int(getattr(run[-1], "source_end_us")),
            target_start_us=0,
            target_end_us=0,
            word_ids=[str(getattr(word, "word_id")) for word in run],
            text="".join(str(getattr(word, "text", "") or "") for word in run),
            decision_ids=sorted(set([*segment.decision_ids, "visual_pacing_hidden_repeat_cleanup"])),
            spoken_source_start_us=None,
            spoken_source_end_us=None,
            clip_source_start_us=None,
            clip_source_end_us=None,
            lead_handle_us=0,
            tail_handle_us=0,
            debug_hints=dict(segment.debug_hints)
            | {
                "visual_pacing_hidden_repeat_dropped_word_ids": [
                    word_id for word_id in segment.word_ids if word_id in dropped_word_ids
                ],
            },
        )
        for run in kept_runs
    ]
    return cleaned_segments, [word_id for word_id in segment.word_ids if word_id in dropped_word_ids]


def _repeated_suffix_island_start(tokens: list[str]) -> int | None:
    max_n = min(6, len(tokens) // 2)
    for n in range(max_n, 1, -1):
        suffix_start = len(tokens) - n
        suffix = tokens[suffix_start:]
        if not all(suffix):
            continue
        for start in range(0, suffix_start - n + 1):
            if tokens[start : start + n] == suffix:
                if is_coordinated_parallel_suffix_repetition(tokens, start, suffix_start, n):
                    continue
                return suffix_start
    if len(tokens) >= 3:
        suffix = tokens[-1]
        if suffix and len(suffix) >= 2:
            for start, token in enumerate(tokens[:-1]):
                if token == suffix and start + 1 < len(tokens) - 1:
                    if is_coordinated_parallel_suffix_repetition(tokens, start, len(tokens) - 1, 1):
                        continue
                    return len(tokens) - 1
    no_repeated_suffix_island = None
    return no_repeated_suffix_island


def _drop_boundary_suffix_prefix_overlaps(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> tuple[list[FinalTimelineSegment], int]:
    word_lookup = {word.word_id: word for word in source_graph.words}
    current = list(segments)
    dropped_count = 0
    while True:
        changed = False
        for index, (left, right) in enumerate(zip(current, current[1:])):
            overlap = _boundary_suffix_prefix_overlap(left.text, right.text)
            if len(overlap) < 2:
                continue
            if is_semantic_label_reuse_boundary(left.text, right.text, overlap):
                continue
            drop_ids = _trailing_word_ids_for_overlap(left, word_lookup, overlap)
            if not drop_ids or len(drop_ids) >= len(left.word_ids):
                continue
            current[index] = _drop_trailing_word_ids(left, word_lookup, drop_ids)
            dropped_count += len(drop_ids)
            changed = True
            break
        if not changed:
            return current, dropped_count
        current = _repack(current)


def _boundary_suffix_prefix_overlap(left_text: str, right_text: str) -> str:
    return boundary_suffix_prefix_overlap(left_text, right_text, max_size=20)


def _trailing_word_ids_for_overlap(
    segment: FinalTimelineSegment,
    word_lookup: dict[str, Any],
    overlap: str,
) -> list[str]:
    return trailing_word_ids_for_suffix_overlap(segment=segment, word_lookup=word_lookup, overlap=overlap)


def _drop_trailing_word_ids(
    segment: FinalTimelineSegment,
    word_lookup: dict[str, Any],
    drop_ids: list[str],
) -> FinalTimelineSegment:
    drop_set = set(drop_ids)
    kept_words = [word_lookup[word_id] for word_id in segment.word_ids if word_id in word_lookup and word_id not in drop_set]
    if not kept_words:
        return segment
    return replace(
        segment,
        source_start_us=int(getattr(kept_words[0], "source_start_us")),
        source_end_us=int(getattr(kept_words[-1], "source_end_us")),
        target_start_us=0,
        target_end_us=0,
        word_ids=[str(getattr(word, "word_id")) for word in kept_words],
        text="".join(str(getattr(word, "text", "") or "") for word in kept_words),
        decision_ids=sorted(set([*segment.decision_ids, "visual_pacing_boundary_overlap_cleanup"])),
        spoken_source_start_us=None,
        spoken_source_end_us=None,
        clip_source_start_us=None,
        clip_source_end_us=None,
        lead_handle_us=0,
        tail_handle_us=0,
        debug_hints=dict(segment.debug_hints)
        | {
            "visual_pacing_boundary_overlap_dropped_word_ids": list(drop_ids),
        },
    )

