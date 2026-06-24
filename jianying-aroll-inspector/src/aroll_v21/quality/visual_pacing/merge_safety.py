from __future__ import annotations

from bisect import bisect_left
from typing import Any

from aroll_v21.ir.models import CanonicalSourceGraph, FinalTimelineSegment

_WORD_RANGE_INDEX_CACHE: dict[int, tuple[tuple[int, int], list[tuple[int, int, int, Any]], list[int], int]] = {}
_WORD_RANGE_INDEX_CACHE_MAX = 16


def _child_segment_records(
    segment: FinalTimelineSegment,
    word_lookup: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw_records = list(segment.debug_hints.get("visual_pacing_child_segments") or [])
    if not raw_records:
        return [
            {
                "segment_id": segment.segment_id,
                "source_start_us": int(segment.source_start_us),
                "source_end_us": int(segment.source_end_us),
                "target_start_us": int(segment.target_start_us),
                "target_end_us": int(segment.target_end_us),
                "word_ids": list(segment.word_ids),
            }
        ]
    if word_lookup is None:
        return [dict(row) for row in raw_records if isinstance(row, dict)]
    kept_word_ids = set(segment.word_ids)
    records: list[dict[str, Any]] = []
    for row in raw_records:
        if not isinstance(row, dict):
            continue
        word_ids = [str(word_id) for word_id in list(row.get("word_ids") or []) if str(word_id) in kept_word_ids]
        if not word_ids:
            continue
        words = [word_lookup[word_id] for word_id in word_ids if word_id in word_lookup]
        if not words:
            continue
        records.append(
            {
                "segment_id": str(row.get("segment_id") or ""),
                "source_start_us": int(words[0].source_start_us),
                "source_end_us": int(words[-1].source_end_us),
                "target_start_us": int(row.get("target_start_us") or segment.target_start_us),
                "target_end_us": int(row.get("target_end_us") or segment.target_end_us),
                "word_ids": word_ids,
            }
        )
    if records:
        return records
    base_records = [
        {
            "segment_id": segment.segment_id,
            "source_start_us": int(segment.source_start_us),
            "source_end_us": int(segment.source_end_us),
            "target_start_us": int(segment.target_start_us),
            "target_end_us": int(segment.target_end_us),
            "word_ids": list(segment.word_ids),
        }
    ]
    return base_records


def _words_overlapping_range(
    source_graph: CanonicalSourceGraph,
    start_us: int,
    end_us: int,
    child_word_ids: set[str],
) -> list[Any]:
    if end_us <= start_us:
        no_words: list[Any] = []
        return no_words
    indexed_words, starts, max_duration_us = _indexed_words_by_source_start(source_graph)
    lower_bound = bisect_left(starts, int(start_us) - max_duration_us)
    upper_bound = bisect_left(starts, int(end_us))
    return [
        row[3]
        for row in indexed_words[lower_bound:upper_bound]
        if str(getattr(row[3], "word_id", "") or "") not in child_word_ids
        and row[2] > int(start_us)
        and row[1] < int(end_us)
    ]


def _indexed_words_by_source_start(source_graph: CanonicalSourceGraph) -> tuple[list[tuple[int, int, int, Any]], list[int], int]:
    words = list(source_graph.words)
    signature = (id(source_graph.words), len(words))
    cache_key = id(source_graph)
    cached = _WORD_RANGE_INDEX_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1], cached[2], cached[3]
    rows = sorted(
        (
            (
                index,
                int(getattr(word, "source_start_us", 0) or 0),
                int(getattr(word, "source_end_us", 0) or 0),
                word,
            )
            for index, word in enumerate(words)
        ),
        key=lambda row: (row[1], row[2], row[0]),
    )
    starts = [row[1] for row in rows]
    max_duration_us = max([max(0, row[2] - row[1]) for row in rows] or [0])
    _WORD_RANGE_INDEX_CACHE[cache_key] = (signature, rows, starts, max_duration_us)
    if len(_WORD_RANGE_INDEX_CACHE) > _WORD_RANGE_INDEX_CACHE_MAX:
        oldest_key = next(iter(_WORD_RANGE_INDEX_CACHE))
        _WORD_RANGE_INDEX_CACHE.pop(oldest_key, None)
    return rows, starts, max_duration_us


def _dropped_segment_ids_for_words(words: list[Any]) -> list[str]:
    ids: set[str] = set()
    for word in words:
        subtitle_index = getattr(word, "subtitle_index", None)
        if subtitle_index is not None:
            ids.add(f"subtitle_{int(subtitle_index):06d}")
            continue
        subtitle_uid = str(getattr(word, "subtitle_uid", "") or "")
        if subtitle_uid:
            ids.add(f"subtitle_{subtitle_uid}")
    return sorted(ids)


def _dropped_cluster_ids_for_words(words: list[Any]) -> list[str]:
    cluster_ids: set[str] = set()
    for word in words:
        hints = getattr(word, "debug_hints", {}) or {}
        if not isinstance(hints, dict):
            continue
        for key in ("repeat_cluster_id", "cluster_id", "final_repeat_cluster_id"):
            value = str(hints.get(key) or "")
            if value:
                cluster_ids.add(value)
    return sorted(cluster_ids)

