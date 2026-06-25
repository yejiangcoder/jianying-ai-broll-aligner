from __future__ import annotations

from dataclasses import replace
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, FinalTimelineSegment

NORMAL_INTRA_SEGMENT_BREATH_GAP_US = 150_000
DENSE_INTRA_SEGMENT_GAP_US = 150_000
DENSE_INTRA_SEGMENT_GAP_MIN_COUNT = 3
DENSE_INTRA_SEGMENT_GAP_MIN_TOTAL_US = 450_000
LARGE_INTRA_SEGMENT_GAP_US = 200_000
VERY_LARGE_INTRA_SEGMENT_GAP_US = 900_000
MIN_SPLIT_SIDE_DURATION_US = 500_000
CUT_DENSITY_PROTECTION_WINDOW_US = 5_000_000
CUT_DENSITY_PROTECTION_MAX_CUTS = 5
CUT_DENSITY_PROTECTION_RELEASE_GAP_US = 300_000
DROPPABLE_BOUNDARY_FILLERS = {"啊", "呃", "嗯", "呐", "呢", "嘛", "吧", "咳"}
DROPPABLE_REPEATED_BOUNDARY_PRONOUNS = {"我", "你", "他", "她", "它", "这", "那"}


def split_large_intra_segment_gaps(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    windows: list[tuple[str, int, int]],
) -> tuple[list[FinalTimelineSegment], dict[str, Any]]:
    word_lookup = {word.word_id: word for word in source_graph.words}
    split_segments: list[FinalTimelineSegment] = []
    candidates: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    max_gap_us = 0
    unsafe_count = 0
    cut_density_protected_segment_ids = _cut_density_protected_segment_ids(segments)

    for segment in segments:
        pieces, segment_candidates, segment_splits = _split_segment_large_gaps(
            segment,
            word_lookup,
            windows,
            cut_density_protected=segment.segment_id in cut_density_protected_segment_ids,
        )
        split_segments.extend(pieces)
        candidates.extend(segment_candidates)
        split_rows.extend(segment_splits)
        max_gap_us = max(max_gap_us, *[int(row.get("gap_us") or 0) for row in segment_candidates], 0)
        unsafe_count += sum(1 for row in segment_candidates if not bool(row.get("applied")))

    return split_segments, {
        "large_intra_segment_gap_candidate_count": len(candidates),
        "large_intra_segment_gap_split_count": len(split_rows),
        "large_intra_segment_gap_unsafe_count": unsafe_count,
        "large_intra_segment_gap_max_us": max_gap_us,
        "large_intra_segment_gap_normal_breath_us": NORMAL_INTRA_SEGMENT_BREATH_GAP_US,
        "dense_intra_segment_gap_threshold_us": DENSE_INTRA_SEGMENT_GAP_US,
        "dense_intra_segment_gap_min_count": DENSE_INTRA_SEGMENT_GAP_MIN_COUNT,
        "dense_intra_segment_gap_min_total_us": DENSE_INTRA_SEGMENT_GAP_MIN_TOTAL_US,
        "large_intra_segment_gap_threshold_us": LARGE_INTRA_SEGMENT_GAP_US,
        "large_intra_segment_gap_min_split_side_duration_us": MIN_SPLIT_SIDE_DURATION_US,
        "large_intra_segment_gap_candidates": candidates,
        "large_intra_segment_gap_splits": split_rows,
    }


def empty_large_intra_segment_gap_report() -> dict[str, Any]:
    return {
        "large_intra_segment_gap_candidate_count": 0,
        "large_intra_segment_gap_split_count": 0,
        "large_intra_segment_gap_unsafe_count": 0,
        "large_intra_segment_gap_max_us": 0,
        "large_intra_segment_gap_normal_breath_us": NORMAL_INTRA_SEGMENT_BREATH_GAP_US,
        "dense_intra_segment_gap_threshold_us": DENSE_INTRA_SEGMENT_GAP_US,
        "dense_intra_segment_gap_min_count": DENSE_INTRA_SEGMENT_GAP_MIN_COUNT,
        "dense_intra_segment_gap_min_total_us": DENSE_INTRA_SEGMENT_GAP_MIN_TOTAL_US,
        "large_intra_segment_gap_threshold_us": LARGE_INTRA_SEGMENT_GAP_US,
        "large_intra_segment_gap_min_split_side_duration_us": MIN_SPLIT_SIDE_DURATION_US,
        "large_intra_segment_gap_candidates": [],
        "large_intra_segment_gap_splits": [],
    }


def _split_segment_large_gaps(
    segment: FinalTimelineSegment,
    word_lookup: dict[str, Any],
    windows: list[tuple[str, int, int]],
    *,
    cut_density_protected: bool = False,
) -> tuple[list[FinalTimelineSegment], list[dict[str, Any]], list[dict[str, Any]]]:
    words = [word_lookup[word_id] for word_id in segment.word_ids if word_id in word_lookup]
    if len(words) < 2:
        return [segment], [], []
    candidates: list[dict[str, Any]] = []
    split_after_indexes: set[int] = set()
    force_drop_word_ids: set[str] = set()
    density_gap_rows: list[tuple[int, Any, Any, int]] = []
    candidate_by_index: dict[int, dict[str, Any]] = {}
    for index, (left_word, right_word) in enumerate(zip(words, words[1:])):
        gap_us = int(getattr(right_word, "source_start_us", 0) or 0) - int(getattr(left_word, "source_end_us", 0) or 0)
        if gap_us > DENSE_INTRA_SEGMENT_GAP_US:
            density_gap_rows.append((index, left_word, right_word, gap_us))
        if gap_us <= NORMAL_INTRA_SEGMENT_BREATH_GAP_US:
            continue
        candidate = _candidate_row(segment, left_word, right_word, gap_us)
        candidate_by_index[index] = candidate
        if _head_false_start_gap_should_drop(words, index, gap_us):
            candidate["applied"] = True
            candidate["reason"] = "head_false_start_gap_drop"
            candidates.append(candidate)
            split_after_indexes.add(index)
            force_drop_word_ids.add(str(getattr(left_word, "word_id", "") or ""))
            continue
        if _tail_single_pronoun_gap_should_drop(words, index, gap_us):
            candidate["applied"] = True
            candidate["reason"] = "tail_single_pronoun_gap_drop"
            candidates.append(candidate)
            split_after_indexes.add(index)
            force_drop_word_ids.add(str(getattr(right_word, "word_id", "") or ""))
            continue
        if gap_us < LARGE_INTRA_SEGMENT_GAP_US:
            candidate["applied"] = False
            candidate["reason"] = "below_split_threshold"
            candidates.append(candidate)
            continue
        if cut_density_protected and gap_us < CUT_DENSITY_PROTECTION_RELEASE_GAP_US:
            candidate["applied"] = False
            candidate["reason"] = "cut_density_window_protected"
            candidates.append(candidate)
            continue
        safety_reason = _split_safety_reason(segment, words, index, windows)
        if safety_reason and not _very_large_gap_can_force_split(words, index, gap_us, safety_reason):
            candidate["applied"] = False
            candidate["reason"] = safety_reason
            candidates.append(candidate)
            continue
        candidate["applied"] = True
        candidate["reason"] = "large_intra_segment_gap_split" if not safety_reason else "very_large_intra_segment_gap_split"
        candidates.append(candidate)
        split_after_indexes.add(index)
    if _dense_gap_cluster_should_split(density_gap_rows):
        existing_indexes = set(candidate_by_index)
        for index, left_word, right_word, gap_us in density_gap_rows:
            if index in split_after_indexes:
                continue
            candidate = candidate_by_index.get(index) or _candidate_row(segment, left_word, right_word, gap_us)
            if _head_false_start_gap_should_drop(words, index, gap_us):
                candidate["applied"] = True
                candidate["reason"] = "head_false_start_gap_drop"
                split_after_indexes.add(index)
                force_drop_word_ids.add(str(getattr(left_word, "word_id", "") or ""))
            elif _tail_single_pronoun_gap_should_drop(words, index, gap_us):
                candidate["applied"] = True
                candidate["reason"] = "tail_single_pronoun_gap_drop"
                split_after_indexes.add(index)
                force_drop_word_ids.add(str(getattr(right_word, "word_id", "") or ""))
            else:
                if cut_density_protected and gap_us < CUT_DENSITY_PROTECTION_RELEASE_GAP_US:
                    candidate["applied"] = False
                    candidate["reason"] = "cut_density_window_protected"
                    if index not in existing_indexes:
                        candidates.append(candidate)
                    continue
                safety_reason = _split_safety_reason(segment, words, index, windows)
                if safety_reason:
                    candidate["applied"] = False
                    candidate["reason"] = safety_reason
                else:
                    candidate["applied"] = True
                    candidate["reason"] = "dense_intra_segment_gap_split"
                    split_after_indexes.add(index)
            if index not in existing_indexes:
                candidates.append(candidate)
    if not split_after_indexes:
        return [segment], candidates, []
    split_after_indexes = _prune_split_indexes_that_create_short_runs(
        words,
        split_after_indexes,
        candidate_by_index,
        force_drop_word_ids,
    )
    if not split_after_indexes:
        return [segment], candidates, []

    runs: list[list[Any]] = []
    current_run: list[Any] = []
    for index, word in enumerate(words):
        current_run.append(word)
        if index in split_after_indexes:
            runs.append(current_run)
            current_run = []
    if current_run:
        runs.append(current_run)
    kept_runs, dropped_boundary_filler_runs = _drop_boundary_filler_runs(runs, force_drop_word_ids)
    if not kept_runs:
        return [segment], candidates, []
    split_rows = [
        {
            "original_segment_id": segment.segment_id,
            "split_segment_count": len(kept_runs),
            "removed_gap_count": len(split_after_indexes),
            "removed_gap_us": int(row.get("gap_us") or 0),
            "left_word_id": str(row.get("left_word_id") or ""),
            "right_word_id": str(row.get("right_word_id") or ""),
            "dropped_boundary_filler_word_ids": [
                str(getattr(word, "word_id", "") or "") for run in dropped_boundary_filler_runs for word in run
            ],
        }
        for row in candidates
        if row.get("applied")
    ]
    return [_segment_from_run(segment, run, dropped_boundary_filler_runs) for run in kept_runs], candidates, split_rows


def _head_false_start_gap_should_drop(words: list[Any], split_after_index: int, gap_us: int) -> bool:
    if split_after_index != 0 or gap_us <= NORMAL_INTRA_SEGMENT_BREATH_GAP_US:
        return False
    left_text = normalize_text(str(getattr(words[split_after_index], "text", "") or ""))
    if len(left_text) != 1 or left_text in DROPPABLE_BOUNDARY_FILLERS:
        return False
    right_text = normalize_text("".join(str(getattr(word, "text", "") or "") for word in words[split_after_index + 1 :]))
    return len(right_text) >= 2 and left_text in right_text[:4]


def _tail_single_pronoun_gap_should_drop(words: list[Any], split_after_index: int, gap_us: int) -> bool:
    if split_after_index != len(words) - 2 or gap_us <= NORMAL_INTRA_SEGMENT_BREATH_GAP_US:
        return False
    right_text = normalize_text(str(getattr(words[split_after_index + 1], "text", "") or ""))
    if right_text not in DROPPABLE_REPEATED_BOUNDARY_PRONOUNS:
        return False
    left_text = normalize_text("".join(str(getattr(word, "text", "") or "") for word in words[: split_after_index + 1]))
    return right_text in left_text[-8:]


def _very_large_gap_can_force_split(words: list[Any], split_after_index: int, gap_us: int, safety_reason: str) -> bool:
    if gap_us < VERY_LARGE_INTRA_SEGMENT_GAP_US:
        return False
    if safety_reason not in {"left_side_too_short", "right_side_too_short"}:
        return False
    left_run = words[: split_after_index + 1]
    right_run = words[split_after_index + 1 :]
    return not _single_char_side_would_survive(left_run) and not _single_char_side_would_survive(right_run)


def _split_safety_reason(
    segment: FinalTimelineSegment,
    words: list[Any],
    split_after_index: int,
    windows: list[tuple[str, int, int]],
) -> str:
    left_run = words[: split_after_index + 1]
    right_run = words[split_after_index + 1 :]
    if not left_run or not right_run:
        return "empty_split_side"
    if _single_char_side_would_survive(left_run):
        return "single_char_left_side_would_survive"
    if _single_char_side_would_survive(right_run):
        return "single_char_right_side_would_survive"
    if _run_duration(left_run) < MIN_SPLIT_SIDE_DURATION_US:
        return "left_side_too_short"
    if _run_duration(right_run) < MIN_SPLIT_SIDE_DURATION_US:
        return "right_side_too_short"
    left_window = _source_window_id_for_range(
        windows,
        int(getattr(left_run[0], "source_start_us", 0) or 0),
        int(getattr(left_run[-1], "source_end_us", 0) or 0),
    )
    right_window = _source_window_id_for_range(
        windows,
        int(getattr(right_run[0], "source_start_us", 0) or 0),
        int(getattr(right_run[-1], "source_end_us", 0) or 0),
    )
    segment_window = _source_window_id_for_range(windows, int(segment.source_start_us), int(segment.source_end_us))
    if not left_window or not right_window or not segment_window:
        return "source_window_unresolved"
    if left_window != right_window or left_window != segment_window:
        return "cross_source_window"
    return ""


def _segment_from_run(
    segment: FinalTimelineSegment,
    words: list[Any],
    dropped_boundary_filler_runs: list[list[Any]],
) -> FinalTimelineSegment:
    dropped_word_ids = [str(getattr(word, "word_id", "") or "") for run in dropped_boundary_filler_runs for word in run]
    return replace(
        segment,
        source_start_us=int(getattr(words[0], "source_start_us")),
        source_end_us=int(getattr(words[-1], "source_end_us")),
        target_start_us=0,
        target_end_us=0,
        word_ids=[str(getattr(word, "word_id")) for word in words],
        text="".join(str(getattr(word, "text", "") or "") for word in words),
        decision_ids=sorted(set([*segment.decision_ids, "visual_pacing_large_intra_segment_gap_split"])),
        spoken_source_start_us=None,
        spoken_source_end_us=None,
        clip_source_start_us=None,
        clip_source_end_us=None,
        lead_handle_us=0,
        tail_handle_us=0,
        debug_hints=dict(segment.debug_hints)
        | {
            "visual_pacing_large_intra_segment_gap_split": True,
            "visual_pacing_large_intra_segment_gap_dropped_boundary_filler_word_ids": dropped_word_ids,
        },
    )


def _candidate_row(segment: FinalTimelineSegment, left_word: Any, right_word: Any, gap_us: int) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "text": segment.text,
        "gap_us": max(0, int(gap_us)),
        "source_start_us": int(getattr(left_word, "source_end_us", 0) or 0),
        "source_end_us": int(getattr(right_word, "source_start_us", 0) or 0),
        "left_word_id": str(getattr(left_word, "word_id", "") or ""),
        "left_word_text": str(getattr(left_word, "text", "") or ""),
        "right_word_id": str(getattr(right_word, "word_id", "") or ""),
        "right_word_text": str(getattr(right_word, "text", "") or ""),
    }


def _cut_density_protected_segment_ids(segments: list[FinalTimelineSegment]) -> set[str]:
    ordered = sorted(segments, key=lambda row: (int(row.target_start_us), int(row.target_end_us), row.segment_id))
    cut_times = [int(segment.target_start_us) for segment in ordered[1:]]
    dense_windows: list[tuple[int, int]] = []
    for index, start in enumerate(cut_times):
        end = start + CUT_DENSITY_PROTECTION_WINDOW_US
        count = 0
        for value in cut_times[index:]:
            if value >= end:
                break
            count += 1
        if count >= CUT_DENSITY_PROTECTION_MAX_CUTS:
            dense_windows.append((start, end))
    if not dense_windows:
        no_protected_segments: set[str] = set()
        return no_protected_segments
    protected: set[str] = set()
    for segment in ordered:
        segment_start = int(segment.target_start_us)
        segment_end = int(segment.target_end_us)
        if any(segment_start < window_end and segment_end > window_start for window_start, window_end in dense_windows):
            protected.add(segment.segment_id)
    return protected


def _dense_gap_cluster_should_split(density_gap_rows: list[tuple[int, Any, Any, int]]) -> bool:
    if len(density_gap_rows) < DENSE_INTRA_SEGMENT_GAP_MIN_COUNT:
        return False
    total_gap_us = sum(max(0, int(row[3])) for row in density_gap_rows)
    return total_gap_us >= DENSE_INTRA_SEGMENT_GAP_MIN_TOTAL_US


def _prune_split_indexes_that_create_short_runs(
    words: list[Any],
    split_after_indexes: set[int],
    candidate_by_index: dict[int, dict[str, Any]],
    force_drop_word_ids: set[str],
) -> set[int]:
    kept = set(split_after_indexes)
    forced = {str(word_id) for word_id in force_drop_word_ids if str(word_id)}
    while True:
        runs = _runs_for_split_indexes(words, kept)
        short_run = _first_non_droppable_short_run(runs, forced)
        if short_run is None:
            return kept
        run_index, start_index, end_index = short_run
        adjacent = []
        if run_index > 0:
            adjacent.append(start_index - 1)
        if run_index + 1 < len(runs):
            adjacent.append(end_index)
        removable = [index for index in adjacent if index in kept and not _split_boundary_is_forced(words, index, forced)]
        if not removable:
            return kept
        remove_index = min(removable, key=lambda index: _split_prune_priority(index, candidate_by_index))
        kept.remove(remove_index)
        candidate = candidate_by_index.get(remove_index)
        if candidate is not None and candidate.get("applied"):
            candidate["applied"] = False
            candidate["reason"] = "short_run_pruned"


def _runs_for_split_indexes(words: list[Any], split_after_indexes: set[int]) -> list[tuple[int, int, list[Any]]]:
    runs: list[tuple[int, int, list[Any]]] = []
    start_index = 0
    current: list[Any] = []
    for index, word in enumerate(words):
        current.append(word)
        if index in split_after_indexes:
            runs.append((start_index, index, current))
            start_index = index + 1
            current = []
    if current:
        runs.append((start_index, len(words) - 1, current))
    return runs


def _first_non_droppable_short_run(
    runs: list[tuple[int, int, list[Any]]],
    forced_word_ids: set[str],
) -> tuple[int, int, int] | None:
    for run_index, (start_index, end_index, run_words) in enumerate(runs):
        word_ids = {str(getattr(word, "word_id", "") or "") for word in run_words}
        if word_ids and word_ids <= forced_word_ids:
            continue
        if _single_char_side_would_survive(run_words) or _run_duration(run_words) < MIN_SPLIT_SIDE_DURATION_US:
            return run_index, start_index, end_index
    no_short_run: tuple[int, int, int] | None = None
    return no_short_run


def _split_boundary_is_forced(words: list[Any], split_after_index: int, forced_word_ids: set[str]) -> bool:
    left_word_id = str(getattr(words[split_after_index], "word_id", "") or "")
    right_word_id = str(getattr(words[split_after_index + 1], "word_id", "") or "") if split_after_index + 1 < len(words) else ""
    return bool((left_word_id and left_word_id in forced_word_ids) or (right_word_id and right_word_id in forced_word_ids))


def _split_prune_priority(index: int, candidate_by_index: dict[int, dict[str, Any]]) -> tuple[int, int]:
    candidate = candidate_by_index.get(index) or {}
    reason = str(candidate.get("reason") or "")
    reason_priority = {
        "dense_intra_segment_gap_split": 0,
        "large_intra_segment_gap_split": 1,
        "very_large_intra_segment_gap_split": 2,
    }.get(reason, 0)
    return reason_priority, int(candidate.get("gap_us") or 0)


def _run_duration(words: list[Any]) -> int:
    return max(0, int(getattr(words[-1], "source_end_us", 0) or 0) - int(getattr(words[0], "source_start_us", 0) or 0))


def _drop_boundary_filler_runs(
    runs: list[list[Any]],
    force_drop_word_ids: set[str] | None = None,
) -> tuple[list[list[Any]], list[list[Any]]]:
    if len(runs) <= 1:
        return runs, []
    forced = {str(word_id) for word_id in (force_drop_word_ids or set()) if str(word_id)}
    kept: list[list[Any]] = []
    dropped: list[list[Any]] = []
    last_index = len(runs) - 1
    for index, run in enumerate(runs):
        run_word_ids = {str(getattr(word, "word_id", "") or "") for word in run}
        if run_word_ids and run_word_ids <= forced:
            dropped.append(run)
            continue
        if index in {0, last_index} and _is_droppable_boundary_filler_run(run):
            dropped.append(run)
            continue
        kept.append(run)
    return kept, dropped


def _single_char_side_would_survive(words: list[Any]) -> bool:
    text = normalize_text("".join(str(getattr(word, "text", "") or "") for word in words))
    return len(text) == 1 and text not in DROPPABLE_BOUNDARY_FILLERS


def _is_droppable_boundary_filler_run(words: list[Any]) -> bool:
    text = normalize_text("".join(str(getattr(word, "text", "") or "") for word in words))
    return text in DROPPABLE_BOUNDARY_FILLERS


def _source_window_id_for_range(windows: list[tuple[str, int, int]], start: int, end: int) -> str:
    for window_id, window_start, window_end in windows:
        if int(window_start) <= int(start) and int(end) <= int(window_end):
            return str(window_id)
    return ""
