from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.contracts import VisualMergeGroup, VisualMergeSafetyReport, VisualPacingReport, contract_to_dict
from aroll_v21.ir.models import CaptionRenderUnit, CanonicalSourceGraph, FinalTimelineSegment
from aroll_v21.quality.tiny_segment_classifier import classify_tiny_segment
from aroll_v21.quality.visual_pacing.cut_density import (
    CUT_DENSITY_WINDOW_US,
    MAX_CUTS_IN_5S,
    _cut_density_blockers,
    _cut_density_report,
    _cut_density_report_from_merge_report,
)
from aroll_v21.quality.visual_pacing.suffix_cleanup import (
    _drop_boundary_suffix_prefix_overlaps,
    _drop_repeated_suffix_islands_by_subtitle,
)
from aroll_v21.quality.visual_pacing.intra_segment_gap import split_large_intra_segment_gaps
from aroll_v21.quality.visual_pacing.timeline_utils import _repack

from aroll_v21.quality.visual_pacing import report as visual_pacing_report_helpers
from aroll_v21.quality.visual_pacing.merge_safety import (
    _child_segment_records,
    _dropped_cluster_ids_for_words,
    _dropped_segment_ids_for_words,
    _words_overlapping_range,
)
from aroll_v21.quality.visual_pacing.report import _percentile, _reason_counts
from aroll_v21.quality.visual_pacing.short_segment_padding import (
    _window_lower_for_segment,
    _window_upper_for_segment,
)

MIN_HARD_SEGMENT_DURATION_US = 300_000
VISUAL_MIN_SEGMENT_DURATION_US = 1_200_000
MAX_SAFE_BRIDGE_GAP_US = 220_000
MAX_UNSPOKEN_BRIDGE_RATIO = 0.20
MAX_SEMANTIC_BRIDGE_SHORT_SEGMENTS = 3
MIN_SEMANTIC_BRIDGE_EXCEPTION_US = 600_000
SEMANTIC_BRIDGE_SHORT_SEGMENT_CAP = 8
SEMANTIC_BRIDGE_SHORT_SEGMENT_RATIO_NUMERATOR = 15
SEMANTIC_BRIDGE_SHORT_SEGMENT_RATIO_DENOMINATOR = 100


class VisualPacingNormalizer:
    def normalize(
        self,
        final_timeline: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
    ) -> tuple[list[FinalTimelineSegment], dict[str, Any]]:
        before_short = _short_count(final_timeline)
        current = list(final_timeline)
        attempted = 0
        merged = 0
        dropped_weak_filler = 0
        merged_weak_filler_micro = 0
        dropped_weak_filler_micro = 0
        unsafe_attempts = 0
        windows = _source_windows(source_graph)
        word_lookup = {word.word_id: word for word in source_graph.words}
        current, micro_report = self._cleanup_weak_filler_micro_segments(current, source_graph, windows, word_lookup)
        if micro_report["merged_weak_filler_micro_segment_count"]:
            attempted += int(micro_report["merged_weak_filler_micro_segment_count"])
            merged += int(micro_report["merged_weak_filler_micro_segment_count"])
            merged_weak_filler_micro += int(micro_report["merged_weak_filler_micro_segment_count"])
        if micro_report["dropped_weak_filler_micro_segment_count"]:
            dropped_weak_filler += int(micro_report["dropped_weak_filler_micro_segment_count"])
            dropped_weak_filler_micro += int(micro_report["dropped_weak_filler_micro_segment_count"])
        unsafe_attempts += int(micro_report["unsafe_micro_merge_attempt_count"])
        while True:
            merge_index, unsafe_count = self._next_merge_index(
                current,
                source_graph,
                windows,
                word_lookup,
            )
            unsafe_attempts += unsafe_count
            if merge_index is None:
                break
            attempted += 1
            current = self._merge_at(current, merge_index)
            merged += 1
        while True:
            drop_index, unsafe_count = self._next_weak_filler_drop_index(current, source_graph, windows, word_lookup)
            unsafe_attempts += unsafe_count
            if drop_index is None:
                break
            current = [*current[:drop_index], *current[drop_index + 1 :]]
            dropped_weak_filler += 1
        current, padded_short_count = _pad_residual_complete_short_segments(current, source_graph)
        current = _repack(current)
        current, hidden_repeat_dropped_word_count, hidden_repeat_split_segment_count = _drop_repeated_suffix_islands_by_subtitle(
            current,
            source_graph,
        )
        current, boundary_overlap_dropped_word_count = _drop_boundary_suffix_prefix_overlaps(current, source_graph)
        current, post_cleanup_padded_short_count = _pad_residual_complete_short_segments(current, source_graph)
        padded_short_count += post_cleanup_padded_short_count
        current, post_micro_report = self._cleanup_weak_filler_micro_segments(current, source_graph, windows, word_lookup)
        if post_micro_report["merged_weak_filler_micro_segment_count"]:
            attempted += int(post_micro_report["merged_weak_filler_micro_segment_count"])
            merged += int(post_micro_report["merged_weak_filler_micro_segment_count"])
            merged_weak_filler_micro += int(post_micro_report["merged_weak_filler_micro_segment_count"])
        if post_micro_report["dropped_weak_filler_micro_segment_count"]:
            dropped_weak_filler += int(post_micro_report["dropped_weak_filler_micro_segment_count"])
            dropped_weak_filler_micro += int(post_micro_report["dropped_weak_filler_micro_segment_count"])
        unsafe_attempts += int(post_micro_report["unsafe_micro_merge_attempt_count"])
        post_cleanup_short_merged = 0
        if (
            hidden_repeat_dropped_word_count
            or boundary_overlap_dropped_word_count
            or post_cleanup_padded_short_count
            or post_micro_report["merged_weak_filler_micro_segment_count"]
            or post_micro_report["dropped_weak_filler_micro_segment_count"]
        ):
            current = _repack(current)
            while True:
                merge_index, unsafe_count = self._next_post_cleanup_micro_merge_index(
                    current,
                    source_graph,
                    windows,
                    word_lookup,
                )
                unsafe_attempts += unsafe_count
                if merge_index is None:
                    break
                attempted += 1
                current = self._merge_at(current, merge_index)
                merged += 1
                post_cleanup_short_merged += 1
            if post_cleanup_short_merged:
                current = _repack(current)
        current, large_intra_segment_gap_report = split_large_intra_segment_gaps(current, source_graph, windows)
        if large_intra_segment_gap_report["large_intra_segment_gap_split_count"]:
            current = _repack(current)
            current, cut_density_relief_merged_count, cut_density_relief_unsafe_count = self._relieve_cut_density_bursts(
                current,
                source_graph,
                windows,
                word_lookup,
            )
            if cut_density_relief_merged_count:
                attempted += cut_density_relief_merged_count
                merged += cut_density_relief_merged_count
                current = _repack(current)
            unsafe_attempts += cut_density_relief_unsafe_count
        current, semantic_bridge_safe_merged_count, semantic_bridge_safe_unsafe_count = self._merge_safe_semantic_bridge_exceptions(
            current,
            source_graph,
            windows,
            word_lookup,
        )
        if semantic_bridge_safe_merged_count:
            attempted += semantic_bridge_safe_merged_count
            merged += semantic_bridge_safe_merged_count
            current = _repack(current)
        unsafe_attempts += semantic_bridge_safe_unsafe_count
        safety_report = _build_visual_merge_safety_report(
            current,
            source_graph,
            unsafe_merge_attempt_count=unsafe_attempts,
        )
        after_short = _short_count(current)
        semantic_bridge_count = sum(
            1
            for segment in current
            if _is_allowed_semantic_bridge_exception(segment) and not _is_visual_gap_split_bridge(segment)
        )
        semantic_bridge_safe_merge_candidates = _semantic_bridge_safe_merge_candidates(current, source_graph)
        cut_density_report = _cut_density_report(current)
        blocking_short_count = _blocking_short_count(current)
        allowed_blocking_short_count = _allowed_blocking_short_count(current)
        blocker_codes = list(safety_report.get("blocker_codes") or [])
        if _short_segments_exceed_gate_limit(
            blocking_short_count=blocking_short_count,
            allowed_blocking_short_count=allowed_blocking_short_count,
            unsafe_merge_attempt_count=unsafe_attempts,
        ):
            blocker_codes.append("V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN")
        semantic_bridge_cap = _semantic_bridge_segment_cap(len(current))
        blocker_codes.extend(
            _semantic_bridge_gate_blockers(
                semantic_bridge_count=semantic_bridge_count,
                semantic_bridge_cap=semantic_bridge_cap,
                safe_merge_candidate_count=len(semantic_bridge_safe_merge_candidates),
            )
        )
        blocker_codes.extend(_cut_density_blockers(cut_density_report))
        report = _visual_report(
            final_timeline=current,
            captions=[],
            gate_passed=not blocker_codes,
            executed=True,
            attempted=attempted,
            merged=merged,
            before_short=before_short,
            after_short=after_short,
            semantic_bridge_count=semantic_bridge_count,
            blocker_codes=blocker_codes,
            hidden_repeat_dropped_word_count=hidden_repeat_dropped_word_count,
            hidden_repeat_split_segment_count=hidden_repeat_split_segment_count,
            boundary_overlap_dropped_word_count=boundary_overlap_dropped_word_count,
            safety_report=safety_report,
            blocking_short_count=blocking_short_count,
            allowed_blocking_short_count=allowed_blocking_short_count,
            dropped_weak_filler_count=dropped_weak_filler,
            merged_weak_filler_micro_segment_count=merged_weak_filler_micro,
            dropped_weak_filler_micro_segment_count=dropped_weak_filler_micro,
            padded_short_count=padded_short_count,
            semantic_bridge_cap=semantic_bridge_cap,
            semantic_bridge_safe_merge_candidates=semantic_bridge_safe_merge_candidates,
            cut_density_report=cut_density_report,
            large_intra_segment_gap_report=large_intra_segment_gap_report,
        )
        return current, report

    def _cleanup_weak_filler_micro_segments(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        windows: list[tuple[str, int, int]],
        word_lookup: dict[str, Any],
    ) -> tuple[list[FinalTimelineSegment], dict[str, int]]:
        current = list(segments)
        merged_count = 0
        dropped_count = 0
        unsafe_count = 0
        while True:
            action = _next_weak_filler_micro_action(current, source_graph, windows, word_lookup)
            unsafe_count += int(action.get("unsafe_count") or 0)
            action_type = str(action.get("action") or "")
            action_index = action.get("index")
            if action_type == "merge" and isinstance(action_index, int):
                current = _repack(self._merge_at(current, action_index))
                merged_count += 1
                continue
            if action_type == "drop" and isinstance(action_index, int):
                current = _repack([*current[:action_index], *current[action_index + 1 :]])
                dropped_count += 1
                continue
            return current, {
                "merged_weak_filler_micro_segment_count": merged_count,
                "dropped_weak_filler_micro_segment_count": dropped_count,
                "unsafe_micro_merge_attempt_count": unsafe_count,
            }

    def _next_merge_index(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        windows: list[tuple[str, int, int]],
        word_lookup: dict[str, Any],
    ) -> tuple[int | None, int]:
        unsafe_count = 0
        for index, segment in enumerate(segments):
            duration = _duration(segment)
            classification = classify_tiny_segment(segment)
            if duration >= VISUAL_MIN_SEGMENT_DURATION_US:
                continue
            candidates = []
            if index > 0:
                candidates.append(index - 1)
            if index + 1 < len(segments):
                candidates.append(index)
            for merge_index in sorted(candidates, key=lambda value: _duration(segments[value]) + _duration(segments[value + 1])):
                left_candidate = segments[merge_index]
                right_candidate = segments[merge_index + 1]
                group = _build_merge_group(
                    left_candidate,
                    right_candidate,
                    source_graph,
                    windows,
                    word_lookup,
                    video_segment_id=segments[merge_index].segment_id,
                )
                if (
                    len(segments) < 10
                    and not _can_merge_across_subtitle_boundary(left_candidate, right_candidate, word_lookup)
                    and not _merge_group_has_redundant_restart_gap(group)
                ):
                    continue
                if group.merge_safe:
                    return merge_index, unsafe_count
                unsafe_reasons = set(group.unsafe_reasons)
                same_window_large_gap = (
                    "bridge_gap_exceeds_threshold" in unsafe_reasons
                    and group.source_window_id not in {"", "source_window_unbounded"}
                )
                boundary_cleanup_pair = _boundary_cleanup_pair(segments[merge_index], segments[merge_index + 1])
                if (
                    (group.dropped_word_ids_crossed and not boundary_cleanup_pair)
                    or (group.dropped_segment_ids_crossed and not boundary_cleanup_pair)
                    or (group.dropped_repeat_cluster_ids_crossed and not boundary_cleanup_pair)
                    or (same_window_large_gap and not boundary_cleanup_pair)
                ):
                    unsafe_count += 1
        no_merge_index = None
        return no_merge_index, unsafe_count

    def _next_post_cleanup_micro_merge_index(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        windows: list[tuple[str, int, int]],
        word_lookup: dict[str, Any],
    ) -> tuple[int | None, int]:
        unsafe_count = 0
        for index, segment in enumerate(segments):
            duration = _duration(segment)
            if duration <= 0 or duration >= MIN_HARD_SEGMENT_DURATION_US:
                continue
            candidates = []
            if index > 0:
                candidates.append(index - 1)
            if index + 1 < len(segments):
                candidates.append(index)
            for merge_index in sorted(candidates, key=lambda value: _duration(segments[value]) + _duration(segments[value + 1])):
                group = _build_merge_group(
                    segments[merge_index],
                    segments[merge_index + 1],
                    source_graph,
                    windows,
                    word_lookup,
                    video_segment_id=segments[merge_index].segment_id,
                )
                if group.merge_safe:
                    return merge_index, unsafe_count
                if _unsafe_micro_merge_attempt(group):
                    unsafe_count += 1
        no_merge_index = None
        return no_merge_index, unsafe_count

    def _relieve_cut_density_bursts(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        windows: list[tuple[str, int, int]],
        word_lookup: dict[str, Any],
    ) -> tuple[list[FinalTimelineSegment], int, int]:
        current = list(segments)
        merged = 0
        unsafe_count = 0
        while True:
            if not _cut_density_blockers(_cut_density_report(current)):
                return current, merged, unsafe_count
            merge_index, iteration_unsafe_count = self._next_cut_density_relief_merge_index(
                current,
                source_graph,
                windows,
                word_lookup,
            )
            unsafe_count += iteration_unsafe_count
            if merge_index is None:
                return current, merged, unsafe_count
            current = _repack(self._merge_at(current, merge_index))
            merged += 1

    def _next_cut_density_relief_merge_index(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        windows: list[tuple[str, int, int]],
        word_lookup: dict[str, Any],
    ) -> tuple[int | None, int]:
        window = _first_cut_density_burst_window(segments)
        if window is None:
            no_merge_index = None
            return no_merge_index, 0
        window_start, window_end = window
        cut_indexes = [
            index
            for index in range(1, len(segments))
            if window_start <= int(segments[index].target_start_us) < window_end
        ]
        candidate_indexes = sorted(
            {index - 1 for index in cut_indexes if 0 <= index - 1 < len(segments) - 1},
            key=lambda index: _cut_density_relief_merge_priority(segments[index], segments[index + 1], index),
        )
        unsafe_count = 0
        for merge_index in candidate_indexes:
            left = segments[merge_index]
            right = segments[merge_index + 1]
            if not (_is_allowed_semantic_bridge_exception(left) or _is_allowed_semantic_bridge_exception(right)):
                continue
            group = _build_merge_group(
                left,
                right,
                source_graph,
                windows,
                word_lookup,
                video_segment_id=left.segment_id,
            )
            if group.merge_safe:
                return merge_index, unsafe_count
            if _unsafe_micro_merge_attempt(group):
                unsafe_count += 1
        no_merge_index = None
        return no_merge_index, unsafe_count

    def _merge_safe_semantic_bridge_exceptions(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        windows: list[tuple[str, int, int]],
        word_lookup: dict[str, Any],
    ) -> tuple[list[FinalTimelineSegment], int, int]:
        current = list(segments)
        merged = 0
        unsafe_count = 0
        while True:
            merge_index, iteration_unsafe_count = self._next_semantic_bridge_exception_merge_index(
                current,
                source_graph,
                windows,
                word_lookup,
            )
            unsafe_count += iteration_unsafe_count
            if merge_index is None:
                return current, merged, unsafe_count
            current = _repack(self._merge_at(current, merge_index))
            merged += 1

    def _next_semantic_bridge_exception_merge_index(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        windows: list[tuple[str, int, int]],
        word_lookup: dict[str, Any],
    ) -> tuple[int | None, int]:
        unsafe_count = 0
        for index, segment in enumerate(segments):
            if not _is_allowed_semantic_bridge_exception(segment):
                continue
            if _is_visual_gap_split_bridge(segment):
                continue
            candidates = []
            if index > 0:
                candidates.append(index - 1)
            if index + 1 < len(segments):
                candidates.append(index)
            for merge_index in sorted(candidates, key=lambda value: _duration(segments[value]) + _duration(segments[value + 1])):
                left = segments[merge_index]
                right = segments[merge_index + 1]
                if _is_visual_gap_split_bridge(left) or _is_visual_gap_split_bridge(right):
                    continue
                if len(segments) < 10 and not _can_merge_across_subtitle_boundary(left, right, word_lookup):
                    continue
                group = _build_merge_group(
                    left,
                    right,
                    source_graph,
                    windows,
                    word_lookup,
                    video_segment_id=left.segment_id,
                )
                if group.merge_safe:
                    return merge_index, unsafe_count
                if _unsafe_micro_merge_attempt(group):
                    unsafe_count += 1
        no_merge_index = None
        return no_merge_index, unsafe_count

    def _next_weak_filler_drop_index(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        windows: list[tuple[str, int, int]],
        word_lookup: dict[str, Any],
    ) -> tuple[int | None, int]:
        unsafe_count = 0
        if len(segments) <= 1:
            return None, unsafe_count
        for index, segment in enumerate(segments):
            classification = classify_tiny_segment(segment)
            if not classification.weak_filler or _duration(segment) >= MIN_HARD_SEGMENT_DURATION_US:
                continue
            candidates = []
            if index > 0:
                candidates.append(index - 1)
            if index + 1 < len(segments):
                candidates.append(index)
            for merge_index in candidates:
                group = _build_merge_group(
                    segments[merge_index],
                    segments[merge_index + 1],
                    source_graph,
                    windows,
                    word_lookup,
                    video_segment_id=segments[merge_index].segment_id,
                )
                if group.merge_safe:
                    return None, unsafe_count
                unsafe_reasons = set(group.unsafe_reasons)
                same_window_large_gap = (
                    "bridge_gap_exceeds_threshold" in unsafe_reasons
                    and group.source_window_id not in {"", "source_window_unbounded"}
                )
                if group.dropped_word_ids_crossed or group.dropped_segment_ids_crossed or group.dropped_repeat_cluster_ids_crossed or same_window_large_gap:
                    unsafe_count += 1
            return index, unsafe_count
        return None, unsafe_count

    def _can_merge(
        self,
        left: FinalTimelineSegment,
        right: FinalTimelineSegment,
        windows: list[tuple[str, int, int]],
    ) -> bool:
        return _same_source_window(left, right, windows)

    def _merge_at(self, segments: list[FinalTimelineSegment], index: int) -> list[FinalTimelineSegment]:
        left = segments[index]
        right = segments[index + 1]
        child_segments = _dedupe_child_segment_records([*_child_segment_records(left), *_child_segment_records(right)])
        merged = replace(
            left,
            source_start_us=min(left.source_start_us, right.source_start_us),
            source_end_us=max(left.source_end_us, right.source_end_us),
            target_start_us=0,
            target_end_us=0,
            word_ids=[*left.word_ids, *right.word_ids],
            text=f"{left.text}{right.text}",
            decision_ids=sorted(set([*left.decision_ids, *right.decision_ids, "visual_pacing_merge"])),
            spoken_source_start_us=None,
            spoken_source_end_us=None,
            clip_source_start_us=None,
            clip_source_end_us=None,
            lead_handle_us=0,
            tail_handle_us=0,
            debug_hints=dict(left.debug_hints)
            | {
                "visual_pacing_child_segments": child_segments,
                "visual_pacing_merged_segment_ids": [
                    *list(left.debug_hints.get("visual_pacing_merged_segment_ids") or [left.segment_id]),
                    *list(right.debug_hints.get("visual_pacing_merged_segment_ids") or [right.segment_id]),
                ],
            },
        )
        return [*segments[:index], merged, *segments[index + 2 :]]


def _dedupe_child_segment_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        word_ids = tuple(str(word_id) for word_id in list(row.get("word_ids") or []))
        key = (
            str(row.get("segment_id") or ""),
            word_ids,
            int(row.get("source_start_us") or 0),
            int(row.get("source_end_us") or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def _gap_is_redundant_restart_fragment(
    gap_words: list[Any],
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    word_lookup: dict[str, Any],
) -> bool:
    if not gap_words:
        return False
    gap_text = normalize_text("".join(str(getattr(word, "text", "") or "") for word in gap_words))
    if len(gap_text) < 2:
        return False
    left_text = normalize_text(_record_text(left_record, word_lookup))
    right_text = normalize_text(_record_text(right_record, word_lookup))
    if not left_text or not right_text:
        return False
    remaining = gap_text
    consumed_right_prefix = False
    consumed_left_suffix = False
    while remaining:
        prefix_len = _common_prefix_len(remaining, right_text)
        if prefix_len >= 2:
            remaining = remaining[prefix_len:]
            consumed_right_prefix = True
            continue
        suffix_len = _common_suffix_len(remaining, left_text)
        if suffix_len >= 2:
            remaining = remaining[: len(remaining) - suffix_len]
            consumed_left_suffix = True
            continue
        return False
    return consumed_right_prefix and consumed_left_suffix


def _record_text(record: dict[str, Any], word_lookup: dict[str, Any]) -> str:
    parts: list[str] = []
    for word_id in list(record.get("word_ids") or []):
        word = word_lookup.get(str(word_id))
        if word is None:
            continue
        parts.append(str(getattr(word, "text", "") or ""))
    return "".join(parts)


def _common_prefix_len(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def _common_suffix_len(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(reversed(left), reversed(right)):
        if left_char != right_char:
            break
        count += 1
    return count


def _build_merge_group(
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    windows: list[tuple[str, int, int]],
    word_lookup: dict[str, Any],
    *,
    video_segment_id: str,
) -> VisualMergeGroup:
    merged_records = _dedupe_child_segment_records([*_child_segment_records(left, word_lookup), *_child_segment_records(right, word_lookup)])
    return _build_group_from_records(
        video_segment_id=video_segment_id,
        records=merged_records,
        base_segment=left,
        source_graph=source_graph,
        windows=windows,
        word_lookup=word_lookup,
    )


def _build_segment_merge_group(
    segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    windows: list[tuple[str, int, int]],
    word_lookup: dict[str, Any],
) -> VisualMergeGroup:
    return _build_group_from_records(
        video_segment_id=segment.segment_id,
        records=_child_segment_records(segment, word_lookup),
        base_segment=segment,
        source_graph=source_graph,
        windows=windows,
        word_lookup=word_lookup,
    )


def _build_group_from_records(
    *,
    video_segment_id: str,
    records: list[dict[str, Any]],
    base_segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    windows: list[tuple[str, int, int]],
    word_lookup: dict[str, Any],
) -> VisualMergeGroup:
    records = _dedupe_child_segment_records(records)
    records = sorted(records, key=lambda row: (int(row.get("source_start_us") or 0), int(row.get("source_end_us") or 0), str(row.get("segment_id") or "")))
    child_ids = [str(row.get("segment_id") or "") for row in records if str(row.get("segment_id") or "")]
    source_start = min([int(row.get("source_start_us") or base_segment.source_start_us) for row in records] or [int(base_segment.source_start_us)])
    source_end = max([int(row.get("source_end_us") or base_segment.source_end_us) for row in records] or [int(base_segment.source_end_us)])
    target_start = int(base_segment.target_start_us)
    target_end = int(base_segment.target_end_us)
    source_window_id = _source_window_id_for_range(windows, source_start, source_end)
    child_word_ids = {str(word_id) for row in records for word_id in list(row.get("word_ids") or [])}
    bridged_gaps: list[dict[str, Any]] = []
    dropped_word_ids: set[str] = set()
    dropped_segment_ids: set[str] = set()
    dropped_cluster_ids: set[str] = set()
    hidden_spans: list[dict[str, Any]] = []
    unsafe_reasons: set[str] = set()
    for left, right in zip(records, records[1:]):
        gap_start = int(left.get("source_end_us") or 0)
        gap_end = int(right.get("source_start_us") or 0)
        gap_us = gap_end - gap_start
        gap_words = _words_overlapping_range(source_graph, gap_start, gap_end, child_word_ids)
        gap_word_ids = [word.word_id for word in gap_words]
        redundant_restart_gap = _gap_is_redundant_restart_fragment(gap_words, left, right, word_lookup)
        if gap_us < 0:
            unsafe_reasons.add("source_ranges_overlap")
        if gap_us > MAX_SAFE_BRIDGE_GAP_US and not redundant_restart_gap:
            unsafe_reasons.add("bridge_gap_exceeds_threshold")
        if gap_word_ids and not redundant_restart_gap:
            unsafe_reasons.add("dropped_words_crossed")
            dropped_word_ids.update(gap_word_ids)
            dropped_segment_ids.update(_dropped_segment_ids_for_words(gap_words))
            dropped_cluster_ids.update(_dropped_cluster_ids_for_words(gap_words))
            hidden_spans.append(
                {
                    "source_start_us": min(int(word.source_start_us) for word in gap_words),
                    "source_end_us": max(int(word.source_end_us) for word in gap_words),
                    "word_ids": gap_word_ids,
                }
            )
        bridged_gaps.append(
            {
                "left_child_segment_id": str(left.get("segment_id") or ""),
                "right_child_segment_id": str(right.get("segment_id") or ""),
                "source_start_us": gap_start,
                "source_end_us": gap_end,
                "duration_us": max(0, gap_us),
                "dropped_word_ids": gap_word_ids,
                "redundant_restart_gap": bool(redundant_restart_gap),
                "counts_for_unspoken_bridge": not bool(redundant_restart_gap),
            }
        )
    counted_gaps = [row for row in bridged_gaps if bool(row.get("counts_for_unspoken_bridge", True))]
    max_gap = max([int(row.get("duration_us") or 0) for row in counted_gaps] or [0])
    total_gap = sum(int(row.get("duration_us") or 0) for row in counted_gaps)
    source_duration = max(1, source_end - source_start)
    ratio = total_gap / source_duration
    if ratio > MAX_UNSPOKEN_BRIDGE_RATIO:
        unsafe_reasons.add("unspoken_bridge_ratio_exceeds_threshold")
    if source_window_id == "" and len(records) > 1:
        unsafe_reasons.add("source_window_unresolved")
    merge_safe = not unsafe_reasons
    return VisualMergeGroup(
        video_segment_id=video_segment_id,
        child_segment_ids=child_ids,
        child_caption_ids=[],
        source_window_id=source_window_id,
        target_start_us=target_start,
        target_end_us=target_end,
        source_start_us=source_start,
        source_end_us=source_end,
        child_spoken_source_ranges=[
            {
                "segment_id": str(row.get("segment_id") or ""),
                "start_us": int(row.get("source_start_us") or 0),
                "end_us": int(row.get("source_end_us") or 0),
            }
            for row in records
        ],
        child_spoken_target_ranges=[
            {
                "segment_id": str(row.get("segment_id") or ""),
                "start_us": int(row.get("target_start_us") or 0),
                "end_us": int(row.get("target_end_us") or 0),
            }
            for row in records
        ],
        bridged_gaps=bridged_gaps,
        dropped_segment_ids_crossed=sorted(dropped_segment_ids),
        dropped_word_ids_crossed=sorted(dropped_word_ids),
        dropped_repeat_cluster_ids_crossed=sorted(dropped_cluster_ids),
        hidden_repeat_spans_crossed=hidden_spans,
        max_bridged_gap_us=max_gap,
        total_bridged_gap_us=total_gap,
        unspoken_bridge_duration_us=total_gap,
        unspoken_bridge_ratio=round(ratio, 6),
        merge_safe=merge_safe,
        unsafe_reasons=sorted(unsafe_reasons),
    )






def _pad_residual_complete_short_segments(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> tuple[list[FinalTimelineSegment], int]:
    word_lookup = {word.word_id: word for word in source_graph.words}
    windows = _source_windows(source_graph)
    current = list(segments)
    padded_count = 0
    for index, segment in enumerate(current):
        duration = _duration(segment)
        classification = classify_tiny_segment(segment)
        if (
            duration <= 0
            or duration >= MIN_SEMANTIC_BRIDGE_EXCEPTION_US
            or classification.weak_filler
            or len(normalize_text(segment.text)) <= 1
        ):
            continue
        needed = MIN_SEMANTIC_BRIDGE_EXCEPTION_US - duration
        padded = _pad_segment_right(segment, current, index, source_graph, windows, needed)
        if padded is None:
            padded = _pad_segment_left(segment, current, index, source_graph, windows, needed)
        if padded is None:
            continue
        current[index] = padded
        padded_count += 1
    return current, padded_count


def _pad_segment_right(
    segment: FinalTimelineSegment,
    segments: list[FinalTimelineSegment],
    index: int,
    source_graph: CanonicalSourceGraph,
    windows: list[tuple[str, int, int]],
    needed_us: int,
) -> FinalTimelineSegment | None:
    if needed_us <= 0 or needed_us > MAX_SAFE_BRIDGE_GAP_US:
        missing: FinalTimelineSegment | None = None
        return missing
    next_start = int(segments[index + 1].source_start_us) if index + 1 < len(segments) else _window_upper_for_segment(segment, windows)
    max_end = min(int(segment.source_end_us) + int(needed_us), next_start)
    if max_end - int(segment.source_end_us) < needed_us:
        missing: FinalTimelineSegment | None = None
        return missing
    if _words_overlapping_range(source_graph, int(segment.source_end_us), max_end, set(segment.word_ids)):
        unsafe: FinalTimelineSegment | None = None
        return unsafe
    return replace(
        segment,
        source_end_us=max_end,
        spoken_source_end_us=max_end,
        target_end_us=int(segment.target_end_us) + needed_us,
        decision_ids=sorted(set([*segment.decision_ids, "visual_pacing_residual_short_padding"])),
    )


def _pad_segment_left(
    segment: FinalTimelineSegment,
    segments: list[FinalTimelineSegment],
    index: int,
    source_graph: CanonicalSourceGraph,
    windows: list[tuple[str, int, int]],
    needed_us: int,
) -> FinalTimelineSegment | None:
    if needed_us <= 0 or needed_us > MAX_SAFE_BRIDGE_GAP_US:
        missing: FinalTimelineSegment | None = None
        return missing
    previous_end = int(segments[index - 1].source_end_us) if index > 0 else _window_lower_for_segment(segment, windows)
    min_start = max(int(segment.source_start_us) - int(needed_us), previous_end)
    if int(segment.source_start_us) - min_start < needed_us:
        missing: FinalTimelineSegment | None = None
        return missing
    if _words_overlapping_range(source_graph, min_start, int(segment.source_start_us), set(segment.word_ids)):
        unsafe: FinalTimelineSegment | None = None
        return unsafe
    return replace(
        segment,
        source_start_us=min_start,
        spoken_source_start_us=min_start,
        target_start_us=max(0, int(segment.target_start_us) - needed_us),
        decision_ids=sorted(set([*segment.decision_ids, "visual_pacing_residual_short_padding"])),
    )




def _duration(segment: FinalTimelineSegment) -> int:
    return max(0, int(segment.target_end_us) - int(segment.target_start_us))


def _first_cut_density_burst_window(segments: list[FinalTimelineSegment]) -> tuple[int, int] | None:
    cut_times = [int(segment.target_start_us) for segment in list(segments)[1:]]
    for index, start in enumerate(cut_times):
        end = start + CUT_DENSITY_WINDOW_US
        count = 0
        for value in cut_times[index:]:
            if value >= end:
                break
            count += 1
        if count > MAX_CUTS_IN_5S:
            return start, end
    no_window: tuple[int, int] | None = None
    return no_window


def _cut_density_relief_merge_priority(left: FinalTimelineSegment, right: FinalTimelineSegment, index: int) -> tuple[int, int, int]:
    bridge = _is_allowed_semantic_bridge_exception(left) or _is_allowed_semantic_bridge_exception(right)
    short = _duration(left) < VISUAL_MIN_SEGMENT_DURATION_US or _duration(right) < VISUAL_MIN_SEGMENT_DURATION_US
    return (0 if bridge else 1, 0 if short else 1, _duration(left) + _duration(right), index)


def _short_count(segments: list[FinalTimelineSegment]) -> int:
    return sum(1 for segment in segments if 0 < _duration(segment) < VISUAL_MIN_SEGMENT_DURATION_US)


def _is_semantic_bridge(segment: FinalTimelineSegment) -> bool:
    return classify_tiny_segment(segment).semantic_bridge


def _is_allowed_semantic_bridge_exception(segment: FinalTimelineSegment) -> bool:
    classification = classify_tiny_segment(segment)
    visual_gap_split_bridge = (
        _is_visual_gap_split_bridge(segment)
        and not classification.weak_filler
        and 2 <= len(normalize_text(segment.text)) <= 10
        and _duration(segment) >= MIN_HARD_SEGMENT_DURATION_US
    )
    return (
        visual_gap_split_bridge
        or (
            classification.semantic_bridge
            and not classification.weak_filler
            and len(normalize_text(segment.text)) > 1
            and _duration(segment) >= MIN_SEMANTIC_BRIDGE_EXCEPTION_US
        )
    )


def _is_visual_gap_split_bridge(segment: FinalTimelineSegment) -> bool:
    return bool((segment.debug_hints or {}).get("visual_pacing_large_intra_segment_gap_split"))


def _boundary_cleanup_pair(left: FinalTimelineSegment, right: FinalTimelineSegment) -> bool:
    decision_ids = {str(decision_id) for decision_id in [*left.decision_ids, *right.decision_ids]}
    return any("boundary_suffix_prefix_overlap_cleanup" in decision_id for decision_id in decision_ids)


def _merge_group_has_redundant_restart_gap(group: VisualMergeGroup) -> bool:
    return any(bool(row.get("redundant_restart_gap")) for row in list(group.bridged_gaps or []))


def _can_merge_across_subtitle_boundary(
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    word_lookup: dict[str, Any],
) -> bool:
    if classify_tiny_segment(left).weak_filler or classify_tiny_segment(right).weak_filler:
        return True
    left_keys = _subtitle_keys(left, word_lookup)
    right_keys = _subtitle_keys(right, word_lookup)
    return not left_keys or not right_keys or bool(left_keys & right_keys)


def _subtitle_keys(segment: FinalTimelineSegment, word_lookup: dict[str, Any]) -> set[tuple[str, int | None]]:
    keys: set[tuple[str, int | None]] = set()
    for word_id in segment.word_ids:
        word = word_lookup.get(word_id)
        if word is None:
            continue
        uid = str(getattr(word, "subtitle_uid", "") or "")
        index = getattr(word, "subtitle_index", None)
        if uid or index is not None:
            keys.add((uid, index))
    return keys


def _next_weak_filler_micro_action(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    windows: list[tuple[str, int, int]],
    word_lookup: dict[str, Any],
) -> dict[str, Any]:
    no_action = {"action": "", "index": -1, "unsafe_count": 0}
    if not segments:
        return no_action
    for index, segment in enumerate(segments):
        if not _is_weak_filler_micro_segment(segment):
            continue
        candidates = []
        if index > 0:
            candidates.append(index - 1)
        if index + 1 < len(segments):
            candidates.append(index)
        unsafe_count = 0
        for merge_index in sorted(candidates, key=lambda value: _duration(segments[value]) + _duration(segments[value + 1])):
            group = _build_merge_group(
                segments[merge_index],
                segments[merge_index + 1],
                source_graph,
                windows,
                word_lookup,
                video_segment_id=segments[merge_index].segment_id,
            )
            if group.merge_safe:
                return {"action": "merge", "index": merge_index, "unsafe_count": unsafe_count}
            if _unsafe_micro_merge_attempt(group):
                unsafe_count += 1
        return {"action": "drop", "index": index, "unsafe_count": unsafe_count}
    return no_action


def _is_weak_filler_micro_segment(segment: FinalTimelineSegment) -> bool:
    classification = classify_tiny_segment(segment)
    return bool(classification.weak_filler) and len(normalize_text(segment.text)) <= 1 and 0 < _duration(segment) < MIN_HARD_SEGMENT_DURATION_US


def _unsafe_micro_merge_attempt(group: VisualMergeGroup) -> bool:
    unsafe_reasons = set(group.unsafe_reasons)
    same_window_large_gap = "bridge_gap_exceeds_threshold" in unsafe_reasons and group.source_window_id not in {"", "source_window_unbounded"}
    return bool(
        group.dropped_word_ids_crossed
        or group.dropped_segment_ids_crossed
        or group.dropped_repeat_cluster_ids_crossed
        or same_window_large_gap
        or "source_window_unresolved" in unsafe_reasons
    )


def _blocking_short_count(segments: list[FinalTimelineSegment]) -> int:
    return sum(
        1
        for segment in segments
        if 0 < _duration(segment) < VISUAL_MIN_SEGMENT_DURATION_US and not _is_allowed_semantic_bridge_exception(segment)
    )


def _allowed_blocking_short_count(segments: list[FinalTimelineSegment]) -> int:
    if len(segments) < 10:
        return MAX_SEMANTIC_BRIDGE_SHORT_SEGMENTS
    return 0


def _residual_visual_short_segments(
    segments: list[FinalTimelineSegment],
    safety_report: dict[str, Any],
    *,
    semantic_bridge_safe_merge_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    groups_by_id = {
        str(row.get("video_segment_id") or ""): row
        for row in list(safety_report.get("merge_groups") or [])
        if isinstance(row, dict)
    }
    safe_merge_by_id = {
        str(row.get("segment_id") or ""): row
        for row in list(semantic_bridge_safe_merge_candidates or [])
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        duration = _duration(segment)
        if duration <= 0 or duration >= VISUAL_MIN_SEGMENT_DURATION_US:
            continue
        classification = classify_tiny_segment(segment)
        group = groups_by_id.get(segment.segment_id) or {}
        allowed_bridge = _is_allowed_semantic_bridge_exception(segment)
        safe_merge_candidate = safe_merge_by_id.get(segment.segment_id)
        semantic_bridge_reason = "large_gap_split_bridge" if _is_visual_gap_split_bridge(segment) else "semantic_bridge_exception"
        rows.append(
            {
                "index": index,
                "segment_id": segment.segment_id,
                "target_start_us": int(segment.target_start_us),
                "target_end_us": int(segment.target_end_us),
                "duration_us": duration,
                "source_start_us": int(segment.source_start_us),
                "source_end_us": int(segment.source_end_us),
                "source_window_id": str(group.get("source_window_id") or ""),
                "word_ids": list(segment.word_ids),
                "text": segment.text,
                "previous_segment_id": segments[index - 1].segment_id if index > 0 else "",
                "next_segment_id": segments[index + 1].segment_id if index + 1 < len(segments) else "",
                "short_segment_status": semantic_bridge_reason if allowed_bridge else "blocking",
                "semantic_bridge_reason": semantic_bridge_reason if allowed_bridge else "",
                "why_not_merge": _semantic_bridge_why_not_merge(
                    allowed_bridge=allowed_bridge,
                    group=group,
                    safe_merge_candidate=safe_merge_candidate,
                ),
                "merge_candidate_reason": classification.merge_candidate_reason,
                "weak_filler": bool(classification.weak_filler),
                "semantic_bridge": bool(allowed_bridge),
                "safe_merge_candidate": bool(safe_merge_candidate),
                "unsafe_reasons": list(group.get("unsafe_reasons") or []),
                "bridged_gap_us": int(group.get("max_bridged_gap_us") or 0),
                "merge_safe": bool(group.get("merge_safe", True)),
            }
        )
    return rows


def _semantic_bridge_why_not_merge(
    *,
    allowed_bridge: bool,
    group: dict[str, Any],
    safe_merge_candidate: dict[str, Any] | None,
) -> list[str]:
    if safe_merge_candidate:
        return ["safe_merge_available"]
    if not allowed_bridge:
        return ["not_semantic_bridge_exception"]
    unsafe_reasons = [str(reason) for reason in list(group.get("unsafe_reasons") or []) if str(reason)]
    if unsafe_reasons:
        return unsafe_reasons
    return ["semantic_bridge_exception_preserved"]


def _semantic_bridge_safe_merge_candidates(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> list[dict[str, Any]]:
    windows = _source_windows(source_graph)
    word_lookup = {word.word_id: word for word in source_graph.words}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, segment in enumerate(segments):
        if not _is_allowed_semantic_bridge_exception(segment):
            continue
        candidates = []
        if index > 0:
            candidates.append(index - 1)
        if index + 1 < len(segments):
            candidates.append(index)
        for merge_index in candidates:
            left = segments[merge_index]
            right = segments[merge_index + 1]
            if _is_visual_gap_split_bridge(left) or _is_visual_gap_split_bridge(right):
                continue
            if len(segments) < 10 and not _can_merge_across_subtitle_boundary(left, right, word_lookup):
                continue
            group = _build_merge_group(
                left,
                right,
                source_graph,
                windows,
                word_lookup,
                video_segment_id=left.segment_id,
            )
            if not group.merge_safe:
                continue
            if segment.segment_id in seen:
                continue
            seen.add(segment.segment_id)
            rows.append(
                {
                    "segment_id": segment.segment_id,
                    "text": segment.text,
                    "duration_us": _duration(segment),
                    "candidate_pair_segment_ids": [left.segment_id, right.segment_id],
                    "reason": "safe_merge_available",
                    "target_start_us": segment.target_start_us,
                    "target_end_us": segment.target_end_us,
                    "source_start_us": segment.source_start_us,
                    "source_end_us": segment.source_end_us,
                }
            )
    return rows


def _semantic_bridge_gate_blockers(
    *,
    semantic_bridge_count: int,
    semantic_bridge_cap: int,
    safe_merge_candidate_count: int,
) -> list[str]:
    if int(semantic_bridge_count) > int(semantic_bridge_cap) or int(safe_merge_candidate_count) > 0:
        return ["V21_VISUAL_SEMANTIC_BRIDGE_ABUSE"]
    no_blockers: list[str] = []
    return no_blockers


def _semantic_bridge_segment_cap(segment_count: int) -> int:
    scaled_cap = (
        int(segment_count) * SEMANTIC_BRIDGE_SHORT_SEGMENT_RATIO_NUMERATOR
        + SEMANTIC_BRIDGE_SHORT_SEGMENT_RATIO_DENOMINATOR
        - 1
    ) // SEMANTIC_BRIDGE_SHORT_SEGMENT_RATIO_DENOMINATOR
    return max(SEMANTIC_BRIDGE_SHORT_SEGMENT_CAP, scaled_cap)




def _short_segments_exceed_gate_limit(
    *,
    blocking_short_count: int,
    allowed_blocking_short_count: int,
    unsafe_merge_attempt_count: int,
) -> bool:
    if int(unsafe_merge_attempt_count) > 0:
        return int(blocking_short_count) > 0
    return int(blocking_short_count) > int(allowed_blocking_short_count)


def _source_windows(source_graph: CanonicalSourceGraph) -> list[tuple[str, int, int]]:
    windows = []
    for index, row in enumerate(source_graph.source_segments, start=1):
        track_type = str(row.get("track_type") or row.get("type") or "").lower()
        if track_type and "video" not in track_type:
            continue
        start = int(row.get("canonical_source_start_us") or row.get("target_start_us") or row.get("source_start_us") or 0)
        end = int(row.get("canonical_source_end_us") or row.get("target_end_us") or row.get("source_end_us") or 0)
        if end > start:
            window_id = str(row.get("source_window_id") or row.get("id") or f"source_window_{index:06d}")
            windows.append((window_id, start, end))
    return sorted(set(windows), key=lambda row: (row[1], row[2], row[0]))


def _source_window_id_for_range(windows: list[tuple[str, int, int]], start: int, end: int) -> str:
    if not windows:
        return "source_window_unbounded"
    matches = [window_id for window_id, window_start, window_end in windows if window_start <= int(start) and int(end) <= window_end]
    if len(matches) == 1:
        return matches[0]
    unresolved_window_id = ""
    return unresolved_window_id


def _same_source_window(left: FinalTimelineSegment, right: FinalTimelineSegment, windows: list[tuple[str, int, int]]) -> bool:
    start = min(int(left.source_start_us), int(right.source_start_us))
    end = max(int(left.source_end_us), int(right.source_end_us))
    if not windows:
        return True
    return any(window_start <= start and end <= window_end for _window_id, window_start, window_end in windows)

def _bind_visual_pacing_report_helpers() -> None:
    visual_pacing_report_helpers.configure_visual_pacing_report_dependencies(globals())
build_visual_pacing_report = visual_pacing_report_helpers.build_visual_pacing_report
_visual_report = visual_pacing_report_helpers._visual_report
_safety_report_from_merge_report = visual_pacing_report_helpers._safety_report_from_merge_report
_empty_safety_report = visual_pacing_report_helpers._empty_safety_report
_safety_report_with_caption_ids = visual_pacing_report_helpers._safety_report_with_caption_ids
_build_visual_merge_safety_report = visual_pacing_report_helpers._build_visual_merge_safety_report

_bind_visual_pacing_report_helpers()
