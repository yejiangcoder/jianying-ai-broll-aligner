from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.contracts import VisualMergeGroup, VisualMergeSafetyReport, VisualPacingReport, contract_to_dict
from aroll_v21.ir.models import CaptionRenderUnit, CanonicalSourceGraph, FinalTimelineSegment
from aroll_v21.quality.safe_boundary import trailing_word_ids_for_suffix_overlap
from aroll_v21.quality.tiny_segment_classifier import classify_tiny_segment


MIN_HARD_SEGMENT_DURATION_US = 300_000
VISUAL_MIN_SEGMENT_DURATION_US = 1_200_000
MAX_SAFE_BRIDGE_GAP_US = 220_000
MAX_UNSPOKEN_BRIDGE_RATIO = 0.20
MAX_SEMANTIC_BRIDGE_SHORT_SEGMENTS = 3
MIN_SEMANTIC_BRIDGE_EXCEPTION_US = 600_000
SEMANTIC_BRIDGE_SHORT_SEGMENT_CAP = 8
SEMANTIC_BRIDGE_SHORT_SEGMENT_RATIO_NUMERATOR = 15
SEMANTIC_BRIDGE_SHORT_SEGMENT_RATIO_DENOMINATOR = 100
CUT_DENSITY_MIN_SEGMENT_COUNT = 10
CUT_DENSITY_WINDOW_US = 5_000_000
MAX_CUTS_PER_MINUTE = 30.0
MAX_CUTS_IN_5S = 5
MAX_BURST_CUT_COUNT = 0


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
        if (
            hidden_repeat_dropped_word_count
            or boundary_overlap_dropped_word_count
            or post_cleanup_padded_short_count
            or post_micro_report["merged_weak_filler_micro_segment_count"]
            or post_micro_report["dropped_weak_filler_micro_segment_count"]
        ):
            current = _repack(current)
        safety_report = _build_visual_merge_safety_report(
            current,
            source_graph,
            unsafe_merge_attempt_count=unsafe_attempts,
        )
        after_short = _short_count(current)
        semantic_bridge_count = sum(1 for segment in current if _is_allowed_semantic_bridge_exception(segment))
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
                if len(segments) < 10 and not _can_merge_across_subtitle_boundary(left_candidate, right_candidate, word_lookup):
                    continue
                group = _build_merge_group(
                    left_candidate,
                    right_candidate,
                    source_graph,
                    windows,
                    word_lookup,
                    video_segment_id=segments[merge_index].segment_id,
                )
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
        child_segments = [*_child_segment_records(left), *_child_segment_records(right)]
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


def build_visual_pacing_report(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    executed: bool = False,
    merge_report: dict[str, Any] | None = None,
    source_graph: CanonicalSourceGraph | None = None,
) -> dict[str, Any]:
    merge_report = dict(merge_report or {})
    timeline_changed_after_visual_pacing = int(merge_report.get("final_visible_repair_action_count") or 0) > 0
    before_short = int(merge_report.get("visual_short_segment_count_lt_1200ms_before") or _short_count(final_timeline))
    after_short = (
        _short_count(final_timeline)
        if timeline_changed_after_visual_pacing
        else int(merge_report.get("visual_short_segment_count_lt_1200ms_after") or _short_count(final_timeline))
    )
    semantic_bridge_count = (
        sum(1 for segment in final_timeline if _is_allowed_semantic_bridge_exception(segment))
        if timeline_changed_after_visual_pacing
        else int(
            merge_report.get("semantic_bridge_short_segment_count")
            or sum(1 for segment in final_timeline if _is_allowed_semantic_bridge_exception(segment))
        )
    )
    if timeline_changed_after_visual_pacing:
        blocking_short_count = _blocking_short_count(final_timeline)
    elif merge_report.get("visual_short_segment_count_lt_1200ms_after_blocking") is not None:
        blocking_short_count = int(merge_report.get("visual_short_segment_count_lt_1200ms_after_blocking") or 0)
    elif merge_report.get("semantic_bridge_short_segment_count") is not None:
        blocking_short_count = max(0, after_short - semantic_bridge_count)
    else:
        blocking_short_count = _blocking_short_count(final_timeline)
    allowed_blocking_short_count = int(
        merge_report.get("visual_pacing_allowed_short_segment_threshold")
        if isinstance(merge_report.get("visual_pacing_allowed_short_segment_threshold"), int)
        else _allowed_blocking_short_count(final_timeline)
    )
    blocker_codes = [] if timeline_changed_after_visual_pacing else list(merge_report.get("visual_pacing_blocker_codes") or [])
    safety_report = (
        _build_visual_merge_safety_report(final_timeline, source_graph, unsafe_merge_attempt_count=0)
        if timeline_changed_after_visual_pacing and source_graph is not None
        else _safety_report_from_merge_report(merge_report, final_timeline, captions)
    )
    if not executed:
        blocker_codes.append("V21_VISUAL_PACING_NOT_EXECUTED")
    unsafe_merge_attempt_count = int(
        safety_report.get("visual_pacing_blocked_unsafe_merge_attempt_count")
        or merge_report.get("visual_pacing_blocked_unsafe_merge_attempt_count")
        or 0
    )
    if _short_segments_exceed_gate_limit(
        blocking_short_count=blocking_short_count,
        allowed_blocking_short_count=allowed_blocking_short_count,
        unsafe_merge_attempt_count=unsafe_merge_attempt_count,
    ):
        blocker_codes.append("V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN")
    if not bool(safety_report.get("gate_passed")):
        blocker_codes.extend(str(code) for code in safety_report.get("blocker_codes") or [])
    semantic_bridge_cap = _semantic_bridge_segment_cap(len(final_timeline))
    if timeline_changed_after_visual_pacing and source_graph is not None:
        safe_merge_candidates = _semantic_bridge_safe_merge_candidates(final_timeline, source_graph)
    elif timeline_changed_after_visual_pacing:
        safe_merge_candidates = []
    else:
        safe_merge_candidates = list(merge_report.get("semantic_bridge_safe_merge_candidates") or [])
    blocker_codes.extend(
        _semantic_bridge_gate_blockers(
            semantic_bridge_count=semantic_bridge_count,
            semantic_bridge_cap=semantic_bridge_cap,
            safe_merge_candidate_count=len(safe_merge_candidates),
        )
    )
    cut_density_report = _cut_density_report_from_merge_report(merge_report, final_timeline)
    blocker_codes.extend(_cut_density_blockers(cut_density_report))
    return _visual_report(
        final_timeline=final_timeline,
        captions=captions,
        gate_passed=not blocker_codes,
        executed=executed,
        attempted=int(merge_report.get("visual_pacing_merge_attempted_count") or 0),
        merged=int(merge_report.get("visual_pacing_merged_count") or 0),
        before_short=before_short,
        after_short=after_short,
        semantic_bridge_count=semantic_bridge_count,
        blocker_codes=sorted(set(blocker_codes)),
        hidden_repeat_dropped_word_count=int(merge_report.get("visual_pacing_hidden_repeat_dropped_word_count") or 0),
        hidden_repeat_split_segment_count=int(merge_report.get("visual_pacing_hidden_repeat_split_segment_count") or 0),
        boundary_overlap_dropped_word_count=int(merge_report.get("visual_pacing_boundary_overlap_dropped_word_count") or 0),
        safety_report=safety_report,
        blocking_short_count=blocking_short_count,
        allowed_blocking_short_count=allowed_blocking_short_count,
        dropped_weak_filler_count=int(merge_report.get("visual_pacing_weak_filler_dropped_count") or 0),
        merged_weak_filler_micro_segment_count=int(merge_report.get("merged_weak_filler_micro_segment_count") or 0),
        dropped_weak_filler_micro_segment_count=int(merge_report.get("dropped_weak_filler_micro_segment_count") or 0),
        padded_short_count=int(merge_report.get("visual_pacing_residual_short_padded_count") or 0),
        semantic_bridge_cap=semantic_bridge_cap,
        semantic_bridge_safe_merge_candidates=safe_merge_candidates,
        cut_density_report=cut_density_report,
    )


def _visual_report(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    gate_passed: bool,
    executed: bool,
    attempted: int,
    merged: int,
    before_short: int,
    after_short: int,
    semantic_bridge_count: int,
    blocker_codes: list[str],
    hidden_repeat_dropped_word_count: int = 0,
    hidden_repeat_split_segment_count: int = 0,
    boundary_overlap_dropped_word_count: int = 0,
    safety_report: dict[str, Any] | None = None,
    blocking_short_count: int | None = None,
    allowed_blocking_short_count: int | None = None,
    dropped_weak_filler_count: int = 0,
    merged_weak_filler_micro_segment_count: int = 0,
    dropped_weak_filler_micro_segment_count: int = 0,
    padded_short_count: int = 0,
    semantic_bridge_cap: int = SEMANTIC_BRIDGE_SHORT_SEGMENT_CAP,
    semantic_bridge_safe_merge_candidates: list[dict[str, Any]] | None = None,
    cut_density_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    durations = [_duration(segment) for segment in final_timeline]
    ratio = round(len(captions) / len(final_timeline), 4) if final_timeline else 0.0
    safety_for_residuals = safety_report or _empty_safety_report()
    safe_merge_candidates = list(semantic_bridge_safe_merge_candidates or [])
    cut_density = cut_density_report or _cut_density_report(final_timeline)
    residual_short_segments = _residual_visual_short_segments(
        final_timeline,
        safety_for_residuals,
        semantic_bridge_safe_merge_candidates=safe_merge_candidates,
    )
    semantic_bridge_details = [row for row in residual_short_segments if bool(row.get("semantic_bridge"))]
    report = contract_to_dict(
        VisualPacingReport(
            gate_passed=gate_passed,
            final_video_segment_count=len(final_timeline),
            caption_count=len(captions),
            visual_short_segment_count_lt_1200ms=after_short,
            median_segment_duration_us=int(median(durations)) if durations else 0,
            p10_segment_duration_us=_percentile(durations, 10),
            caption_per_video_segment_ratio=ratio,
            blocker_codes=sorted(set(blocker_codes)),
        )
    )
    report.update(
        {
            "visual_pacing_executed": executed,
            "visual_pacing_merge_attempted_count": attempted,
            "visual_pacing_merged_count": merged,
            "visual_short_segment_count_lt_1200ms_before": before_short,
            "visual_short_segment_count_lt_1200ms_after": after_short,
            "visual_short_segment_count_lt_1200ms_after_blocking": _blocking_short_count(final_timeline)
            if blocking_short_count is None
            else int(blocking_short_count),
            "semantic_bridge_short_segment_count": semantic_bridge_count,
            "visual_pacing_allowed_short_segment_threshold": _allowed_blocking_short_count(final_timeline)
            if allowed_blocking_short_count is None
            else int(allowed_blocking_short_count),
            "visual_pacing_allowed_short_segment_policy": "explicit_semantic_bridge_only",
            "visual_pacing_blocker_codes": sorted(set(blocker_codes)),
            "visual_pacing_hidden_repeat_dropped_word_count": hidden_repeat_dropped_word_count,
            "visual_pacing_hidden_repeat_split_segment_count": hidden_repeat_split_segment_count,
            "visual_pacing_boundary_overlap_dropped_word_count": boundary_overlap_dropped_word_count,
            "visual_pacing_weak_filler_dropped_count": int(dropped_weak_filler_count),
            "merged_weak_filler_micro_segment_count": int(merged_weak_filler_micro_segment_count),
            "dropped_weak_filler_micro_segment_count": int(dropped_weak_filler_micro_segment_count),
            "visual_pacing_residual_short_padded_count": int(padded_short_count),
            "residual_visual_short_segments": residual_short_segments,
            "semantic_bridge_short_segment_details": semantic_bridge_details,
            "semantic_bridge_reason_counts": _reason_counts(semantic_bridge_details),
            "semantic_bridge_cap": int(semantic_bridge_cap),
            "semantic_bridge_safe_merge_candidate_count": len(safe_merge_candidates),
            "semantic_bridge_safe_merge_candidates": safe_merge_candidates,
            "cuts_per_minute": float(cut_density.get("cuts_per_minute") or 0.0),
            "max_cuts_in_5s": int(cut_density.get("max_cuts_in_5s") or 0),
            "burst_cut_count": int(cut_density.get("burst_cut_count") or 0),
            "cut_density_gate_enabled": bool(cut_density.get("enabled")),
            "cut_density_gate_passed": bool(cut_density.get("gate_passed")),
            "cut_density_thresholds": dict(cut_density.get("thresholds") or {}),
            "cut_density_window_us": CUT_DENSITY_WINDOW_US,
        }
    )
    safety_report = _safety_report_with_caption_ids(safety_report or _empty_safety_report(), captions)
    report.update(
        {
            "visual_merge_safety_gate_passed": bool(safety_report.get("gate_passed")),
            "unsafe_merge_group_count": int(safety_report.get("unsafe_merge_group_count") or 0),
            "dropped_content_reintroduced_count": int(safety_report.get("dropped_content_reintroduced_count") or 0),
            "max_bridged_gap_us": int(safety_report.get("max_bridged_gap_us") or 0),
            "total_bridged_gap_us": int(safety_report.get("total_bridged_gap_us") or 0),
            "unspoken_bridge_ratio": float(safety_report.get("unspoken_bridge_ratio") or 0.0),
            "visual_merge_safety_report": safety_report,
            "visual_merge_groups": list(safety_report.get("merge_groups") or []),
        }
    )
    return report


def _safety_report_from_merge_report(
    merge_report: dict[str, Any],
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
) -> dict[str, Any]:
    existing = merge_report.get("visual_merge_safety_report")
    if isinstance(existing, dict):
        return _safety_report_with_caption_ids(existing, captions)
    merged_count = int(merge_report.get("visual_pacing_merged_count") or 0)
    ratio = len(captions) / len(final_timeline) if final_timeline else 0.0
    if merged_count > 0 or ratio > 10.0:
        return contract_to_dict(
            VisualMergeSafetyReport(
                gate_passed=False,
                unsafe_merge_group_count=1,
                blocker_codes=["V21_VISUAL_PACING_MISSING_MERGE_SAFETY_PROOF"],
            )
        )
    return _empty_safety_report()


def _empty_safety_report() -> dict[str, Any]:
    return contract_to_dict(VisualMergeSafetyReport())


def _safety_report_with_caption_ids(
    safety_report: dict[str, Any],
    captions: list[CaptionRenderUnit],
) -> dict[str, Any]:
    if not captions:
        return dict(safety_report)
    caption_ids_by_segment: dict[str, list[str]] = {}
    for caption in captions:
        segment_id = str(caption.containing_video_segment_id or "")
        if not segment_id:
            for timeline_segment_id in caption.timeline_segment_ids:
                caption_ids_by_segment.setdefault(str(timeline_segment_id), []).append(caption.caption_id)
            continue
        caption_ids_by_segment.setdefault(segment_id, []).append(caption.caption_id)
    payload = dict(safety_report)
    groups = []
    for row in list(payload.get("merge_groups") or []):
        group = dict(row)
        segment_id = str(group.get("video_segment_id") or "")
        group["child_caption_ids"] = sorted(set(caption_ids_by_segment.get(segment_id) or []))
        groups.append(group)
    payload["merge_groups"] = groups
    return payload


def _build_visual_merge_safety_report(
    segments: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    *,
    unsafe_merge_attempt_count: int,
) -> dict[str, Any]:
    windows = _source_windows(source_graph)
    word_lookup = {word.word_id: word for word in source_graph.words}
    groups = [
        _build_segment_merge_group(segment, source_graph, windows, word_lookup)
        for segment in segments
    ]
    unsafe_count = sum(1 for group in groups if not group.merge_safe)
    dropped_content_count = sum(1 for group in groups if group.dropped_word_ids_crossed or group.dropped_segment_ids_crossed)
    max_gap = max([group.max_bridged_gap_us for group in groups] or [0])
    total_gap = sum(group.total_bridged_gap_us for group in groups)
    ratio = max([group.unspoken_bridge_ratio for group in groups] or [0.0])
    blocker_codes: list[str] = []
    if unsafe_count or dropped_content_count or max_gap > MAX_SAFE_BRIDGE_GAP_US or ratio > MAX_UNSPOKEN_BRIDGE_RATIO:
        blocker_codes.append("V21_VISUAL_PACING_UNSAFE_MERGE")
    report = contract_to_dict(
        VisualMergeSafetyReport(
            gate_passed=not blocker_codes,
            merge_groups=groups,
            unsafe_merge_group_count=unsafe_count,
            dropped_content_reintroduced_count=dropped_content_count,
            max_bridged_gap_us=max_gap,
            total_bridged_gap_us=total_gap,
            unspoken_bridge_ratio=round(ratio, 6),
            blocker_codes=blocker_codes,
        )
    )
    report["visual_pacing_blocked_unsafe_merge_attempt_count"] = int(unsafe_merge_attempt_count)
    report["max_safe_bridge_gap_us"] = MAX_SAFE_BRIDGE_GAP_US
    report["max_unspoken_bridge_ratio"] = MAX_UNSPOKEN_BRIDGE_RATIO
    return report


def _build_merge_group(
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    windows: list[tuple[str, int, int]],
    word_lookup: dict[str, Any],
    *,
    video_segment_id: str,
) -> VisualMergeGroup:
    merged_records = [*_child_segment_records(left, word_lookup), *_child_segment_records(right, word_lookup)]
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
        if gap_us < 0:
            unsafe_reasons.add("source_ranges_overlap")
        if gap_us > MAX_SAFE_BRIDGE_GAP_US:
            unsafe_reasons.add("bridge_gap_exceeds_threshold")
        if gap_word_ids:
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
            }
        )
    max_gap = max([int(row.get("duration_us") or 0) for row in bridged_gaps] or [0])
    total_gap = sum(int(row.get("duration_us") or 0) for row in bridged_gaps)
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
    return [
        word
        for word in source_graph.words
        if word.word_id not in child_word_ids
        and int(word.source_end_us) > int(start_us)
        and int(word.source_start_us) < int(end_us)
    ]


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
                return suffix_start
    if len(tokens) >= 3:
        suffix = tokens[-1]
        if suffix and len(suffix) >= 2:
            for start, token in enumerate(tokens[:-1]):
                if token == suffix and start + 1 < len(tokens) - 1:
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
    left = normalize_text(str(left_text or ""))
    right = normalize_text(str(right_text or ""))
    max_size = min(len(left), len(right), 20)
    for size in range(max_size, 1, -1):
        candidate = left[-size:]
        if right.startswith(candidate):
            return candidate
    return ""


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


def _window_upper_for_segment(segment: FinalTimelineSegment, windows: list[tuple[str, int, int]]) -> int:
    for _window_id, start, end in windows:
        if int(start) <= int(segment.source_start_us) and int(segment.source_end_us) <= int(end):
            return int(end)
    return int(segment.source_end_us)


def _window_lower_for_segment(segment: FinalTimelineSegment, windows: list[tuple[str, int, int]]) -> int:
    for _window_id, start, end in windows:
        if int(start) <= int(segment.source_start_us) and int(segment.source_end_us) <= int(end):
            return int(start)
    return int(segment.source_start_us)


def _duration(segment: FinalTimelineSegment) -> int:
    return max(0, int(segment.target_end_us) - int(segment.target_start_us))


def _short_count(segments: list[FinalTimelineSegment]) -> int:
    return sum(1 for segment in segments if 0 < _duration(segment) < VISUAL_MIN_SEGMENT_DURATION_US)


def _is_semantic_bridge(segment: FinalTimelineSegment) -> bool:
    return classify_tiny_segment(segment).semantic_bridge


def _is_allowed_semantic_bridge_exception(segment: FinalTimelineSegment) -> bool:
    classification = classify_tiny_segment(segment)
    return (
        classification.semantic_bridge
        and not classification.weak_filler
        and len(normalize_text(segment.text)) > 1
        and _duration(segment) >= MIN_SEMANTIC_BRIDGE_EXCEPTION_US
    )


def _boundary_cleanup_pair(left: FinalTimelineSegment, right: FinalTimelineSegment) -> bool:
    decision_ids = {str(decision_id) for decision_id in [*left.decision_ids, *right.decision_ids]}
    return any("boundary_suffix_prefix_overlap_cleanup" in decision_id for decision_id in decision_ids)


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
                "short_segment_status": "semantic_bridge_exception" if allowed_bridge else "blocking",
                "semantic_bridge_reason": "semantic_bridge_exception" if allowed_bridge else "",
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


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("merge_candidate_reason") or row.get("short_segment_status") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


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


def _cut_density_report_from_merge_report(
    merge_report: dict[str, Any],
    final_timeline: list[FinalTimelineSegment],
) -> dict[str, Any]:
    report = _cut_density_report(final_timeline)
    for key in ("cuts_per_minute", "max_cuts_in_5s", "burst_cut_count"):
        if key in merge_report:
            report[key] = merge_report[key]
    report["gate_passed"] = not _cut_density_failed(report)
    return report


def _cut_density_report(segments: list[FinalTimelineSegment]) -> dict[str, Any]:
    ordered = sorted(segments, key=lambda row: (int(row.target_start_us), int(row.target_end_us), row.segment_id))
    cut_times = [int(segment.target_start_us) for segment in ordered[1:]]
    timeline_start = min([int(segment.target_start_us) for segment in ordered] or [0])
    timeline_end = max([int(segment.target_end_us) for segment in ordered] or [0])
    duration_us = max(0, timeline_end - timeline_start)
    cuts_per_minute = round((len(cut_times) * 60_000_000 / duration_us), 4) if duration_us > 0 else 0.0
    max_cuts_in_window = 0
    burst_cut_count = 0
    for index, start in enumerate(cut_times):
        end = start + CUT_DENSITY_WINDOW_US
        count = 0
        for value in cut_times[index:]:
            if value >= end:
                break
            count += 1
        max_cuts_in_window = max(max_cuts_in_window, count)
        if count > MAX_CUTS_IN_5S:
            burst_cut_count += 1
    report = {
        "enabled": len(ordered) >= CUT_DENSITY_MIN_SEGMENT_COUNT,
        "cut_count": len(cut_times),
        "timeline_duration_us": duration_us,
        "cuts_per_minute": cuts_per_minute,
        "max_cuts_in_5s": max_cuts_in_window,
        "burst_cut_count": burst_cut_count,
        "thresholds": {
            "min_segment_count": CUT_DENSITY_MIN_SEGMENT_COUNT,
            "max_cuts_per_minute": MAX_CUTS_PER_MINUTE,
            "max_cuts_in_5s": MAX_CUTS_IN_5S,
            "max_burst_cut_count": MAX_BURST_CUT_COUNT,
            "window_us": CUT_DENSITY_WINDOW_US,
        },
    }
    report["gate_passed"] = not _cut_density_failed(report)
    return report


def _cut_density_failed(report: dict[str, Any]) -> bool:
    if not bool(report.get("enabled")):
        return False
    return (
        float(report.get("cuts_per_minute") or 0.0) > MAX_CUTS_PER_MINUTE
        or int(report.get("max_cuts_in_5s") or 0) > MAX_CUTS_IN_5S
        or int(report.get("burst_cut_count") or 0) > MAX_BURST_CUT_COUNT
    )


def _cut_density_blockers(report: dict[str, Any]) -> list[str]:
    if _cut_density_failed(report):
        return ["V21_VISUAL_CUT_DENSITY_FAILED"]
    no_blockers: list[str] = []
    return no_blockers


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


def _repack(segments: list[FinalTimelineSegment]) -> list[FinalTimelineSegment]:
    cursor = 0
    repacked = []
    for index, segment in enumerate(segments, start=1):
        duration = max(0, int(segment.source_end_us) - int(segment.source_start_us))
        repacked.append(
            replace(
                segment,
                segment_id=f"v21_seg_{index:06d}",
                target_start_us=cursor,
                target_end_us=cursor + duration,
            )
        )
        cursor += duration
    return repacked


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return int(round(ordered[lower] * (1 - weight) + ordered[upper] * weight))
