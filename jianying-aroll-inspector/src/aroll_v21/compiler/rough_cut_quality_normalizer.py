from __future__ import annotations

from dataclasses import replace
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import Blocker, CanonicalSourceGraph, DecisionPlan, FinalTimelineSegment


MIN_HARD_SEGMENT_DURATION_US = 300_000
MIN_SOFT_SEGMENT_DURATION_US = 500_000
MIN_CAPTION_CHARS = 3
PREFERRED_MIN_SEGMENT_DURATION_US = 700_000
DEFAULT_LEAD_HANDLE_US = 220_000
DEFAULT_TAIL_HANDLE_US = 220_000
SOURCE_GAP_MERGE_LIMIT_US = 1_500_000
WEAK_FILLER_MICRO_TEXTS = {"呃", "嗯", "啊", "额", "呐", "哎", "诶", "哦", "噢", "喔", "唉"}
STRUCTURAL_FUNCTION_MICRO_TEXTS = {"的", "就", "是", "了", "在"}


class RoughCutQualityNormalizer:
    """Compile-time rough-cut quality normalizer.

    This normalizes the final timeline before captions/materials are derived.
    It only merges whole final segments and annotates handle ranges; validators
    remain read-only.
    """

    def normalize(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        decision_plan: DecisionPlan,
        *,
        emit_residual_blockers: bool = True,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker]]:
        if not segments:
            return segments, []
        word_lookup = {word.word_id: word for word in source_graph.words}
        source_windows = self._source_windows(source_graph)
        merged = self._merge_micro_segments(list(segments), word_lookup, decision_plan, source_windows)
        finalized = self._finalize_residual_micro_segments(merged, word_lookup, decision_plan, source_windows)
        repacked = self._repack_target_timeline(finalized)
        handled = self._apply_source_handles(repacked, source_graph)
        blockers = self._residual_micro_segment_blockers(handled, word_lookup) if emit_residual_blockers else []
        return handled, blockers

    def _merge_micro_segments(
        self,
        segments: list[FinalTimelineSegment],
        word_lookup: dict[str, Any],
        decision_plan: DecisionPlan,
        source_windows: list[tuple[int, int]],
    ) -> list[FinalTimelineSegment]:
        current = list(segments)
        max_rounds = max(1, len(current) * 2)
        for _round in range(max_rounds):
            merge_index = self._next_merge_index(current, word_lookup, source_windows)
            if merge_index is None:
                break
            left = current[merge_index]
            right = current[merge_index + 1]
            merged = self._merge_pair(left, right)
            decision_plan.decision_trace.append(
                {
                    "route": "rough_cut_quality_normalizer",
                    "stage": "final_timeline_pre_emit",
                    "decision": "merge_micro_segments",
                    "applied": True,
                    "left_segment_id": left.segment_id,
                    "right_segment_id": right.segment_id,
                    "merged_text_length": len(normalize_text(merged.text)),
                    "reason": "whole-segment phrase grouping for short rough-cut fragments",
                }
            )
            current = current[:merge_index] + [merged] + current[merge_index + 2 :]
        return current

    def _finalize_residual_micro_segments(
        self,
        segments: list[FinalTimelineSegment],
        word_lookup: dict[str, Any],
        decision_plan: DecisionPlan,
        source_windows: list[tuple[int, int]],
    ) -> list[FinalTimelineSegment]:
        current = list(segments)
        max_rounds = max(1, len(current) * 2)
        for _round in range(max_rounds):
            drop_index: int | None = None
            merge_index: int | None = None
            for index, segment in self._residual_micro_segments(current):
                if self._can_drop_weak_filler_micro_segment(segment, word_lookup):
                    drop_index = index
                    break
                if self._can_drop_residual_prefix_before_next(current, index):
                    drop_index = index
                    break
                candidate_index = self._residual_merge_index(current, index, word_lookup, source_windows)
                if candidate_index is not None:
                    merge_index = candidate_index
                    break
            if drop_index is not None:
                dropped = current[drop_index]
                next_segment = current[drop_index + 1] if drop_index + 1 < len(current) else None
                decision = (
                    "drop_weak_filler_micro_segment"
                    if self._can_drop_weak_filler_micro_segment(dropped, word_lookup)
                    else "drop_residual_prefix_segment"
                )
                decision_plan.decision_trace.append(
                    {
                        "route": (
                            "residual_prefix_containment_drop"
                            if decision == "drop_residual_prefix_segment"
                            else "rough_cut_quality_normalizer"
                        ),
                        "component": "rough_cut_quality_normalizer",
                        "stage": "final_timeline_pre_emit",
                        "decision": decision,
                        "applied": True,
                        "dropped_segment_id": dropped.segment_id,
                        "dropped_text": dropped.text,
                        "next_segment_id": next_segment.segment_id if next_segment is not None else "",
                        "next_text": next_segment.text if next_segment is not None else "",
                        "reason": "weak_filler_micro_segment" if decision == "drop_weak_filler_micro_segment" else "residual_text_is_prefix_of_next_text",
                    }
                )
                current = current[:drop_index] + current[drop_index + 1 :]
                continue
            if merge_index is None:
                break
            left = current[merge_index]
            right = current[merge_index + 1]
            merged = self._merge_pair(left, right)
            decision_plan.decision_trace.append(
                {
                    "route": "rough_cut_quality_normalizer",
                    "stage": "final_timeline_pre_emit",
                    "decision": "final_sweep_merge_residual_micro_segment",
                    "applied": True,
                    "left_segment_id": left.segment_id,
                    "right_segment_id": right.segment_id,
                    "merged_text_length": len(normalize_text(merged.text)),
                    "reason": "final whole-segment sweep for hard rough-cut residuals",
                }
            )
            current = current[:merge_index] + [merged] + current[merge_index + 2 :]
        return current

    def _can_drop_residual_prefix_before_next(
        self,
        segments: list[FinalTimelineSegment],
        index: int,
    ) -> bool:
        if index + 1 >= len(segments):
            return False
        current_text = normalize_text(segments[index].text)
        next_text = normalize_text(segments[index + 1].text)
        if current_text in STRUCTURAL_FUNCTION_MICRO_TEXTS:
            return False
        return bool(current_text and len(next_text) > len(current_text) and next_text.startswith(current_text))

    def _can_drop_weak_filler_micro_segment(
        self,
        segment: FinalTimelineSegment,
        word_lookup: dict[str, Any],
    ) -> bool:
        text = normalize_text(segment.text)
        if text not in WEAK_FILLER_MICRO_TEXTS:
            return False
        if not self._is_micro_duration(segment):
            return False
        if len(segment.word_ids) > 2:
            return False
        for word_id in segment.word_ids:
            word = word_lookup.get(word_id)
            word_text = normalize_text(str(getattr(word, "text", "") or "")) if word is not None else ""
            if word_text and word_text not in WEAK_FILLER_MICRO_TEXTS:
                return False
        return True

    def _residual_micro_segments(
        self,
        segments: list[FinalTimelineSegment],
    ) -> list[tuple[int, FinalTimelineSegment]]:
        return [(index, segment) for index, segment in enumerate(segments) if self._is_hard_residual(segment)]

    def _residual_merge_index(
        self,
        segments: list[FinalTimelineSegment],
        index: int,
        word_lookup: dict[str, Any],
        source_windows: list[tuple[int, int]],
    ) -> int | None:
        candidates: list[tuple[int, int]] = []
        if index > 0 and self._can_merge(segments[index - 1], segments[index], word_lookup, source_windows):
            candidates.append((self._merge_score(segments[index - 1], segments[index], word_lookup), index - 1))
        if index + 1 < len(segments) and self._can_merge(segments[index], segments[index + 1], word_lookup, source_windows):
            candidates.append((self._merge_score(segments[index], segments[index + 1], word_lookup), index))
        if not candidates:
            missing: int | None = None
            return missing
        candidates.sort(key=lambda row: row[0])
        return candidates[0][1]

    def _residual_micro_segment_blockers(
        self,
        segments: list[FinalTimelineSegment],
        word_lookup: dict[str, Any],
    ) -> list[Blocker]:
        blockers: list[Blocker] = []
        for index, segment in self._residual_micro_segments(segments):
            blockers.append(
                Blocker(
                    code="ROUGH_CUT_RESIDUAL_MICRO_SEGMENT_UNMERGEABLE",
                    message="final timeline contains a hard rough-cut residual that cannot be safely merged",
                    layer="compiler",
                    severity="write_blocker",
                    context=self._residual_context(segments, index, word_lookup),
                )
            )
        return blockers

    def _residual_context(
        self,
        segments: list[FinalTimelineSegment],
        index: int,
        word_lookup: dict[str, Any],
    ) -> dict[str, Any]:
        segment = segments[index]
        prev_segment = segments[index - 1] if index > 0 else None
        next_segment = segments[index + 1] if index + 1 < len(segments) else None
        prev_gap_us = int(segment.source_start_us) - int(prev_segment.source_end_us) if prev_segment is not None else None
        next_gap_us = int(next_segment.source_start_us) - int(segment.source_end_us) if next_segment is not None else None
        return {
            "index": index,
            "segment_id": segment.segment_id,
            "text": segment.text,
            "normalized_text": normalize_text(segment.text),
            "word_ids": list(segment.word_ids),
            "decision_ids": list(segment.decision_ids),
            "duration_us": int(segment.target_end_us) - int(segment.target_start_us),
            "duration_ms": (int(segment.target_end_us) - int(segment.target_start_us)) / 1000,
            "source_material_id": segment.source_material_id,
            "source_segment_id": segment.source_segment_id,
            "source_start_us": segment.source_start_us,
            "source_end_us": segment.source_end_us,
            "prev_gap_us": prev_gap_us,
            "next_gap_us": next_gap_us,
            "prev": self._neighbor_context(prev_segment, segment, word_lookup) if prev_segment is not None else None,
            "next": self._neighbor_context(segment, next_segment, word_lookup) if next_segment is not None else None,
            "merge_policy": {
                "whole_segment_only": True,
                "source_time_continuity_required": True,
                "source_gap_merge_limit_us": SOURCE_GAP_MERGE_LIMIT_US,
            },
        }

    def _neighbor_context(
        self,
        left: FinalTimelineSegment,
        right: FinalTimelineSegment,
        word_lookup: dict[str, Any],
    ) -> dict[str, Any]:
        gap_us = int(right.source_start_us) - int(left.source_end_us)
        return {
            "left_segment_id": left.segment_id,
            "right_segment_id": right.segment_id,
            "left_text": left.text,
            "right_text": right.text,
            "left_source_segment_id": left.source_segment_id,
            "right_source_segment_id": right.source_segment_id,
            "left_source_end_us": left.source_end_us,
            "right_source_start_us": right.source_start_us,
            "gap_us": gap_us,
            "can_merge": self._can_merge(left, right, word_lookup, self._window_keys_from_words(word_lookup)),
        }

    def _next_merge_index(
        self,
        segments: list[FinalTimelineSegment],
        word_lookup: dict[str, Any],
        source_windows: list[tuple[int, int]],
    ) -> int | None:
        merge_index: int | None = None
        for index, segment in enumerate(segments):
            if not self._needs_merge(segment):
                continue
            candidates: list[tuple[int, int]] = []
            if index > 0 and self._can_merge(segments[index - 1], segment, word_lookup, source_windows):
                candidates.append((self._merge_score(segments[index - 1], segment, word_lookup), index - 1))
            if index + 1 < len(segments) and self._can_merge(segment, segments[index + 1], word_lookup, source_windows):
                candidates.append((self._merge_score(segment, segments[index + 1], word_lookup), index))
            if candidates:
                candidates.sort(key=lambda row: row[0])
                return candidates[0][1]
        return merge_index

    def _needs_merge(self, segment: FinalTimelineSegment) -> bool:
        return self._is_micro_duration(segment) or self._is_tiny_text(segment)

    def _is_hard_residual(self, segment: FinalTimelineSegment) -> bool:
        return self._is_micro_duration(segment) or len(normalize_text(segment.text)) <= 1

    def _can_merge(
        self,
        left: FinalTimelineSegment,
        right: FinalTimelineSegment,
        word_lookup: dict[str, Any],
        source_windows: list[tuple[int, int]],
    ) -> bool:
        left_window = self._window_for_range(source_windows, int(left.source_start_us), int(left.source_end_us))
        right_window = self._window_for_range(source_windows, int(right.source_start_us), int(right.source_end_us))
        if left_window is not None and right_window is not None and left_window != right_window:
            return False
        if int(right.source_start_us) < int(left.source_end_us):
            return True
        gap = int(right.source_start_us) - int(left.source_end_us)
        if gap > SOURCE_GAP_MERGE_LIMIT_US:
            return False
        left_keys = self._subtitle_keys(left, word_lookup)
        right_keys = self._subtitle_keys(right, word_lookup)
        if left_keys and right_keys and left_keys & right_keys:
            return True
        return (
            self._is_tiny_text(left)
            or self._is_tiny_text(right)
            or self._is_micro_phrase(left)
            or self._is_micro_phrase(right)
        )

    def _window_keys_from_words(self, word_lookup: dict[str, Any]) -> list[tuple[int, int]]:
        windows: set[tuple[int, int]] = set()
        for word in word_lookup.values():
            start = int(getattr(word, "source_start_us", 0) or 0)
            end = int(getattr(word, "source_end_us", 0) or 0)
            if end > start:
                windows.add((start, end))
        return sorted(windows)

    def _merge_score(
        self,
        left: FinalTimelineSegment,
        right: FinalTimelineSegment,
        word_lookup: dict[str, Any],
    ) -> int:
        score = max(0, int(right.source_start_us) - int(left.source_end_us))
        left_keys = self._subtitle_keys(left, word_lookup)
        right_keys = self._subtitle_keys(right, word_lookup)
        if left_keys and right_keys and left_keys & right_keys:
            score -= 1_000_000
        return score

    def _subtitle_keys(self, segment: FinalTimelineSegment, word_lookup: dict[str, Any]) -> set[tuple[str, int | None]]:
        keys: set[tuple[str, int | None]] = set()
        for word_id in segment.word_ids:
            word = word_lookup.get(word_id)
            if word is None:
                continue
            uid = str(word.subtitle_uid or "")
            if uid or word.subtitle_index is not None:
                keys.add((uid, word.subtitle_index))
        return keys

    def _is_micro_duration(self, segment: FinalTimelineSegment) -> bool:
        duration = int(segment.target_end_us) - int(segment.target_start_us)
        return duration < MIN_HARD_SEGMENT_DURATION_US

    def _is_tiny_text(self, segment: FinalTimelineSegment) -> bool:
        return len(normalize_text(segment.text)) <= 2

    def _is_micro_phrase(self, segment: FinalTimelineSegment) -> bool:
        duration = int(segment.target_end_us) - int(segment.target_start_us)
        text_length = len(normalize_text(segment.text))
        return duration < MIN_HARD_SEGMENT_DURATION_US and text_length <= MIN_CAPTION_CHARS

    def _merge_pair(self, left: FinalTimelineSegment, right: FinalTimelineSegment) -> FinalTimelineSegment:
        decision_ids = sorted(set(left.decision_ids + right.decision_ids + ["rough_cut_quality_merge"]))
        return replace(
            left,
            source_start_us=min(left.source_start_us, right.source_start_us),
            source_end_us=max(left.source_end_us, right.source_end_us),
            target_start_us=min(left.target_start_us, right.target_start_us),
            target_end_us=max(left.target_end_us, right.target_end_us),
            word_ids=list(left.word_ids) + list(right.word_ids),
            text=f"{left.text}{right.text}",
            decision_ids=decision_ids,
            spoken_source_start_us=None,
            spoken_source_end_us=None,
            clip_source_start_us=None,
            clip_source_end_us=None,
            lead_handle_us=0,
            tail_handle_us=0,
        )

    def _repack_target_timeline(self, segments: list[FinalTimelineSegment]) -> list[FinalTimelineSegment]:
        repacked: list[FinalTimelineSegment] = []
        cursor = 0
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

    def _apply_source_handles(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
    ) -> list[FinalTimelineSegment]:
        lower, upper = self._source_bounds(source_graph)
        source_windows = self._source_windows(source_graph)
        handled: list[FinalTimelineSegment] = []
        for index, segment in enumerate(segments):
            spoken_start = int(segment.source_start_us)
            spoken_end = int(segment.source_end_us)
            window = self._window_for_range(source_windows, spoken_start, spoken_end)
            window_lower, window_upper = window if window is not None else (lower, upper)
            clip_start = max(window_lower, spoken_start - DEFAULT_LEAD_HANDLE_US)
            clip_end = min(window_upper, spoken_end + DEFAULT_TAIL_HANDLE_US)
            if index > 0:
                previous = handled[-1]
                previous_clip_end = int(previous.clip_source_end_us if previous.clip_source_end_us is not None else previous.source_end_us)
                if clip_start < previous_clip_end:
                    clip_start = spoken_start
            if index + 1 < len(segments):
                next_segment = segments[index + 1]
                if clip_end > int(next_segment.source_start_us):
                    clip_end = spoken_end
            lead = max(0, spoken_start - clip_start)
            tail = max(0, clip_end - spoken_end)
            debug_hints = dict(segment.debug_hints or {})
            debug_hints.update(
                {
                    "safe_handle_policy_enabled": True,
                    "safe_handle_source_window_start_us": int(window_lower),
                    "safe_handle_source_window_end_us": int(window_upper),
                    "safe_handle_requested_lead_us": int(DEFAULT_LEAD_HANDLE_US),
                    "safe_handle_requested_tail_us": int(DEFAULT_TAIL_HANDLE_US),
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

    def _source_bounds(self, source_graph: CanonicalSourceGraph) -> tuple[int, int]:
        starts: list[int] = []
        ends: list[int] = []
        for segment in source_graph.source_segments:
            start, end = self._canonical_segment_bounds(segment)
            if end > start:
                starts.append(start)
                ends.append(end)
        if starts and ends:
            return min(starts), max(ends)
        word_starts = [int(word.source_start_us) for word in source_graph.words]
        word_ends = [int(word.source_end_us) for word in source_graph.words]
        return (min(word_starts, default=0), max(word_ends, default=0))

    def _source_bounds_by_segment(self, source_graph: CanonicalSourceGraph) -> dict[str, tuple[int, int]]:
        bounds: dict[str, tuple[int, int]] = {}
        for segment in source_graph.source_segments:
            segment_id = str(segment.get("id") or segment.get("source_segment_id") or "")
            if not segment_id:
                continue
            start, end = self._canonical_segment_bounds(segment)
            if end <= start:
                duration = self._time_value(segment, "duration_us")
                if duration is not None and duration > 0:
                    end = start + duration
            if end > start:
                bounds[segment_id] = (int(start), int(end))
        return bounds

    def _source_windows(self, source_graph: CanonicalSourceGraph) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        for segment in source_graph.source_segments:
            if "video" not in str(segment.get("track_type") or segment.get("type") or "").lower():
                continue
            start, end = self._canonical_segment_bounds(segment)
            if end > start:
                windows.append((start, end))
        return sorted(set(windows))

    def _window_for_range(self, windows: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
        matches = [(window_start, window_end) for window_start, window_end in windows if window_start <= start and end <= window_end]
        return matches[0] if len(matches) == 1 else None

    def _canonical_segment_bounds(self, segment: dict[str, Any]) -> tuple[int, int]:
        for start_key, end_key in (
            ("canonical_source_start_us", "canonical_source_end_us"),
            ("target_start_us", "target_end_us"),
            ("source_start_us", "source_end_us"),
        ):
            start = self._time_value(segment, start_key)
            end = self._time_value(segment, end_key)
            if start is not None and end is not None and end > start:
                return int(start), int(end)
        target_timerange = segment.get("target_timerange")
        if isinstance(target_timerange, dict):
            start = int(target_timerange.get("start") or 0)
            duration = int(target_timerange.get("duration") or 0)
            end = int(target_timerange.get("end") or (start + duration if duration else 0))
            if end > start:
                return start, end
        source_timerange = segment.get("source_timerange")
        if isinstance(source_timerange, dict):
            start = int(source_timerange.get("start") or 0)
            duration = int(source_timerange.get("duration") or 0)
            end = int(source_timerange.get("end") or (start + duration if duration else 0))
            if end > start:
                return start, end
        return 0, 0

    def _time_value(self, row: dict[str, Any], key: str) -> int | None:
        value = row.get(key)
        if value is None:
            missing: int | None = None
            return missing
        try:
            return int(value)
        except (TypeError, ValueError):
            invalid: int | None = None
            return invalid
