from __future__ import annotations

from dataclasses import replace

from aroll_adjacent_modifier_semantic_redundancy_gate import detect_adjacent_modifier_semantic_redundancy
from aroll_text_normalize import normalize_text
from aroll_v21.compiler.rough_cut_quality_normalizer import RoughCutQualityNormalizer
from aroll_v21.decision.deterministic_baseline_policy import DeterministicBaselinePolicy
from aroll_v21.decision.semantic_decision_planner import FORBIDDEN_DEEPSEEK_FIELDS
from aroll_v21.decision.final_target_repeat_resolver import FinalTargetRepeatResolver
from aroll_v21.ir.models import Blocker, CanonicalSourceGraph, DecisionPlan, FinalTimelineSegment
from aroll_v21.quality.boundary_overlap import (
    boundary_suffix_prefix_overlap,
    is_semantic_label_reuse_boundary,
)


class FinalTimelineCompiler:
    """Single-pass compiler from source graph + semantic decisions to final timeline."""

    def __init__(self, *, rough_cut_normalizer: RoughCutQualityNormalizer | None = None) -> None:
        self.rough_cut_normalizer = rough_cut_normalizer or RoughCutQualityNormalizer()
        self.baseline_policy = DeterministicBaselinePolicy()

    def compile(
        self,
        source_graph: CanonicalSourceGraph,
        decision_plan: DecisionPlan,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker]]:
        blockers: list[Blocker] = []
        if decision_plan.blocked:
            return [], list(decision_plan.blockers)

        units_by_id = {unit.unit_id: unit for unit in source_graph.edit_units}
        drop_unit_ids: set[str] = set()
        dropped_word_ids: set[str] = set()
        decision_ids_by_word: dict[str, list[str]] = {}
        for decision in decision_plan.decisions:
            for unit_id in decision.drop_unit_ids:
                unit = units_by_id.get(unit_id)
                if unit is None:
                    blockers.append(
                        Blocker(
                            code="DECISION_DROP_UNIT_NOT_FOUND",
                            message="decision references a missing edit unit",
                            layer="compiler",
                            context={"decision_id": decision.decision_id, "unit_id": unit_id},
                        )
                    )
                    continue
                if unit.cut_policy == "unsafe":
                    blockers.append(
                        Blocker(
                            code="UNSAFE_EDIT_UNIT_DROP_BLOCKED",
                            message="compiler refuses to drop an unsafe edit unit",
                            layer="compiler",
                            context={"decision_id": decision.decision_id, "unit_id": unit_id},
                        )
                    )
                    continue
                drop_unit_ids.add(unit_id)
                for word_id in unit.word_ids:
                    decision_ids_by_word.setdefault(word_id, []).append(decision.decision_id)

        for split in decision_plan.split_decisions:
            unit = units_by_id.get(split.unit_id)
            if unit is None:
                blockers.append(
                    Blocker(
                        code="UNIT_SPLIT_UNIT_NOT_FOUND",
                        message="split decision references a missing edit unit",
                        layer="compiler",
                        context={"split_id": split.split_id, "unit_id": split.unit_id},
                    )
                )
                continue
            if split.requires_human_review:
                blockers.append(
                    Blocker(
                        code="UNIT_SPLIT_REQUIRES_HUMAN_REVIEW",
                        message="split decision requires human review before compilation",
                        layer="compiler",
                        context={"split_id": split.split_id, "unit_id": split.unit_id},
                    )
                )
                continue
            if unit.cut_policy == "unsafe":
                blockers.append(
                    Blocker(
                        code="UNIT_SPLIT_UNSAFE_BOUNDARY",
                        message="compiler refuses to split an unsafe edit unit",
                        layer="compiler",
                        context={"split_id": split.split_id, "unit_id": split.unit_id},
                    )
                )
                continue
            unit_word_ids = set(unit.word_ids)
            drop_ids = set(split.drop_word_ids)
            keep_ids = set(split.keep_word_ids)
            if not drop_ids or not drop_ids <= unit_word_ids:
                blockers.append(
                    Blocker(
                        code="UNIT_SPLIT_UNKNOWN_WORD",
                        message="split decision references unknown drop word ids",
                        layer="compiler",
                        context={"split_id": split.split_id, "drop_word_ids": split.drop_word_ids},
                    )
                )
                continue
            if not keep_ids or not keep_ids <= unit_word_ids or drop_ids & keep_ids:
                blockers.append(
                    Blocker(
                        code="UNIT_SPLIT_INVALID_KEEP_WORDS",
                        message="split decision has invalid keep word ids",
                        layer="compiler",
                        context={"split_id": split.split_id, "keep_word_ids": split.keep_word_ids},
                    )
                )
                continue
            dropped_word_ids.update(drop_ids)
            for word_id in drop_ids:
                decision_ids_by_word.setdefault(word_id, []).append(split.split_id)
            for word_id in keep_ids:
                decision_ids_by_word.setdefault(word_id, []).append(split.split_id)

        if blockers:
            return [], blockers

        dropped_word_ids.update({word_id for unit_id in drop_unit_ids for word_id in units_by_id[unit_id].word_ids})
        kept_words = [word for word in source_graph.words if word.word_id not in dropped_word_ids]
        source_order_blockers = self._source_order_blockers(kept_words)
        if source_order_blockers:
            return [], source_order_blockers
        word_to_unit_id = {
            word_id: unit.unit_id
            for unit in source_graph.edit_units
            for word_id in unit.word_ids
        }
        kept_words.sort(key=lambda word: (word.source_start_us, word.source_end_us, word.word_id))
        segments: list[FinalTimelineSegment] = []
        target_cursor = 0
        for group in self._groups(kept_words, word_to_unit_id):
            group_blockers = self._group_blockers(group)
            if group_blockers:
                blockers.extend(group_blockers)
                continue
            source_start = group[0].source_start_us
            source_end = group[-1].source_end_us
            duration = source_end - source_start
            if duration <= 0:
                blockers.append(
                    Blocker(
                        code="FINAL_SEGMENT_EMPTY_SOURCE_RANGE",
                        message="compiled final segment has empty source range",
                        layer="compiler",
                        context={"word_ids": [word.word_id for word in group]},
                    )
                )
                continue
            word_ids = [word.word_id for word in group]
            decision_ids = sorted({decision_id for word_id in word_ids for decision_id in decision_ids_by_word.get(word_id, [])})
            segments.append(
                FinalTimelineSegment(
                    segment_id=f"v21_seg_{len(segments) + 1:06d}",
                    source_material_id="",
                    source_segment_id=None,
                    source_start_us=source_start,
                    source_end_us=source_end,
                    target_start_us=target_cursor,
                    target_end_us=target_cursor + duration,
                    word_ids=word_ids,
                    text="".join(word.text for word in group),
                    decision_ids=decision_ids,
                )
            )
            target_cursor += duration
        if not segments and source_graph.words:
            blockers.append(Blocker("FINAL_TIMELINE_EMPTY", "compiler produced no final timeline segments", "compiler"))
        if blockers:
            return segments, blockers
        segments, source_window_blockers = self._split_segments_by_source_windows(segments, source_graph)
        if source_window_blockers:
            return segments, source_window_blockers
        segments, pre_emit_blockers = self._pre_emit_boundary_prefix_normalization(segments, decision_plan)
        segments, final_target_blockers = FinalTargetRepeatResolver().resolve(segments, decision_plan)
        segments, rough_cut_blockers = self.rough_cut_normalizer.normalize(
            segments,
            source_graph,
            decision_plan,
            emit_residual_blockers=False,
        )
        segments, modifier_blockers, modifier_changed = self._final_modifier_redundancy_semantic_pass(
            segments,
            source_graph,
            decision_plan,
        )
        if modifier_changed:
            segments, rough_cut_after_modifier_blockers = self.rough_cut_normalizer.normalize(
                segments,
                source_graph,
                decision_plan,
                emit_residual_blockers=False,
            )
            rough_cut_blockers.extend(rough_cut_after_modifier_blockers)
        segments, adjacent_duplicate_blockers = self._post_normalizer_adjacent_exact_duplicate_cleanup(segments, decision_plan)
        segments, suffix_prefix_blockers = self._final_cjk_boundary_suffix_prefix_overlap_cleanup(
            segments,
            source_graph,
            decision_plan,
        )
        segments, repeated_island_blockers = self._final_repeated_island_suffix_cleanup(
            segments,
            source_graph,
            decision_plan,
        )
        segments, final_rough_cut_blockers = self.rough_cut_normalizer.normalize(
            segments,
            source_graph,
            decision_plan,
        )
        segments, late_final_target_blockers = FinalTargetRepeatResolver().resolve(segments, decision_plan)
        if late_final_target_blockers:
            final_rough_after_late_target_blockers: list[Blocker] = []
        else:
            segments, final_rough_after_late_target_blockers = self.rough_cut_normalizer.normalize(
                segments,
                source_graph,
                decision_plan,
            )
        final_source_window_blockers = self._final_source_window_blockers(segments, source_graph)
        return (
            segments,
            source_window_blockers
            + pre_emit_blockers
            + final_target_blockers
            + rough_cut_blockers
            + modifier_blockers
            + adjacent_duplicate_blockers
            + suffix_prefix_blockers
            + repeated_island_blockers
            + final_rough_cut_blockers
            + late_final_target_blockers
            + final_rough_after_late_target_blockers
            + final_source_window_blockers
        )

    def _split_segments_by_source_windows(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker]]:
        windows = self._source_windows(source_graph)
        if not windows:
            return segments, []
        word_lookup = {word.word_id: word for word in source_graph.words}
        split_segments: list[FinalTimelineSegment] = []
        blockers: list[Blocker] = []
        for segment in segments:
            if self._range_window_count(windows, segment.source_start_us, segment.source_end_us) == 1:
                split_segments.append(segment)
                continue
            grouped: list[list[str]] = []
            current_window: tuple[int, int] | None = None
            for word_id in segment.word_ids:
                word = word_lookup.get(word_id)
                if word is None:
                    continue
                word_window = self._window_for_range(windows, int(word.source_start_us), int(word.source_end_us))
                if word_window is None:
                    grouped = []
                    break
                if current_window != word_window:
                    grouped.append([])
                    current_window = word_window
                grouped[-1].append(word_id)
            if blockers:
                continue
            grouped = [group for group in grouped if group]
            if len(grouped) <= 1:
                blockers.append(
                    Blocker(
                        "V21_FINAL_SEGMENT_CROSSES_PRIMARY_SOURCE_WINDOW",
                        "final segment crosses primary video source windows but cannot be split on word boundaries",
                        "compiler",
                        context={"segment_id": segment.segment_id, "word_ids": segment.word_ids},
                    )
                )
                continue
            for group in grouped:
                words = [word_lookup[word_id] for word_id in group if word_id in word_lookup]
                if not words:
                    continue
                split_segments.append(
                    replace(
                        segment,
                        source_start_us=min(word.source_start_us for word in words),
                        source_end_us=max(word.source_end_us for word in words),
                        target_start_us=0,
                        target_end_us=0,
                        word_ids=list(group),
                        text="".join(word.text for word in words),
                    )
                )
        if blockers:
            return split_segments, blockers
        return self._repack_segments(split_segments), []

    def _final_source_window_blockers(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
    ) -> list[Blocker]:
        windows = self._source_windows(source_graph)
        if not windows:
            return list()
        blockers: list[Blocker] = []
        for segment in segments:
            start = int(segment.clip_source_start_us if segment.clip_source_start_us is not None else segment.source_start_us)
            end = int(segment.clip_source_end_us if segment.clip_source_end_us is not None else segment.source_end_us)
            count = self._range_window_count(windows, start, end)
            if count == 0:
                blockers.append(
                    Blocker(
                        "V21_FINAL_SEGMENT_SOURCE_WINDOW_UNRESOLVED",
                        "final segment clip range is not covered by a primary source window",
                        "compiler",
                        context={"segment_id": segment.segment_id, "clip_source_start_us": start, "clip_source_end_us": end},
                    )
                )
            elif count > 1:
                blockers.append(
                    Blocker(
                        "V21_FINAL_SEGMENT_SOURCE_WINDOW_AMBIGUOUS",
                        "final segment clip range is covered by multiple primary source windows",
                        "compiler",
                        context={"segment_id": segment.segment_id, "clip_source_start_us": start, "clip_source_end_us": end},
                    )
                )
        return blockers

    def _source_windows(self, source_graph: CanonicalSourceGraph) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        for row in source_graph.source_segments:
            if "video" not in str(row.get("track_type") or row.get("type") or "").lower():
                continue
            start = int(row.get("canonical_source_start_us") or row.get("target_start_us") or row.get("source_start_us") or 0)
            end = int(row.get("canonical_source_end_us") or row.get("target_end_us") or row.get("source_end_us") or 0)
            if end > start:
                windows.append((start, end))
        return sorted(set(windows))

    def _window_for_range(self, windows: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
        matches = [window for window in windows if window[0] <= start and end <= window[1]]
        return matches[0] if len(matches) == 1 else None

    def _range_window_count(self, windows: list[tuple[int, int]], start: int, end: int) -> int:
        return sum(1 for window_start, window_end in windows if window_start <= int(start) and int(end) <= window_end)

    def _repack_segments(self, segments: list[FinalTimelineSegment]) -> list[FinalTimelineSegment]:
        cursor = 0
        repacked: list[FinalTimelineSegment] = []
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

    def _pre_emit_boundary_prefix_normalization(
        self,
        segments: list[FinalTimelineSegment],
        decision_plan: DecisionPlan,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker]]:
        if len(segments) < 2:
            return segments, []
        current = list(segments)
        blockers: list[Blocker] = []
        seen_blocker_pairs: set[tuple[str, str]] = set()
        while True:
            dropped_indices: set[int] = set()
            for index, (left, right) in enumerate(zip(current, current[1:])):
                left_text = normalize_text(left.text)
                right_text = normalize_text(right.text)
                if not left_text or left_text == right_text or not right_text.startswith(left_text):
                    continue
                if not self._safe_pre_emit_boundary_prefix_drop(left, right):
                    pair_key = (
                        left.text,
                        right.text,
                        str(left.source_start_us),
                        str(left.source_end_us),
                        str(right.source_start_us),
                        str(right.source_end_us),
                    )
                    if pair_key not in seen_blocker_pairs:
                        seen_blocker_pairs.add(pair_key)
                        blockers.append(
                            Blocker(
                                code="BOUNDARY_PREFIX_CONTAINMENT_REQUIRES_HUMAN_REVIEW",
                                message="final timeline contains prefix containment that is not safe for automatic pre-emit drop",
                                layer="compiler",
                                severity="write_blocker",
                                context={
                                    "left_segment_id": left.segment_id,
                                    "right_segment_id": right.segment_id,
                                    "left_text": left.text,
                                    "right_text": right.text,
                                },
                            )
                        )
                    continue
                dropped_indices.add(index)
                decision_plan.decision_trace.append(
                    {
                        "route": "boundary_prefix_containment",
                        "stage": "final_timeline_pre_emit",
                        "left_text": left.text,
                        "right_text": right.text,
                        "decision": "drop_left_keep_right",
                        "applied": True,
                        "reason": "right segment is strict prefix extension of left segment",
                        "source": "local_policy",
                        "left_segment_id": left.segment_id,
                        "right_segment_id": right.segment_id,
                    }
                )
            if not dropped_indices:
                return current, blockers
            kept = [segment for index, segment in enumerate(current) if index not in dropped_indices]
            current = self._repack_target_timeline(kept)

    def _safe_pre_emit_boundary_prefix_drop(self, left: FinalTimelineSegment, right: FinalTimelineSegment) -> bool:
        left_text = normalize_text(left.text)
        right_text = normalize_text(right.text)
        if not right_text.startswith(left_text) or right_text == left_text:
            return False
        if not left.word_ids or not right.word_ids:
            return False
        return True

    def _post_normalizer_adjacent_exact_duplicate_cleanup(
        self,
        segments: list[FinalTimelineSegment],
        decision_plan: DecisionPlan,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker]]:
        if len(segments) < 2:
            return segments, []
        current = list(segments)
        while True:
            dropped_indices: set[int] = set()
            for index, (left, right) in enumerate(zip(current, current[1:])):
                left_text = normalize_text(left.text)
                right_text = normalize_text(right.text)
                if not left_text or left_text != right_text:
                    continue
                if not left.word_ids or not right.word_ids:
                    continue
                left_duration = int(left.source_end_us) - int(left.source_start_us)
                right_duration = int(right.source_end_us) - int(right.source_start_us)
                drop_index = index if left_duration <= right_duration else index + 1
                dropped = current[drop_index]
                kept = right if drop_index == index else left
                dropped_indices.add(drop_index)
                decision_plan.decision_trace.append(
                    {
                        "route": "adjacent_exact_duplicate_cleanup",
                        "stage": "final_timeline_pre_emit",
                        "left_text": left.text,
                        "right_text": right.text,
                        "decision": "drop_left" if drop_index == index else "drop_right",
                        "applied": True,
                        "dropped_segment_id": dropped.segment_id,
                        "kept_segment_id": kept.segment_id,
                        "reason": "adjacent final segments have identical normalized text",
                        "source": "local_policy",
                    }
                )
                break
            if not dropped_indices:
                return current, []
            current = self._repack_target_timeline([segment for index, segment in enumerate(current) if index not in dropped_indices])

    def _final_repeated_island_suffix_cleanup(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        decision_plan: DecisionPlan,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker]]:
        word_lookup = {word.word_id: word for word in source_graph.words}
        current: list[FinalTimelineSegment] = []
        blockers: list[Blocker] = []
        changed = False
        for segment in segments:
            cleaned_segments, dropped_word_ids, blocker = self._drop_repeated_suffix_islands_by_subtitle(segment, word_lookup)
            if blocker is not None:
                blockers.append(blocker)
                current.append(segment)
                continue
            if dropped_word_ids:
                changed = True
                decision_plan.decision_trace.append(
                    {
                        "route": "hidden_audio_repeat",
                        "stage": "final_timeline_pre_emit",
                        "segment_id": segment.segment_id,
                        "decision": "drop_repeated_suffix_island",
                        "applied": True,
                        "dropped_word_ids": dropped_word_ids,
                        "reason": "drop repeated trailing word island inside one final segment",
                        "source": "local_policy",
                    }
                )
            current.extend(cleaned_segments)
        return (self._repack_target_timeline(current) if changed else current), blockers

    def _drop_repeated_suffix_islands_by_subtitle(
        self,
        segment: FinalTimelineSegment,
        word_lookup: dict[str, object],
    ) -> tuple[list[FinalTimelineSegment], list[str], Blocker | None]:
        words = [word_lookup[word_id] for word_id in segment.word_ids if word_id in word_lookup]
        if len(words) < 3:
            return [segment], [], None
        dropped_word_ids: set[str] = set()
        group: list[object] = []
        group_key: object = object()
        for word in [*words, None]:
            key = (
                getattr(word, "subtitle_index", None),
                getattr(word, "subtitle_uid", None),
            ) if word is not None else object()
            if group and key != group_key:
                tokens = [normalize_text(str(getattr(item, "text", "") or "")) for item in group]
                drop_start = self._repeated_suffix_island_start(tokens)
                if drop_start is not None:
                    dropped_word_ids.update(str(getattr(item, "word_id")) for item in group[drop_start:])
                group = []
            if word is not None:
                group.append(word)
                group_key = key
        if not dropped_word_ids:
            return [segment], [], None
        kept_runs: list[list[object]] = []
        current_run: list[object] = []
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
            return [segment], [], Blocker(
                code="HIDDEN_REPEAT_SUFFIX_CLEANUP_EMPTY_RANGE",
                message="repeated suffix cleanup would drop the entire segment",
                layer="compiler",
                context={"segment_id": segment.segment_id, "word_ids": list(segment.word_ids)},
            )
        cleaned_segments = [
            replace(
                segment,
                source_start_us=int(getattr(run[0], "source_start_us")),
                source_end_us=int(getattr(run[-1], "source_end_us")),
                word_ids=[str(getattr(word, "word_id")) for word in run],
                text="".join(str(getattr(word, "text", "") or "") for word in run),
                decision_ids=sorted(set([*segment.decision_ids, "drop_repeated_suffix_island"])),
                spoken_source_start_us=None,
                spoken_source_end_us=None,
                clip_source_start_us=None,
                clip_source_end_us=None,
                lead_handle_us=0,
                tail_handle_us=0,
            )
            for run in kept_runs
        ]
        return cleaned_segments, [word_id for word_id in segment.word_ids if word_id in dropped_word_ids], None

    def _drop_repeated_suffix_island(
        self,
        segment: FinalTimelineSegment,
        word_lookup: dict[str, object],
    ) -> tuple[FinalTimelineSegment, Blocker | None]:
        words = [word_lookup[word_id] for word_id in segment.word_ids if word_id in word_lookup]
        tokens = [normalize_text(str(getattr(word, "text", "") or "")) for word in words]
        if len(tokens) < 4:
            return segment, None
        drop_start = self._repeated_suffix_island_start(tokens)
        if drop_start is None:
            return segment, None
        kept_words = words[:drop_start]
        dropped_words = words[drop_start:]
        if not kept_words or not dropped_words:
            return segment, None
        kept_text = "".join(str(getattr(word, "text", "") or "") for word in kept_words)
        if not normalize_text(kept_text):
            return segment, None
        new_end = int(getattr(kept_words[-1], "source_end_us"))
        if new_end <= int(segment.source_start_us):
            return segment, Blocker(
                code="HIDDEN_REPEAT_SUFFIX_CLEANUP_EMPTY_RANGE",
                message="repeated suffix cleanup would produce an empty final segment",
                layer="compiler",
                context={"segment_id": segment.segment_id, "word_ids": list(segment.word_ids)},
            )
        return replace(
            segment,
            source_end_us=new_end,
            target_end_us=int(segment.target_start_us) + max(0, new_end - int(segment.source_start_us)),
            word_ids=[str(getattr(word, "word_id")) for word in kept_words],
            text=kept_text,
            decision_ids=sorted(set([*segment.decision_ids, "drop_repeated_suffix_island"])),
            spoken_source_start_us=None,
            spoken_source_end_us=None,
            clip_source_start_us=None,
            clip_source_end_us=None,
            lead_handle_us=0,
            tail_handle_us=0,
        ), None

    def _repeated_suffix_island_start(self, tokens: list[str]) -> int | None:
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
        no_suffix_island = None
        return no_suffix_island

    def _final_cjk_boundary_suffix_prefix_overlap_cleanup(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        decision_plan: DecisionPlan,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker]]:
        if len(segments) < 2:
            return segments, []
        word_lookup = {word.word_id: word for word in source_graph.words}
        current = list(segments)
        blockers: list[Blocker] = []
        while True:
            changed = False
            for index, (left, right) in enumerate(zip(current, current[1:])):
                overlap = self._boundary_suffix_prefix_overlap(left.text, right.text)
                if len(overlap) < 2:
                    continue
                if is_semantic_label_reuse_boundary(left.text, right.text, overlap):
                    continue
                drop_word_ids = self._trailing_word_ids_for_overlap(left, word_lookup, overlap)
                if not drop_word_ids:
                    blockers.append(
                        Blocker(
                            code="BOUNDARY_SUFFIX_PREFIX_OVERLAP_WORD_BINDING_MISSING",
                            message="boundary suffix-prefix overlap could not be bound to trailing whole word ids",
                            layer="compiler",
                            context={
                                "left_segment_id": left.segment_id,
                                "right_segment_id": right.segment_id,
                                "left_text": left.text,
                                "right_text": right.text,
                                "overlap_text": overlap,
                                "left_word_ids": left.word_ids,
                            },
                        )
                    )
                    return current, blockers
                drop_set = set(drop_word_ids)
                kept_word_ids = [word_id for word_id in left.word_ids if word_id not in drop_set]
                left_before = left.text
                if kept_word_ids:
                    kept_words = [word_lookup[word_id] for word_id in kept_word_ids if word_id in word_lookup]
                    if len(kept_words) != len(kept_word_ids):
                        blockers.append(
                            Blocker(
                                code="BOUNDARY_SUFFIX_PREFIX_OVERLAP_WORD_BINDING_MISSING",
                                message="boundary suffix-prefix overlap cleanup lost source word bindings",
                                layer="compiler",
                                context={
                                    "left_segment_id": left.segment_id,
                                    "right_segment_id": right.segment_id,
                                    "overlap_text": overlap,
                                    "kept_word_ids": kept_word_ids,
                                },
                            )
                        )
                        return current, blockers
                    updated_left = replace(
                        left,
                        source_end_us=int(getattr(kept_words[-1], "source_end_us")),
                        word_ids=kept_word_ids,
                        text="".join(str(getattr(word, "text")) for word in kept_words),
                        decision_ids=sorted(set(left.decision_ids + ["final_cjk_boundary_suffix_prefix_overlap_cleanup"])),
                        spoken_source_start_us=None,
                        spoken_source_end_us=None,
                        clip_source_start_us=None,
                        clip_source_end_us=None,
                        lead_handle_us=0,
                        tail_handle_us=0,
                    )
                    current[index] = updated_left
                    left_after = updated_left.text
                else:
                    current = [segment for position, segment in enumerate(current) if position != index]
                    left_after = ""
                decision_plan.decision_trace.append(
                    {
                        "route": "final_cjk_boundary_suffix_prefix_overlap_cleanup",
                        "stage": "final_timeline_pre_emit",
                        "decision": "drop_left_overlap_suffix",
                        "applied": True,
                        "left_segment_id": left.segment_id,
                        "right_segment_id": right.segment_id,
                        "overlap_text": overlap,
                        "dropped_word_ids": drop_word_ids,
                        "left_text_before": left_before,
                        "left_text_after": left_after,
                        "right_text": right.text,
                        "reason": "left suffix repeats right prefix at final subtitle boundary",
                    }
                )
                current = self._repack_target_timeline(current)
                changed = True
                break
            if not changed:
                return current, []

    def _boundary_suffix_prefix_overlap(self, left_text: str, right_text: str) -> str:
        return boundary_suffix_prefix_overlap(left_text, right_text)

    def _trailing_word_ids_for_overlap(
        self,
        segment: FinalTimelineSegment,
        word_lookup: dict[str, object],
        overlap: str,
    ) -> list[str]:
        selected: list[str] = []
        empty_selection: list[str] = []
        selected_text = ""
        for word_id in reversed(segment.word_ids):
            word = word_lookup.get(word_id)
            if word is None:
                return empty_selection
            selected.insert(0, word_id)
            selected_text = normalize_text(str(getattr(word, "text", "") or "")) + selected_text
            if len(selected_text) >= len(overlap):
                break
        return selected if selected_text == overlap else empty_selection

    def _final_modifier_redundancy_semantic_pass(
        self,
        segments: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        decision_plan: DecisionPlan,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker], bool]:
        if not segments:
            return segments, [], False
        word_lookup = {word.word_id: word for word in source_graph.words}
        current = list(segments)
        blockers: list[Blocker] = []
        unresolved_ids = set(decision_plan.modifier_redundancy_unresolved_cluster_ids)
        newly_unresolved_ids: set[str] = set()
        accepted_ids = set(decision_plan.modifier_redundancy_accepted_cluster_ids)
        changed = False
        candidates = self._final_modifier_candidates(current)
        for offset, candidate in enumerate(candidates):
            existing_payload_cluster_id = self._existing_modifier_payload_cluster_id(decision_plan, candidate)
            if existing_payload_cluster_id:
                unresolved_ids.add(existing_payload_cluster_id)
                continue
            cluster_id = self._modifier_cluster_id(decision_plan, offset)
            row = self._semantic_decision_row(decision_plan, cluster_id)
            if row is None:
                if cluster_id not in unresolved_ids:
                    unresolved_ids.add(cluster_id)
                    newly_unresolved_ids.add(cluster_id)
                    self._append_modifier_semantic_request(decision_plan, cluster_id, candidate)
                    decision_plan.blockers.append(
                        Blocker(
                            code="FINAL_MODIFIER_REDUNDANCY_SEMANTIC_DECISION_REQUIRED",
                            message="final modifier redundancy requires explicit semantic decision",
                            layer="decision",
                            severity="write_blocker",
                            context={
                                "cluster_id": cluster_id,
                                "repeat_type": "modifier_redundancy",
                                "type": "single_variant_modifier_redundancy",
                                "allows_dry_run_discovery": True,
                            },
                        )
                    )
                continue
            forbidden = sorted(FORBIDDEN_DEEPSEEK_FIELDS & set(row.keys()))
            if forbidden:
                blockers.append(
                    Blocker(
                        code="SEMANTIC_DECISION_HAS_PHYSICAL_FIELDS",
                        message="semantic decisions json contains forbidden physical timeline/material fields",
                        layer="compiler",
                        context={"cluster_id": cluster_id, "forbidden_fields": forbidden},
                    )
                )
                continue
            decision = str(row.get("decision") or "").strip()
            if decision == "keep_all":
                if cluster_id not in unresolved_ids:
                    unresolved_ids.add(cluster_id)
                    newly_unresolved_ids.add(cluster_id)
                    self._append_modifier_semantic_request(decision_plan, cluster_id, candidate)
                decision_plan.blockers.append(
                    Blocker(
                        code="V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
                        message="fatal modifier redundancy cannot be accepted with keep_all",
                        layer="decision",
                        severity="write_blocker",
                        context={"cluster_id": cluster_id, "repeat_type": "modifier_redundancy", "decision": "keep_all"},
                    )
                )
                decision_plan.decision_trace.append(
                    {
                        "route": "final_modifier_redundancy",
                        "cluster_id": cluster_id,
                        "decision": "keep_all_rejected",
                        "applied": False,
                        "source": "SemanticDecisionsJson",
                        "validator_effect": "fatal_modifier_redundancy_unresolved",
                    }
                )
                continue
            if decision == "requires_human_review" or bool(row.get("requires_human_review")):
                if cluster_id not in unresolved_ids:
                    unresolved_ids.add(cluster_id)
                    newly_unresolved_ids.add(cluster_id)
                    self._append_modifier_semantic_request(decision_plan, cluster_id, candidate)
                    decision_plan.blockers.append(
                        Blocker(
                            code="FINAL_MODIFIER_REDUNDANCY_REQUIRES_HUMAN_REVIEW",
                            message="final modifier redundancy decision requires human review",
                            layer="decision",
                            severity="write_blocker",
                            context={"cluster_id": cluster_id},
                        )
                    )
                continue
            if decision != "drop_redundant_modifier":
                blockers.append(
                    Blocker(
                        code="SEMANTIC_DECISION_SCHEMA_INVALID",
                        message="semantic decisions json uses an unsupported modifier redundancy decision",
                        layer="compiler",
                        context={"cluster_id": cluster_id, "decision": decision},
                    )
                )
                continue
            segment_index = int(candidate.get("segment_index") or 0) - 1
            if not (0 <= segment_index < len(current)):
                blockers.append(
                    Blocker(
                        code="MODIFIER_REDUNDANCY_WORD_BINDING_MISSING",
                        message="modifier redundancy candidate does not map to a final segment",
                        layer="compiler",
                        context={"cluster_id": cluster_id},
                    )
                )
                continue
            updated, binding_blocker = self._drop_redundant_modifier_from_segment(current[segment_index], candidate, word_lookup, cluster_id)
            if binding_blocker:
                blockers.append(binding_blocker)
                continue
            current[segment_index] = updated
            changed = True
            decision_plan.decision_trace.append(
                {
                    "route": "final_modifier_redundancy",
                    "stage": "final_timeline_pre_emit",
                    "cluster_id": cluster_id,
                    "decision": "drop_redundant_modifier",
                    "applied": True,
                    "source": "SemanticDecisionsJson",
                    "reason": str(row.get("reason") or "drop redundant modifier before same head"),
                }
            )
        self._set_plan_list(decision_plan.modifier_redundancy_unresolved_cluster_ids, sorted(unresolved_ids))
        self._set_plan_list(decision_plan.modifier_redundancy_accepted_cluster_ids, sorted(accepted_ids))
        if unresolved_ids:
            object.__setattr__(decision_plan, "semantic_unresolved_count", int(decision_plan.semantic_unresolved_count) + len(newly_unresolved_ids))
            object.__setattr__(decision_plan, "requires_human_review", True)
            object.__setattr__(decision_plan, "write_allowed", False)
            object.__setattr__(decision_plan, "dry_run_continued_for_discovery", True)
        if changed:
            current = self._repack_target_timeline(current)
        return current, blockers, changed

    def _existing_modifier_payload_cluster_id(self, decision_plan: DecisionPlan, candidate: dict[str, object]) -> str:
        candidate_texts = {
            normalize_text(str(candidate.get("text") or "")),
            normalize_text(str(candidate.get("phrase") or "")),
        }
        candidate_texts = {text for text in candidate_texts if text}
        if not candidate_texts:
            return ""
        for payload in decision_plan.semantic_request_payloads:
            if str(payload.get("repeat_type") or "") != "modifier_redundancy":
                continue
            payload_texts = {
                normalize_text(str(payload.get("text") or "")),
                normalize_text(str(payload.get("phrase") or "")),
            }
            for evidence in payload.get("local_evidence") or []:
                if not isinstance(evidence, dict):
                    continue
                metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), dict) else {}
                payload_candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
                payload_texts.add(normalize_text(str(payload_candidate.get("raw_phrase") or "")))
                payload_texts.add(normalize_text(str(payload_candidate.get("phrase") or "")))
            if candidate_texts & {text for text in payload_texts if text}:
                return str(payload.get("cluster_id") or "")
        return ""

    def _final_modifier_candidates(self, segments: list[FinalTimelineSegment]) -> list[dict[str, object]]:
        rows = [
            {
                "fragment_id": segment.segment_id,
                "fragment_text": segment.text,
                "text": segment.text,
            }
            for segment in segments
        ]
        candidates = []
        for candidate in detect_adjacent_modifier_semantic_redundancy(rows):
            if str(candidate.get("severity") or "fatal") != "fatal":
                continue
            if str(candidate.get("scope") or "") != "intra_subtitle":
                continue
            row_index = int(candidate.get("row_index") or 0)
            if not (1 <= row_index <= len(segments)):
                continue
            row = dict(candidate)
            row["segment_index"] = row_index
            row["segment_id"] = segments[row_index - 1].segment_id
            row["word_ids"] = list(segments[row_index - 1].word_ids)
            row["source_start_us"] = int(segments[row_index - 1].source_start_us)
            row["source_end_us"] = int(segments[row_index - 1].source_end_us)
            row["target_start_us"] = int(segments[row_index - 1].target_start_us)
            row["target_end_us"] = int(segments[row_index - 1].target_end_us)
            candidates.append(row)
        return candidates

    def _modifier_cluster_id(self, decision_plan: DecisionPlan, offset: int) -> str:
        expected = f"repeat_{2000 + offset:06d}"
        for row in decision_plan.semantic_decision_rows:
            if str(row.get("cluster_id") or "") == expected:
                return expected
        existing_ids = {
            str(row.get("cluster_id") or "")
            for row in decision_plan.semantic_request_payloads
        }
        existing_ids.update(decision_plan.modifier_redundancy_unresolved_cluster_ids)
        existing_ids.update(decision_plan.modifier_redundancy_accepted_cluster_ids)
        candidate = expected
        while candidate in existing_ids:
            offset += 1
            candidate = f"repeat_{2000 + offset:06d}"
        return candidate

    def _semantic_decision_row(self, decision_plan: DecisionPlan, cluster_id: str) -> dict[str, object] | None:
        missing: dict[str, object] | None = None
        for row in decision_plan.semantic_decision_rows:
            if str(row.get("cluster_id") or "") == cluster_id:
                return row
        if self.baseline_policy.is_enabled(decision_plan):
            return self.baseline_policy.decision_for_missing_cluster(
                cluster_id,
                cluster_type="modifier_redundancy",
                context={
                    "reason": "deterministic baseline refuses fatal modifier redundancy; semantic repair/drop required",
                    "confidence": 0.65,
                },
            )
        return missing

    def _deterministic_baseline_enabled(self, decision_plan: DecisionPlan) -> bool:
        return self.baseline_policy.is_enabled(decision_plan)

    def _append_modifier_semantic_request(
        self,
        decision_plan: DecisionPlan,
        cluster_id: str,
        candidate: dict[str, object],
    ) -> None:
        existing = {str(row.get("cluster_id") or "") for row in decision_plan.semantic_request_payloads}
        if cluster_id in existing:
            return
        left_modifier = normalize_text(str(candidate.get("left_modifier") or ""))
        right_modifier = normalize_text(str(candidate.get("right_modifier") or ""))
        decision_plan.semantic_request_payloads.append(
            {
                "issue_id": cluster_id,
                "cluster_id": cluster_id,
                "issue_type": "modifier_redundancy",
                "severity": "fatal",
                "repeat_type": "modifier_redundancy",
                "type": "single_variant_modifier_redundancy",
                "text": str(candidate.get("text") or candidate.get("phrase") or ""),
                "text_before": str(candidate.get("text") or candidate.get("phrase") or ""),
                "text_after": "",
                "candidate_segment_ids": [str(candidate.get("segment_id") or "")],
                "candidate_caption_ids": [],
                "word_ids": [str(word_id) for word_id in candidate.get("word_ids") or [] if str(word_id)],
                "source_start_us": int(candidate.get("source_start_us") or 0),
                "source_end_us": int(candidate.get("source_end_us") or 0),
                "target_start_us": int(candidate.get("target_start_us") or 0),
                "target_end_us": int(candidate.get("target_end_us") or 0),
                "modifiers": [
                    {"role": "redundant_modifier", "text": f"{left_modifier}的", "position": "left"},
                    {"role": "kept_modifier", "text": f"{right_modifier}的", "position": "right"},
                ],
                "head": normalize_text(str(candidate.get("head_text") or "")),
                "allowed_decisions": [
                    "drop_redundant_modifier",
                    "requires_human_review",
                    "no_decision",
                ],
                "recommended_action": "repair_text",
                "suggested_for_rough_cut": "drop_redundant_modifier",
                "why_local_policy_cannot_decide": "final timeline modifier redundancy requires explicit repair/drop; keep_all is forbidden",
                "local_context": {
                    "candidate": dict(candidate),
                    "final_segment_id": str(candidate.get("segment_id") or ""),
                },
                "local_evidence": [
                    {
                        "evidence_type": "adjacent_modifier_semantic_redundancy",
                        "reason": str(candidate.get("reason") or ""),
                        "metadata": {
                            "candidate": {
                                "type": "single_variant_modifier_redundancy",
                                "raw_phrase": str(candidate.get("phrase") or ""),
                                "modifiers": [
                                    {"role": "redundant_modifier", "text": f"{left_modifier}的", "position": "left"},
                                    {"role": "kept_modifier", "text": f"{right_modifier}的", "position": "right"},
                                ],
                                "head": normalize_text(str(candidate.get("head_text") or "")),
                            }
                        },
                    }
                ],
                "required_decision_schema": {
                    "decision": "drop_redundant_modifier | requires_human_review | no_decision",
                    "reason": "",
                    "confidence": 0.0,
                    "requires_human_review": False,
                },
                "fatal_modifier_redundancy_keep_all_allowed": False,
            }
        )
        decision_plan.decision_trace.append(
            {
                "route": "final_modifier_redundancy",
                "stage": "final_timeline_pre_emit",
                "cluster_id": cluster_id,
                "decision": "semantic_request_emitted",
                "applied": True,
                "reason": "final timeline contains single-variant modifier redundancy requiring explicit decision",
            }
        )

    def _drop_redundant_modifier_from_segment(
        self,
        segment: FinalTimelineSegment,
        candidate: dict[str, object],
        word_lookup: dict[str, object],
        cluster_id: str,
    ) -> tuple[FinalTimelineSegment, Blocker | None]:
        left_modifier = normalize_text(str(candidate.get("left_modifier") or ""))
        redundant_text = f"{left_modifier}的" if left_modifier else ""
        segment_text = normalize_text(segment.text)
        start_char = segment_text.find(redundant_text)
        if not redundant_text or start_char < 0:
            return segment, Blocker(
                code="MODIFIER_REDUNDANCY_WORD_BINDING_MISSING",
                message="could not locate redundant modifier text inside final segment",
                layer="compiler",
                context={"cluster_id": cluster_id, "segment_id": segment.segment_id},
            )
        drop_word_ids = self._word_ids_for_char_span(segment, word_lookup, start_char, start_char + len(redundant_text))
        if not drop_word_ids:
            return segment, Blocker(
                code="MODIFIER_REDUNDANCY_WORD_BINDING_MISSING",
                message="could not bind redundant modifier to whole word ids",
                layer="compiler",
                context={"cluster_id": cluster_id, "segment_id": segment.segment_id},
            )
        drop_set = set(drop_word_ids)
        kept_words = [word_lookup[word_id] for word_id in segment.word_ids if word_id in word_lookup and word_id not in drop_set]
        if not kept_words:
            return segment, Blocker(
                code="MODIFIER_REDUNDANCY_WORD_BINDING_MISSING",
                message="modifier redundancy decision would drop the entire segment",
                layer="compiler",
                context={"cluster_id": cluster_id, "segment_id": segment.segment_id},
            )
        return replace(
            segment,
            source_start_us=int(getattr(kept_words[0], "source_start_us")),
            source_end_us=int(getattr(kept_words[-1], "source_end_us")),
            word_ids=[str(getattr(word, "word_id")) for word in kept_words],
            text="".join(str(getattr(word, "text")) for word in kept_words),
            decision_ids=sorted(set(segment.decision_ids + [cluster_id, "drop_redundant_modifier"])),
            spoken_source_start_us=None,
            spoken_source_end_us=None,
            clip_source_start_us=None,
            clip_source_end_us=None,
            lead_handle_us=0,
            tail_handle_us=0,
        ), None

    def _word_ids_for_char_span(
        self,
        segment: FinalTimelineSegment,
        word_lookup: dict[str, object],
        start_char: int,
        end_char: int,
    ) -> list[str]:
        cursor = 0
        selected: list[str] = []
        for word_id in segment.word_ids:
            word = word_lookup.get(word_id)
            text = normalize_text(str(getattr(word, "text", "") or ""))
            if not text:
                continue
            word_start = cursor
            word_end = cursor + len(text)
            if start_char <= word_start and word_end <= end_char:
                selected.append(word_id)
            elif word_start < end_char and start_char < word_end:
                partial_overlap: list[str] = []
                return partial_overlap
            cursor = word_end
        return selected

    def _set_plan_list(self, target: list[str], values: list[str]) -> None:
        target.clear()
        target.extend(values)

    def _repack_target_timeline(self, segments: list[FinalTimelineSegment]) -> list[FinalTimelineSegment]:
        repacked: list[FinalTimelineSegment] = []
        target_cursor = 0
        for index, segment in enumerate(segments, start=1):
            duration = segment.source_end_us - segment.source_start_us
            repacked.append(
                replace(
                    segment,
                    segment_id=f"v21_seg_{index:06d}",
                    target_start_us=target_cursor,
                    target_end_us=target_cursor + duration,
                )
            )
            target_cursor += duration
        return repacked

    def _groups(self, words, word_to_unit_id: dict[str, str]):
        current = []
        last_end = None
        last_unit = None
        for word in words:
            unit_id = word_to_unit_id.get(word.word_id)
            if (
                current
                and (
                    unit_id != last_unit
                    or (last_end is not None and word.source_start_us > last_end + 80_000)
                )
            ):
                yield current
                current = []
            current.append(word)
            last_end = word.source_end_us
            last_unit = unit_id
        if current:
            yield current

    def _source_order_blockers(self, words) -> list[Blocker]:
        blockers: list[Blocker] = []
        ordered = [word for word in words if word.subtitle_index is not None]
        ordered.sort(key=lambda word: (int(word.subtitle_index or 0), word.source_start_us, word.word_id))
        previous: tuple[int, str] | None = None
        for word in ordered:
            if previous is not None and word.source_start_us < previous[0]:
                blockers.append(
                    Blocker(
                        "FINAL_TIMELINE_SEGMENT_UNSAFE_WORD_ORDER",
                        "word source times are not monotonic in subtitle order",
                        "compiler",
                        context={
                            "word_id": word.word_id,
                            "prev_word_id": previous[1],
                            "subtitle_index": word.subtitle_index,
                            "source_start_us": word.source_start_us,
                            "previous_source_start_us": previous[0],
                        },
                    )
                )
                break
            previous = (word.source_start_us, word.word_id)
        return blockers

    def _group_blockers(self, group) -> list[Blocker]:
        blockers: list[Blocker] = []
        subtitle_indices = [int(word.subtitle_index) for word in group if word.subtitle_index is not None]
        unique_indices = sorted(set(subtitle_indices))
        text = "".join(word.text for word in group)
        if len(group) > 20:
            blockers.append(
                Blocker(
                    "FINAL_TIMELINE_SEGMENT_OVERSIZED_WORD_COUNT",
                    "compiled segment contains too many words to be a safe edit unit",
                    "compiler",
                    context={"word_count": len(group), "word_ids": [word.word_id for word in group[:30]]},
                )
            )
        if len(text) > 60:
            blockers.append(
                Blocker(
                    "FINAL_TIMELINE_SEGMENT_OVERSIZED_TEXT",
                    "compiled segment text is too long to be a safe edit unit",
                    "compiler",
                    context={"text_length": len(text), "text": text[:80]},
                )
            )
        if unique_indices:
            span = unique_indices[-1] - unique_indices[0]
            contiguous = unique_indices == list(range(unique_indices[0], unique_indices[-1] + 1))
            if span > 3 or (not contiguous and len(unique_indices) > 2):
                blockers.append(
                    Blocker(
                        "FINAL_TIMELINE_SEGMENT_MIXED_SUBTITLE_INDICES",
                        "compiled segment mixes too many unrelated subtitle indices",
                        "compiler",
                        context={"subtitle_indices": unique_indices[:30], "subtitle_index_span": span},
                    )
                )
        last_end = None
        for word in group:
            if last_end is not None and word.source_start_us < last_end:
                blockers.append(
                    Blocker(
                        "FINAL_TIMELINE_SEGMENT_UNSAFE_WORD_ORDER",
                        "compiled segment word source ranges overlap or go backwards",
                        "compiler",
                        context={"word_id": word.word_id, "source_start_us": word.source_start_us, "previous_source_end_us": last_end},
                    )
                )
                break
            last_end = word.source_end_us
        return blockers
