from __future__ import annotations

from typing import Any

from aroll_v21.quality.repeated_suffix_island import is_coordinated_parallel_suffix_repetition

PREDICATE_CONTINUATION_STARTS = ("就", "会", "是", "永远", "可以", "能", "要", "应该", "必须", "只能", "才", "都", "也", "还")


def configure_compiler_dependencies(dependencies: dict[str, Any]) -> None:
    globals().update(dependencies)


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
    for index, segment in enumerate(segments):
        next_word_after_segment = _first_segment_word(segments[index + 1], word_lookup) if index + 1 < len(segments) else None
        cleaned_segments, dropped_word_ids, blocker = self._drop_repeated_suffix_islands_by_subtitle(
            segment,
            word_lookup,
            next_word_after_segment=next_word_after_segment,
        )
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
    *,
    next_word_after_segment: object | None = None,
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
            boundary_next_word = word if word is not None else next_word_after_segment
            next_token = normalize_text(str(getattr(boundary_next_word, "text", "") or "")) if boundary_next_word is not None else ""
            drop_start = self._repeated_suffix_island_start(tokens, next_token=next_token)
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


def _first_segment_word(segment: FinalTimelineSegment, word_lookup: dict[str, object]) -> object | None:
    for word_id in list(segment.word_ids or []):
        if word_id in word_lookup:
            return word_lookup[word_id]
    return None


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


def _repeated_suffix_island_start(self, tokens: list[str], next_token: str = "") -> int | None:
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
                    if _repeated_suffix_is_boundary_subject(tokens, len(tokens) - 1, next_token):
                        continue
                    return len(tokens) - 1
    no_suffix_island = None
    return no_suffix_island


def _repeated_suffix_is_boundary_subject(tokens: list[str], suffix_start: int, next_token: str) -> bool:
    if suffix_start != len(tokens) - 1:
        return False
    suffix = normalize_text(tokens[suffix_start])
    if not _looks_like_nominal_subject(suffix):
        return False
    following = normalize_text(next_token)
    if not following:
        return False
    return any(following.startswith(prefix) or prefix.startswith(following) for prefix in PREDICATE_CONTINUATION_STARTS)


def _looks_like_nominal_subject(text: str) -> bool:
    if not (2 <= len(text) <= 6):
        return False
    if not all("\u4e00" <= char <= "\u9fff" for char in text):
        return False
    if text.endswith(("的", "地", "得", "了", "着", "过", "是", "就")):
        return False
    return True
