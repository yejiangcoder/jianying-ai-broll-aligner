from __future__ import annotations

from typing import Any


def configure_rule_dependencies(dependencies: dict[str, Any]) -> None:
    globals().update(dependencies)


CONNECTOR_INTRUSION_NEXT_WORDS = ("所以", "但是", "然后", "因为", "就是", "其实")


MIN_CONNECTOR_INTRUSION_SIDE_GAP_US = 180_000


MAX_CONNECTOR_INTRUSION_WORD_DURATION_US = 450_000


MIN_CONNECTOR_INTRUSION_REMAINING_CHARS = 6


CONNECTOR_RESTART_WORDS = ("但", "但是", "可", "可是", "那", "那么", "然后", "所以", "因为", "如果", "就是")


CONNECTOR_RESTART_INTRUSION_WORDS = ("哪", "那", "啊", "呀", "呃", "嗯", "诶", "哎", "唉", "额")


MAX_CONNECTOR_RESTART_INTRUSION_DURATION_US = 450_000


MIN_CONNECTOR_RESTART_REMAINING_CHARS = 6


MIN_REPEATED_OBJECT_HEAD_GAP_US = 120_000


MIN_REPEATED_OBJECT_REMAINING_CHARS = 6


COMPLETED_PREDICATE_PREFIXES = ("全是", "都是", "全都是", "全部是", "尽是", "就是")


MIN_COMPLETED_PREDICATE_RESTART_PREFIX_CHARS = 3


MIN_COMPLETED_PREDICATE_RESTART_SHARED_CHARS = 2


def _repair_connector_single_word_intrusion(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for segment in _ordered_segments(final_timeline):
        if len(segment.word_ids) < 4:
            continue
        for index in range(1, len(segment.word_ids) - 1):
            previous = words_by_id.get(segment.word_ids[index - 1])
            current = words_by_id.get(segment.word_ids[index])
            next_word = words_by_id.get(segment.word_ids[index + 1])
            if previous is None or current is None or next_word is None:
                continue
            current_text = normalize_text(str(getattr(current, "text", "") or ""))
            next_text = normalize_text(str(getattr(next_word, "text", "") or ""))
            if len(current_text) != 1 or next_text not in CONNECTOR_INTRUSION_NEXT_WORDS:
                continue
            current_duration_us = int(getattr(current, "source_end_us", 0) or 0) - int(getattr(current, "source_start_us", 0) or 0)
            if current_duration_us <= 0 or current_duration_us > MAX_CONNECTOR_INTRUSION_WORD_DURATION_US:
                continue
            left_gap_us = int(getattr(current, "source_start_us", 0) or 0) - int(getattr(previous, "source_end_us", 0) or 0)
            right_gap_us = int(getattr(next_word, "source_start_us", 0) or 0) - int(getattr(current, "source_end_us", 0) or 0)
            if left_gap_us < MIN_CONNECTOR_INTRUSION_SIDE_GAP_US or right_gap_us < MIN_CONNECTOR_INTRUSION_SIDE_GAP_US:
                continue
            remaining = [word_id for pos, word_id in enumerate(segment.word_ids) if pos != index]
            remaining_text = normalize_text(_text_from_word_ids(remaining, source_graph))
            if len(remaining_text) < MIN_CONNECTOR_INTRUSION_REMAINING_CHARS:
                continue
            repaired = _drop_contiguous_word_ids_from_timeline(
                final_timeline,
                source_graph,
                [segment.word_ids[index]],
                "connector_single_word_intrusion",
            )
            if repaired is None:
                continue
            return _RepairStep(
                final_timeline=repaired,
                captions=[],
                timeline_changed=True,
                action=_action(
                    "connector_single_word_intrusion",
                    "trim_single_word_intrusion_before_connector",
                    pass_index,
                    {
                        "caption_id": "",
                        "related_caption_id": "",
                        "reason": "single isolated word before a discourse connector is likely ASR intrusion",
                        "overlap_text": current_text + next_text,
                    },
                    affected_segment_id=segment.segment_id,
                    dropped_word_ids=[segment.word_ids[index]],
                    dropped_text=str(getattr(current, "text", "") or ""),
                    connector_text=str(getattr(next_word, "text", "") or ""),
                    left_gap_us=left_gap_us,
                    right_gap_us=right_gap_us,
                ),
            )
    no_step: _RepairStep | None = None
    return no_step


def _repair_connector_filler_restart(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for segment in _ordered_segments(final_timeline):
        if len(segment.word_ids) < 4:
            continue
        for index in range(0, len(segment.word_ids) - 2):
            first = words_by_id.get(segment.word_ids[index])
            filler = words_by_id.get(segment.word_ids[index + 1])
            restart = words_by_id.get(segment.word_ids[index + 2])
            if first is None or filler is None or restart is None:
                continue
            first_text = normalize_text(str(getattr(first, "text", "") or ""))
            filler_text = normalize_text(str(getattr(filler, "text", "") or ""))
            restart_text = normalize_text(str(getattr(restart, "text", "") or ""))
            if first_text not in CONNECTOR_RESTART_WORDS or restart_text != first_text:
                continue
            if filler_text not in CONNECTOR_RESTART_INTRUSION_WORDS:
                continue
            filler_duration_us = int(getattr(filler, "source_end_us", 0) or 0) - int(getattr(filler, "source_start_us", 0) or 0)
            if filler_duration_us <= 0 or filler_duration_us > MAX_CONNECTOR_RESTART_INTRUSION_DURATION_US:
                continue
            remaining_word_ids = [word_id for pos, word_id in enumerate(segment.word_ids) if pos not in {index, index + 1}]
            remaining_text = normalize_text(_text_from_word_ids(remaining_word_ids, source_graph))
            if len(remaining_text) < MIN_CONNECTOR_RESTART_REMAINING_CHARS:
                continue
            drop_word_ids = [segment.word_ids[index], segment.word_ids[index + 1]]
            repaired = _drop_contiguous_word_ids_from_timeline(
                final_timeline,
                source_graph,
                drop_word_ids,
                "connector_filler_restart",
            )
            if repaired is None:
                continue
            return _RepairStep(
                final_timeline=repaired,
                captions=[],
                timeline_changed=True,
                action=_action(
                    "connector_filler_restart",
                    "trim_connector_filler_before_restart",
                    pass_index,
                    {
                        "caption_id": "",
                        "related_caption_id": "",
                        "reason": "discourse connector is restarted after a short filler intrusion",
                        "overlap_text": f"{first_text}{filler_text}{restart_text}",
                    },
                    affected_segment_id=segment.segment_id,
                    dropped_word_ids=drop_word_ids,
                    dropped_text=f"{first_text}{filler_text}",
                    restart_word_id=segment.word_ids[index + 2],
                    restart_text=restart_text,
                    filler_duration_us=filler_duration_us,
                ),
            )
    no_step: _RepairStep | None = None
    return no_step


def _repair_subject_prefix_completed_predicate_restart(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    ordered = _ordered_segments(final_timeline)
    for left, right in zip(ordered, ordered[1:]):
        candidate = _subject_prefix_completed_predicate_restart_candidate(left, right, words_by_id)
        if candidate is None:
            continue
        repaired = _drop_contiguous_word_ids_from_timeline(
            final_timeline,
            source_graph,
            list(candidate["drop_word_ids"]),
            "subject_prefix_completed_predicate_restart",
        )
        if repaired is None:
            continue
        return _RepairStep(
            final_timeline=repaired,
            captions=[],
            timeline_changed=True,
            action=_action(
                "subject_prefix_completed_predicate_restart",
                "trim_abandoned_predicate_after_subject_prefix",
                pass_index,
                {
                    "caption_id": "",
                    "related_caption_id": "",
                    "reason": "subject prefix is followed by an abandoned predicate before the next segment completes it",
                    "overlap_text": str(candidate["shared_text"]),
                },
                affected_segment_ids=[left.segment_id, right.segment_id],
                preserved_prefix_word_ids=list(candidate["prefix_word_ids"]),
                preserved_prefix_text=str(candidate["prefix_text"]),
                dropped_word_ids=list(candidate["drop_word_ids"]),
                dropped_text=str(candidate["drop_text"]),
                completed_predicate_segment_id=right.segment_id,
                completed_predicate_prefix=str(candidate["completed_prefix"]),
            ),
        )
    no_step: _RepairStep | None = None
    return no_step


def _subject_prefix_completed_predicate_restart_candidate(
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    words_by_id: dict[str, Any],
) -> dict[str, Any] | None:
    left_ids = [str(word_id) for word_id in list(left.word_ids or []) if str(word_id)]
    right_ids = [str(word_id) for word_id in list(right.word_ids or []) if str(word_id)]
    if len(left_ids) < 4 or len(right_ids) < 3:
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    left_words = [words_by_id.get(word_id) for word_id in left_ids]
    right_words = [words_by_id.get(word_id) for word_id in right_ids]
    if any(word is None for word in [*left_words, *right_words]):
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    left_texts = [normalize_text(str(getattr(word, "text", "") or "")) for word in left_words]
    right_texts = [normalize_text(str(getattr(word, "text", "") or "")) for word in right_words]
    completed_prefix_len = _completed_predicate_prefix_word_count(right_texts)
    if completed_prefix_len <= 0:
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    right_tail = normalize_text("".join(right_texts[completed_prefix_len:]))
    if len(right_tail) < MIN_COMPLETED_PREDICATE_RESTART_SHARED_CHARS + 1:
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    for marker_index in range(len(left_texts) - 1):
        marker = left_texts[marker_index]
        if marker != "是":
            continue
        prefix_text = normalize_text("".join(left_texts[:marker_index]))
        left_tail = normalize_text("".join(left_texts[marker_index + 1 :]))
        if len(prefix_text) < MIN_COMPLETED_PREDICATE_RESTART_PREFIX_CHARS:
            continue
        if len(left_tail) < MIN_COMPLETED_PREDICATE_RESTART_SHARED_CHARS:
            continue
        if not right_tail.startswith(left_tail):
            continue
        return {
            "prefix_word_ids": left_ids[:marker_index],
            "prefix_text": prefix_text,
            "drop_word_ids": left_ids[marker_index:],
            "drop_text": normalize_text("".join(left_texts[marker_index:])),
            "shared_text": left_tail,
            "completed_prefix": normalize_text("".join(right_texts[:completed_prefix_len])),
        }
    no_candidate: dict[str, Any] | None = None
    return no_candidate


def _completed_predicate_prefix_word_count(texts: list[str]) -> int:
    joined = ""
    max_count = min(4, len(texts))
    for count in range(1, max_count + 1):
        joined += normalize_text(texts[count - 1])
        if joined in COMPLETED_PREDICATE_PREFIXES:
            return count
    return 0


def _repair_repeated_object_head_tail(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for segment in _ordered_segments(final_timeline):
        if len(segment.word_ids) < 5:
            continue
        first = words_by_id.get(segment.word_ids[0])
        second = words_by_id.get(segment.word_ids[1])
        last = words_by_id.get(segment.word_ids[-1])
        if first is None or second is None or last is None:
            continue
        first_text = normalize_text(str(getattr(first, "text", "") or ""))
        last_text = normalize_text(str(getattr(last, "text", "") or ""))
        if not (2 <= len(first_text) <= 4) or len(last_text) != 1:
            continue
        if not first_text.startswith(last_text):
            continue
        text = normalize_text(str(segment.text or ""))
        if not text.endswith(last_text) or f"的{last_text}" not in text:
            continue
        source_gap_us = int(getattr(second, "source_start_us", 0) or 0) - int(getattr(first, "source_end_us", 0) or 0)
        if source_gap_us < MIN_REPEATED_OBJECT_HEAD_GAP_US:
            continue
        first_subtitle_uid = str(getattr(first, "subtitle_uid", "") or "")
        second_subtitle_uid = str(getattr(second, "subtitle_uid", "") or "")
        if first_subtitle_uid and second_subtitle_uid and first_subtitle_uid == second_subtitle_uid:
            continue
        remaining_word_ids = list(segment.word_ids[1:])
        remaining_text = normalize_text(_text_from_word_ids(remaining_word_ids, source_graph))
        if len(remaining_text) < MIN_REPEATED_OBJECT_REMAINING_CHARS:
            continue
        repaired = _trim_word_ids_from_timeline(final_timeline, source_graph, [segment.word_ids[0]])
        if repaired is None:
            continue
        return _RepairStep(
            final_timeline=repaired,
            captions=[],
            timeline_changed=True,
            action=_action(
                "leading_object_head_repeated_as_tail",
                "trim_repeated_object_head",
                pass_index,
                {
                    "caption_id": "",
                    "related_caption_id": "",
                    "reason": "leading object label repeats as the final syntactic head",
                    "overlap_text": last_text,
                },
                affected_segment_id=segment.segment_id,
                dropped_word_ids=[segment.word_ids[0]],
                dropped_text=str(getattr(first, "text", "") or ""),
                tail_text=str(getattr(last, "text", "") or ""),
                source_gap_us=source_gap_us,
            ),
        )
    no_step: _RepairStep | None = None
    return no_step
