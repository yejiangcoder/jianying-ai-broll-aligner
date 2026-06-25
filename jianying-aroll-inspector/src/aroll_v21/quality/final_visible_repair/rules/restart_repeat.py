from __future__ import annotations

from typing import Any


def configure_rule_dependencies(dependencies: dict[str, Any]) -> None:
    globals().update(dependencies)


def _repair_same_segment_de_duplicate_prefix(
    final_timeline: list[FinalTimelineSegment],
    current: CaptionRenderUnit,
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    if normalize_text(str(candidate.get("reason") or "")) != "dangling_de_prefix":
        no_step: _RepairStep | None = None
        return no_step
    current_ids = _caption_segment_ids(current)
    if len(current_ids) != 1:
        no_step: _RepairStep | None = None
        return no_step
    segment = next((row for row in final_timeline if row.segment_id == current_ids[0]), None)
    if segment is None:
        no_step: _RepairStep | None = None
        return no_step
    segment_word_ids = list(segment.word_ids)
    caption_word_ids = list(current.word_ids)
    if not caption_word_ids or not _is_suffix(segment_word_ids, caption_word_ids):
        no_step: _RepairStep | None = None
        return no_step
    prefix_word_ids = segment_word_ids[: len(segment_word_ids) - len(caption_word_ids)]
    if not prefix_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    words_by_id = {word.word_id: word for word in source_graph.words}
    first_word = words_by_id.get(caption_word_ids[0])
    if normalize_text(str(getattr(first_word, "text", "") or "")) != "的":
        no_step: _RepairStep | None = None
        return no_step
    after_de_word_ids = caption_word_ids[1:]
    duplicate_len = _leading_duplicate_word_count(prefix_word_ids, after_de_word_ids, source_graph)
    if duplicate_len <= 0:
        no_step: _RepairStep | None = None
        return no_step
    remaining_caption_word_ids = after_de_word_ids[duplicate_len:]
    if not remaining_caption_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    repaired_word_ids = [*prefix_word_ids, *remaining_caption_word_ids]
    repaired_segments = _segments_with_word_ids_preserving_effective_speed(
        segment,
        repaired_word_ids,
        source_graph,
        "same_segment_de_duplicate_prefix_trim",
        existing_segment_ids={row.segment_id for row in final_timeline},
    )
    if repaired_segments is None:
        no_step: _RepairStep | None = None
        return no_step
    repaired: list[FinalTimelineSegment] = []
    for row in final_timeline:
        if row.segment_id == segment.segment_id:
            repaired.extend(repaired_segments)
            continue
        repaired.append(row)
    dropped_word_ids = [caption_word_ids[0], *after_de_word_ids[:duplicate_len]]
    return _RepairStep(
        final_timeline=repaired,
        captions=[],
        timeline_changed=True,
        action=_action(
            "dangling_prefix_suffix",
            "trim_same_segment_de_duplicate_prefix",
            pass_index,
            candidate,
            affected_caption_ids=[current.caption_id],
            trimmed_segment_id=segment.segment_id,
            materialized_segment_ids=[row.segment_id for row in repaired_segments],
            dropped_word_ids=dropped_word_ids,
            duplicate_prefix_text=_text_from_word_ids(after_de_word_ids[:duplicate_len], source_graph),
            remaining_word_ids=repaired_word_ids,
        ),
    )


def _leading_duplicate_word_count(
    prefix_word_ids: list[str],
    after_de_word_ids: list[str],
    source_graph: CanonicalSourceGraph,
) -> int:
    max_len = min(len(prefix_word_ids), len(after_de_word_ids))
    for count in range(max_len, 0, -1):
        left_text = normalize_text(_text_from_word_ids(prefix_word_ids[-count:], source_graph))
        right_text = normalize_text(_text_from_word_ids(after_de_word_ids[:count], source_graph))
        if left_text and left_text == right_text:
            return count
    return 0


def _repair_dangling_pronoun_modal_suffix(
    final_timeline: list[FinalTimelineSegment],
    caption: CaptionRenderUnit,
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    if str(candidate.get("reason") or "") != "dangling_pronoun_modal_suffix":
        no_step: _RepairStep | None = None
        return no_step
    suffix = _dangling_pronoun_modal_suffix(normalize_text(caption.text))
    if not suffix:
        no_step: _RepairStep | None = None
        return no_step
    drop_word_ids = _trailing_word_ids_for_text(caption.word_ids, source_graph, suffix)
    if not drop_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    if len(drop_word_ids) >= len(caption.word_ids):
        no_step: _RepairStep | None = None
        return no_step
    repaired_timeline = _trim_word_ids_from_timeline(final_timeline, source_graph, drop_word_ids)
    if repaired_timeline is None:
        no_step: _RepairStep | None = None
        return no_step
    return _RepairStep(
        final_timeline=repaired_timeline,
        captions=[],
        timeline_changed=True,
        action=_action(
            "dangling_prefix_suffix",
            "trim_dangling_suffix_tail",
            pass_index,
            candidate,
            affected_caption_ids=[caption.caption_id],
            dropped_word_ids=drop_word_ids,
            drop_text=suffix,
        ),
    )


def _trim_asr_restart_prefix(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    caption = _caption_by_id(captions, str(candidate.get("caption_id") or ""))
    if caption is None:
        no_step: _RepairStep | None = None
        return no_step
    repeated_prefix = normalize_text(str(candidate.get("repeated_prefix") or ""))
    if not repeated_prefix:
        no_step: _RepairStep | None = None
        return no_step
    drop_text = f"{repeated_prefix}就"
    drop_word_ids = _leading_word_ids_for_text(caption.word_ids, source_graph, drop_text)
    if not drop_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    repaired_timeline = _trim_word_ids_from_timeline(final_timeline, source_graph, drop_word_ids)
    if repaired_timeline is None:
        no_step: _RepairStep | None = None
        return no_step
    return _RepairStep(
        final_timeline=repaired_timeline,
        captions=captions,
        timeline_changed=True,
        action=_action(
            "semantic_garbage_or_asr_suspect",
            "trim_repeated_prefix",
            pass_index,
            candidate,
            affected_caption_ids=[caption.caption_id],
            dropped_word_ids=drop_word_ids,
            recheck_decision="trim_repeated_prefix",
        ),
    )


def _trim_restart_repeat_visible_prefix(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    drop_text = normalize_text(str(candidate.get("drop_text") or ""))
    if not drop_text:
        no_step: _RepairStep | None = None
        return no_step
    caption = _caption_by_id(captions, str(candidate.get("caption_id") or ""))
    if caption is None:
        no_step: _RepairStep | None = None
        return no_step
    drop_word_ids = _leading_word_ids_for_text(caption.word_ids, source_graph, drop_text)
    if not drop_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    repaired_timeline = _trim_word_ids_from_timeline(final_timeline, source_graph, drop_word_ids)
    if repaired_timeline is None:
        repaired_timeline = _drop_contiguous_word_ids_from_timeline(
            final_timeline,
            source_graph,
            drop_word_ids,
            "restart_repeat_visible_prefix_trim",
        )
    if repaired_timeline is None:
        no_step: _RepairStep | None = None
        return no_step
    return _RepairStep(
        final_timeline=repaired_timeline,
        captions=captions,
        timeline_changed=True,
        action=_action(
            "restart_repeat_visible",
            "trim_restart_prefix",
            pass_index,
            candidate,
            affected_caption_ids=[caption.caption_id],
            dropped_word_ids=drop_word_ids,
            drop_text=drop_text,
        ),
    )


def _drop_restart_repeat_word_span(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    pattern = str(candidate.get("pattern") or "")
    repair_reason_by_pattern = {
        "internal_prefix_restart": "internal_prefix_restart_repair",
        "abandoned_clause_restart": "abandoned_clause_restart_repair",
        "negative_predicate_restart": "negative_predicate_restart_repair",
        "partial_phrase_restart": "partial_phrase_restart_repair",
        "partial_phrase_restart_tail_mismatch": "partial_phrase_restart_repair",
    }
    repair_reason = repair_reason_by_pattern.get(pattern)
    if repair_reason is None:
        no_step: _RepairStep | None = None
        return no_step
    drop_text = normalize_text(str(candidate.get("drop_text") or ""))
    if not drop_text:
        no_step: _RepairStep | None = None
        return no_step
    if pattern == "internal_prefix_restart" and normalize_text(str(candidate.get("text") or "")).startswith(drop_text):
        no_step: _RepairStep | None = None
        return no_step
    window_captions = _candidate_window_captions(captions, candidate)
    if not window_captions:
        no_step: _RepairStep | None = None
        return no_step
    word_ids = [word_id for caption in window_captions for word_id in caption.word_ids]
    drop_word_ids = _contiguous_word_ids_for_text(word_ids, source_graph, drop_text)
    if not drop_word_ids:
        no_step: _RepairStep | None = None
        return no_step
    repaired_timeline = _drop_contiguous_word_ids_from_timeline(
        final_timeline,
        source_graph,
        drop_word_ids,
        repair_reason,
    )
    if repaired_timeline is None:
        no_step: _RepairStep | None = None
        return no_step
    return _RepairStep(
        final_timeline=repaired_timeline,
        captions=captions,
        timeline_changed=True,
        action=_action(
            "restart_repeat_visible",
            _restart_repeat_drop_decision(pattern),
            pass_index,
            candidate,
            affected_caption_ids=[caption.caption_id for caption in window_captions],
            dropped_word_ids=drop_word_ids,
            drop_text=drop_text,
        ),
    )


def _restart_repeat_drop_decision(pattern: str) -> str:
    if pattern == "negative_predicate_restart":
        return "drop_negative_predicate_restart_span"
    if pattern == "abandoned_clause_restart":
        return "drop_abandoned_clause_restart_span"
    if pattern == "internal_prefix_restart":
        return "drop_internal_prefix_restart_span"
    return "drop_partial_phrase_restart_span"


def _partial_previous_tail_match(
    visible: CaptionRenderUnit,
    source_captions: list[CaptionRenderUnit],
) -> tuple[list[str], list[str], str, str] | None:
    first = source_captions[0]
    later_word_ids = [word_id for caption in source_captions[1:] for word_id in caption.word_ids]
    visible_word_ids = list(visible.word_ids)
    if not later_word_ids or not _is_suffix(visible_word_ids, later_word_ids):
        no_match: tuple[list[str], list[str], str, str] | None = None
        return no_match
    first_tail_word_ids = visible_word_ids[: len(visible_word_ids) - len(later_word_ids)]
    if not first_tail_word_ids or not _is_suffix(list(first.word_ids), first_tail_word_ids):
        no_match: tuple[list[str], list[str], str, str] | None = None
        return no_match
    first_prefix_word_ids = list(first.word_ids)[: len(first.word_ids) - len(first_tail_word_ids)]
    later_text = "".join(caption.text for caption in source_captions[1:])
    visible_text = str(visible.text or "")
    text_match = _partial_tail_visible_text_match(visible_text, str(first.text or ""), later_text)
    if text_match is None:
        no_match: tuple[list[str], list[str], str, str] | None = None
        return no_match
    tail_text, prefix_text = text_match
    return first_tail_word_ids, first_prefix_word_ids, tail_text, prefix_text


def _partial_tail_visible_text_match(
    visible_text: str,
    first_text: str,
    later_text: str,
) -> tuple[str, str] | None:
    if not normalize_text(later_text):
        no_match: tuple[str, str] | None = None
        return no_match
    for later_visible_text in _right_boundary_text_options_after_non_de_left(later_text):
        tail_text = ""
        if visible_text.endswith(later_visible_text):
            tail_text = visible_text[: max(0, len(visible_text) - len(later_visible_text))]
        else:
            tail_text = _normalized_prefix_before_suffix(visible_text, later_visible_text)
        if not normalize_text(tail_text):
            continue
        prefix_text = _text_before_suffix(first_text, tail_text)
        if prefix_text is None:
            continue
        expected_visible = _join_visible_boundary_text(tail_text, later_text)
        if normalize_text(visible_text) == normalize_text(expected_visible):
            return tail_text, prefix_text
    no_match: tuple[str, str] | None = None
    return no_match
