from __future__ import annotations

from typing import Any


def configure_rule_dependencies(dependencies: dict[str, Any]) -> None:
    globals().update(dependencies)


MAX_SOURCE_BOUNDARY_PREFIX_GAP_US = 600_000


MAX_SOURCE_BOUNDARY_COMPOUND_GAP_US = 120_000


SOURCE_BOUNDARY_FUNCTION_PREFIXES = ("就", "也", "还", "才", "又", "再", "都", "只", "却", "仍", "便")


SOURCE_BOUNDARY_PREFIX_DEPENDENT_STARTS = ("有", "能", "敢", "会", "要", "把", "让", "给", "对", "在", "被", "将", "成", "可以")


SOURCE_BOUNDARY_COMPOUND_SUFFIXES = (
    "区",
    "圈",
    "群",
    "场",
    "端",
    "口",
    "线",
    "面",
    "点",
    "处",
    "侧",
    "边",
)


LEGAL_REDUPLICATION_AMOUNT_UNITS = tuple("个只件台部杯瓶份次块元毛角分钱万千百十亿年月天小时分钟公里米斤克岁平")
LEGAL_REDUPLICATION_NUMERAL_PREFIXES = tuple("零〇一二两三四五六七八九十百千万亿半几多0123456789")
LEGAL_REDUPLICATION_FALSE_START_SINGLE_CHARS = set("我你他她它这那就会又也还再都才要想能该")
MAX_OMITTED_REDUPLICATION_SOURCE_GAP_US = 220_000


TRUNCATED_COMPOUND_TAIL_MAX_GAP_US = 180_000
TRUNCATED_COMPOUND_TAIL_MAX_WORD_DURATION_US = 160_000
TRUNCATED_COMPOUND_TAIL_EXCLUDED_CHARS = set("的一了着过啊呢吗吧嘛就都也")


MIN_TRANSFERRED_PREFIX_TARGET_US = 80_000


MAX_TRANSFERRED_PREFIX_TARGET_US = 500_000


def _transfer_leading_function_prefix_to_previous_caption(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    previous_index: int,
    current_index: int,
    source_graph: CanonicalSourceGraph,
    candidate: dict[str, Any],
    pass_index: int,
) -> _RepairStep | None:
    if normalize_text(str(candidate.get("reason") or "")) != "dangling_de_prefix":
        no_step: _RepairStep | None = None
        return no_step
    if previous_index < 0 or current_index <= previous_index or current_index >= len(captions):
        no_step: _RepairStep | None = None
        return no_step
    previous = captions[previous_index]
    current = captions[current_index]
    if str(previous.containing_video_segment_id or "") != str(current.containing_video_segment_id or ""):
        no_step: _RepairStep | None = None
        return no_step
    if int(current.target_start_us) < int(previous.target_end_us):
        no_step: _RepairStep | None = None
        return no_step
    if not current.word_ids or len(current.word_ids) < 2:
        no_step: _RepairStep | None = None
        return no_step
    words_by_id = {word.word_id: word for word in source_graph.words}
    leading_word = words_by_id.get(current.word_ids[0])
    if leading_word is None:
        no_step: _RepairStep | None = None
        return no_step
    leading_text = str(getattr(leading_word, "text", "") or "")
    if normalize_text(leading_text) != "的":
        no_step: _RepairStep | None = None
        return no_step
    remaining_word_ids = list(current.word_ids[1:])
    remaining_text = _text_from_word_ids(remaining_word_ids, source_graph)
    if not normalize_text(remaining_text):
        no_step: _RepairStep | None = None
        return no_step
    previous_text = _join_visible_boundary_text(str(previous.text or ""), leading_text)
    if len(normalize_text(previous_text)) > HARD_MAX_CHARS:
        no_step: _RepairStep | None = None
        return no_step
    if not bool(build_final_caption_visible_repeat_gate([replace(current, text=remaining_text, word_ids=remaining_word_ids)]).get("gate_passed")):
        no_step: _RepairStep | None = None
        return no_step
    boundary_us = _target_boundary_after_leading_word(current, source_graph)
    if boundary_us is None:
        no_step: _RepairStep | None = None
        return no_step
    if boundary_us - int(previous.target_start_us) < MIN_REBALANCED_CAPTION_DURATION_US:
        no_step: _RepairStep | None = None
        return no_step
    if int(current.target_end_us) - boundary_us < MIN_REBALANCED_CAPTION_DURATION_US:
        no_step: _RepairStep | None = None
        return no_step
    leading_uid = str(getattr(leading_word, "subtitle_uid", "") or "")
    remaining_uids = [
        str(getattr(words_by_id[word_id], "subtitle_uid", "") or "")
        for word_id in remaining_word_ids
        if word_id in words_by_id and str(getattr(words_by_id[word_id], "subtitle_uid", "") or "")
    ]
    previous_repaired = replace(
        previous,
        word_ids=[*previous.word_ids, current.word_ids[0]],
        text=previous_text,
        target_end_us=boundary_us,
        source_subtitle_uids=_unique([*previous.source_subtitle_uids, leading_uid]),
        spoken_source_end_us=int(getattr(leading_word, "source_end_us", 0) or 0),
    )
    first_remaining = words_by_id.get(remaining_word_ids[0])
    current_repaired = replace(
        current,
        word_ids=remaining_word_ids,
        text=remaining_text,
        target_start_us=boundary_us,
        source_subtitle_uids=_unique(remaining_uids or list(current.source_subtitle_uids)),
        spoken_source_start_us=int(getattr(first_remaining, "source_start_us", 0) or 0) if first_remaining is not None else current.spoken_source_start_us,
    )
    repaired = list(captions)
    repaired[previous_index] = previous_repaired
    repaired[current_index] = current_repaired
    return _RepairStep(
        final_timeline=final_timeline,
        captions=repaired,
        timeline_changed=False,
        action=_action(
            "dangling_prefix_suffix",
            "transfer_leading_function_prefix_to_previous_caption",
            pass_index,
            candidate,
            affected_caption_ids=[previous.caption_id, current.caption_id],
            transferred_word_ids=[current.word_ids[0]],
            transferred_text=leading_text,
            previous_caption_text=previous_repaired.text,
            current_caption_text=current_repaired.text,
            boundary_target_us=boundary_us,
        ),
    )


def _target_boundary_after_leading_word(
    caption: CaptionRenderUnit,
    source_graph: CanonicalSourceGraph,
) -> int | None:
    if not caption.word_ids:
        no_boundary: int | None = None
        return no_boundary
    words_by_id = {word.word_id: word for word in source_graph.words}
    words = [words_by_id[word_id] for word_id in caption.word_ids if word_id in words_by_id]
    if not words:
        no_boundary: int | None = None
        return no_boundary
    duration_us = max(0, int(caption.target_end_us) - int(caption.target_start_us))
    if duration_us <= MIN_REBALANCED_CAPTION_DURATION_US:
        no_boundary: int | None = None
        return no_boundary
    source_span_us = max(1, int(getattr(words[-1], "source_end_us", 0) or 0) - int(getattr(words[0], "source_start_us", 0) or 0))
    leading_source_us = max(
        1,
        int(getattr(words[0], "source_end_us", 0) or 0) - int(getattr(words[0], "source_start_us", 0) or 0),
    )
    scaled_us = round(duration_us * leading_source_us / source_span_us)
    transfer_us = max(MIN_TRANSFERRED_PREFIX_TARGET_US, min(MAX_TRANSFERRED_PREFIX_TARGET_US, int(scaled_us)))
    max_transfer_us = duration_us - MIN_REBALANCED_CAPTION_DURATION_US
    if max_transfer_us < MIN_TRANSFERRED_PREFIX_TARGET_US:
        no_boundary: int | None = None
        return no_boundary
    transfer_us = min(transfer_us, max_transfer_us)
    return int(caption.target_start_us) + transfer_us


def _repair_source_boundary_prefix_gap(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    ordered_words = list(source_graph.words)
    index_by_word_id = {word.word_id: index for index, word in enumerate(ordered_words)}
    for segment in _ordered_segments(final_timeline):
        prefix_candidate = _source_boundary_prefix_candidate(
            segment,
            final_timeline,
            words_by_id,
            ordered_words,
            index_by_word_id,
        )
        if prefix_candidate is None:
            continue
        repaired = _apply_source_boundary_prefix_candidate(final_timeline, segment, prefix_candidate, source_graph)
        if repaired is None:
            continue
        prefix_word = prefix_candidate.word
        return _RepairStep(
            final_timeline=repaired,
            captions=[],
            timeline_changed=True,
            action=_action(
                "source_boundary_prefix_gap",
                "prepend_source_boundary_prefix",
                pass_index,
                {
                    "caption_id": "",
                    "related_caption_id": "",
                    "reason": "source-aware boundary prefix was omitted before a dependent visible caption start",
                    "overlap_text": normalize_text(str(getattr(prefix_word, "text", "") or "")),
                },
                affected_segment_id=segment.segment_id,
                prepended_word_id=prefix_word.word_id,
                prepended_text=prefix_word.text,
                transferred_from_segment_id=prefix_candidate.transfer_from_segment_id,
            ),
        )
    no_step: _RepairStep | None = None
    return no_step


def _repair_omitted_legal_reduplication_word(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    ordered_words = list(source_graph.words)
    index_by_word_id = {word.word_id: index for index, word in enumerate(ordered_words)}
    selected_word_ids = {str(word_id) for segment in final_timeline for word_id in list(segment.word_ids or [])}
    for segment in _ordered_segments(final_timeline):
        segment_word_ids = [str(word_id) for word_id in list(segment.word_ids or []) if str(word_id)]
        for position, word_id in enumerate(segment_word_ids):
            word = words_by_id.get(word_id)
            source_index = index_by_word_id.get(word_id)
            if word is None or source_index is None or source_index <= 0:
                continue
            previous_word = ordered_words[source_index - 1]
            previous_word_id = str(getattr(previous_word, "word_id", "") or "")
            if not previous_word_id or previous_word_id in selected_word_ids:
                continue
            if not _omitted_legal_reduplication_pair(previous_word, word, segment_word_ids[position + 1 :], words_by_id):
                continue
            repaired_word_ids = [*segment_word_ids[:position], previous_word_id, *segment_word_ids[position:]]
            repaired_segment = _segment_with_word_ids_preserving_effective_speed(
                segment,
                repaired_word_ids,
                source_graph,
                "restore_omitted_legal_reduplication",
            )
            if repaired_segment is None:
                continue
            repaired = [
                repaired_segment if row.segment_id == segment.segment_id else row
                for row in final_timeline
            ]
            return _RepairStep(
                final_timeline=repaired,
                captions=[],
                timeline_changed=True,
                action=_action(
                    "omitted_legal_reduplication",
                    "restore_omitted_legal_reduplication_word",
                    pass_index,
                    {
                        "caption_id": "",
                        "related_caption_id": "",
                        "reason": "source graph contains a legal reduplication before an amount or modifier suffix but the first token was omitted",
                        "overlap_text": normalize_text(str(getattr(word, "text", "") or "")),
                    },
                    affected_segment_id=segment.segment_id,
                    inserted_word_id=previous_word_id,
                    inserted_text=str(getattr(previous_word, "text", "") or ""),
                    before_word_id=word_id,
                    repaired_word_ids=repaired_word_ids,
                ),
            )
    no_step: _RepairStep | None = None
    return no_step


def _omitted_legal_reduplication_pair(
    previous_word: Any,
    current_word: Any,
    following_word_ids: list[str],
    words_by_id: dict[str, Any],
) -> bool:
    previous_text = normalize_text(str(getattr(previous_word, "text", "") or ""))
    current_text = normalize_text(str(getattr(current_word, "text", "") or ""))
    if not previous_text or previous_text != current_text:
        return False
    if len(previous_text) == 1 and previous_text in LEGAL_REDUPLICATION_FALSE_START_SINGLE_CHARS:
        return False
    if not all("\u3400" <= char <= "\u9fff" for char in previous_text):
        return False
    previous_end = int(getattr(previous_word, "source_end_us", 0) or 0)
    current_start = int(getattr(current_word, "source_start_us", 0) or 0)
    if current_start < previous_end or current_start - previous_end > MAX_OMITTED_REDUPLICATION_SOURCE_GAP_US:
        return False
    if str(getattr(previous_word, "source_material_id", "") or "") != str(getattr(current_word, "source_material_id", "") or ""):
        return False
    if str(getattr(previous_word, "source_segment_id", "") or "") != str(getattr(current_word, "source_segment_id", "") or ""):
        return False
    following_text = normalize_text(
        "".join(
            str(getattr(words_by_id[word_id], "text", "") or "")
            for word_id in following_word_ids[:4]
            if word_id in words_by_id
        )
    )
    return _legal_reduplication_following_text(following_text)


def _legal_reduplication_following_text(following_text: str) -> bool:
    if not following_text:
        return False
    if following_text.startswith(("的", "地", "得")):
        return True
    if following_text[0] not in LEGAL_REDUPLICATION_NUMERAL_PREFIXES:
        return False
    tail = following_text[1:8]
    if not tail:
        return False
    return any(char in LEGAL_REDUPLICATION_AMOUNT_UNITS for char in tail)


def _source_boundary_prefix_candidate(
    segment: FinalTimelineSegment,
    final_timeline: list[FinalTimelineSegment],
    words_by_id: dict[str, Any],
    ordered_words: list[Any],
    index_by_word_id: dict[str, int],
) -> _SourceBoundaryPrefixCandidate | None:
    if not segment.word_ids:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    first_word_id = segment.word_ids[0]
    first_word = words_by_id.get(first_word_id)
    first_index = index_by_word_id.get(first_word_id)
    if first_word is None or first_index is None or first_index <= 0:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    current_text = normalize_text(str(segment.text or ""))
    if not _source_boundary_prefix_dependent_start(current_text):
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    prefix_word = ordered_words[first_index - 1]
    prefix_word_id = str(getattr(prefix_word, "word_id", "") or "")
    prefix_text = normalize_text(str(getattr(prefix_word, "text", "") or ""))
    if prefix_text not in SOURCE_BOUNDARY_FUNCTION_PREFIXES:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    prefix_material_id = str(getattr(prefix_word, "source_material_id", "") or "")
    segment_material_id = str(segment.source_material_id or "")
    if prefix_material_id and segment_material_id and prefix_material_id != segment_material_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    prefix_segment_id = str(getattr(prefix_word, "source_segment_id", "") or "")
    segment_source_id = str(segment.source_segment_id or "")
    if prefix_segment_id and segment_source_id and prefix_segment_id != segment_source_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    first_material_id = str(getattr(first_word, "source_material_id", "") or "")
    if first_material_id and segment_material_id and first_material_id != segment_material_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    first_segment_id = str(getattr(first_word, "source_segment_id", "") or "")
    if first_segment_id and segment_source_id and first_segment_id != segment_source_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    source_gap_us = int(getattr(first_word, "source_start_us", 0)) - int(getattr(prefix_word, "source_end_us", 0))
    if source_gap_us < -80_000 or source_gap_us > MAX_SOURCE_BOUNDARY_PREFIX_GAP_US:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    if abs(int(segment.source_start_us) - int(getattr(first_word, "source_start_us", segment.source_start_us))) > 80_000:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    containing_segments = [row for row in final_timeline if prefix_word_id in list(row.word_ids)]
    if not containing_segments:
        return _SourceBoundaryPrefixCandidate(word=prefix_word)
    if len(containing_segments) != 1:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    transfer_segment = containing_segments[0]
    if transfer_segment.segment_id == segment.segment_id:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    if not _is_suffix(list(transfer_segment.word_ids), [prefix_word_id]):
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    if int(transfer_segment.target_end_us) > int(segment.target_start_us) + 80_000:
        no_candidate: _SourceBoundaryPrefixCandidate | None = None
        return no_candidate
    return _SourceBoundaryPrefixCandidate(word=prefix_word, transfer_from_segment_id=transfer_segment.segment_id)


def _repair_source_boundary_compound_suffix_gap(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    candidate = _source_boundary_compound_candidate(final_timeline, source_graph)
    if candidate is None:
        no_step: _RepairStep | None = None
        return no_step
    repaired = _merge_source_boundary_compound_segments(final_timeline, candidate, source_graph)
    if repaired is None:
        no_step: _RepairStep | None = None
        return no_step
    return _RepairStep(
        final_timeline=repaired,
        captions=[],
        timeline_changed=True,
        action=_action(
            "source_boundary_compound_suffix",
            "merge_source_boundary_compound_suffix",
            pass_index,
            {
                "caption_id": "",
                "related_caption_id": "",
                "reason": "source-aware lexical suffix belongs with the previous visible word",
                "overlap_text": f"{getattr(candidate.left_word, 'text', '')}{getattr(candidate.right_word, 'text', '')}",
            },
            affected_segment_ids=[candidate.left_segment.segment_id, candidate.right_segment.segment_id],
            suffix_word_id=str(getattr(candidate.right_word, "word_id", "") or ""),
            suffix_text=str(getattr(candidate.right_word, "text", "") or ""),
        ),
    )


def _repair_source_boundary_truncated_compound_tail(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    candidate = _source_boundary_truncated_compound_tail_candidate(final_timeline, source_graph)
    if candidate is None:
        no_step: _RepairStep | None = None
        return no_step
    segment = candidate["segment"]
    appended_word_id = str(candidate["append_word_id"])
    repaired_segment = _segment_with_word_ids_preserving_effective_speed(
        segment,
        [*list(segment.word_ids), appended_word_id],
        source_graph,
        "source_boundary_truncated_compound_tail_append",
    )
    if repaired_segment is None:
        no_step: _RepairStep | None = None
        return no_step
    repaired = [
        repaired_segment if row.segment_id == segment.segment_id else row
        for row in final_timeline
    ]
    return _RepairStep(
        final_timeline=repaired,
        captions=[],
        timeline_changed=True,
        action=_action(
            "source_boundary_truncated_compound_tail",
            "append_source_boundary_compound_tail",
            pass_index,
            {
                "caption_id": "",
                "related_caption_id": "",
                "reason": "single-character lexical tail is immediately completed by the next source word",
                "overlap_text": str(candidate["tail_text"]),
            },
            affected_segment_ids=[segment.segment_id],
            appended_word_id=appended_word_id,
            appended_word_text=str(candidate["append_word_text"]),
            source_gap_us=int(candidate["source_gap_us"]),
        ),
    )


def _source_boundary_truncated_compound_tail_candidate(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> dict[str, Any] | None:
    used_word_ids = {str(word_id) for segment in final_timeline for word_id in list(segment.word_ids)}
    words_by_id = {word.word_id: word for word in source_graph.words}
    ordered_words = sorted(
        source_graph.words,
        key=lambda word: (int(getattr(word, "source_start_us", 0) or 0), int(getattr(word, "source_end_us", 0) or 0)),
    )
    for segment in _ordered_segments(final_timeline):
        if not segment.word_ids:
            continue
        tail_word_id = str(segment.word_ids[-1])
        tail_word = words_by_id.get(tail_word_id)
        if tail_word is None:
            continue
        tail_text = normalize_text(str(getattr(tail_word, "text", "") or ""))
        if len(tail_text) != 1 or tail_text in TRUNCATED_COMPOUND_TAIL_EXCLUDED_CHARS:
            continue
        tail_duration_us = int(getattr(tail_word, "source_end_us", 0) or 0) - int(getattr(tail_word, "source_start_us", 0) or 0)
        if tail_duration_us <= 0 or tail_duration_us > TRUNCATED_COMPOUND_TAIL_MAX_WORD_DURATION_US:
            continue
        next_word = _next_unselected_source_word(tail_word, ordered_words, used_word_ids)
        if next_word is None:
            continue
        next_text = normalize_text(str(getattr(next_word, "text", "") or ""))
        if len(next_text) < 2:
            continue
        return {
            "segment": segment,
            "append_word_id": str(getattr(next_word, "word_id", "") or ""),
            "append_word_text": str(getattr(next_word, "text", "") or ""),
            "tail_text": tail_text,
            "source_gap_us": int(getattr(next_word, "source_start_us", 0) or 0) - int(getattr(tail_word, "source_end_us", 0) or 0),
        }
    no_candidate: dict[str, Any] | None = None
    return no_candidate


def _next_unselected_source_word(
    tail_word: Any,
    ordered_words: list[Any],
    used_word_ids: set[str],
) -> Any | None:
    tail_end_us = int(getattr(tail_word, "source_end_us", 0) or 0)
    tail_material = str(getattr(tail_word, "source_material_id", "") or "")
    tail_segment = str(getattr(tail_word, "source_segment_id", "") or "")
    tail_subtitle_uid = str(getattr(tail_word, "subtitle_uid", "") or "")
    tail_subtitle_index = getattr(tail_word, "subtitle_index", None)
    for word in ordered_words:
        word_id = str(getattr(word, "word_id", "") or "")
        if not word_id or word_id in used_word_ids:
            continue
        word_start_us = int(getattr(word, "source_start_us", 0) or 0)
        if word_start_us < tail_end_us:
            continue
        gap_us = word_start_us - tail_end_us
        if gap_us > TRUNCATED_COMPOUND_TAIL_MAX_GAP_US:
            break
        if tail_material and str(getattr(word, "source_material_id", "") or "") not in {"", tail_material}:
            continue
        if tail_segment and str(getattr(word, "source_segment_id", "") or "") not in {"", tail_segment}:
            continue
        word_subtitle_uid = str(getattr(word, "subtitle_uid", "") or "")
        word_subtitle_index = getattr(word, "subtitle_index", None)
        same_subtitle = (
            bool(tail_subtitle_uid and word_subtitle_uid and tail_subtitle_uid == word_subtitle_uid)
            or (tail_subtitle_index is not None and tail_subtitle_index == word_subtitle_index)
        )
        if not same_subtitle:
            continue
        return word
    no_word: Any | None = None
    return no_word


def _source_boundary_compound_candidate(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> _SourceBoundaryCompoundCandidate | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    ordered = _ordered_segments(final_timeline)
    for left, right in zip(ordered, ordered[1:]):
        if not left.word_ids or not right.word_ids:
            continue
        if len(normalize_text(f"{left.text}{right.text}")) > HARD_MAX_CHARS:
            continue
        if int(right.target_end_us) - int(left.target_start_us) > HARD_MAX_DURATION_US:
            continue
        left_word = words_by_id.get(left.word_ids[-1])
        right_word = words_by_id.get(right.word_ids[0])
        if left_word is None or right_word is None:
            continue
        if not _source_boundary_compound_words_match(left_word, right_word):
            continue
        if not _safe_merge_segments(left, right, source_graph):
            continue
        return _SourceBoundaryCompoundCandidate(
            left_segment=left,
            right_segment=right,
            left_word=left_word,
            right_word=right_word,
        )
    no_candidate: _SourceBoundaryCompoundCandidate | None = None
    return no_candidate


def _source_boundary_compound_words_match(left_word: Any, right_word: Any) -> bool:
    left_text = normalize_text(str(getattr(left_word, "text", "") or ""))
    right_text = normalize_text(str(getattr(right_word, "text", "") or ""))
    if len(left_text) < 2 or right_text not in SOURCE_BOUNDARY_COMPOUND_SUFFIXES:
        return False
    source_gap_us = int(getattr(right_word, "source_start_us", 0)) - int(getattr(left_word, "source_end_us", 0))
    if source_gap_us < -80_000 or source_gap_us > MAX_SOURCE_BOUNDARY_COMPOUND_GAP_US:
        return False
    left_material = str(getattr(left_word, "source_material_id", "") or "")
    right_material = str(getattr(right_word, "source_material_id", "") or "")
    if left_material and right_material and left_material != right_material:
        return False
    left_segment = str(getattr(left_word, "source_segment_id", "") or "")
    right_segment = str(getattr(right_word, "source_segment_id", "") or "")
    if left_segment and right_segment and left_segment != right_segment:
        return False
    return True


def _merge_source_boundary_compound_segments(
    final_timeline: list[FinalTimelineSegment],
    candidate: _SourceBoundaryCompoundCandidate,
    source_graph: CanonicalSourceGraph,
) -> list[FinalTimelineSegment] | None:
    left = candidate.left_segment
    right = candidate.right_segment
    index_by_id = {segment.segment_id: index for index, segment in enumerate(final_timeline)}
    left_index = index_by_id.get(left.segment_id)
    right_index = index_by_id.get(right.segment_id)
    if left_index is None or right_index is None or right_index != left_index + 1:
        no_repair: list[FinalTimelineSegment] | None = None
        return no_repair
    merged = _merged_segment_pair_preserving_effective_speed(
        left,
        right,
        source_graph,
        "source_boundary_compound_suffix_merge",
    )
    return [*final_timeline[:left_index], merged, *final_timeline[right_index + 1 :]]


def _source_boundary_prefix_dependent_start(text: str) -> bool:
    if not text:
        return False
    return any(text.startswith(prefix) for prefix in SOURCE_BOUNDARY_PREFIX_DEPENDENT_STARTS)


def _apply_source_boundary_prefix_candidate(
    final_timeline: list[FinalTimelineSegment],
    segment: FinalTimelineSegment,
    candidate: _SourceBoundaryPrefixCandidate,
    source_graph: CanonicalSourceGraph,
) -> list[FinalTimelineSegment] | None:
    prefix_word = candidate.word
    prefix_word_id = str(getattr(prefix_word, "word_id", "") or "")
    if not prefix_word_id:
        no_repair: list[FinalTimelineSegment] | None = None
        return no_repair
    repaired: list[FinalTimelineSegment] = []
    changed = False
    for row in final_timeline:
        if candidate.transfer_from_segment_id and row.segment_id == candidate.transfer_from_segment_id:
            remaining_word_ids = [word_id for word_id in row.word_ids if word_id != prefix_word_id]
            if not remaining_word_ids:
                no_repair: list[FinalTimelineSegment] | None = None
                return no_repair
            trimmed = _segment_with_word_ids_preserving_effective_speed(row, remaining_word_ids, source_graph, "source_boundary_prefix_transfer")
            if trimmed is None:
                no_repair: list[FinalTimelineSegment] | None = None
                return no_repair
            repaired.append(trimmed)
            changed = True
            continue
        if row.segment_id != segment.segment_id:
            repaired.append(row)
            continue
        word_ids = [prefix_word_id, *row.word_ids]
        text = _text_from_word_ids(word_ids, source_graph)
        if not normalize_text(text):
            no_repair: list[FinalTimelineSegment] | None = None
            return no_repair
        source_start_us = int(getattr(prefix_word, "source_start_us", row.source_start_us))
        source_end_us = int(row.source_end_us)
        target_duration_us = _target_duration_preserving_effective_speed(row, source_start_us, source_end_us)
        repaired.append(
            replace(
                row,
                source_start_us=source_start_us,
                target_end_us=int(row.target_start_us) + target_duration_us,
                word_ids=word_ids,
                text=text,
                spoken_source_start_us=source_start_us,
                clip_source_start_us=source_start_us
                if row.clip_source_start_us is not None
                else row.clip_source_start_us,
                debug_hints={
                    **dict(row.debug_hints or {}),
                    "final_visible_repair": "source_boundary_prefix_prepend",
                    "prepended_word_id": str(getattr(prefix_word, "word_id", "") or ""),
                },
            )
        )
        changed = True
    if changed:
        return repaired
    no_repair: list[FinalTimelineSegment] | None = None
    return no_repair
