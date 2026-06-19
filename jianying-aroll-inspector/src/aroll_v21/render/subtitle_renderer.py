from __future__ import annotations

from dataclasses import replace

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.boundary_overlap import is_explanatory_term_reuse, is_semantic_label_reuse_boundary
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.subtitle_readability import (
    HARD_MAX_CHARS,
    HARD_MAX_DURATION_US,
    MIN_DURATION_US,
    TARGET_MAX_CHARS,
    fit_groups_to_segment_duration,
    merge_tiny_display_fragments,
    split_words_for_display,
)


MIN_CAPTION_DURATION_US = 300_000
DETERMINISTIC_REPEAT_REASONS = {"containment_repeat", "prefix_suffix_overlap", "ngram_repeat"}


class SubtitleRenderer:
    def render(
        self,
        final_timeline: list[FinalTimelineSegment],
        source_graph: CanonicalSourceGraph,
        *,
        style_template_id: str = "canonical_caption_template",
    ) -> list[CaptionRenderUnit]:
        words_by_id = {word.word_id: word for word in source_graph.words}
        captions: list[CaptionRenderUnit] = []
        caption_index = 1
        for segment in final_timeline:
            words = [words_by_id[word_id] for word_id in segment.word_ids if word_id in words_by_id]
            previous_caption_end = int(segment.target_start_us)
            groups = fit_groups_to_segment_duration(
                merge_tiny_display_fragments(split_words_for_display(words)),
                max(0, int(segment.target_end_us) - int(segment.target_start_us)),
                min_duration_us=MIN_DURATION_US,
            )
            for group in groups:
                if not group:
                    continue
                source_uids: list[str] = []
                for word in group:
                    if word.subtitle_uid and word.subtitle_uid not in source_uids:
                        source_uids.append(word.subtitle_uid)
                caption_start = segment.target_start_us + max(0, int(group[0].source_start_us) - int(segment.source_start_us))
                caption_start = max(caption_start, previous_caption_end)
                caption_end = segment.target_start_us + max(0, int(group[-1].source_end_us) - int(segment.source_start_us))
                caption_end = max(caption_start + MIN_DURATION_US, caption_end)
                if caption_end > segment.target_end_us:
                    caption_start = max(segment.target_start_us, previous_caption_end, segment.target_end_us - MIN_DURATION_US)
                    caption_end = segment.target_end_us
                captions.append(
                    CaptionRenderUnit(
                        caption_id=f"v21_cap_{caption_index:06d}",
                        timeline_segment_ids=[segment.segment_id],
                        word_ids=[word.word_id for word in group],
                        text="".join(word.text for word in group),
                        target_start_us=caption_start,
                        target_end_us=caption_end,
                        source_subtitle_uids=source_uids,
                        style_template_id=style_template_id,
                        spoken_source_start_us=int(group[0].source_start_us),
                        spoken_source_end_us=int(group[-1].source_end_us),
                        containing_video_segment_id=segment.segment_id,
                    )
                )
                caption_index += 1
                previous_caption_end = caption_end
        cleaned = _cleanup_caption_units(captions, {segment.segment_id: segment for segment in final_timeline})
        repaired = _repair_visible_caption_repeats(cleaned)
        return _renumber_captions(repaired)

    def _caption_word_groups(self, words):
        subtitle_groups = []
        current = []
        current_subtitle = None
        for word in words:
            key = word.subtitle_index if word.subtitle_index is not None else word.subtitle_uid
            if current and key != current_subtitle:
                subtitle_groups.append(current)
                current = []
            current.append(word)
            current_subtitle = key
        if current:
            subtitle_groups.append(current)
        chunks = []
        for group in subtitle_groups:
            chunks.extend(self._split_display_group(group))
        return self._merge_tiny_chunks(chunks)

    def _split_display_group(self, words):
        chunks = []
        current = []
        for word in words:
            text = "".join(item.text for item in current) + word.text
            duration = int(word.source_end_us) - int(current[0].source_start_us if current else word.source_start_us)
            if current and (len(text) > TARGET_MAX_CHARS or duration > HARD_MAX_DURATION_US):
                chunks.append(current)
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(current)
        return chunks

    def _merge_tiny_chunks(self, chunks):
        merged = []
        index = 0
        while index < len(chunks):
            chunk = chunks[index]
            text = "".join(word.text for word in chunk)
            if len(text) < 2 and index + 1 < len(chunks):
                candidate = chunk + chunks[index + 1]
                if len("".join(word.text for word in candidate)) <= HARD_MAX_CHARS:
                    merged.append(candidate)
                    index += 2
                    continue
            if len(text) < 2 and merged:
                candidate = merged[-1] + chunk
                if len("".join(word.text for word in candidate)) <= HARD_MAX_CHARS:
                    merged[-1] = candidate
                    index += 1
                    continue
            merged.append(chunk)
            index += 1
        return merged

    def _fit_groups_to_segment_duration(self, groups, segment_duration_us: int):
        groups = [list(group) for group in groups if group]
        while len(groups) > 1 and len(groups) * MIN_CAPTION_DURATION_US > int(segment_duration_us):
            merge_index = min(
                range(len(groups) - 1),
                key=lambda index: len("".join(word.text for word in groups[index] + groups[index + 1])),
            )
            groups = [*groups[:merge_index], groups[merge_index] + groups[merge_index + 1], *groups[merge_index + 2 :]]
        return groups


def _cleanup_caption_units(
    captions: list[CaptionRenderUnit],
    segments_by_id: dict[str, FinalTimelineSegment],
) -> list[CaptionRenderUnit]:
    current = list(captions)
    while True:
        changed = False
        for index, caption in enumerate(current):
            if not _caption_needs_cleanup(caption):
                continue
            merged = _merge_caption_with_neighbor(current, index)
            if merged is not None:
                current = merged
                changed = True
                break
            extended = _extend_caption_inside_container(current, index, segments_by_id)
            if extended is not None and extended != caption:
                current[index] = extended
                changed = True
                break
        if not changed:
            return current


def _caption_needs_cleanup(caption: CaptionRenderUnit) -> bool:
    return _caption_duration(caption) < MIN_DURATION_US or 0 < len(str(caption.text or "").strip()) <= 3


def _merge_caption_with_neighbor(
    captions: list[CaptionRenderUnit],
    index: int,
) -> list[CaptionRenderUnit] | None:
    for neighbor_index in (index + 1, index - 1):
        if neighbor_index < 0 or neighbor_index >= len(captions):
            continue
        merged = _merged_caption(captions[index], captions[neighbor_index])
        if merged is None:
            continue
        keep_index = min(index, neighbor_index)
        drop_index = max(index, neighbor_index)
        rows = list(captions)
        rows[keep_index] = merged
        return [*rows[:drop_index], *rows[drop_index + 1 :]]
    no_merge: list[CaptionRenderUnit] | None = None
    return no_merge


def _merged_caption(left: CaptionRenderUnit, right: CaptionRenderUnit) -> CaptionRenderUnit | None:
    rows = sorted([left, right], key=lambda row: (row.target_start_us, row.target_end_us, row.caption_id))
    first, second = rows
    if str(first.containing_video_segment_id or "") != str(second.containing_video_segment_id or ""):
        incompatible: CaptionRenderUnit | None = None
        return incompatible
    text = f"{first.text}{second.text}"
    if len(text.strip()) > HARD_MAX_CHARS:
        too_long: CaptionRenderUnit | None = None
        return too_long
    if int(second.target_start_us) < int(first.target_end_us):
        overlapping: CaptionRenderUnit | None = None
        return overlapping
    target_start_us = int(first.target_start_us)
    target_end_us = int(second.target_end_us)
    compact_window = _compact_overlong_tiny_caption_merge_window(first, second)
    if compact_window is not None:
        target_start_us, target_end_us = compact_window
    elif target_end_us - target_start_us > HARD_MAX_DURATION_US:
        too_long: CaptionRenderUnit | None = None
        return too_long
    return replace(
        first,
        timeline_segment_ids=sorted(set([*first.timeline_segment_ids, *second.timeline_segment_ids])),
        word_ids=[*first.word_ids, *second.word_ids],
        text=text,
        target_start_us=target_start_us,
        target_end_us=target_end_us,
        source_subtitle_uids=_unique([*first.source_subtitle_uids, *second.source_subtitle_uids]),
        spoken_source_start_us=min(
            value
            for value in [first.spoken_source_start_us, second.spoken_source_start_us]
            if value is not None
        )
        if first.spoken_source_start_us is not None or second.spoken_source_start_us is not None
        else None,
        spoken_source_end_us=max(
            value
            for value in [first.spoken_source_end_us, second.spoken_source_end_us]
            if value is not None
        )
        if first.spoken_source_end_us is not None or second.spoken_source_end_us is not None
        else None,
        containing_video_segment_id=first.containing_video_segment_id,
    )


def _compact_overlong_tiny_caption_merge_window(
    first: CaptionRenderUnit,
    second: CaptionRenderUnit,
) -> tuple[int, int] | None:
    target_start_us = int(first.target_start_us)
    target_end_us = int(second.target_end_us)
    if target_end_us - target_start_us <= HARD_MAX_DURATION_US:
        no_compaction: tuple[int, int] | None = None
        return no_compaction
    first_text_len = len(normalize_text(str(first.text or "")))
    second_text_len = len(normalize_text(str(second.text or "")))
    if first_text_len <= 1 and int(second.target_end_us) - int(second.target_start_us) <= HARD_MAX_DURATION_US:
        compact_start = max(target_start_us, target_end_us - HARD_MAX_DURATION_US)
        if compact_start <= int(first.target_end_us):
            return compact_start, target_end_us
    if second_text_len <= 1 and int(first.target_end_us) - int(first.target_start_us) <= HARD_MAX_DURATION_US:
        compact_end = min(target_end_us, target_start_us + HARD_MAX_DURATION_US)
        if compact_end >= int(second.target_start_us):
            return target_start_us, compact_end
    no_compaction: tuple[int, int] | None = None
    return no_compaction


def _extend_caption_inside_container(
    captions: list[CaptionRenderUnit],
    index: int,
    segments_by_id: dict[str, FinalTimelineSegment],
) -> CaptionRenderUnit | None:
    caption = captions[index]
    segment = segments_by_id.get(str(caption.containing_video_segment_id or ""))
    if segment is None or _caption_duration(caption) >= MIN_DURATION_US:
        not_extendable: CaptionRenderUnit | None = None
        return not_extendable
    previous_end = int(segment.target_start_us)
    next_start = int(segment.target_end_us)
    if index > 0 and str(captions[index - 1].containing_video_segment_id or "") == str(caption.containing_video_segment_id or ""):
        previous_end = max(previous_end, int(captions[index - 1].target_end_us))
    if index + 1 < len(captions) and str(captions[index + 1].containing_video_segment_id or "") == str(caption.containing_video_segment_id or ""):
        next_start = min(next_start, int(captions[index + 1].target_start_us))
    start = int(caption.target_start_us)
    end = int(caption.target_end_us)
    if start + MIN_DURATION_US <= next_start:
        return replace(caption, target_end_us=start + MIN_DURATION_US)
    if end - MIN_DURATION_US >= previous_end:
        return replace(caption, target_start_us=end - MIN_DURATION_US)
    no_extension: CaptionRenderUnit | None = None
    return no_extension


def _caption_duration(caption: CaptionRenderUnit) -> int:
    return max(0, int(caption.target_end_us) - int(caption.target_start_us))


def _repair_visible_caption_repeats(captions: list[CaptionRenderUnit]) -> list[CaptionRenderUnit]:
    current = sorted(captions, key=lambda row: (row.target_start_us, row.target_end_us, row.caption_id))
    while True:
        repaired = _repair_next_deterministic_visible_repeat(current)
        if repaired is None:
            return current
        current = repaired


def _repair_next_deterministic_visible_repeat(captions: list[CaptionRenderUnit]) -> list[CaptionRenderUnit] | None:
    gate = build_final_caption_visible_repeat_gate(captions)
    for candidate in list(gate.get("visible_repeat_candidates") or []):
        reason = str(candidate.get("reason") or "")
        if reason not in DETERMINISTIC_REPEAT_REASONS:
            continue
        repaired = _repair_repeat_candidate(captions, candidate, reason)
        if repaired is not None and len(repaired) < len(captions):
            return repaired
    no_repair: list[CaptionRenderUnit] | None = None
    return no_repair


def _repair_repeat_candidate(
    captions: list[CaptionRenderUnit],
    candidate: dict,
    reason: str,
) -> list[CaptionRenderUnit] | None:
    caption_id = str(candidate.get("caption_id") or "")
    related_caption_id = str(candidate.get("related_caption_id") or "")
    index_by_id = {caption.caption_id: index for index, caption in enumerate(captions)}
    left_index = index_by_id.get(caption_id)
    right_index = index_by_id.get(related_caption_id)
    if left_index is None or right_index is None or left_index == right_index:
        no_repair: list[CaptionRenderUnit] | None = None
        return no_repair
    left = captions[left_index]
    right = captions[right_index]
    if not set(left.word_ids).intersection(set(right.word_ids)) and _disjoint_repeat_should_keep_both(left, right):
        no_repair: list[CaptionRenderUnit] | None = None
        return no_repair
    drop_index = _deterministic_repeat_drop_index(left, right, reason, left_index, right_index)
    if drop_index is None:
        no_repair: list[CaptionRenderUnit] | None = None
        return no_repair
    return [*captions[:drop_index], *captions[drop_index + 1 :]]


def _disjoint_repeat_should_keep_both(left: CaptionRenderUnit, right: CaptionRenderUnit) -> bool:
    left_text = normalize_text(left.text)
    right_text = normalize_text(right.text)
    if is_explanatory_term_reuse(left_text, right_text) or is_explanatory_term_reuse(right_text, left_text):
        return True
    overlap = left_text if right_text.startswith(left_text) else right_text if left_text.startswith(right_text) else ""
    if overlap and is_semantic_label_reuse_boundary(left_text, right_text, overlap):
        return True
    if overlap and is_semantic_label_reuse_boundary(right_text, left_text, overlap):
        return True
    return False


def _deterministic_repeat_drop_index(
    left: CaptionRenderUnit,
    right: CaptionRenderUnit,
    reason: str,
    left_index: int,
    right_index: int,
) -> int | None:
    left_text = normalize_text(left.text)
    right_text = normalize_text(right.text)
    if not left_text or not right_text:
        no_drop: int | None = None
        return no_drop
    if reason == "containment_repeat":
        if left_text == right_text:
            return _less_informative_caption_index(left, right, left_index, right_index)
        if left_text in right_text:
            return left_index
        if right_text in left_text:
            return right_index
        no_drop: int | None = None
        return no_drop
    if reason == "prefix_suffix_overlap":
        return _less_informative_caption_index(left, right, left_index, right_index)
    if reason == "ngram_repeat":
        return _less_informative_caption_index(left, right, left_index, right_index)
    no_drop: int | None = None
    return no_drop


def _less_informative_caption_index(
    left: CaptionRenderUnit,
    right: CaptionRenderUnit,
    left_index: int,
    right_index: int,
) -> int:
    left_text = normalize_text(left.text)
    right_text = normalize_text(right.text)
    if len(left_text) != len(right_text):
        return left_index if len(left_text) < len(right_text) else right_index
    left_duration = _caption_duration(left)
    right_duration = _caption_duration(right)
    if left_duration != right_duration:
        return left_index if left_duration < right_duration else right_index
    return max(left_index, right_index)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _renumber_captions(captions: list[CaptionRenderUnit]) -> list[CaptionRenderUnit]:
    return [
        replace(caption, caption_id=f"v21_cap_{index:06d}")
        for index, caption in enumerate(sorted(captions, key=lambda row: (row.target_start_us, row.target_end_us, row.caption_id)), start=1)
    ]
