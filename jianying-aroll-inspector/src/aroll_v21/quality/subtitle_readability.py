from __future__ import annotations

from typing import Any


TARGET_MIN_CHARS = 10
TARGET_MAX_CHARS = 18
HARD_MAX_CHARS = 20
MIN_DURATION_US = 500_000
TARGET_MAX_DURATION_US = 3_000_000
HARD_MAX_DURATION_US = 3_500_000
SUBTITLE_GAP_US = 20_000
MAX_CAPTIONS_LE_3_CHARS = 3
CAPTION_DENSITY_WINDOW_US = 5_000_000
MAX_CAPTIONS_IN_5S = 8
MAX_COMPOUND_BOUNDARY_SOURCE_GAP_US = 120_000
BOUND_COMPOUND_SUFFIXES = (
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


def text_len(text: str) -> int:
    return len(str(text or "").strip())


def split_words_for_display(words: list[Any]) -> list[list[Any]]:
    subtitle_groups: list[list[Any]] = []
    current: list[Any] = []
    current_subtitle: object = object()
    for word in words:
        key = getattr(word, "subtitle_index", None)
        if key is None:
            key = getattr(word, "subtitle_uid", None)
        if current and key != current_subtitle:
            subtitle_groups.append(current)
            current = []
        current.append(word)
        current_subtitle = key
    if current:
        subtitle_groups.append(current)

    chunks: list[list[Any]] = []
    for group in subtitle_groups:
        chunks.extend(_split_group_for_display(group))
    return merge_compound_boundary_fragments(chunks)


def merge_tiny_display_fragments(chunks: list[list[Any]]) -> list[list[Any]]:
    merged: list[list[Any]] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if _chunk_text_len(chunk) < 2 and index + 1 < len(chunks):
            candidate = chunk + chunks[index + 1]
            if _chunk_text_len(candidate) <= HARD_MAX_CHARS and _chunk_duration_us(candidate) <= HARD_MAX_DURATION_US:
                merged.append(candidate)
                index += 2
                continue
        if _chunk_text_len(chunk) < 2 and merged:
            candidate = merged[-1] + chunk
            if _chunk_text_len(candidate) <= HARD_MAX_CHARS and _chunk_duration_us(candidate) <= HARD_MAX_DURATION_US:
                merged[-1] = candidate
                index += 1
                continue
        merged.append(chunk)
        index += 1
    return merged


def merge_compound_boundary_fragments(chunks: list[list[Any]]) -> list[list[Any]]:
    merged: list[list[Any]] = []
    for chunk in chunks:
        if not chunk:
            continue
        if merged and _should_merge_compound_boundary(merged[-1], chunk):
            merged[-1] = [*merged[-1], *chunk]
            continue
        merged.append(chunk)
    return merged


def fit_groups_to_segment_duration(
    groups: list[list[Any]],
    segment_duration_us: int,
    *,
    min_duration_us: int = MIN_DURATION_US,
) -> list[list[Any]]:
    current = [list(group) for group in groups if group]
    while len(current) > 1 and len(current) * int(min_duration_us) > int(segment_duration_us):
        merge_index = min(
            range(len(current) - 1),
            key=lambda index: _chunk_text_len(current[index] + current[index + 1]),
        )
        current = [*current[:merge_index], current[merge_index] + current[merge_index + 1], *current[merge_index + 2 :]]
    return current


def subtitle_interval_report(captions: list[Any]) -> dict[str, Any]:
    rows = sorted(captions, key=lambda row: (int(getattr(row, "target_start_us", 0)), int(getattr(row, "target_end_us", 0))))
    overlap_count = 0
    gap_violation_count = 0
    too_short_count = 0
    too_long_count = 0
    hard_max_char_count = 0
    captions_le_3_chars = 0
    max_chars = 0
    tiny_caption_details: list[dict[str, Any]] = []
    hard_max_char_details: list[dict[str, Any]] = []
    too_short_details: list[dict[str, Any]] = []
    too_long_details: list[dict[str, Any]] = []
    previous_end: int | None = None
    for row in rows:
        start = int(getattr(row, "target_start_us", 0))
        end = int(getattr(row, "target_end_us", 0))
        duration = max(0, end - start)
        chars = text_len(str(getattr(row, "text", "") or ""))
        max_chars = max(max_chars, chars)
        if previous_end is not None and start < previous_end:
            overlap_count += 1
        elif previous_end is not None and start < previous_end + SUBTITLE_GAP_US:
            gap_violation_count += 1
        if duration < MIN_DURATION_US:
            too_short_count += 1
            too_short_details.append(_caption_detail(row, duration_us=duration, chars=chars))
        if duration > HARD_MAX_DURATION_US:
            too_long_count += 1
            too_long_details.append(_caption_detail(row, duration_us=duration, chars=chars))
        if chars > HARD_MAX_CHARS:
            hard_max_char_count += 1
            hard_max_char_details.append(_caption_detail(row, duration_us=duration, chars=chars))
        if 0 < chars <= 3:
            captions_le_3_chars += 1
            tiny_caption_details.append(_caption_detail(row, duration_us=duration, chars=chars))
        previous_end = max(previous_end or end, end)
    max_captions_in_window, burst_count = _caption_burst_metrics(rows)
    span_us = max(0, int(getattr(rows[-1], "target_end_us", 0)) - int(getattr(rows[0], "target_start_us", 0))) if rows else 0
    captions_per_minute = round(len(rows) * 60_000_000 / span_us, 6) if span_us > 0 else 0.0
    blocker_codes: list[str] = []
    if overlap_count:
        blocker_codes.append("V21_SUBTITLE_INTERVAL_OVERLAP")
    if too_long_count:
        blocker_codes.append("V21_SUBTITLE_TOO_LONG")
    if hard_max_char_count:
        blocker_codes.append("V21_SUBTITLE_HARD_MAX_CHARS")
    if captions_le_3_chars > MAX_CAPTIONS_LE_3_CHARS:
        blocker_codes.append("V21_SUBTITLE_TINY_CAPTION_DENSITY")
    if burst_count:
        blocker_codes.append("V21_SUBTITLE_CAPTION_DENSITY_BURST")
    return {
        "target_min_chars": TARGET_MIN_CHARS,
        "target_max_chars": TARGET_MAX_CHARS,
        "hard_max_chars": HARD_MAX_CHARS,
        "min_duration_us": MIN_DURATION_US,
        "target_max_duration_us": TARGET_MAX_DURATION_US,
        "hard_max_duration_us": HARD_MAX_DURATION_US,
        "subtitle_gap_us": SUBTITLE_GAP_US,
        "captions_le_3_chars_cap": MAX_CAPTIONS_LE_3_CHARS,
        "caption_density_window_us": CAPTION_DENSITY_WINDOW_US,
        "max_captions_in_5s_threshold": MAX_CAPTIONS_IN_5S,
        "subtitle_interval_overlap_count": overlap_count,
        "subtitle_interval_gap_violation_count": gap_violation_count,
        "subtitle_interval_too_short_count": too_short_count,
        "subtitle_interval_too_long_count": too_long_count,
        "subtitle_hard_max_char_count": hard_max_char_count,
        "subtitle_max_chars": max_chars,
        "captions_le_3_chars": captions_le_3_chars,
        "caption_density_per_minute": captions_per_minute,
        "max_captions_in_5s": max_captions_in_window,
        "caption_burst_density_count": burst_count,
        "tiny_caption_details": tiny_caption_details,
        "subtitle_hard_max_char_details": hard_max_char_details,
        "subtitle_too_short_details": too_short_details,
        "subtitle_too_long_details": too_long_details,
        "subtitle_readability_gate_passed": not blocker_codes,
        "gate_passed": not blocker_codes,
        "blocker_codes": blocker_codes,
    }


def _caption_burst_metrics(rows: list[Any]) -> tuple[int, int]:
    starts = [int(getattr(row, "target_start_us", 0)) for row in rows]
    max_count = 0
    burst_count = 0
    for start in starts:
        end = start + CAPTION_DENSITY_WINDOW_US
        count = sum(1 for value in starts if start <= value < end)
        max_count = max(max_count, count)
        if count > MAX_CAPTIONS_IN_5S:
            burst_count += 1
    return max_count, burst_count


def _caption_detail(row: Any, *, duration_us: int, chars: int) -> dict[str, Any]:
    return {
        "caption_id": str(getattr(row, "caption_id", "") or ""),
        "text": str(getattr(row, "text", "") or ""),
        "target_start_us": int(getattr(row, "target_start_us", 0)),
        "target_end_us": int(getattr(row, "target_end_us", 0)),
        "duration_us": int(duration_us),
        "chars": int(chars),
        "containing_video_segment_id": getattr(row, "containing_video_segment_id", None),
    }


def _split_group_for_display(words: list[Any]) -> list[list[Any]]:
    chunks: list[list[Any]] = []
    current: list[Any] = []
    for word in words:
        candidate = [*current, word]
        if current and (_chunk_text_len(candidate) > TARGET_MAX_CHARS or _chunk_duration_us(candidate) > HARD_MAX_DURATION_US):
            chunks.append(current)
            current = [word]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _chunk_text_len(chunk: list[Any]) -> int:
    return text_len("".join(str(getattr(word, "text", "") or "") for word in chunk))


def _chunk_duration_us(chunk: list[Any]) -> int:
    if not chunk:
        return 0
    return max(0, int(getattr(chunk[-1], "source_end_us", 0)) - int(getattr(chunk[0], "source_start_us", 0)))


def _should_merge_compound_boundary(left: list[Any], right: list[Any]) -> bool:
    if not left or not right:
        return False
    if _chunk_text_len(left + right) > HARD_MAX_CHARS:
        return False
    if _chunk_duration_us(left + right) > HARD_MAX_DURATION_US:
        return False
    left_word = left[-1]
    right_word = right[0]
    left_text = str(getattr(left_word, "text", "") or "").strip()
    right_text = str(getattr(right_word, "text", "") or "").strip()
    if len(left_text) < 2 or right_text not in BOUND_COMPOUND_SUFFIXES:
        return False
    gap_us = int(getattr(right_word, "source_start_us", 0)) - int(getattr(left_word, "source_end_us", 0))
    if gap_us < -80_000 or gap_us > MAX_COMPOUND_BOUNDARY_SOURCE_GAP_US:
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
