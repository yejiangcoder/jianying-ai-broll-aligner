from __future__ import annotations

from aroll_text_normalize import normalize_text
from aroll_v21.contracts import CaptionAlignmentReport, contract_to_dict
from aroll_v21.ir.models import CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.subtitle_readability import subtitle_interval_report


MIN_CAPTION_DURATION_US = 300_000
MAX_MULTI_SEGMENT_CAPTION_TARGET_GAP_US = 120_000


def build_caption_alignment_report(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    visible_caption_track_count: int | None = None,
    caption_lane_count: int | None = None,
) -> dict:
    segments_by_id = {segment.segment_id: segment for segment in final_timeline}
    outside_count = 0
    without_container_count = 0
    orphan_caption_count = 0
    floating_caption_count = 0
    one_char_count = 0
    overlap_count = 0
    too_short_count = 0
    cross_window_count = 0
    residual_too_short_captions: list[dict] = []
    residual_one_char_captions: list[dict] = []
    previous_end: int | None = None
    interval = subtitle_interval_report(captions)
    input_order = [(int(caption.target_start_us), int(caption.target_end_us), str(caption.caption_id)) for caption in captions]
    render_order_stable = input_order == sorted(input_order)
    for caption in sorted(captions, key=lambda row: (row.target_start_us, row.target_end_us, row.caption_id)):
        if not caption.containing_video_segment_id and not caption.timeline_segment_ids:
            orphan_caption_count += 1
        container_ids = [caption.containing_video_segment_id] if caption.containing_video_segment_id else list(caption.timeline_segment_ids)
        containers = [segments_by_id[segment_id] for segment_id in container_ids if segment_id in segments_by_id]
        if not containers:
            without_container_count += 1
            floating_caption_count += 1
        elif not _caption_target_covered_by_containers(caption, containers):
            outside_count += 1
            floating_caption_count += 1
        if containers:
            crosses_primary_window = False
            spoken_start = caption.spoken_source_start_us
            spoken_end = caption.spoken_source_end_us
            caption_words = set(caption.word_ids)
            container_words = {word_id for segment in containers for word_id in segment.word_ids}
            if caption_words:
                if not caption_words <= container_words:
                    crosses_primary_window = True
            elif spoken_start is not None and spoken_end is not None:
                if not any(segment.source_start_us <= spoken_start and spoken_end <= segment.source_end_us for segment in containers):
                    crosses_primary_window = True
            if crosses_primary_window:
                cross_window_count += 1
        text = normalize_text(caption.text)
        if len(text) == 1:
            one_char_count += 1
            residual_one_char_captions.append(
                {
                    "caption_id": caption.caption_id,
                    "text": caption.text,
                    "target_start_us": int(caption.target_start_us),
                    "target_end_us": int(caption.target_end_us),
                    "duration_us": int(caption.target_end_us) - int(caption.target_start_us),
                    "containing_video_segment_id": caption.containing_video_segment_id,
                    "word_ids": list(caption.word_ids),
                }
            )
        if int(caption.target_end_us) - int(caption.target_start_us) < MIN_CAPTION_DURATION_US:
            too_short_count += 1
            residual_too_short_captions.append(
                {
                    "caption_id": caption.caption_id,
                    "text": caption.text,
                    "target_start_us": int(caption.target_start_us),
                    "target_end_us": int(caption.target_end_us),
                    "duration_us": int(caption.target_end_us) - int(caption.target_start_us),
                    "containing_video_segment_id": caption.containing_video_segment_id,
                    "word_ids": list(caption.word_ids),
                }
            )
        if previous_end is not None and caption.target_start_us < previous_end:
            overlap_count += 1
        previous_end = max(previous_end or caption.target_end_us, caption.target_end_us)
    blocker_codes = []
    if outside_count:
        blocker_codes.append("V21_CAPTION_OUTSIDE_VIDEO_SEGMENT")
    if overlap_count:
        blocker_codes.append("V21_CAPTION_OVERLAP")
    if one_char_count:
        blocker_codes.append("V21_ONE_CHAR_CAPTION")
    if too_short_count:
        blocker_codes.append("V21_CAPTION_TOO_SHORT")
    if without_container_count:
        blocker_codes.append("V21_CAPTION_WITHOUT_VIDEO_CONTAINER")
    if cross_window_count:
        blocker_codes.append("V21_CAPTION_CROSSES_PRIMARY_SOURCE_WINDOW")
    effective_visible_track_count = int(visible_caption_track_count if visible_caption_track_count is not None else (1 if captions else 0))
    effective_lane_count = int(caption_lane_count if caption_lane_count is not None else (1 if captions else 0))
    gui_gate_passed = (
        effective_visible_track_count == 1
        and effective_lane_count == 1
        and orphan_caption_count == 0
        and floating_caption_count == 0
        and without_container_count == 0
        and render_order_stable
    )
    if not gui_gate_passed:
        blocker_codes.append("V21_CAPTION_GUI_TRACK_GATE_FAILED")
    readability_codes = [str(code) for code in interval.get("blocker_codes") or []]
    if readability_codes:
        blocker_codes.extend(readability_codes)
        blocker_codes.append("V21_SUBTITLE_READABILITY_GATE_FAILED")
    blocker_codes = _unique(blocker_codes)
    report = contract_to_dict(
        CaptionAlignmentReport(
            gate_passed=not blocker_codes,
            caption_count=len(captions),
            caption_outside_video_count=outside_count,
            caption_overlap_count=overlap_count,
            caption_too_short_count=too_short_count,
            one_char_caption_count=one_char_count,
            caption_without_video_container_count=without_container_count,
            caption_cross_primary_window_count=cross_window_count,
            blocker_codes=blocker_codes,
        )
    )
    report.update(
        {
            "caption_too_short_count": too_short_count,
            "caption_cross_primary_window_count": cross_window_count,
            "caption_alignment_ok": not blocker_codes,
            "caption_gui_track_gate_passed": gui_gate_passed,
            "visible_caption_track_count": effective_visible_track_count,
            "caption_lane_count": effective_lane_count,
            "orphan_caption_count": orphan_caption_count,
            "floating_caption_count": floating_caption_count,
            "caption_render_order_stable": render_order_stable,
            "subtitle_readability_report": interval,
            "subtitle_readability_gate_passed": bool(interval.get("subtitle_readability_gate_passed")),
            "subtitle_interval_overlap_count": int(interval.get("subtitle_interval_overlap_count") or 0),
            "subtitle_interval_gap_violation_count": int(interval.get("subtitle_interval_gap_violation_count") or 0),
            "subtitle_interval_too_short_count": int(interval.get("subtitle_interval_too_short_count") or 0),
            "subtitle_interval_too_long_count": int(interval.get("subtitle_interval_too_long_count") or 0),
            "subtitle_hard_max_char_count": int(interval.get("subtitle_hard_max_char_count") or 0),
            "captions_le_3_chars": int(interval.get("captions_le_3_chars") or 0),
            "captions_le_3_chars_cap": int(interval.get("captions_le_3_chars_cap") or 0),
            "caption_density_per_minute": float(interval.get("caption_density_per_minute") or 0.0),
            "max_captions_in_5s": int(interval.get("max_captions_in_5s") or 0),
            "caption_burst_density_count": int(interval.get("caption_burst_density_count") or 0),
            "caption_density_window_us": int(interval.get("caption_density_window_us") or 0),
            "max_captions_in_5s_threshold": int(interval.get("max_captions_in_5s_threshold") or 0),
            "tiny_caption_details": list(interval.get("tiny_caption_details") or []),
            "subtitle_hard_max_char_details": list(interval.get("subtitle_hard_max_char_details") or []),
            "subtitle_too_short_details": list(interval.get("subtitle_too_short_details") or []),
            "subtitle_too_long_details": list(interval.get("subtitle_too_long_details") or []),
            "residual_too_short_captions": residual_too_short_captions,
            "residual_one_char_captions": residual_one_char_captions,
        }
    )
    return report


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _caption_target_covered_by_containers(caption: CaptionRenderUnit, containers: list[FinalTimelineSegment]) -> bool:
    start = int(caption.target_start_us)
    end = int(caption.target_end_us)
    if any(int(segment.target_start_us) <= start and end <= int(segment.target_end_us) for segment in containers):
        return True
    ordered = sorted(containers, key=lambda segment: (int(segment.target_start_us), int(segment.target_end_us), segment.segment_id))
    if len(ordered) < 2 or int(ordered[0].target_start_us) > start or int(ordered[-1].target_end_us) < end:
        return False
    cursor = start
    for segment in ordered:
        segment_start = int(segment.target_start_us)
        segment_end = int(segment.target_end_us)
        if segment_end <= cursor:
            continue
        if segment_start > cursor + MAX_MULTI_SEGMENT_CAPTION_TARGET_GAP_US:
            return False
        cursor = max(cursor, segment_end)
        if cursor >= end:
            return True
    return cursor >= end
