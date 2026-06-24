from __future__ import annotations

from aroll_v21.ir.models import FinalTimelineSegment

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

