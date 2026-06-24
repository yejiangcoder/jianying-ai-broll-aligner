from __future__ import annotations

from dataclasses import replace

from aroll_v21.ir.models import FinalTimelineSegment

def _repack(segments: list[FinalTimelineSegment]) -> list[FinalTimelineSegment]:
    cursor = 0
    repacked = []
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

