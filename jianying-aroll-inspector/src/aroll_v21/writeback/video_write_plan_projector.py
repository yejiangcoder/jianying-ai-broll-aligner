from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SafeHandleProjection:
    policy_enabled: bool
    source_start_us: int
    source_end_us: int
    target_start_us: int
    target_end_us: int
    lead_requested_us: int = 0
    tail_requested_us: int = 0
    lead_applied_us: int = 0
    tail_applied_us: int = 0
    blocked_reasons: list[str] = field(default_factory=list)


class SafeHandlePolicy:
    def project(self, final_segment: Any) -> SafeHandleProjection:
        spoken_start = int(
            final_segment.spoken_source_start_us
            if getattr(final_segment, "spoken_source_start_us", None) is not None
            else final_segment.source_start_us
        )
        spoken_end = int(
            final_segment.spoken_source_end_us
            if getattr(final_segment, "spoken_source_end_us", None) is not None
            else final_segment.source_end_us
        )
        target_start = int(final_segment.target_start_us)
        target_end = int(final_segment.target_end_us)
        debug_hints = dict(getattr(final_segment, "debug_hints", None) or {})
        if not bool(debug_hints.get("safe_handle_policy_enabled")):
            return SafeHandleProjection(
                policy_enabled=False,
                source_start_us=spoken_start,
                source_end_us=spoken_end,
                target_start_us=target_start,
                target_end_us=target_end,
            )
        clip_start = int(
            final_segment.clip_source_start_us
            if getattr(final_segment, "clip_source_start_us", None) is not None
            else spoken_start
        )
        clip_end = int(
            final_segment.clip_source_end_us
            if getattr(final_segment, "clip_source_end_us", None) is not None
            else spoken_end
        )
        window_start = int(debug_hints.get("safe_handle_source_window_start_us", clip_start))
        window_end = int(debug_hints.get("safe_handle_source_window_end_us", clip_end))
        forbidden_ranges = [
            row
            for row in list(debug_hints.get("safe_handle_forbidden_source_ranges") or [])
            if isinstance(row, dict)
        ]
        source_start = spoken_start
        source_end = spoken_end
        projected_target_start = target_start
        projected_target_end = target_end
        blocked_reasons: list[str] = []
        lead_requested = max(0, spoken_start - clip_start)
        tail_requested = max(0, clip_end - spoken_end)
        if lead_requested:
            proposed_source_start = clip_start
            proposed_target_start = target_start - lead_requested
            reason = _blocked_handle_reason(
                handle_start_us=proposed_source_start,
                handle_end_us=spoken_start,
                source_window_start_us=window_start,
                source_window_end_us=window_end,
                forbidden_ranges=forbidden_ranges,
                proposed_target_start_us=proposed_target_start,
                proposed_target_end_us=target_start,
                previous_target_end_us=debug_hints.get("safe_handle_previous_target_end_us"),
                next_target_start_us=None,
            )
            if reason:
                blocked_reasons.append(f"lead:{reason}")
            else:
                source_start = proposed_source_start
                projected_target_start = proposed_target_start
        if tail_requested:
            proposed_source_end = clip_end
            proposed_target_end = target_end + tail_requested
            reason = _blocked_handle_reason(
                handle_start_us=spoken_end,
                handle_end_us=proposed_source_end,
                source_window_start_us=window_start,
                source_window_end_us=window_end,
                forbidden_ranges=forbidden_ranges,
                proposed_target_start_us=target_end,
                proposed_target_end_us=proposed_target_end,
                previous_target_end_us=None,
                next_target_start_us=debug_hints.get("safe_handle_next_target_start_us"),
            )
            if reason:
                blocked_reasons.append(f"tail:{reason}")
            else:
                source_end = proposed_source_end
                projected_target_end = proposed_target_end
        return SafeHandleProjection(
            policy_enabled=True,
            source_start_us=source_start,
            source_end_us=source_end,
            target_start_us=projected_target_start,
            target_end_us=projected_target_end,
            lead_requested_us=lead_requested,
            tail_requested_us=tail_requested,
            lead_applied_us=max(0, spoken_start - source_start),
            tail_applied_us=max(0, source_end - spoken_end),
            blocked_reasons=blocked_reasons,
        )


def project_video_segment_from_template(
    template: dict[str, Any],
    final_segment: Any,
    index: int,
    speed: float,
    *,
    safe_handle_policy: SafeHandlePolicy | None = None,
) -> dict[str, Any]:
    row = deepcopy(template)
    row.pop("_resolved_current_material_template", None)
    handle_projection = (safe_handle_policy or SafeHandlePolicy()).project(final_segment)
    source_start_us = handle_projection.source_start_us
    source_end_us = handle_projection.source_end_us
    material_start = source_timeline_to_material_time(
        source_start_us,
        timerange_start(template.get("target_timerange")),
        timerange_start(template.get("source_timerange")),
        speed,
    )
    material_end = source_timeline_to_material_time(
        source_end_us,
        timerange_start(template.get("target_timerange")),
        timerange_start(template.get("source_timerange")),
        speed,
    )
    row["id"] = f"v21_video_segment_{index:06d}"
    resolved_material_id = str(template.get("material_id") or template.get("materialId") or final_segment.source_material_id)
    row["material_id"] = resolved_material_id
    if "materialId" in row:
        row["materialId"] = resolved_material_id
    row["source_timerange"] = {
        "start": material_start,
        "duration": max(0, material_end - material_start),
    }
    row["target_timerange"] = {
        "start": handle_projection.target_start_us,
        "duration": max(0, handle_projection.target_end_us - handle_projection.target_start_us),
    }
    row["_safe_handle_projection"] = {
        "safe_handle_policy_enabled": handle_projection.policy_enabled,
        "source_start_us": handle_projection.source_start_us,
        "source_end_us": handle_projection.source_end_us,
        "target_start_us": handle_projection.target_start_us,
        "target_end_us": handle_projection.target_end_us,
        "lead_handle_requested_us": handle_projection.lead_requested_us,
        "tail_handle_requested_us": handle_projection.tail_requested_us,
        "lead_handle_applied_us": handle_projection.lead_applied_us,
        "tail_handle_applied_us": handle_projection.tail_applied_us,
        "handle_blocked_reasons": list(handle_projection.blocked_reasons),
    }
    return row


def source_timeline_to_material_time(
    source_timeline_time_us: int,
    segment_target_start_us: int,
    segment_source_start_us: int,
    speed: float | int | str | None,
) -> int:
    display_offset = int(source_timeline_time_us) - int(segment_target_start_us)
    return int(segment_source_start_us) + int(round(display_offset * float(speed or 1.0)))


def timerange_start(value: Any) -> int:
    return int(value.get("start") or 0) if isinstance(value, dict) else 0


def timerange_duration(value: Any) -> int:
    return int(value.get("duration") or 0) if isinstance(value, dict) else 0


def safe_handle_report_from_projected_segments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    projections = [
        row.get("_safe_handle_projection")
        for row in rows
        if isinstance(row.get("_safe_handle_projection"), dict)
    ]
    reason_counts: dict[str, int] = {}
    for projection in projections:
        for reason in projection.get("handle_blocked_reasons") or []:
            reason_text = str(reason)
            reason_counts[reason_text] = reason_counts.get(reason_text, 0) + 1
    return {
        "safe_handle_policy_enabled": any(bool(row.get("safe_handle_policy_enabled")) for row in projections),
        "lead_handle_requested_count": sum(1 for row in projections if int(row.get("lead_handle_requested_us") or 0) > 0),
        "tail_handle_requested_count": sum(1 for row in projections if int(row.get("tail_handle_requested_us") or 0) > 0),
        "lead_handle_applied_count": sum(1 for row in projections if int(row.get("lead_handle_applied_us") or 0) > 0),
        "tail_handle_applied_count": sum(1 for row in projections if int(row.get("tail_handle_applied_us") or 0) > 0),
        "segments_with_no_lead_handle": sum(
            1
            for row in projections
            if bool(row.get("safe_handle_policy_enabled"))
            and int(row.get("lead_handle_applied_us") or 0) <= 0
        ),
        "segments_with_no_tail_handle": sum(
            1
            for row in projections
            if bool(row.get("safe_handle_policy_enabled"))
            and int(row.get("tail_handle_applied_us") or 0) <= 0
        ),
        "handle_blocked_count": sum(len(row.get("handle_blocked_reasons") or []) for row in projections),
        "handle_blocked_reasons": dict(sorted(reason_counts.items())),
    }


def _blocked_handle_reason(
    *,
    handle_start_us: int,
    handle_end_us: int,
    source_window_start_us: int,
    source_window_end_us: int,
    forbidden_ranges: list[dict[str, Any]],
    proposed_target_start_us: int,
    proposed_target_end_us: int,
    previous_target_end_us: Any,
    next_target_start_us: Any,
) -> str:
    if int(handle_start_us) < int(source_window_start_us) or int(handle_end_us) > int(source_window_end_us):
        return "crosses_source_window"
    for row in forbidden_ranges:
        start = int(row.get("start_us") or row.get("source_start_us") or 0)
        end = int(row.get("end_us") or row.get("source_end_us") or 0)
        if int(handle_start_us) < end and start < int(handle_end_us):
            reason = str(row.get("reason") or "crosses_dropped_content")
            return reason
    if int(proposed_target_start_us) < 0:
        return "target_before_zero"
    if previous_target_end_us is not None and int(proposed_target_start_us) < int(previous_target_end_us):
        return "target_overlap_previous"
    if next_target_start_us is not None and int(proposed_target_end_us) > int(next_target_start_us):
        return "target_overlap_next"
    no_reason = ""
    return no_reason
