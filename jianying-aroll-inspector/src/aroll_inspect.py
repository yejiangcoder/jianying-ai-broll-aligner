from __future__ import annotations

import argparse
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from aroll_attached_effects_preservation import inspect_attached_effects
from aroll_root_mirror import root_mirrors_timeline_id
from aroll_runtime_paths import get_aroll_runs_dir
from aroll_speed_mapping import EPSILON, source_to_target_delta
from jy_bridge import (
    AI_TRACK_NAME,
    DEFAULT_JY_DRAFTC,
    assert_all_project_timeline_files_match_folder_ids,
    assert_layout_has_no_duplicate_timeline_ids,
    assert_timeline_content_id,
    decrypt,
    norm_text,
    read_json,
    resolve_timeline_id,
    write_json,
)


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = get_aroll_runs_dir()
FRAME_TOLERANCE_US = 80_000
NEAR_SCORE_DELTA = 10
PHOTO_BROLL_SEGMENT_THRESHOLD = 5


def clean_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def timerange_start(timerange: dict[str, Any] | None) -> int:
    return safe_int((timerange or {}).get("start"), 0)


def timerange_duration(timerange: dict[str, Any] | None) -> int:
    return safe_int((timerange or {}).get("duration"), 0)


def segment_start(segment: dict[str, Any]) -> int:
    return timerange_start(segment.get("target_timerange") or {})


def segment_duration(segment: dict[str, Any]) -> int:
    return timerange_duration(segment.get("target_timerange") or {})


def segment_end(segment: dict[str, Any]) -> int:
    return segment_start(segment) + segment_duration(segment)


def total_target_duration(segments: list[dict[str, Any]]) -> int:
    if not segments:
        return 0
    return max(segment_end(segment) for segment in segments)


def material_text(material: dict[str, Any]) -> str:
    text = str(material.get("recognize_text") or "")
    if text:
        return text
    content = material.get("content")
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return str(parsed.get("text") or "")
        except Exception:
            return ""
    return ""


def material_index(data: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {
        str(material.get("id") or ""): material
        for material in (data.get("materials") or {}).get(key, [])
        if material.get("id")
    }


def referenced_materials(
    segment: dict[str, Any],
    material_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    refs = segment.get("extra_material_refs") or []
    if not isinstance(refs, list):
        return []
    return [material_map[ref] for ref in refs if ref in material_map]


def has_any_key_deep(value: Any, names: set[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in names:
                return True
            if has_any_key_deep(child, names):
                return True
    elif isinstance(value, list):
        return any(has_any_key_deep(child, names) for child in value)
    return False


def contains_effect_like_ref(
    segment: dict[str, Any],
    data: dict[str, Any],
    speed_ids: set[str],
) -> bool:
    refs = set(str(ref) for ref in (segment.get("extra_material_refs") or []) if ref)
    if not refs:
        return False
    materials = data.get("materials") or {}
    safe_ref_keys = {
        "canvases",
        "placeholder_infos",
        "speeds",
        "sound_channel_mappings",
        "material_colors",
        "vocal_separations",
        "material_animations",
        "audio_fades",
        "beats",
        "loudnesses",
        "realtime_denoises",
    }
    known_safe_ids: set[str] = set(speed_ids)
    for key in safe_ref_keys:
        for material in materials.get(key, []) if isinstance(materials.get(key, []), list) else []:
            if material.get("id"):
                known_safe_ids.add(str(material.get("id")))
    suspicious_keys = {"effects", "adjusts", "filters", "video_effects", "sticker_animations"}
    for key in suspicious_keys:
        for material in materials.get(key, []) if isinstance(materials.get(key, []), list) else []:
            if str(material.get("id") or "") in refs:
                return True
    return bool(refs - known_safe_ids)


def speed_report_for_segment(
    segment: dict[str, Any],
    speeds_by_id: dict[str, dict[str, Any]],
) -> tuple[bool, bool, bool, bool, list[str], list[str]]:
    details = speed_details_for_segment(segment, speeds_by_id)
    return (
        bool(details["algorithm_1x_safe"]),
        bool(details["has_curve_speed"]),
        bool(details["has_reverse"]),
        bool(details["source_target_ratio_safe"]),
        list(details["fatal_reasons"]),
        list(details["warnings"]),
    )


def speed_details_for_segment(
    segment: dict[str, Any],
    speeds_by_id: dict[str, dict[str, Any]],
    max_allowed_speed: float = 1.25,
) -> dict[str, Any]:
    warnings: list[str] = []
    fatal_reasons: list[str] = []
    has_curve_speed = False
    has_reverse = False
    ratio_safe = True
    numeric_values: list[float] = []

    speed_refs = referenced_materials(segment, speeds_by_id)
    if not speed_refs:
        warnings.append("SEGMENT_SPEED_MATERIAL_MISSING_ASSUME_1X")
        numeric_values.append(1.0)
    for speed in speed_refs:
        speed_value = speed.get("speed")
        if speed_value is None:
            warnings.append("SEGMENT_SPEED_FIELD_MISSING_ASSUME_1X")
            numeric_speed = 1.0
        else:
            try:
                numeric_speed = float(speed_value)
            except Exception:
                numeric_speed = 0.0
                fatal_reasons.append("MAIN_VIDEO_SPEED_UNREADABLE")
        numeric_values.append(numeric_speed)
        if numeric_speed < 0:
            has_reverse = True
            fatal_reasons.append("MAIN_VIDEO_HAS_REVERSE")
        if numeric_speed == 0:
            fatal_reasons.append("MAIN_VIDEO_HAS_ZERO_SPEED")
        if has_any_key_deep(speed, {"curve_speed", "curveSpeed", "speed_curve", "curve"}):
            has_curve_speed = True
            fatal_reasons.append("MAIN_VIDEO_HAS_CURVE_SPEED")
        if has_any_key_deep(speed, {"reverse", "is_reverse", "isReverse"}):
            has_reverse = True
            fatal_reasons.append("MAIN_VIDEO_HAS_REVERSE")

    unique_values: list[float] = []
    for value in numeric_values:
        if not any(abs(value - existing) <= EPSILON for existing in unique_values):
            unique_values.append(value)
    speed_mode = "none" if not speed_refs else "constant"
    if has_curve_speed:
        speed_mode = "curve"
    elif has_reverse:
        speed_mode = "reverse"
    elif len(unique_values) > 1:
        speed_mode = "mixed"
    constant_speed = unique_values[0] if len(unique_values) == 1 else None
    speed_supported = (
        speed_mode in {"none", "constant"}
        and constant_speed is not None
        and 0.5 <= abs(float(constant_speed)) <= max_allowed_speed
        and not has_curve_speed
        and not has_reverse
    )
    if speed_mode == "mixed":
        fatal_reasons.append("MAIN_VIDEO_HAS_MIXED_SPEED")
    if constant_speed is not None and abs(float(constant_speed)) > max_allowed_speed:
        fatal_reasons.append("MAIN_VIDEO_SPEED_EXCEEDS_MAX_ALLOWED")

    source_timerange = segment.get("source_timerange")
    target_timerange = segment.get("target_timerange")
    if not isinstance(source_timerange, dict):
        ratio_safe = False
        fatal_reasons.append("MAIN_VIDEO_MISSING_SOURCE_TIMERANGE")
    if not isinstance(target_timerange, dict):
        ratio_safe = False
        fatal_reasons.append("MAIN_VIDEO_MISSING_TARGET_TIMERANGE")
    source_duration = timerange_duration(source_timerange or {})
    target_duration = timerange_duration(target_timerange or {})
    if source_duration < 0:
        has_reverse = True
        fatal_reasons.append("MAIN_VIDEO_HAS_REVERSE")
    if source_duration <= 0:
        ratio_safe = False
        fatal_reasons.append("MAIN_VIDEO_MISSING_SOURCE_TIMERANGE")
    if target_duration <= 0:
        ratio_safe = False
        fatal_reasons.append("MAIN_VIDEO_MISSING_TARGET_TIMERANGE")
    expected_target_duration = source_duration
    if source_duration > 0 and constant_speed:
        expected_target_duration = source_to_target_delta(source_duration, constant_speed)
    if source_duration > 0 and target_duration > 0 and abs(expected_target_duration - target_duration) > FRAME_TOLERANCE_US:
        ratio_safe = False
        fatal_reasons.append("MAIN_VIDEO_SOURCE_TARGET_DURATION_MISMATCH")

    speed_requires_mapping = bool(constant_speed is not None and abs(float(constant_speed) - 1.0) > EPSILON)
    return {
        "speed_mode": speed_mode,
        "constant_speed": constant_speed,
        "speed_values": unique_values,
        "speed_supported": speed_supported and ratio_safe,
        "speed_safe_for_aroll": speed_supported and ratio_safe,
        "speed_requires_mapping": speed_requires_mapping,
        "algorithm_1x_safe": (not speed_requires_mapping) and speed_supported and ratio_safe,
        "has_curve_speed": has_curve_speed,
        "has_reverse": has_reverse,
        "source_target_ratio_safe": ratio_safe,
        "expected_target_duration_us": expected_target_duration,
        "fatal_reasons": sorted(set(fatal_reasons)),
        "warnings": warnings,
    }


def summarize_material(material: dict[str, Any]) -> dict[str, Any]:
    return {
        "material_id": str(material.get("id") or ""),
        "material_name": str(material.get("material_name") or material.get("name") or ""),
        "material_path": clean_path(material.get("path")),
        "material_type": str(material.get("type") or ""),
        "duration": safe_int(material.get("duration"), 0),
        "has_audio": bool(material.get("has_audio", False)),
    }


def inspect_video_tracks(
    data: dict[str, Any],
    main_track_index: int = -1,
    main_material_path: str = "",
    max_allowed_speed: float = 1.25,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str], list[str], bool]:
    videos_by_id = material_index(data, "videos")
    speeds_by_id = material_index(data, "speeds")
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    main_material_norm = clean_path(main_material_path).lower()
    for track_index, track in enumerate(data.get("tracks") or []):
        if track.get("type") != "video":
            continue
        segments = track.get("segments") or []
        materials = [
            videos_by_id.get(str(segment.get("material_id") or ""), {})
            for segment in segments
        ]
        material_summaries = [summarize_material(material) for material in materials if material]
        material_types = [row["material_type"] for row in material_summaries]
        is_photo_only = bool(material_summaries) and all(t == "photo" for t in material_types)
        looks_like_ai_broll = str(track.get("name") or "") == AI_TRACK_NAME
        photo_count = sum(1 for t in material_types if t == "photo")
        video_count = sum(1 for t in material_types if t not in {"", "photo"})
        paths = [row["material_path"].lower() for row in material_summaries]
        looks_like_raw_aroll = any(path.endswith((".mp4", ".mov", ".m4v")) for path in paths) and video_count > 0
        if main_material_norm and any(main_material_norm in path for path in paths):
            looks_like_raw_aroll = True

        reject_reasons: list[str] = []
        segment_warnings: list[str] = []
        speed_safe = True
        has_curve_speed = False
        has_reverse = False
        source_target_ratio_safe = True
        speed_supported = True
        speed_safe_for_aroll = True
        speed_requires_mapping = False
        speed_modes: set[str] = set()
        speed_values: list[float] = []
        for segment in segments:
            details = speed_details_for_segment(segment, speeds_by_id, max_allowed_speed=max_allowed_speed)
            speed_safe = speed_safe and bool(details["algorithm_1x_safe"])
            has_curve_speed = has_curve_speed or bool(details["has_curve_speed"])
            has_reverse = has_reverse or bool(details["has_reverse"])
            source_target_ratio_safe = source_target_ratio_safe and bool(details["source_target_ratio_safe"])
            speed_supported = speed_supported and bool(details["speed_supported"])
            speed_safe_for_aroll = speed_safe_for_aroll and bool(details["speed_safe_for_aroll"])
            speed_requires_mapping = speed_requires_mapping or bool(details["speed_requires_mapping"])
            speed_modes.add(str(details["speed_mode"]))
            for value in details["speed_values"]:
                if not any(abs(float(value) - existing) <= EPSILON for existing in speed_values):
                    speed_values.append(float(value))
            reject_reasons.extend(details["fatal_reasons"])
            segment_warnings.extend(details["warnings"])
        warnings.extend(f"video_track_{track_index}:{warning}" for warning in sorted(set(segment_warnings)))

        aggregate_speed_mode = "none"
        if has_curve_speed:
            aggregate_speed_mode = "curve"
        elif has_reverse:
            aggregate_speed_mode = "reverse"
        elif len(speed_values) > 1:
            aggregate_speed_mode = "mixed"
        elif speed_values and abs(speed_values[0] - 1.0) > EPSILON:
            aggregate_speed_mode = "constant"
        constant_speed = speed_values[0] if len(speed_values) == 1 else None

        if is_photo_only:
            reject_reasons.append("PHOTO_ONLY_VIDEO_TRACK")
        if looks_like_ai_broll:
            reject_reasons.append("AI_BROLL_TRACK")
        if not looks_like_raw_aroll and not is_photo_only and segments:
            reject_reasons.append("NOT_RAW_AROLL_LIKE")

        total_duration = total_target_duration(segments)
        score = 0
        score += min(total_duration // 1_000_000, 3600)
        if looks_like_raw_aroll:
            score += 1000
        if is_photo_only:
            score -= 2000
        if looks_like_ai_broll:
            score -= 3000
        if not speed_safe or has_curve_speed or has_reverse or not source_target_ratio_safe:
            score -= 1000
        if main_track_index == track_index:
            score += 5000
        if main_material_norm and any(main_material_norm in path for path in paths):
            score += 5000

        candidates.append(
            {
                "track_index": track_index,
                "track_id": str(track.get("id") or ""),
                "track_name": str(track.get("name") or ""),
                "segment_count": len(segments),
                "total_target_duration_us": total_duration,
                "materials": material_summaries,
                "is_photo_only": is_photo_only,
                "looks_like_ai_broll": looks_like_ai_broll,
                "looks_like_raw_aroll": looks_like_raw_aroll,
                "speed_safe": speed_safe,
                "speed_mode": aggregate_speed_mode,
                "constant_speed": constant_speed,
                "speed_values": speed_values,
                "speed_supported": speed_supported,
                "speed_safe_for_aroll": speed_safe_for_aroll,
                "speed_requires_mapping": speed_requires_mapping,
                "has_curve_speed": has_curve_speed,
                "has_reverse": has_reverse,
                "source_target_ratio_safe": source_target_ratio_safe,
                "candidate_score": int(score),
                "reject_reasons": sorted(set(reject_reasons)),
            }
        )

    fatal_reasons: list[str] = []
    selected: dict[str, Any] | None = None
    viable = [
        candidate
        for candidate in candidates
        if not candidate["is_photo_only"]
        and not candidate["looks_like_ai_broll"]
        and candidate["segment_count"] > 0
        and candidate["looks_like_raw_aroll"]
    ]

    if main_track_index >= 0:
        selected = next((candidate for candidate in candidates if candidate["track_index"] == main_track_index), None)
        if selected is None:
            fatal_reasons.append("MAIN_VIDEO_TRACK_INDEX_NOT_FOUND")
    elif main_material_norm:
        matched = [
            candidate
            for candidate in viable
            if any(main_material_norm in row["material_path"].lower() for row in candidate["materials"])
        ]
        if len(matched) == 1:
            selected = matched[0]
        elif len(matched) > 1:
            fatal_reasons.append("MAIN_VIDEO_MATERIAL_MATCH_NOT_UNIQUE")
        else:
            fatal_reasons.append("MAIN_VIDEO_MATERIAL_PATH_NOT_FOUND")
    elif viable:
        ranked = sorted(viable, key=lambda row: row["candidate_score"], reverse=True)
        selected = ranked[0]
        if len(ranked) > 1 and ranked[0]["candidate_score"] - ranked[1]["candidate_score"] <= NEAR_SCORE_DELTA:
            selected = None
            fatal_reasons.append("MAIN_VIDEO_TRACK_NOT_UNIQUE")
    else:
        fatal_reasons.append("MAIN_VIDEO_TRACK_NOT_FOUND")

    main_speed_safe = (
        bool(selected)
        and bool(selected.get("speed_safe_for_aroll"))
        and bool(selected.get("speed_supported"))
        and not selected.get("has_curve_speed")
        and not selected.get("has_reverse")
        and bool(selected.get("source_target_ratio_safe"))
    )
    if selected:
        for reason in selected.get("reject_reasons") or []:
            if reason.startswith("MAIN_VIDEO_"):
                fatal_reasons.append(reason)
        if not selected.get("speed_safe_for_aroll") or not selected.get("speed_supported"):
            if selected.get("speed_requires_mapping"):
                fatal_reasons.append("MAIN_VIDEO_SPEED_MAPPING_UNSUPPORTED")
            else:
                fatal_reasons.append("MAIN_VIDEO_SPEED_UNSAFE")
        if selected.get("has_curve_speed"):
            fatal_reasons.append("MAIN_VIDEO_HAS_CURVE_SPEED")
        if selected.get("has_reverse"):
            fatal_reasons.append("MAIN_VIDEO_HAS_REVERSE")
        if not selected.get("source_target_ratio_safe"):
            fatal_reasons.append("MAIN_VIDEO_SOURCE_TARGET_DURATION_MISMATCH")

    return candidates, selected, sorted(set(fatal_reasons)), sorted(set(warnings)), main_speed_safe


def subtitle_timeline(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    texts_by_id = material_index(data, "texts")
    text_tracks: list[dict[str, Any]] = []
    selected_track: dict[str, Any] | None = None
    selected_track_index = -1
    selected_nonempty = -1

    for track_index, track in enumerate(data.get("tracks") or []):
        if track.get("type") != "text":
            continue
        segments = track.get("segments") or []
        nonempty = 0
        total_duration = total_target_duration(segments)
        for segment in segments:
            material = texts_by_id.get(str(segment.get("material_id") or ""), {})
            if material_text(material).strip():
                nonempty += 1
        summary = {
            "track_index": track_index,
            "track_id": str(track.get("id") or ""),
            "track_name": str(track.get("name") or ""),
            "segment_count": len(segments),
            "nonempty_text_count": nonempty,
            "total_target_duration_us": total_duration,
            "selected_as_subtitle_track": False,
        }
        text_tracks.append(summary)
        if selected_track is None or (nonempty, len(segments)) > (selected_nonempty, len(selected_track.get("segments") or [])):
            selected_track = track
            selected_track_index = track_index
            selected_nonempty = nonempty

    rows: list[dict[str, Any]] = []
    if selected_track is None:
        return rows, text_tracks, None

    for row in text_tracks:
        if row["track_index"] == selected_track_index:
            row["selected_as_subtitle_track"] = True

    selected_segments = sorted(
        selected_track.get("segments") or [],
        key=lambda segment: segment_start(segment),
    )
    for index, segment in enumerate(selected_segments, start=1):
        material = texts_by_id.get(str(segment.get("material_id") or ""), {})
        text = material_text(material)
        start_us = segment_start(segment)
        duration_us = segment_duration(segment)
        rows.append(
            {
                "subtitle_uid": f"sub_{index:06d}",
                "subtitle_index": index,
                "track_index": selected_track_index,
                "track_id": str(selected_track.get("id") or ""),
                "text_segment_id": str(segment.get("id") or ""),
                "text_material_id": str(segment.get("material_id") or ""),
                "subtitle_text": text,
                "norm_text": norm_text(text),
                "start_us": start_us,
                "duration_us": duration_us,
                "end_us": start_us + duration_us,
                "segment": deepcopy(segment),
                "material": deepcopy(material),
            }
        )
    return rows, text_tracks, next((row for row in text_tracks if row["selected_as_subtitle_track"]), None)


def inspect_audio_tracks(data: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, bool, list[str]]:
    audios_by_id = material_index(data, "audios")
    rows: list[dict[str, Any]] = []
    fatal_reasons: list[str] = []

    for track_index, track in enumerate(data.get("tracks") or []):
        if track.get("type") != "audio":
            continue
        segments = track.get("segments") or []
        materials: list[dict[str, Any]] = []
        source_signatures: set[str] = set()
        looks_like_bgm_or_sfx = False
        for segment in segments:
            material = audios_by_id.get(str(segment.get("material_id") or ""), {})
            if material:
                summary = summarize_material(material)
                materials.append(summary)
                source_signatures.add((summary["material_path"] or summary["material_name"] or summary["material_id"]).lower())
                material_type = summary["material_type"].lower()
                path = summary["material_path"].lower()
                name = summary["material_name"].lower()
                if any(token in material_type for token in ["music", "sound", "effect", "record"]):
                    looks_like_bgm_or_sfx = True
                if any(token in path or token in name for token in ["music", "sound", "effect", "record", "combination", "bgm", "sfx"]):
                    looks_like_bgm_or_sfx = True
        looks_like_extracted_main_audio = False
        can_sync = False
        reject_reasons: list[str] = []
        if looks_like_bgm_or_sfx:
            reject_reasons.append("AUDIO_LOOKS_LIKE_BGM_OR_SFX")
        if len(source_signatures) > 1:
            reject_reasons.append("AUDIO_MULTIPLE_SOURCES")
        # Phase 1 remains conservative: only a clearly single-source non-music audio can be marked syncable.
        if len(source_signatures) == 1 and not looks_like_bgm_or_sfx and len(rows) == 0:
            looks_like_extracted_main_audio = True
            can_sync = True
        else:
            reject_reasons.append("AUDIO_NOT_PROVEN_MAIN_SOURCE")
        rows.append(
            {
                "track_index": track_index,
                "track_id": str(track.get("id") or ""),
                "track_name": str(track.get("name") or ""),
                "segment_count": len(segments),
                "total_target_duration_us": total_target_duration(segments),
                "materials": materials,
                "looks_like_extracted_main_audio": looks_like_extracted_main_audio,
                "looks_like_bgm_or_sfx": looks_like_bgm_or_sfx,
                "can_sync_with_aroll_edl": can_sync,
                "reject_reasons": sorted(set(reject_reasons)),
            }
        )

    has_independent_audio_track = bool(rows)
    has_complex_audio = False
    if len(rows) > 1:
        has_complex_audio = True
        fatal_reasons.append("AUDIO_MULTIPLE_TRACKS")
    for row in rows:
        if not row["can_sync_with_aroll_edl"] or row["looks_like_bgm_or_sfx"]:
            has_complex_audio = True
            fatal_reasons.extend(row["reject_reasons"])
    return rows, has_independent_audio_track, has_complex_audio, sorted(set(fatal_reasons))


def inspect_filter_tracks(data: dict[str, Any], main_total_duration_us: int) -> tuple[list[dict[str, Any]], bool, bool, list[str]]:
    rows: list[dict[str, Any]] = []
    fatal_reasons: list[str] = []
    has_global_filter = False
    has_complex_filter = False
    for track_index, track in enumerate(data.get("tracks") or []):
        if track.get("type") != "filter":
            continue
        segments = track.get("segments") or []
        reject_reasons: list[str] = []
        is_global_filter = False
        is_complex_filter = False
        if len(segments) == 1:
            segment = segments[0]
            start = segment_start(segment)
            duration = segment_duration(segment)
            if start == 0 and duration >= max(0, main_total_duration_us - FRAME_TOLERANCE_US):
                is_global_filter = True
            else:
                is_complex_filter = True
                reject_reasons.append("FILTER_NOT_COVERING_MAIN_TRACK")
        elif len(segments) > 1:
            is_complex_filter = True
            reject_reasons.append("FILTER_MULTIPLE_SEGMENTS")
        has_global_filter = has_global_filter or is_global_filter
        has_complex_filter = has_complex_filter or is_complex_filter
        fatal_reasons.extend(reject_reasons)
        rows.append(
            {
                "track_index": track_index,
                "track_id": str(track.get("id") or ""),
                "track_name": str(track.get("name") or ""),
                "segment_count": len(segments),
                "is_global_filter": is_global_filter,
                "is_complex_filter": is_complex_filter,
                "total_target_duration_us": total_target_duration(segments),
                "reject_reasons": sorted(set(reject_reasons)),
            }
        )
    return rows, has_global_filter, has_complex_filter, sorted(set(fatal_reasons))


def detect_existing_broll(data: dict[str, Any], video_candidates: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for candidate in video_candidates:
        if candidate["looks_like_ai_broll"]:
            reasons.append("EXISTING_AI_BROLL_TRACK")
        if candidate["is_photo_only"] and candidate["segment_count"] >= PHOTO_BROLL_SEGMENT_THRESHOLD:
            reasons.append("EXISTING_PHOTO_ONLY_BROLL_TRACK")
    total_photo_segments = sum(
        candidate["segment_count"]
        for candidate in video_candidates
        if candidate["is_photo_only"]
    )
    if total_photo_segments >= PHOTO_BROLL_SEGMENT_THRESHOLD:
        reasons.append("EXISTING_MANY_PHOTO_SEGMENTS")
    return bool(reasons), sorted(set(reasons))


def selected_video_track(data: dict[str, Any], selected: dict[str, Any] | None) -> dict[str, Any] | None:
    if not selected:
        return None
    video_tracks = [track for track in data.get("tracks") or [] if track.get("type") == "video"]
    for track in video_tracks:
        if str(track.get("id") or "") == selected.get("track_id"):
            return track
    return None


def run_speed_mapping_checker(
    main_track: dict[str, Any],
    subtitles: list[dict[str, Any]],
    max_allowed_speed: float,
) -> dict[str, Any]:
    from aroll_speed_self_test import run_speed_mapping_self_test

    return run_speed_mapping_self_test(main_track, subtitles, max_allowed_speed)


def attached_effect_check(data: dict[str, Any], selected: dict[str, Any] | None) -> tuple[bool, list[str], list[str], dict[str, Any]]:
    report = inspect_attached_effects(data, selected)
    has_attached_effects = int(report.get("attached_ref_count") or 0) > 0
    fatal_reasons: list[str] = []
    if report.get("fatal_reasons") or not bool(report.get("attached_refs_cloneable", True)):
        fatal_reasons.append("MAIN_VIDEO_HAS_UNRECOGNIZED_ATTACHED_EFFECT_REFS")
    warnings = [str(row) for row in (report.get("warnings") or [])]
    if has_attached_effects and not warnings:
        warnings.append("ATTACHED_REFS_PRESENT_CLONE_REQUIRED")
    return has_attached_effects, sorted(set(fatal_reasons)), sorted(set(warnings)), report


def attached_effect_warnings(data: dict[str, Any], selected: dict[str, Any] | None) -> tuple[bool, list[str]]:
    has_attached_effects, _fatal_reasons, warnings, _report = attached_effect_check(data, selected)
    return has_attached_effects, warnings


def run_checks(
    draft_dir: Path,
    jy_draftc: Path,
    run_dir: Path,
    data: dict[str, Any],
    encrypted_path: Path,
    timeline_id: str,
) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "timeline_content_id_matches_folder": True,
        "project_timeline_files_match_folder_ids": True,
        "timeline_layout_has_no_duplicate_ids": True,
    }
    fatal_reasons: list[str] = []
    try:
        assert_timeline_content_id(data, timeline_id, encrypted_path)
    except Exception as exc:
        checks["timeline_content_id_matches_folder"] = False
        fatal_reasons.append(f"TIMELINE_CONTENT_ID_MISMATCH:{exc}")
    try:
        assert_layout_has_no_duplicate_timeline_ids(draft_dir)
    except Exception as exc:
        checks["timeline_layout_has_no_duplicate_ids"] = False
        fatal_reasons.append(f"TIMELINE_LAYOUT_DUPLICATE_IDS:{exc}")
    try:
        assert_all_project_timeline_files_match_folder_ids(draft_dir, jy_draftc, run_dir)
    except Exception as exc:
        checks["project_timeline_files_match_folder_ids"] = False
        fatal_reasons.append(f"PROJECT_TIMELINE_FILES_ID_MISMATCH:{exc}")
    return checks, fatal_reasons


def build_report(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = args.runtime / f"aroll_inspect_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    timeline_id, timeline_name = resolve_timeline_id(args.draft_dir, args.timeline_name)
    timeline_dir = args.draft_dir / "Timelines" / timeline_id
    encrypted_path = timeline_dir / "draft_content.json"
    plain_path = run_dir / "draft_content.dec.json"
    decrypt(args.jy_draftc, encrypted_path, plain_path)
    data = read_json(plain_path)

    fatal_reasons: list[str] = []
    warnings: list[str] = []
    timeline_checks, timeline_fatals = run_checks(args.draft_dir, args.jy_draftc, run_dir, data, encrypted_path, timeline_id)
    fatal_reasons.extend(timeline_fatals)

    root_exists = (args.draft_dir / "draft_content.json").exists()
    root_mirror_required = False
    root_mirror_matches = False
    if root_exists:
        try:
            root_mirror_required = root_mirrors_timeline_id(args.draft_dir, args.jy_draftc, run_dir, timeline_id)
            root_mirror_matches = root_mirror_required
        except Exception as exc:
            warnings.append(f"ROOT_MIRROR_CHECK_FAILED:{exc}")

    video_candidates, selected_main, video_fatals, video_warnings, main_video_speed_safe = inspect_video_tracks(
        data,
        main_track_index=args.main_video_track_index,
        main_material_path=args.main_material_path,
        max_allowed_speed=float(getattr(args, "max_allowed_speed", 1.25)),
    )
    fatal_reasons.extend(video_fatals)
    warnings.extend(video_warnings)

    subtitles, text_tracks, selected_text_track = subtitle_timeline(data)
    nonempty_subtitles = [row for row in subtitles if str(row.get("subtitle_text") or "").strip()]
    if not selected_text_track:
        fatal_reasons.append("TEXT_TRACK_NOT_FOUND")
    elif not nonempty_subtitles:
        fatal_reasons.append("SUBTITLE_TEXT_NOT_READABLE")

    speed_mapping_self_test: dict[str, Any] = {}
    speed_mapping_required = bool((selected_main or {}).get("speed_requires_mapping"))
    speed_mapping_validated = not speed_mapping_required
    if selected_main and speed_mapping_required:
        warnings.append("MAIN_VIDEO_SPEED_REQUIRES_MAPPING")
        main_track = selected_video_track(data, selected_main)
        if not main_track:
            fatal_reasons.append("MAIN_VIDEO_SPEED_MAPPING_UNSUPPORTED")
            speed_mapping_self_test = {
                "passed": False,
                "fatal_reasons": ["MAIN_VIDEO_TRACK_NOT_FOUND_FOR_SPEED_MAPPING"],
            }
        else:
            try:
                speed_mapping_self_test = run_speed_mapping_checker(
                    main_track,
                    subtitles,
                    float(getattr(args, "max_allowed_speed", 1.25)),
                )
            except Exception as exc:
                fatal_reasons.append("MAIN_VIDEO_SPEED_MAPPING_UNSUPPORTED")
                speed_mapping_self_test = {
                    "passed": False,
                    "fatal_reasons": ["SPEED_MAPPING_CHECKER_UNAVAILABLE"],
                    "error": str(exc),
                }
            else:
                speed_mapping_validated = bool(speed_mapping_self_test.get("passed"))
                if not speed_mapping_validated:
                    fatal_reasons.append("MAIN_VIDEO_SPEED_MAPPING_SELF_TEST_FAILED")
                    fatal_reasons.extend(str(reason) for reason in (speed_mapping_self_test.get("fatal_reasons") or []))

    audio_tracks, has_independent_audio_track, has_complex_audio, audio_fatals = inspect_audio_tracks(data)
    fatal_reasons.extend(f"AUDIO:{reason}" for reason in audio_fatals)

    main_total_duration = int((selected_main or {}).get("total_target_duration_us") or 0)
    filter_tracks, has_global_filter, has_complex_filter, filter_fatals = inspect_filter_tracks(data, main_total_duration)
    fatal_reasons.extend(f"FILTER:{reason}" for reason in filter_fatals)

    has_existing_broll, broll_reasons = detect_existing_broll(data, video_candidates)
    if has_existing_broll:
        fatal_reasons.extend(broll_reasons)

    has_attached_effects, attached_fatals, attached_warnings, attached_report = attached_effect_check(data, selected_main)
    if has_attached_effects:
        fatal_reasons.extend(attached_fatals)
        warnings.extend(attached_warnings)

    fatal_reasons = sorted(set(str(reason) for reason in fatal_reasons if reason))
    warnings = sorted(set(str(warning) for warning in warnings if warning))
    can_aroll_rewrite = (
        not fatal_reasons
        and bool(selected_main)
        and bool(selected_text_track)
        and bool(nonempty_subtitles)
        and not has_existing_broll
        and not has_complex_audio
        and not has_complex_filter
        and main_video_speed_safe
        and speed_mapping_validated
        and all(timeline_checks.values())
    )

    subtitle_path = run_dir / "subtitle_timeline.json"
    report_path = run_dir / "aroll_inspect_report.json"
    write_json(subtitle_path, subtitles)
    report = {
        "draft_dir": str(args.draft_dir),
        "timeline_id": timeline_id,
        "timeline_name": timeline_name,
        "runtime_dir": str(run_dir),
        "draft_content_dec_path": str(plain_path),
        "root_mirror": {
            "root_draft_content_exists": root_exists,
            "root_mirror_required": root_mirror_required,
            "root_mirror_timeline_id_matches_target": root_mirror_matches,
        },
        "timeline_id_checks": timeline_checks,
        "main_video_track_candidates": video_candidates,
        "selected_main_video_track": selected_main,
        "audio_tracks": audio_tracks,
        "text_tracks": text_tracks,
        "filter_tracks": filter_tracks,
        "attached_effects_report": attached_report,
        "has_existing_broll": has_existing_broll,
        "has_attached_effects": has_attached_effects,
        "has_independent_audio_track": has_independent_audio_track,
        "has_complex_audio": has_complex_audio,
        "has_global_filter": has_global_filter,
        "has_complex_filter": has_complex_filter,
        "main_video_speed_safe": main_video_speed_safe,
        "main_video_speed_mapping_required": speed_mapping_required,
        "main_video_speed_mapping_validated": speed_mapping_validated,
        "speed_mapping_self_test": speed_mapping_self_test,
        "can_aroll_rewrite": can_aroll_rewrite,
        "fatal_reasons": fatal_reasons,
        "warnings": warnings,
    }
    write_json(report_path, report)
    return run_dir, report_path, subtitle_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Jianying A-Roll draft structure inspector.")
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--timeline-name", default="")
    parser.add_argument("--main-video-track-index", type=int, default=-1)
    parser.add_argument("--main-material-path", default="")
    parser.add_argument("--jy-draftc", type=Path, default=DEFAULT_JY_DRAFTC)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--max-allowed-speed", type=float, default=1.25)
    args = parser.parse_args()

    run_dir, report_path, subtitle_path = build_report(args)
    report = read_json(report_path)
    print(f"status={'ok' if report.get('can_aroll_rewrite') else 'blocked'}")
    print(f"runtime={run_dir}")
    print(f"report={report_path}")
    print(f"subtitle_timeline={subtitle_path}")
    if report.get("fatal_reasons"):
        print("fatal_reasons=" + ",".join(report["fatal_reasons"]))
    if report.get("warnings"):
        print("warnings=" + ",".join(report["warnings"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
