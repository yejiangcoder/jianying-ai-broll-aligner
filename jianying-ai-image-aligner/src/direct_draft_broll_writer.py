from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from draft_runtime_binding import (
    DraftRuntimeBinding,
    assert_all_project_timeline_files_match_folder_ids,
    assert_layout_has_no_duplicate_timeline_ids,
    assert_timeline_content_id,
    decrypt,
    encrypt,
    read_json,
    sha256,
    write_json,
)
from runtime_paths import get_runs_dir


SEC = 1_000_000
DEFAULT_JY_DRAFTC = None
DEFAULT_RUNTIME = get_runs_dir()
AI_TRACK_NAME = "AI_BROLL"
MIN_MATCH_CONFIDENCE = 0.45
SOURCE_DURATION_TOLERANCE_US = 1


def guid() -> str:
    return str(uuid.uuid4()).upper()


def normalize_image_id(value: str) -> str:
    match = re.search(r"\d+", str(value or ""))
    if not match:
        return ""
    digits = match.group(0)
    return digits.zfill(2) if len(digits) < 2 else digits


def image_id_sort_key(image_id: str) -> tuple[int, str]:
    return (int(normalize_image_id(image_id) or 0), normalize_image_id(image_id))


def image_id_from_filename(path: Path) -> str:
    match = re.search(r"(?:^|_)AI_(\d+)(?:_|$)", path.stem)
    return normalize_image_id(match.group(1)) if match else ""


def parse_id_list(value: str) -> set[str]:
    ids = set()
    for part in re.split(r"[,，\s]+", value or ""):
        image_id = normalize_image_id(part)
        if image_id:
            ids.add(image_id)
    return ids


def clean_text(text: str) -> str:
    return (text or "").strip().strip("“”\"' ")


def norm_text(text: str) -> str:
    text = (text or "").lower()
    table = str.maketrans(
        "，。！？；：、“”‘’（）【】《》—…·　,.!?;:\"'()[]<>-",
        " " * len("，。！？；：、“”‘’（）【】《》—…·　,.!?;:\"'()[]<>-"),
    )
    text = text.translate(table)
    text = re.sub(r"\s+", "", text)
    text = text.replace("国南", "国男").replace("0", "零")
    return text


def text_score(target: str, candidate: str) -> float:
    a = norm_text(target)
    b = norm_text(candidate)
    return text_score_norm(a, b)


def text_score_norm(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b:
        return min(0.99, 0.88 + len(a) / max(1, len(b)) * 0.11)
    if b in a and len(b) >= 4:
        return min(0.90, 0.65 + len(b) / max(1, len(a)) * 0.25)
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    overlap = len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
    dice = dice_score(a, b)
    return max(seq * 0.55 + overlap * 0.25 + dice * 0.20, overlap * 0.78, dice * 0.82)


def bigrams(value: str) -> set[str]:
    if not value:
        return set()
    if len(value) == 1:
        return {value}
    return {value[i : i + 2] for i in range(len(value) - 1)}


def dice_score(a: str, b: str) -> float:
    left = bigrams(a)
    right = bigrams(b)
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / max(1, len(left) + len(right))


def match_method_for_norm(target_norm: str, candidate_norm: str) -> str:
    if not target_norm or not candidate_norm:
        return "empty"
    if target_norm == candidate_norm:
        return "exact"
    if target_norm in candidate_norm:
        return "target_contains"
    if candidate_norm in target_norm and len(candidate_norm) >= 4:
        return "candidate_contains"
    return "normalized_fuzzy"


def match_method_rank(method: str) -> int:
    return {
        "exact": 4,
        "target_contains": 3,
        "candidate_contains": 2,
        "normalized_fuzzy": 1,
        "empty": 0,
    }.get(method, 0)


def normalized_image_files(image_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(image_dir.glob("*.png")):
        image_id = image_id_from_filename(path)
        if image_id and image_id not in files:
            files[image_id] = path
    return dict(sorted(files.items(), key=lambda item: image_id_sort_key(item[0])))


def find_image(image_dir: Path, image_id: str, files: dict[str, Path] | None = None) -> Path:
    image_id = normalize_image_id(image_id)
    files = files or normalized_image_files(image_dir)
    if image_id not in files:
        raise FileNotFoundError(f"缺少 AI 图片 {image_id}: {image_dir}")
    return files[image_id]


def broll_table_ai_ids(broll_path: Path) -> list[str]:
    ids = []
    for line in broll_path.read_text("utf-8").splitlines():
        if not line.startswith("|") or "AI静态图" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] not in {"序号", "---"}:
            image_id = normalize_image_id(cells[0])
            if image_id:
                ids.append(image_id)
    return sorted(set(ids), key=image_id_sort_key)


def broll_static_list_ids(broll_path: Path) -> list[str]:
    text = broll_path.read_text("utf-8")
    ids = []
    for match in re.finditer(r"(?ms)^【(\d+)】\s*(.*?)(?=^【\d+】|^#\s*\d+\.|\Z)", text):
        body = match.group(2)
        if "画面名称：" in body and "台词落点：" in body:
            ids.append(normalize_image_id(match.group(1)))
    return sorted(set(ids), key=image_id_sort_key)


def parse_broll_items(broll_path: Path, image_dir: Path) -> list[dict[str, Any]]:
    text = broll_path.read_text("utf-8")
    image_files = normalized_image_files(image_dir)
    by_id: dict[str, dict[str, Any]] = {}

    def add_item(image_id: str, target_text: str, image_name: str, source: str, prompt: str = "") -> None:
        target_text = clean_text(target_text)
        if not image_id or not target_text:
            return
        try:
            path = find_image(image_dir, image_id, image_files)
        except FileNotFoundError:
            return
        old = by_id.get(image_id)
        if old and old.get("source") == "ai_static_list":
            return
        by_id[image_id] = {
            "image_id": image_id,
            "image_path": path,
            "image_name": clean_text(image_name) or path.stem,
            "target_text": target_text,
            "source": source,
            "prompt": prompt,
        }

    for line in text.splitlines():
        if not line.startswith("|") or "AI静态图" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] in {"序号", "---"}:
            continue
        image_id = normalize_image_id(cells[0])
        if not image_id:
            continue
        # Current and historical B-roll tables may contain either:
        # 序号/优先级/台词落点/B-roll类型/画面设计/音效
        # or 序号/优先级/对齐台词序号/对齐台词起句/台词落点/B-roll类型/...
        if len(cells) >= 8 and cells[5] == "AI静态图":
            target_text = cells[3] or cells[4]
            prompt = cells[6]
        else:
            target_text = cells[2]
            prompt = cells[4] if len(cells) > 4 else ""
        add_item(image_id, target_text, "", "broll_table", prompt)

    for match in re.finditer(r"(?ms)^【(\d+)】\s*(.*?)(?=^【\d+】|^#\s*\d+\.|\Z)", text):
        image_id = normalize_image_id(match.group(1))
        body = match.group(2)
        if "画面名称：" not in body:
            continue
        quote = re.search(r"^台词落点：(.+)$", body, flags=re.MULTILINE)
        align_quote = re.search(r"^对齐台词起句(?:（可选）)?：(.+)$", body, flags=re.MULTILINE)
        name = re.search(r"^画面名称：(.+)$", body, flags=re.MULTILINE)
        if not quote:
            continue
        prompt = ""
        prompt_match = re.search(r"^画面方向：(.+)$", body, flags=re.MULTILINE)
        if prompt_match:
            prompt = prompt_match.group(1)
        add_item(
            image_id,
            align_quote.group(1) if align_quote else quote.group(1),
            clean_text(name.group(1)) if name else "",
            "ai_static_list",
            prompt,
        )

    items = list(by_id.values())
    items.sort(key=lambda row: image_id_sort_key(row["image_id"]))
    return items


def unnormalized_png_files(image_dir: Path) -> list[Path]:
    return [path for path in sorted(image_dir.glob("*.png")) if not image_id_from_filename(path)]


def validate_broll_image_contract(broll_path: Path, image_dir: Path) -> tuple[list[str], dict[str, Path]]:
    image_files = normalized_image_files(image_dir)
    image_ids = list(image_files.keys())
    invalid_images = unnormalized_png_files(image_dir)
    if invalid_images:
        names = [path.name for path in invalid_images]
        raise RuntimeError(f"AI 图片目录存在不规范 PNG 文件名，必须包含 _AI_<number>_：{names}")
    if not image_ids:
        raise RuntimeError(f"AI 图片目录没有规范命名图片：{image_dir}")
    table_ids = broll_table_ai_ids(broll_path)
    static_list_ids = broll_static_list_ids(broll_path)
    if not table_ids:
        raise RuntimeError("B-roll 设计稿没有可校验的 AI静态图表格编号")
    if not static_list_ids:
        raise RuntimeError("B-roll 设计稿没有可校验的 AI 静态图清单编号")
    if table_ids != image_ids:
        raise RuntimeError(f"B-roll 表格 AI 编号与图片目录不一致：table={table_ids}, images={image_ids}")
    if static_list_ids != image_ids:
        raise RuntimeError(f"AI 静态图清单编号与图片目录不一致：static_list={static_list_ids}, images={image_ids}")
    parsed_ids = [item["image_id"] for item in parse_broll_items(broll_path, image_dir)]
    if parsed_ids != image_ids:
        raise RuntimeError(f"B-roll 解析 AI 编号与图片目录不一致：parsed={parsed_ids}, images={image_ids}")
    return image_ids, image_files


def visual_slot_rows(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        if raw.get("ready_for_image_alignment") is False:
            raise RuntimeError("visual_slot_plan 未通过上游校验：ready_for_image_alignment=false")
        rows = []
        for key in ("slots", "visual_slots", "image_slots"):
            if isinstance(raw.get(key), list):
                rows = raw[key]
                break
    else:
        rows = []
    if not rows:
        raise RuntimeError("visual_slot_plan.json 缺少 slots 列表")
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("visual_slot_plan.json 的 slots 必须是对象列表")
    return rows


def int_field(row: dict[str, Any], *names: str, required: bool = True) -> int:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        return int(value)
    if required:
        raise RuntimeError(f"slot 缺少字段：{'/'.join(names)}")
    return 0


def resolve_slot_image_path(
    row: dict[str, Any],
    image_files: dict[str, Path],
    image_dir: Path,
    plan_dir: Path,
) -> tuple[str, Path]:
    raw_path = str(row.get("image_path") or "").strip()
    raw_id = str(row.get("image_id") or row.get("slot_id") or "").strip()
    path_image_id = image_id_from_filename(Path(raw_path)) if raw_path else ""
    declared_image_id = normalize_image_id(raw_id)
    if declared_image_id and path_image_id and declared_image_id != path_image_id:
        raise RuntimeError(
            "slot image_id 与 image_path 文件名编号不一致："
            f"slot={row.get('slot_id')} image_id={declared_image_id} image_path_id={path_image_id}"
        )
    image_id = path_image_id or declared_image_id
    if image_id not in image_files:
        raise RuntimeError(f"slot 图片编号不在规范图片目录中：slot={row.get('slot_id')} image_id={image_id}")
    expected_path = image_files[image_id]
    if not raw_path:
        return image_id, expected_path
    candidate = Path(raw_path)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates = [plan_dir / candidate, image_dir / candidate]
    existing = next((path for path in candidates if path.exists()), None)
    if not existing:
        raise FileNotFoundError(f"slot image_path 不存在：slot={row.get('slot_id')} image_path={raw_path}")
    if existing.resolve() != expected_path.resolve():
        raise RuntimeError(
            "slot image_path 与当前 ImageDir 的规范图片不一致："
            f"slot={row.get('slot_id')} plan={existing} image_dir={expected_path}"
        )
    return image_id, expected_path


def load_visual_slot_plan(plan_path: Path, image_dir: Path, image_files: dict[str, Path]) -> list[dict[str, Any]]:
    raw = read_json(plan_path)
    rows = visual_slot_rows(raw)
    slots = []
    seen_slot_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        slot_id = clean_text(str(row.get("slot_id") or f"broll_{index:03d}"))
        if slot_id in seen_slot_ids:
            raise RuntimeError(f"visual_slot_plan slot_id 重复：{slot_id}")
        seen_slot_ids.add(slot_id)
        image_id, image_path = resolve_slot_image_path(row, image_files, image_dir, plan_path.parent)
        target_start_us = int_field(row, "target_start_us", "start_us")
        target_end_us = int_field(row, "target_end_us", "end_us", required=False)
        duration_us = int_field(row, "duration_us", required=False)
        if target_end_us <= 0 and duration_us > 0:
            target_end_us = target_start_us + duration_us
        if target_end_us <= target_start_us:
            raise RuntimeError(f"slot 时间区间非法：slot={slot_id} start={target_start_us} end={target_end_us}")
        source_start_us = int_field(row, "source_start_us", required=False)
        source_end_us = int_field(row, "source_end_us", required=False)
        if source_end_us and source_end_us < source_start_us:
            raise RuntimeError(f"slot source 区间非法：slot={slot_id}")
        container_ids = row.get("container_video_segment_ids") or row.get("container_segment_ids") or []
        if isinstance(container_ids, str):
            container_ids = [part.strip() for part in re.split(r"[,，\s]+", container_ids) if part.strip()]
        if not isinstance(container_ids, list) or not container_ids:
            raise RuntimeError(f"slot 缺少 container_video_segment_ids：slot={slot_id}")
        slot = {
            "slot_id": slot_id,
            "image_id": image_id,
            "image_path": image_path,
            "image_name": clean_text(str(row.get("image_name") or image_path.stem)),
            "text": clean_text(str(row.get("text") or row.get("target_text") or "")),
            "target_text": clean_text(str(row.get("text") or row.get("target_text") or "")),
            "start_us": target_start_us,
            "end_us": target_end_us,
            "duration_us": target_end_us - target_start_us,
            "source_start_us": source_start_us,
            "source_end_us": source_end_us,
            "container_video_segment_ids": [str(value) for value in container_ids],
            "match_method": "visual_slot_plan",
            "confidence": row.get("confidence", row.get("match_confidence", "")),
            "matched_window": row.get("matched_window", ""),
            "window_size": row.get("window_size", ""),
            "window_start_index": row.get("window_start_index", ""),
            "window_end_index": row.get("window_end_index", ""),
            "subtitle_index": row.get("subtitle_index", ""),
            "subtitle_text": row.get("subtitle_text", row.get("text", "")),
            "nudged_us": 0,
        }
        slot["start_sec"] = slot["start_us"] / SEC
        slot["duration_sec"] = slot["duration_us"] / SEC
        slots.append(slot)
    slots.sort(key=lambda row: (int(row["start_us"]), image_id_sort_key(row["image_id"])))
    return slots


def validate_slot_plan_ids(slots: list[dict[str, Any]], image_ids: list[str]) -> None:
    slot_ids = [slot["image_id"] for slot in sorted(slots, key=lambda row: image_id_sort_key(row["image_id"]))]
    if slot_ids != image_ids:
        raise RuntimeError(f"visual_slot_plan 图片编号与 ImageDir 不一致：slots={slot_ids}, images={image_ids}")
    if len(set(slot_ids)) != len(slot_ids):
        raise RuntimeError(f"visual_slot_plan 图片编号重复：{slot_ids}")


def validate_no_slot_overlaps(slots: list[dict[str, Any]]) -> None:
    ordered = sorted(slots, key=lambda row: (int(row["start_us"]), int(row["end_us"])))
    for left, right in zip(ordered, ordered[1:]):
        if int(left["end_us"]) > int(right["start_us"]):
            raise RuntimeError(
                "visual_slot_plan 存在同轨重叠 slot："
                f"{left['slot_id']}({left['start_us']}-{left['end_us']}) "
                f"{right['slot_id']}({right['start_us']}-{right['end_us']})"
            )


def validate_slot_confidence(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    low = []
    for slot in slots:
        value = slot.get("confidence")
        if value is None or value == "":
            continue
        try:
            if float(value) < MIN_MATCH_CONFIDENCE:
                low.append(slot)
        except (TypeError, ValueError):
            raise RuntimeError(f"slot confidence 不是数字：slot={slot['slot_id']} confidence={value}")
    if low:
        ids = ",".join(slot["image_id"] for slot in low)
        raise RuntimeError(f"visual_slot_plan 存在低置信字幕匹配，停止写入：{ids}")
    return low


def draft_video_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    videos = {m["id"]: m for m in data["materials"].get("videos", [])}
    rows = []
    for track_index, track in enumerate(data.get("tracks", [])):
        if track.get("type") != "video" or track.get("name") == AI_TRACK_NAME:
            continue
        for segment in track.get("segments", []):
            timerange = segment.get("target_timerange") or {}
            start = int(timerange.get("start") or 0)
            duration = int(timerange.get("duration") or 0)
            if duration <= 0:
                continue
            material = videos.get(segment.get("material_id"), {})
            rows.append(
                {
                    "segment_id": str(segment.get("id") or ""),
                    "track_index": track_index,
                    "track_name": track.get("name"),
                    "material_id": segment.get("material_id"),
                    "material_name": material.get("material_name"),
                    "start_us": start,
                    "end_us": start + duration,
                    "duration_us": duration,
                }
            )
    return rows


def interval_is_contained(start_us: int, end_us: int, intervals: list[tuple[int, int]]) -> bool:
    if not intervals:
        return False
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return any(start_us >= start and end_us <= end for start, end in merged)


def assert_slots_inside_video_segments(slots: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    video_rows = draft_video_segments(data)
    if not video_rows:
        raise RuntimeError("当前 active timeline 没有可承载图片的 video segment")
    by_id = {row["segment_id"]: row for row in video_rows if row["segment_id"]}
    final_video_end = max(row["end_us"] for row in video_rows)
    errors = []
    for slot in slots:
        start_us = int(slot["start_us"])
        end_us = int(slot["end_us"])
        if end_us > final_video_end:
            errors.append(f"{slot['slot_id']}:IMAGE_AFTER_FINAL_VIDEO_END")
            continue
        missing = [sid for sid in slot["container_video_segment_ids"] if sid not in by_id]
        if missing:
            errors.append(f"{slot['slot_id']}:MISSING_CONTAINER_VIDEO_SEGMENT={missing}")
            continue
        intervals = [(by_id[sid]["start_us"], by_id[sid]["end_us"]) for sid in slot["container_video_segment_ids"]]
        if not interval_is_contained(start_us, end_us, intervals):
            errors.append(f"{slot['slot_id']}:SLOT_CROSSES_VIDEO_GAP_OR_EXCEEDS_CONTAINER")
    if errors:
        raise RuntimeError("visual_slot_plan 与当前 active timeline video segment 不一致：" + "; ".join(errors))
    return {
        "final_video_end_us": final_video_end,
        "container_video_segment_count": len(video_rows),
    }


def subtitle_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    text_materials = {m["id"]: m for m in data["materials"].get("texts", [])}
    text_tracks = [t for t in data.get("tracks", []) if t.get("type") == "text"]
    if not text_tracks:
        raise RuntimeError("草稿里没有 text track")
    track = max(text_tracks, key=lambda t: len(t.get("segments", [])))
    rows = []
    for index, segment in enumerate(
        sorted(track.get("segments", []), key=lambda s: int((s.get("target_timerange") or {}).get("start") or 0)),
        start=1,
    ):
        material = text_materials.get(segment.get("material_id"), {})
        text = material.get("recognize_text") or ""
        if not text:
            try:
                text = json.loads(material.get("content") or "{}").get("text") or ""
            except Exception:
                text = ""
        timerange = segment.get("target_timerange") or {}
        rows.append(
            {
                "subtitle_index": index,
                "subtitle_text": text,
                "start_us": int(timerange.get("start") or 0),
                "duration_us": int(timerange.get("duration") or 0),
            }
        )
    return rows


def build_subtitle_windows(rows: list[dict[str, Any]], max_window: int = 7) -> list[dict[str, Any]]:
    subtitle_windows: list[dict[str, Any]] = []
    for i in range(len(rows)):
        combined = ""
        for width in range(1, max_window + 1):
            if i + width > len(rows):
                break
            combined += rows[i + width - 1]["subtitle_text"]
            normalized = norm_text(combined)
            if not normalized:
                continue
            subtitle_windows.append(
                {
                    "row": rows[i],
                    "rows_span": rows[i : i + width],
                    "norm_pieces": [norm_text(row["subtitle_text"]) for row in rows[i : i + width]],
                    "start_index": i,
                    "end_index": i + width - 1,
                    "width": width,
                    "text": combined,
                    "norm_text": normalized,
                }
            )
    return subtitle_windows


def anchor_subtitle_window(window: dict[str, Any], target_norm: str, method: str) -> dict[str, Any]:
    if method not in {"exact", "target_contains"} or not target_norm:
        return window
    position = str(window["norm_text"]).find(target_norm)
    if position < 0:
        return window
    cursor = 0
    for offset, piece in enumerate(window.get("norm_pieces") or []):
        next_cursor = cursor + len(piece)
        if position < next_cursor or (piece and position == cursor):
            anchored = dict(window)
            anchored["row"] = window["rows_span"][offset]
            anchored["start_index"] = int(window["start_index"]) + offset
            anchored["anchor_offset"] = offset
            anchored["width"] = int(window["end_index"]) - int(anchored["start_index"]) + 1
            return anchored
        cursor = next_cursor
    return window


def match_item_to_subtitle_window(
    item: dict[str, Any],
    subtitle_windows: list[dict[str, Any]],
    min_start_index: int,
) -> dict[str, Any]:
    target_norm = norm_text(item["target_text"])
    for window in subtitle_windows:
        method = match_method_for_norm(target_norm, window["norm_text"])
        score = text_score_norm(target_norm, window["norm_text"])
        candidate = anchor_subtitle_window(window, target_norm, method)
        if int(candidate["start_index"]) < min_start_index:
            continue
        if method in {"exact", "target_contains"} and score >= 0.86:
            return {
                **candidate,
                "score": score,
                "method": method,
            }

    best: dict[str, Any] | None = None
    best_key: tuple[float, int, int, int] | None = None
    for window in subtitle_windows:
        method = match_method_for_norm(target_norm, window["norm_text"])
        score = text_score_norm(target_norm, window["norm_text"])
        candidate = anchor_subtitle_window(window, target_norm, method)
        if int(candidate["start_index"]) < min_start_index:
            continue
        key = (
            round(score, 6),
            match_method_rank(method),
            -int(candidate["width"]),
            -int(candidate["start_index"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                **candidate,
                "score": score,
                "method": method,
            }
    if best is None:
        raise RuntimeError(f"无法匹配图片 {item['image_id']}")
    return best


def map_items(items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subtitle_windows = build_subtitle_windows(rows)
    mapped = []
    min_start_index = 0
    for item in items:
        match = match_item_to_subtitle_window(item, subtitle_windows, min_start_index)
        row = match["row"]
        score = float(match["score"])
        method = f"subtitle_text_window_global:{match['method']}"
        duration_us = int(row.get("duration_us") or 0)
        if score >= MIN_MATCH_CONFIDENCE:
            min_start_index = max(min_start_index, int(match["start_index"]))
        mapped.append(
            {
                **item,
                "subtitle_index": row["subtitle_index"],
                "subtitle_text": row["subtitle_text"],
                "start_us": row["start_us"],
                "duration_us": duration_us,
                "match_method": method,
                "confidence": round(float(score), 4),
                "matched_window": match["text"],
                "window_size": match["width"],
                "window_start_index": int(match["start_index"]) + 1,
                "window_end_index": int(match["end_index"]) + 1,
            }
        )

    # A single Jianying track cannot contain overlapping clips. Preserve subtitle
    # starts when possible and only nudge direct collisions forward.
    last_end = -1
    for row in mapped:
        original = row["start_us"]
        participates_in_track = float(row.get("confidence", 0)) >= MIN_MATCH_CONFIDENCE
        if participates_in_track and original < last_end:
            row["start_us"] = last_end
            row["match_method"] += "+nonoverlap_nudge"
        row["end_us"] = row["start_us"] + int(row["duration_us"])
        row["start_sec"] = row["start_us"] / SEC
        row["duration_sec"] = int(row["duration_us"]) / SEC
        row["nudged_us"] = row["start_us"] - original
        if participates_in_track:
            last_end = row["end_us"]
    return mapped


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        data = f.read(24)
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return 1920, 1080


def remove_existing_ai(data: dict[str, Any], image_dir: Path) -> dict[str, set[str]]:
    image_root = str(image_dir).replace("\\", "/")
    ai_material_ids = set()
    removed_names = set()
    for track in data.get("tracks", []):
        if track.get("name") == AI_TRACK_NAME:
            for segment in track.get("segments", []):
                if segment.get("material_id"):
                    ai_material_ids.add(segment.get("material_id"))
    for material in data["materials"].get("videos", []):
        path = str(material.get("path") or "").replace("\\", "/")
        if image_root in path:
            ai_material_ids.add(material.get("id"))

    if ai_material_ids:
        kept_tracks = []
        for track in data.get("tracks", []):
            if track.get("name") == AI_TRACK_NAME:
                continue
            old_segments = track.get("segments", [])
            new_segments = [seg for seg in old_segments if seg.get("material_id") not in ai_material_ids]
            if len(new_segments) != len(old_segments):
                track["segments"] = new_segments
            kept_tracks.append(track)
        data["tracks"] = kept_tracks
        for material in data["materials"].get("videos", []):
            if material.get("id") in ai_material_ids and material.get("material_name"):
                removed_names.add(str(material.get("material_name")))
        data["materials"]["videos"] = [
            material for material in data["materials"].get("videos", []) if material.get("id") not in ai_material_ids
        ]
    return {"ids": ai_material_ids, "names": removed_names}


def helper_materials(refs: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    return {
        "speeds": [{"id": refs["speed"], "type": "speed"}],
        "placeholder_infos": [{"id": refs["placeholder"], "type": "placeholder_info", "meta_type": "none"}],
        "canvases": [{"id": refs["canvas"], "type": "canvas_color"}],
        "material_animations": [{"id": refs["animation"], "type": "sticker_animation"}],
        "sound_channel_mappings": [{"id": refs["sound"], "type": ""}],
        "material_colors": [{"id": refs["color"]}],
        "vocal_separations": [{"id": refs["vocal"], "type": "vocal_separation"}],
    }


def append_ai_track(data: dict[str, Any], mapped: list[dict[str, Any]]) -> None:
    for key in [
        "videos",
        "speeds",
        "placeholder_infos",
        "canvases",
        "material_animations",
        "sound_channel_mappings",
        "material_colors",
        "vocal_separations",
    ]:
        data["materials"].setdefault(key, [])

    tracks = data.get("tracks", [])
    text_track_indexes = [index for index, track in enumerate(tracks) if track.get("type") == "text"]
    filter_tri = [
        int(seg.get("track_render_index") or 0)
        for track in data.get("tracks", [])
        if track.get("type") == "filter"
        for seg in track.get("segments", [])
        if "track_render_index" in seg
    ]
    ai_track_render_index = max(filter_tri, default=2) + 1
    new_text_tri = ai_track_render_index + 1
    for track in tracks:
        if track.get("type") != "text":
            continue
        for seg in track.get("segments", []):
            if int(seg.get("track_render_index") or 0) <= ai_track_render_index:
                seg["track_render_index"] = new_text_tri

    ai_track = {"id": guid(), "type": "video", "flag": 2, "segments": [], "name": AI_TRACK_NAME}
    ai_render_index = 12000

    for row in mapped:
        material_id = guid()
        segment_id = guid()
        refs = {
            "speed": guid(),
            "placeholder": guid(),
            "canvas": guid(),
            "animation": guid(),
            "sound": guid(),
            "color": guid(),
            "vocal": guid(),
        }
        width, height = png_size(row["image_path"])
        data["materials"]["videos"].append(
            {
                "id": material_id,
                "type": "photo",
                "duration": 10_800_000_000,
                "path": str(row["image_path"]).replace("\\", "/"),
                "has_audio": False,
                "width": width,
                "height": height,
                "material_id": hashlib.md5(str(row["image_path"]).encode("utf-8")).hexdigest(),
                "material_name": row["image_path"].name,
                "crop": {},
                "stable": {"time_range": {}},
                "matting": {"path": ""},
                "check_flag": 62978047,
                "video_algorithm": {"path": "", "story_video_modify_video_config": {}},
                "is_unified_beauty_mode": True,
                "is_set_beauty_mode": True,
                "beauty_face_auto_preset": {},
                "video_mask_stroke": {"resource_id": "", "path": "", "type": ""},
                "video_mask_shadow": {"resource_id": "", "path": ""},
            }
        )
        for key, values in helper_materials(refs).items():
            data["materials"][key].extend(values)
        ai_track["segments"].append(
            {
                "id": segment_id,
                "source_timerange": {"duration": int(row["duration_us"])},
                "target_timerange": {"start": int(row["start_us"]), "duration": int(row["duration_us"])},
                "render_timerange": {},
                "clip": {"scale": {"x": 1.0, "y": 1.0}, "transform": {"x": 0.0, "y": 0.0}, "flip": {}},
                "uniform_scale": {},
                "material_id": material_id,
                "extra_material_refs": list(refs.values()),
                "render_index": ai_render_index,
                "enable_hsl": False,
                "track_render_index": ai_track_render_index,
                "hdr_settings": {"mode": 1},
                "responsive_layout": {},
                "enable_adjust_mask": False,
                "source": "segmentsourcenormal",
            }
        )
        row["draft_material_id"] = material_id
        row["draft_segment_id"] = segment_id

    insert_index = min(text_track_indexes) if text_track_indexes else len(tracks)
    tracks.insert(insert_index, ai_track)


def update_key_value(draft_dir: Path, mapped: list[dict[str, Any]], removed_ai: dict[str, set[str]] | None = None) -> None:
    path = draft_dir / "key_value.json"
    data = read_json(path) if path.exists() else {}
    ai_names = {row["image_path"].name for row in mapped}
    removed_ids = (removed_ai or {}).get("ids", set())
    removed_names = (removed_ai or {}).get("names", set())
    data = {
        key: value
        for key, value in data.items()
        if key not in removed_ids
        and not (
            isinstance(value, dict)
            and value.get("materialName") in ai_names.union(removed_names)
        )
    }
    for rank, row in enumerate(mapped, start=1):
        material_hash = hashlib.md5(str(row["image_path"]).encode("utf-8")).hexdigest()
        base = {
            "filter_category": "",
            "filter_detail": "",
            "is_brand": 0,
            "is_from_artist_shop": 0,
            "is_vip": "0",
            "keywordSource": "",
            "materialCategory": "media",
            "materialId": material_hash,
            "materialName": row["image_path"].name,
            "materialSubcategory": "local",
            "materialSubcategoryId": "",
            "materialThirdcategory": "导入",
            "materialThirdcategoryId": "",
            "material_copyright": "",
            "material_is_purchased": "",
            "rank": str(900 + rank),
            "rec_id": "",
            "requestId": "",
            "role": "",
            "searchId": "",
            "searchKeyword": "",
            "team_id": "",
            "textTemplateVersion": "",
        }
        data[row["draft_material_id"]] = {**base, "segmentId": row["draft_material_id"]}
        data[row["draft_segment_id"]] = {**base, "segmentId": row["draft_segment_id"]}
    write_json(path, data)


def inspect_written_ai(data: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    image_root = str(image_dir).replace("\\", "/")
    videos = {m["id"]: m for m in data["materials"].get("videos", [])}
    ai_tracks = []
    ai_segments = []
    ai_segment_rows = []
    filter_layers = []
    text_layers = []
    text_track_summaries = []
    for index, track in enumerate(data.get("tracks", [])):
        layers = sorted(
            set(
                seg.get("track_render_index")
                for seg in track.get("segments", [])
                if seg.get("track_render_index") is not None
            )
        )
        if track.get("type") == "filter":
            filter_layers.extend(layers)
        if track.get("type") == "text":
            text_layers.extend(layers)
            text_track_summaries.append(
                {
                    "track_index": index,
                    "name": track.get("name"),
                    "count": len(track.get("segments", [])),
                    "track_render_layers": layers,
                }
            )
        rows = []
        for segment in track.get("segments", []):
            material = videos.get(segment.get("material_id"))
            if not material:
                continue
            name = material.get("material_name") or ""
            path = str(material.get("path") or "").replace("\\", "/")
            if track.get("name") == AI_TRACK_NAME or image_root in path:
                image_id = image_id_from_filename(Path(name)) or image_id_from_filename(Path(path))
                rows.append((segment, material))
                timerange = segment.get("target_timerange") or {}
                source_timerange = segment.get("source_timerange") or {}
                start_us = int(timerange.get("start") or 0)
                duration_us = int(timerange.get("duration") or 0)
                ai_segment_rows.append(
                    {
                        "image_id": image_id,
                        "material_name": name,
                        "path": material.get("path"),
                        "segment_id": segment.get("id"),
                        "material_id": segment.get("material_id"),
                        "track_index": index,
                        "track_name": track.get("name"),
                        "track_type": track.get("type"),
                        "track_render_index": segment.get("track_render_index"),
                        "start_us": start_us,
                        "end_us": start_us + duration_us,
                        "duration_us": duration_us,
                        "source_duration_us": int(source_timerange.get("duration") or 0),
                    }
                )
        if rows:
            ai_tracks.append(
                {
                    "track_index": index,
                    "name": track.get("name"),
                    "type": track.get("type"),
                    "flag": track.get("flag"),
                    "count": len(rows),
                    "track_render_layers": sorted(set(seg.get("track_render_index") for seg, _ in rows)),
                }
            )
            ai_segments.extend(rows)

    missing_paths = []
    durations = sorted(set(row["duration_us"] for row in ai_segment_rows))
    ordered = sorted(ai_segment_rows, key=lambda row: row["start_us"])
    overlaps = []
    for left, right in zip(ordered, ordered[1:]):
        if int(left["end_us"]) > int(right["start_us"]):
            overlaps.append((left.get("material_name"), right.get("material_name")))
    for _, material in ai_segments:
        if not Path(str(material.get("path") or "").replace("/", "\\")).exists():
            missing_paths.append(material.get("material_name") or material.get("path"))

    main_text_track = max(text_track_summaries, key=lambda row: row["count"], default=None)
    return {
        "ai_track_count": len(ai_tracks),
        "ai_tracks": ai_tracks,
        "ai_segment_count": len(ai_segment_rows),
        "ai_segment_rows": sorted(ai_segment_rows, key=lambda row: (row["start_us"], row["image_id"])),
        "durations_us": durations,
        "missing_paths": missing_paths,
        "overlaps": overlaps,
        "filter_layers": sorted(set(filter_layers)),
        "text_layers": sorted(set(text_layers)),
        "text_tracks": text_track_summaries,
        "main_text_track": main_text_track,
    }


def post_write_actual_image_audit(
    data: dict[str, Any],
    slots: list[dict[str, Any]],
    image_dir: Path,
) -> dict[str, Any]:
    written = inspect_written_ai(data, image_dir)
    slot_by_id = {slot["image_id"]: slot for slot in slots}
    written_by_id: dict[str, dict[str, Any]] = {}
    duplicate_written_ids = []
    for row in written["ai_segment_rows"]:
        image_id = row.get("image_id") or ""
        if image_id in written_by_id:
            duplicate_written_ids.append(image_id)
        written_by_id[image_id] = row

    precision_rows = []
    errors = []
    source_duration_mismatch_count = 0
    for image_id, slot in sorted(slot_by_id.items(), key=lambda item: image_id_sort_key(item[0])):
        actual = written_by_id.get(image_id)
        if not actual:
            errors.append(f"MISSING_WRITTEN_IMAGE={image_id}")
            precision_rows.append(
                {
                    "image_id": image_id,
                    "status": "missing",
                    "expected_start_us": slot["start_us"],
                    "expected_end_us": slot["end_us"],
                    "actual_start_us": None,
                    "actual_end_us": None,
                }
            )
            continue
        status = "ok"
        if int(actual["start_us"]) != int(slot["start_us"]):
            status = "failed"
        if int(actual["end_us"]) != int(slot["end_us"]):
            status = "failed"
        if int(actual["duration_us"]) != int(slot["duration_us"]):
            status = "failed"
        source_duration_delta_us = abs(int(actual.get("source_duration_us") or 0) - int(slot["duration_us"]))
        if source_duration_delta_us > SOURCE_DURATION_TOLERANCE_US:
            status = "failed"
            source_duration_mismatch_count += 1
        if status != "ok":
            errors.append(f"SLOT_PRECISION_MISMATCH={image_id}")
        precision_rows.append(
            {
                "image_id": image_id,
                "status": status,
                "expected_start_us": slot["start_us"],
                "expected_end_us": slot["end_us"],
                "expected_duration_us": slot["duration_us"],
                "actual_start_us": actual["start_us"],
                "actual_end_us": actual["end_us"],
                "actual_duration_us": actual["duration_us"],
                "actual_source_duration_us": actual.get("source_duration_us"),
                "source_duration_delta_us": source_duration_delta_us,
                "track_render_index": actual["track_render_index"],
            }
        )

    unexpected_ids = sorted(set(written_by_id) - set(slot_by_id), key=image_id_sort_key)
    if unexpected_ids:
        errors.append("UNEXPECTED_AI_BROLL_RESIDUE=" + ",".join(unexpected_ids))
    if duplicate_written_ids:
        errors.append("DUPLICATE_WRITTEN_IMAGE_IDS=" + ",".join(sorted(set(duplicate_written_ids), key=image_id_sort_key)))
    if written["ai_segment_count"] != len(slots):
        errors.append(f"WRITTEN_IMAGE_COUNT={written['ai_segment_count']} SLOT_COUNT={len(slots)}")
    if written["ai_track_count"] != 1:
        errors.append(f"AI_TRACK_COUNT={written['ai_track_count']}")
    if written["missing_paths"]:
        errors.append("MISSING_IMAGE_PATHS")
    if written["overlaps"]:
        errors.append("AI_BROLL_OVERLAPS")
    try:
        video_audit = assert_slots_inside_video_segments(slots, data)
    except RuntimeError as exc:
        video_audit = {"video_container_audit_error": str(exc)}
        errors.append("VIDEO_CONTAINER_AUDIT_FAILED")
    final_video_end_us = int(video_audit.get("final_video_end_us") or 0)
    image_after_final_video_end_count = (
        sum(1 for row in written["ai_segment_rows"] if final_video_end_us and int(row["end_us"]) > final_video_end_us)
        if final_video_end_us
        else 0
    )

    ai_layers = written["ai_tracks"][0]["track_render_layers"] if written["ai_tracks"] else []
    if written["filter_layers"] and ai_layers and max(written["filter_layers"]) >= min(ai_layers):
        errors.append("AI_NOT_ABOVE_FILTER")
    if written["text_layers"] and ai_layers and max(ai_layers) >= min(written["text_layers"]):
        errors.append("AI_NOT_BELOW_TEXT")

    return {
        "image_slot_count": len(slots),
        "written_image_segment_count": written["ai_segment_count"],
        "slot_precision_mismatch_count": sum(1 for row in precision_rows if row["status"] != "ok"),
        "source_duration_mismatch_count": source_duration_mismatch_count,
        "source_duration_tolerance_us": SOURCE_DURATION_TOLERANCE_US,
        "image_after_final_video_end_count": image_after_final_video_end_count,
        "slot_precision_rows": precision_rows,
        "written_ai": written,
        "video_audit": video_audit,
        "no_old_ai_broll_residue": not unexpected_ids and not duplicate_written_ids and written["ai_track_count"] == 1,
        "hard_errors": errors,
        "post_write_actual_image_audit_gate_passed": not errors,
    }


def write_report(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "broll_exec_plan.csv"
    fields = [
        "slot_id",
        "image_id",
        "image_path",
        "start_sec",
        "end_sec",
        "duration_sec",
        "start_us",
        "end_us",
        "duration_us",
        "source_start_us",
        "source_end_us",
        "container_video_segment_ids",
        "subtitle_index",
        "subtitle_text",
        "target_text",
        "match_method",
        "confidence",
        "matched_window",
        "window_size",
        "window_start_index",
        "window_end_index",
        "nudged_us",
        "draft_segment_id",
        "draft_material_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: image_id_sort_key(r["image_id"])):
            output = {field: row.get(field, "") for field in fields}
            output["end_sec"] = int(row.get("end_us") or 0) / SEC if row.get("end_us") is not None else ""
            if isinstance(output.get("container_video_segment_ids"), list):
                output["container_video_segment_ids"] = ",".join(output["container_video_segment_ids"])
            writer.writerow(output)
    return path


def write_confirmation_sheet(
    out_dir: Path,
    args: argparse.Namespace,
    timeline_id: str,
    timeline_name: str,
    image_ids: list[str],
    slots: list[dict[str, Any]],
    video_audit: dict[str, Any],
) -> Path:
    can_execute = len(slots) == len(image_ids)
    path = out_dir / "preflight_confirmation.md"
    sample_rows = sorted(slots, key=lambda row: image_id_sort_key(row["image_id"]))[:12]
    lines = [
        "# 剪映 AI B-roll 施工确认单",
        "",
        "## 输入确认",
        "",
        f"- 当前工程：`{args.draft_dir}`",
        f"- 内部草稿范围：`{timeline_name}`",
        f"- 当前时间线 ID：`{timeline_id}`",
        f"- B-ROLL 设计稿：`{args.broll}`",
        f"- AI 静态图目录：`{args.image_dir}`",
        f"- visual_slot_plan：`{args.visual_slot_plan}`",
        "",
        "## 当前工程时间线读取",
        "",
        f"- final_video_end_us：`{video_audit['final_video_end_us']}`",
        f"- container_video_segment_count：`{video_audit['container_video_segment_count']}`",
        "- 对齐来源：visual_slot_plan target_start_us / target_end_us",
        "",
        "## B-ROLL 与图片读取",
        "",
        f"- 规范命名 AI 图片：`{len(image_ids)}`",
        f"- visual slots：`{len(slots)}`",
        "",
        "## 匹配结论",
        "",
        "- 字幕语义匹配：由上游 A-Roll/B-Roll QC 输出的 visual_slot_plan 承担，本阶段不重新猜字幕轨",
        f"- 是否允许写入：`{'YES' if can_execute else 'NO'}`",
        "",
        "## 前 12 张抽样",
        "",
        "| 图片 | start_sec | end_sec | duration_sec | slot | 台词 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in sample_rows:
        lines.append(
            "| {image_id} | {start_sec:.3f} | {end_sec:.3f} | {duration_sec:.3f} | {slot_id} | {target_text} |".format(
                image_id=row["image_id"],
                start_sec=float(row["start_sec"]),
                end_sec=int(row["end_us"]) / SEC,
                duration_sec=float(row["duration_sec"]),
                slot_id=str(row.get("slot_id", "")).replace("|", "/"),
                target_text=str(row.get("target_text", "")).replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", "utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write AI B-roll photo clips from visual_slot_plan into Jianying encrypted draft.")
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--broll", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--visual-slot-plan", type=Path, required=True)
    parser.add_argument("--jy-draftc", type=Path, default=DEFAULT_JY_DRAFTC)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--preflight-only", action="store_true", help="只生成施工确认单，不写入草稿。")
    parser.add_argument("--confirm-write", action="store_true", help="确认当前工程/B-ROLL/字幕匹配后才允许写入草稿。")
    args = parser.parse_args()

    for label, path in {
        "draft_dir": args.draft_dir,
        "broll": args.broll,
        "image_dir": args.image_dir,
        "visual_slot_plan": args.visual_slot_plan,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} 不存在：{path}")

    run_id = time.strftime("direct_write_%Y%m%d_%H%M%S")
    out_dir = args.runtime / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    binding = DraftRuntimeBinding.bind(args.draft_dir, args.jy_draftc, out_dir)
    plain_path = out_dir / "draft_content.dec.json"
    modified_plain = out_dir / "draft_content.modified.json"
    encrypted_out = out_dir / "draft_content.encrypted.json"

    print(f"draft_dir={args.draft_dir}")
    print(f"timeline_id={binding.timeline_id}")
    print(f"draft_scope={binding.timeline_name}")
    print(f"broll={args.broll}")
    print(f"image_dir={args.image_dir}")
    print(f"visual_slot_plan={args.visual_slot_plan}")
    print(f"jy_draftc={binding.jy_draftc}")

    data = binding.decrypt_timeline(plain_path)
    image_ids, image_files = validate_broll_image_contract(args.broll, args.image_dir)
    slots = load_visual_slot_plan(args.visual_slot_plan, args.image_dir, image_files)
    validate_slot_plan_ids(slots, image_ids)
    validate_no_slot_overlaps(slots)
    validate_slot_confidence(slots)
    video_audit = assert_slots_inside_video_segments(slots, data)
    confirmation = write_confirmation_sheet(
        out_dir,
        args,
        binding.timeline_id,
        binding.timeline_name,
        image_ids,
        slots,
        video_audit,
    )
    report = write_report(out_dir, slots)
    print(f"ai_images={len(image_ids)}")
    print(f"visual_slots={len(slots)}")
    print(f"final_video_end_us={video_audit['final_video_end_us']}")
    print(f"confirmation={confirmation}")
    print(f"report={report}")
    print(f"work_dir={out_dir}")
    if args.preflight_only:
        print("PREFLIGHT_ONLY_NO_DRAFT_WRITE")
        return 0
    if not args.confirm_write:
        raise RuntimeError("缺少 --confirm-write。必须先确认 preflight_confirmation.md 后才允许写入草稿。")

    removed_ai = remove_existing_ai(data, args.image_dir)
    append_ai_track(data, slots)
    write_json(modified_plain, data)
    encrypt(binding.jy_draftc, modified_plain, encrypted_out)

    encrypted_text = encrypted_out.read_text("utf-8")

    def key_value_writer() -> None:
        update_key_value(args.draft_dir, slots, removed_ai)

    def post_write_validator() -> dict[str, Any]:
        assert_all_project_timeline_files_match_folder_ids(args.draft_dir, binding.jy_draftc, out_dir)
        post_plain = out_dir / "post_write_actual.dec.json"
        post_data = binding.decrypt_timeline(post_plain)
        audit = post_write_actual_image_audit(post_data, slots, args.image_dir)
        root_mirror_consistent = True
        if binding.mirrors_root:
            root_mirror_consistent = sha256(binding.root_content) == sha256(binding.timeline_content)
        audit["root_timeline_mirror_consistent"] = root_mirror_consistent
        if not root_mirror_consistent:
            audit["hard_errors"].append("ROOT_TIMELINE_MIRROR_INCONSISTENT")
            audit["post_write_actual_image_audit_gate_passed"] = False
        audit_path = out_dir / "post_write_actual_image_audit.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), "utf-8")
        if not audit["post_write_actual_image_audit_gate_passed"]:
            raise RuntimeError("post-write actual image audit failed: " + ",".join(audit["hard_errors"]))
        return {
            "post_write_actual_image_audit": str(audit_path),
            "root_timeline_mirror_consistent": root_mirror_consistent,
        }

    transaction = binding.write_encrypted_transaction(
        encrypted_text,
        key_value_writer,
        out_dir,
        post_write_validator=post_write_validator,
    )

    for changed_path in transaction["changed_paths"]:
        print(f"wrote={changed_path}")
    print(f"only_specified_draft_written={transaction['only_specified_draft_written']}")
    print(f"root_timeline_mirror_consistent={transaction['root_timeline_mirror_consistent']}")
    print(f"post_write_actual_image_audit={transaction['post_write_actual_image_audit']}")
    print(f"written_segments={len(slots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
