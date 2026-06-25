from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aroll_runtime_paths import get_aligner_root, get_runtime_root  # noqa: E402


DEFAULT_RUN_ROOT = get_runtime_root() / "subtitle_recalibration_runs"
DEFAULT_CURRENT_DRAFT_STATE = get_runtime_root() / "video_pipeline" / "current_draft.json"
DEFAULT_JY_DRAFTC = Path(
    os.environ.get("JY_DRAFTC")
    or os.environ.get("JY_DRAFTC_EXE")
    or get_aligner_root() / "vendor" / "jy-draftc-bin" / "jy-draftc-amd64-windows" / "jy-draftc.exe"
)
DEFAULT_MAX_FILL_GAP_US = 2_000_000
DEFAULT_GAP_THRESHOLD_US = 80_000

PROFANITY_PATTERNS = [
    r"(?i)der",
    r"蠢货",
    r"傻[逼比B]",
    r"煞笔",
    r"妈的",
    r"他妈的?",
    r"卧槽",
    r"我操",
    r"艹",
]

SCRIPT_MATCH_DROP_RE = re.compile(r"[\s，,。.!！?？、：:；;“”\"'《》<>（）()\[\]【】…·\-|｜/\\]+")


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def run_command(args: list[str | Path], *, cwd: Path | None = None) -> None:
    process = subprocess.run([str(item) for item in args], cwd=str(cwd) if cwd else None, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit code {process.returncode}: {' '.join(str(item) for item in args)}")


def load_current_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return dict(payload) if isinstance(payload, dict) else {}


def normalize_path_text(value: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(value))).replace("/", "\\").rstrip("\\").lower()


def resolve_draft_dir(value: str, current_state: dict[str, Any]) -> Path:
    if value.strip():
        return Path(value)
    state_draft = str(current_state.get("draft_dir") or "").strip()
    if not state_draft:
        raise SystemExit("DraftDir is required when current_draft.json has no draft_dir.")
    return Path(state_draft)


def resolve_timeline_dir(draft_dir: Path, timeline_id: str = "") -> Path:
    timelines_root = draft_dir / "Timelines"
    if not timelines_root.exists():
        raise SystemExit(f"Timelines directory not found: {timelines_root}")
    if timeline_id:
        candidate = timelines_root / timeline_id
        if candidate.exists():
            return candidate
        raise SystemExit(f"Requested timeline_id not found under draft: {timeline_id}")
    candidates = [path for path in timelines_root.iterdir() if path.is_dir() and (path / "draft_content.json").exists()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"No timeline draft_content.json found under: {timelines_root}")
    return max(candidates, key=lambda path: (path / "draft_content.json").stat().st_mtime)


def stable_dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def non_subtitle_signature(data: dict[str, Any]) -> str:
    reduced = copy.deepcopy(data)
    reduced["tracks"] = [track for track in reduced.get("tracks", []) if track.get("type") != "text"]
    reduced.get("materials", {}).pop("texts", None)
    return hashlib.sha256(stable_dump(reduced)).hexdigest()


def timerange(segment: dict[str, Any]) -> tuple[int, int]:
    value = segment.get("target_timerange") or segment.get("timerange") or {}
    start = int(value.get("start") or 0)
    duration = int(value.get("duration") or 0)
    end = int(value.get("end") or (start + duration if duration else start))
    return start, max(start, end)


def remove_display_breaks(value: str) -> str:
    return str(value or "").replace("\r", "").replace("\n", "").strip()


def text_from_material(material: dict[str, Any]) -> str:
    value = str(material.get("recognize_text") or "").strip()
    if value:
        return remove_display_breaks(value)
    for key in ("content", "base_content"):
        try:
            payload = json.loads(material.get(key) or "{}")
        except Exception:
            continue
        value = str(payload.get("text") or "").strip()
        if value:
            return remove_display_breaks(value)
    return ""


def extract_caption_track(data: dict[str, Any], *, allow_multiple_text_tracks: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    texts = {str(material.get("id")): material for material in data.get("materials", {}).get("texts", [])}
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for track in data.get("tracks", []):
        if track.get("type") != "text":
            continue
        rows = []
        for segment in track.get("segments", []):
            material = texts.get(str(segment.get("material_id") or ""))
            if not material:
                continue
            text = text_from_material(material)
            if not text:
                continue
            rows.append({"segment": segment, "material": material, "text": text})
        if rows:
            candidates.append((track, sorted(rows, key=lambda row: timerange(row["segment"])[0])))
    if not candidates:
        raise SystemExit("No non-empty caption/text track found in draft.")
    if len(candidates) > 1 and not allow_multiple_text_tracks:
        counts = [{"track_id": track.get("id"), "caption_count": len(rows)} for track, rows in candidates]
        raise SystemExit(f"Multiple non-empty text tracks found; pass --allow-multiple-text-tracks to use the largest one. {counts}")
    return max(candidates, key=lambda item: len(item[1]))


def script_text_from_markdown(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    raw = path.read_text("utf-8", errors="replace")
    lines = []
    capture = False
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if re.match(r"^\*\*Jackson[：:]\*\*", line, flags=re.IGNORECASE):
            capture = True
            continue
        if line.startswith("## ") or line.startswith("# "):
            capture = False
            continue
        if not capture:
            continue
        if not line or line.startswith(">") or line.startswith("**[") or line.startswith("["):
            continue
        line = re.sub(r"^\s*[-*]\s+", "", line)
        lines.append(line)
    return "\n".join(lines) if lines else raw


def has_term(script_text: str, term: str) -> bool:
    return term.lower() in script_text.lower()


def compact_match_text(value: str) -> str:
    return SCRIPT_MATCH_DROP_RE.sub("", normalize_domain_terms(str(value or ""), "")).lower()


def script_units(script_text: str) -> list[str]:
    units: list[str] = []
    for raw_line in str(script_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*]\s+", "", line)
        for part in re.split(r"(?<=[。！？!?])", line):
            part = part.strip()
            if 4 <= len(compact_match_text(part)) <= 60:
                units.append(part)
    return list(dict.fromkeys(units))


def split_phrase_chunks(value: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"[，,、：:；;。！？!?]+", value) if chunk.strip()]
    return chunks or [value.strip()]


def trim_leading_clause(value: str) -> str:
    result = value.strip()
    for marker in ("是", "叫", "成"):
        if marker in result and result.rfind(marker) < len(result) - 1:
            result = result[result.rfind(marker) + 1 :].strip()
    return result


def script_phrase_candidates(script_text: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for unit in script_units(script_text):
        chunks = split_phrase_chunks(unit)
        phrase_values = [unit]
        for start in range(len(chunks)):
            for end in range(start + 1, len(chunks) + 1):
                joined = "、".join(chunks[start:end])
                phrase_values.append(joined)
                trimmed_first = trim_leading_clause(chunks[start])
                if trimmed_first and trimmed_first != chunks[start]:
                    phrase_values.append("、".join([trimmed_first, *chunks[start + 1 : end]]))
        for value in phrase_values:
            value = value.strip()
            compact = compact_match_text(value)
            if not (4 <= len(compact) <= 30) or compact in seen:
                continue
            seen.add(compact)
            candidates.append({"text": value, "compact": compact, "unit": unit})
    return candidates


def longest_common_substring_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_char in left:
        current = [0] * (len(right) + 1)
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def rescue_with_script_phrase(
    text: str,
    candidates: list[dict[str, str]],
    *,
    script_compact: str = "",
) -> tuple[str, dict[str, Any] | None]:
    compact = compact_match_text(text)
    if len(compact) < 4:
        return text, None
    if script_compact and compact in script_compact:
        return text, None
    if any(compact and compact in candidate["compact"] for candidate in candidates):
        return text, None

    best: tuple[float, dict[str, str], int] | None = None
    for candidate in candidates:
        candidate_compact = candidate["compact"]
        if candidate_compact == compact or candidate_compact in compact:
            continue
        length_ratio = len(candidate_compact) / max(1, len(compact))
        if length_ratio < 0.65 or length_ratio > 1.85:
            continue
        overlap = longest_common_substring_length(compact, candidate_compact)
        if overlap < 4:
            continue
        shorter_share = overlap / min(len(compact), len(candidate_compact))
        longer_share = overlap / max(len(compact), len(candidate_compact))
        if shorter_share < 0.70 or longer_share < 0.55:
            continue
        score = overlap * 3 + min(len(compact), len(candidate_compact)) - abs(len(compact) - len(candidate_compact))
        if best is None or score > best[0]:
            best = (float(score), candidate, overlap)

    if best is None:
        return text, None
    _score, candidate, overlap = best
    return candidate["text"], {
        "reason": "script_phrase_rescue",
        "matched_script_unit": candidate["unit"],
        "overlap_chars": overlap,
    }


def normalize_domain_terms(text: str, script_text: str = "") -> str:
    result = text.strip().replace("　", " ")
    if has_term(script_text, "AIR") or not script_text:
        result = result.replace("A2", "AIR").replace("L2", "AIR").replace("二阶段", "AIR阶段")
    if has_term(script_text, "PRO") or not script_text:
        result = re.sub(r"(?i)\bpro\b", "PRO", result)
    if has_term(script_text, "Logo") or not script_text:
        result = result.replace("LOGO", "Logo")
    result = result.replace("0瑕疵", "零瑕疵").replace("0噪点", "零噪点")
    result = result.replace("360P全损", "360P画质")
    result = result.replace("吃的干干净净", "吃得干干净净")
    result = result.replace("粗糙的像", "粗糙得像")
    result = result.replace("喷的再亮", "喷得再亮")
    result = result.replace("利落的收紧", "利落收紧")
    result = result.replace("利落的劈开", "利落地劈开")
    result = result.replace("真实的肩宽", "真实肩宽")
    result = result.replace("唯一的动作", "唯一动作")
    result = result.replace("为了服务你本来", "为了服务你本来")
    result = re.sub(r"\s+", " ", result).strip()
    return result


def mask_profanity(text: str, patterns: list[str] | None = None) -> str:
    result = text
    for pattern in patterns or PROFANITY_PATTERNS:
        result = re.sub(pattern, "**", result)
    return result


def apply_user_corrections(text: str, index: int, material_id: str, corrections: dict[str, str]) -> str:
    for key in (str(index), f"index:{index}", material_id, f"material:{material_id}", text):
        if key in corrections:
            return str(corrections[key])
    return text


def display_text(text: str, *, max_line_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_line_chars:
        return text
    forced = [
        ("：", True),
        ("，", True),
        ("、", True),
        ("？", True),
        ("！", True),
        (" ", False),
    ]
    for sep, keep_sep in forced:
        positions = [match.start() + (1 if keep_sep else 0) for match in re.finditer(re.escape(sep), text)]
        if not positions:
            continue
        position = min(positions, key=lambda value: abs(value - len(text) / 2))
        if 4 <= position <= len(text) - 4:
            return text[:position] + "\n" + text[position:].lstrip()

    target = len(text) // 2
    candidates = range(max(4, target - 8), min(len(text) - 4, target + 8) + 1)
    for position in sorted(candidates, key=lambda value: abs(value - target)):
        left = text[position - 1]
        right = text[position]
        if re.match(r"[A-Za-z0-9.%]", left) or re.match(r"[A-Za-z0-9.%]", right):
            continue
        return text[:position] + "\n" + text[position:]
    position = target
    return text[:position] + "\n" + text[position:]


def display_lines_ok(visible_text: str, *, max_line_chars: int) -> bool:
    lines = [line for line in str(visible_text or "").splitlines() if line.strip()]
    if not lines:
        return False
    return all(len(line.strip()) <= max_line_chars + 4 for line in lines)


def choose_visible_text(
    *,
    old_plain: str,
    new_plain: str,
    old_visible: str,
    script_text: str,
    mask_dirty_words: bool,
    layout: bool,
    max_line_chars: int,
) -> str:
    if not layout:
        return new_plain

    if "\n" in old_visible and remove_display_breaks(old_visible) == old_plain:
        rewritten_lines = []
        for line in old_visible.splitlines():
            rewritten = normalize_domain_terms(line, script_text)
            if mask_dirty_words:
                rewritten = mask_profanity(rewritten)
            rewritten_lines.append(rewritten)
        candidate = "\n".join(rewritten_lines)
        if remove_display_breaks(candidate) == new_plain and display_lines_ok(candidate, max_line_chars=max_line_chars):
            return candidate

    if "\n" in old_visible and remove_display_breaks(old_visible) == new_plain and display_lines_ok(
        old_visible, max_line_chars=max_line_chars
    ):
        return old_visible

    return display_text(new_plain, max_line_chars=max_line_chars)


def update_content_json(raw: str, visible_text: str) -> str:
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        payload = {"text": visible_text, "styles": []}
    payload["text"] = visible_text
    visible_len = len(visible_text)
    for style in payload.get("styles") or []:
        if isinstance(style, dict) and isinstance(style.get("range"), list) and len(style["range"]) == 2:
            style["range"] = [0, visible_len]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def gap_fill_changes(rows: list[dict[str, Any]], *, threshold_us: int, max_fill_gap_us: int) -> list[dict[str, Any]]:
    changes = []
    for index, (left, right) in enumerate(zip(rows, rows[1:]), start=1):
        left_start, left_end = timerange(left["segment"])
        right_start, _right_end = timerange(right["segment"])
        gap = right_start - left_end
        if gap <= threshold_us or gap > max_fill_gap_us:
            continue
        new_duration = right_start - left_start
        if new_duration <= 0:
            continue
        left["segment"].setdefault("target_timerange", {})["duration"] = new_duration
        changes.append(
            {
                "kind": "gap_fill",
                "after_caption_index": index,
                "before_caption_index": index + 1,
                "segment_id": left["segment"].get("id"),
                "old_end_us": left_end,
                "new_end_us": right_start,
                "gap_filled_us": gap,
                "text": text_from_material(left["material"]),
                "next_text": text_from_material(right["material"]),
            }
        )
    return changes


def validate_no_text_overlaps(rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    issues = []
    for index, (left, right) in enumerate(zip(rows, rows[1:]), start=1):
        _left_start, left_end = timerange(left["segment"])
        right_start, _right_end = timerange(right["segment"])
        if right_start < left_end:
            issues.append({"index": index, "left_end_us": left_end, "right_start_us": right_start})
    return issues


def remaining_gap_count(rows: list[dict[str, Any]], threshold_us: int) -> int:
    count = 0
    for left, right in zip(rows, rows[1:]):
        _left_start, left_end = timerange(left["segment"])
        right_start, _right_end = timerange(right["segment"])
        if right_start - left_end > threshold_us:
            count += 1
    return count


def load_corrections(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = read_json(path)
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    if isinstance(payload, list):
        result = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            replacement = str(item.get("new_text") or item.get("replacement") or "")
            if not replacement:
                continue
            for key in ("index", "material_id", "segment_id", "old_text"):
                if item.get(key) is not None:
                    result[str(item[key])] = replacement
        return result
    raise SystemExit(f"Unsupported correction map shape: {path}")


def jianying_is_running() -> bool:
    if os.name != "nt":
        return False
    result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq JianyingPro.exe"], text=True, capture_output=True)
    return "JianyingPro.exe" in result.stdout


def build_cleanup_plan(
    data: dict[str, Any],
    *,
    script_text: str,
    corrections: dict[str, str],
    mask_dirty_words: bool,
    layout: bool,
    fill_gaps: bool,
    gap_threshold_us: int,
    max_fill_gap_us: int,
    max_line_chars: int,
    allow_multiple_text_tracks: bool,
    script_phrase_rescue_mode: str = "suggest",
) -> tuple[dict[str, Any], dict[str, Any]]:
    before_signature = non_subtitle_signature(data)
    track, rows = extract_caption_track(data, allow_multiple_text_tracks=allow_multiple_text_tracks)
    if script_phrase_rescue_mode not in {"off", "suggest", "apply"}:
        raise ValueError(f"Unsupported script_phrase_rescue_mode: {script_phrase_rescue_mode}")
    phrase_candidates = script_phrase_candidates(script_text) if script_phrase_rescue_mode != "off" else []
    script_compact = compact_match_text(script_text) if script_phrase_rescue_mode != "off" else ""
    text_changes = []
    layout_changes = []
    script_suggestions = []
    for index, row in enumerate(rows, start=1):
        material = row["material"]
        old_plain = text_from_material(material)
        text_reason: dict[str, Any] = {}
        new_plain = apply_user_corrections(old_plain, index, str(material.get("id") or ""), corrections)
        new_plain = normalize_domain_terms(new_plain, script_text)
        rescued_plain, rescue_reason = rescue_with_script_phrase(new_plain, phrase_candidates, script_compact=script_compact)
        if rescue_reason:
            suggestion = {
                "kind": "script_phrase_suggestion",
                "index": index,
                "segment_id": row["segment"].get("id"),
                "material_id": material.get("id"),
                "old_text": new_plain,
                "suggested_text": rescued_plain,
                **rescue_reason,
            }
            if script_phrase_rescue_mode == "apply":
                new_plain = rescued_plain
                text_reason.update(rescue_reason)
            else:
                script_suggestions.append(suggestion)
        if mask_dirty_words:
            masked_plain = mask_profanity(new_plain)
            if masked_plain != new_plain and not text_reason:
                text_reason["reason"] = "profanity_mask"
            new_plain = masked_plain
        old_visible = old_plain
        try:
            old_visible = str(json.loads(material.get("content") or "{}").get("text") or old_plain)
        except Exception:
            old_visible = old_plain
        visible = choose_visible_text(
            old_plain=old_plain,
            new_plain=new_plain,
            old_visible=old_visible,
            script_text=script_text,
            mask_dirty_words=mask_dirty_words,
            layout=layout,
            max_line_chars=max_line_chars,
        )
        if new_plain != old_plain:
            text_changes.append(
                {
                    "kind": "text_correction",
                    "index": index,
                    "segment_id": row["segment"].get("id"),
                    "material_id": material.get("id"),
                    "old_text": old_plain,
                    "new_text": new_plain,
                }
                | text_reason
            )
        if visible != old_visible:
            layout_changes.append(
                {
                    "kind": "display_layout",
                    "index": index,
                    "segment_id": row["segment"].get("id"),
                    "material_id": material.get("id"),
                    "old_display_text": old_visible,
                    "new_display_text": visible,
                }
            )
        material["recognize_text"] = new_plain
        material["content"] = update_content_json(material.get("content") or "{}", visible)
        material["base_content"] = update_content_json(material.get("base_content") or material.get("content") or "{}", visible)

    gap_changes = gap_fill_changes(rows, threshold_us=gap_threshold_us, max_fill_gap_us=max_fill_gap_us) if fill_gaps else []
    after_signature = non_subtitle_signature(data)
    if before_signature != after_signature:
        raise RuntimeError("non-subtitle draft signature changed; refusing to continue")
    overlap_issues = validate_no_text_overlaps(rows)
    if overlap_issues:
        raise RuntimeError(f"subtitle overlaps introduced: {overlap_issues[:10]}")
    report = {
        "schema_version": "manual_refined_caption_cleanup_report.v1",
        "created_at": iso_now(),
        "caption_track_id": track.get("id"),
        "caption_count": len(rows),
        "text_correction_count": len(text_changes),
        "display_layout_count": len(layout_changes),
        "gap_fill_count": len(gap_changes),
        "remaining_gap_count": remaining_gap_count(rows, gap_threshold_us),
        "gap_threshold_us": gap_threshold_us,
        "max_fill_gap_us": max_fill_gap_us,
        "script_phrase_rescue_mode": script_phrase_rescue_mode,
        "script_phrase_suggestion_count": len(script_suggestions),
        "non_subtitle_signature": before_signature,
        "changes": [*text_changes, *layout_changes, *gap_changes],
        "suggestions": script_suggestions,
    }
    return data, report


def write_preview(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"caption_count={report['caption_count']}",
        f"text_correction_count={report['text_correction_count']}",
        f"display_layout_count={report['display_layout_count']}",
        f"gap_fill_count={report['gap_fill_count']}",
        f"remaining_gap_count={report['remaining_gap_count']}",
        f"script_phrase_rescue_mode={report.get('script_phrase_rescue_mode', 'off')}",
        f"script_phrase_suggestion_count={report.get('script_phrase_suggestion_count', 0)}",
        "",
    ]
    for change in report["changes"]:
        kind = change.get("kind")
        if kind == "text_correction":
            suffix = f" ({change['reason']})" if change.get("reason") else ""
            lines.extend(
                [
                    f"[text] #{change['index']:03d}{suffix}",
                    f"  OLD: {change['old_text']}",
                    f"  NEW: {change['new_text']}",
                ]
            )
            if change.get("matched_script_unit"):
                lines.append(f"  SCRIPT: {change['matched_script_unit']}")
        elif kind == "display_layout":
            lines.extend(
                [
                    f"[layout] #{change['index']:03d}",
                    f"  OLD: {str(change['old_display_text']).replace(chr(10), ' / ')}",
                    f"  NEW: {str(change['new_display_text']).replace(chr(10), ' / ')}",
                ]
            )
        elif kind == "gap_fill":
            lines.extend(
                [
                    f"[gap] #{change['after_caption_index']:03d}->#{change['before_caption_index']:03d} "
                    f"{change['gap_filled_us'] / 1_000_000:.3f}s",
                    f"  HOLD: {change['text']}",
                    f"  NEXT: {change['next_text']}",
                ]
            )
    for suggestion in report.get("suggestions") or []:
        lines.extend(
            [
                f"[suggest] #{suggestion['index']:03d} ({suggestion.get('reason', 'script_phrase_suggestion')})",
                f"  OLD: {suggestion['old_text']}",
                f"  NEW: {suggestion['suggested_text']}",
                f"  SCRIPT: {suggestion.get('matched_script_unit', '')}",
            ]
        )
    path.write_text("\n".join(lines), "utf-8")


def post_write_validate(jy_draftc: Path, written_path: Path, run_dir: Path, before_dec: Path, threshold_us: int) -> dict[str, Any]:
    post_dec = run_dir / "post_write.dec.json"
    run_command([jy_draftc, "--dec", written_path, post_dec])
    before = read_json(before_dec)
    after = read_json(post_dec)
    track, rows = extract_caption_track(after, allow_multiple_text_tracks=True)
    return {
        "post_write_dec": str(post_dec),
        "video_tracks_equal": [t for t in before.get("tracks", []) if t.get("type") != "text"]
        == [t for t in after.get("tracks", []) if t.get("type") != "text"],
        "video_materials_equal": before.get("materials", {}).get("videos") == after.get("materials", {}).get("videos"),
        "caption_track_id": track.get("id"),
        "caption_count": len(rows),
        "overlap_count": len(validate_no_text_overlaps(rows)),
        "remaining_gap_count": remaining_gap_count(rows, threshold_us),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean up human-refined Jianying captions without touching video tracks.")
    parser.add_argument("--draft-dir", default="")
    parser.add_argument("--timeline-id", default="")
    parser.add_argument("--user-script-path", type=Path)
    parser.add_argument("--correction-map-json", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--current-draft-state", type=Path, default=DEFAULT_CURRENT_DRAFT_STATE)
    parser.add_argument("--jy-draftc", type=Path, default=DEFAULT_JY_DRAFTC)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manual-refine-confirmed", action="store_true")
    parser.add_argument("--allow-open-jianying", action="store_true")
    parser.add_argument("--allow-multiple-text-tracks", action="store_true")
    parser.add_argument("--no-mask-profanity", action="store_true")
    parser.add_argument("--no-script-phrase-rescue", action="store_true")
    parser.add_argument("--script-phrase-rescue-mode", choices=("off", "suggest", "apply"), default="suggest")
    parser.add_argument("--no-layout", action="store_true")
    parser.add_argument("--no-fill-gaps", action="store_true")
    parser.add_argument("--gap-threshold-us", type=int, default=DEFAULT_GAP_THRESHOLD_US)
    parser.add_argument("--max-fill-gap-us", type=int, default=DEFAULT_MAX_FILL_GAP_US)
    parser.add_argument("--max-line-chars", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_state = load_current_state(args.current_draft_state)
    draft_dir = resolve_draft_dir(str(args.draft_dir or ""), current_state)
    timeline_dir = resolve_timeline_dir(draft_dir, args.timeline_id or str(current_state.get("timeline_id") or ""))
    root_draft_content = draft_dir / "draft_content.json"
    timeline_draft_content = timeline_dir / "draft_content.json"
    if not root_draft_content.exists() or not timeline_draft_content.exists():
        raise SystemExit("Both root and timeline draft_content.json must exist.")
    if not args.jy_draftc.exists():
        raise SystemExit(f"jy-draftc does not exist: {args.jy_draftc}")

    run_dir = args.run_dir or (args.run_root / f"manual_refined_caption_cleanup_{now_stamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    current_dec = run_dir / "current_timeline.dec.json"
    output_dec = run_dir / "caption_cleanup.dec.json"
    output_enc = run_dir / "caption_cleanup.enc.json"

    run_command([args.jy_draftc, "--dec", timeline_draft_content, current_dec])
    data = read_json(current_dec)
    script_text = script_text_from_markdown(args.user_script_path)
    script_phrase_rescue_mode = "off" if args.no_script_phrase_rescue else args.script_phrase_rescue_mode
    cleaned, report = build_cleanup_plan(
        data,
        script_text=script_text,
        corrections=load_corrections(args.correction_map_json),
        mask_dirty_words=not args.no_mask_profanity,
        layout=not args.no_layout,
        fill_gaps=not args.no_fill_gaps,
        gap_threshold_us=args.gap_threshold_us,
        max_fill_gap_us=args.max_fill_gap_us,
        max_line_chars=args.max_line_chars,
        allow_multiple_text_tracks=args.allow_multiple_text_tracks,
        script_phrase_rescue_mode=script_phrase_rescue_mode,
    )
    report.update(
        {
            "draft_dir": str(draft_dir),
            "timeline_dir": str(timeline_dir),
            "root_draft_content": str(root_draft_content),
            "timeline_draft_content": str(timeline_draft_content),
            "user_script_path": str(args.user_script_path or ""),
            "run_dir": str(run_dir),
            "apply_requested": bool(args.apply),
            "manual_refine_confirmed": bool(args.manual_refine_confirmed),
            "output_dec": str(output_dec),
            "output_enc": str(output_enc),
        }
    )
    output_dec.write_text(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")), "utf-8")
    run_command([args.jy_draftc, "--enc", output_dec, output_enc])

    blockers = []
    if args.apply and not args.manual_refine_confirmed:
        blockers.append({"code": "MANUAL_REFINE_CONFIRMED_REQUIRED", "message": "--manual-refine-confirmed is required for --apply."})
    if args.apply and jianying_is_running() and not args.allow_open_jianying:
        blockers.append({"code": "JIANYING_IS_RUNNING", "message": "Close Jianying before applying, or pass --allow-open-jianying knowingly."})
    report["blockers"] = blockers
    report["ready_to_apply"] = not blockers

    if args.apply and not blockers:
        backup_dir = run_dir / "backups_before_apply"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(timeline_draft_content, backup_dir / "timeline_draft_content.before_caption_cleanup.json")
        shutil.copy2(root_draft_content, backup_dir / "root_draft_content.before_caption_cleanup.json")
        shutil.copy2(output_enc, timeline_draft_content)
        shutil.copy2(output_enc, root_draft_content)
        report["applied"] = True
        report["backup_dir"] = str(backup_dir)
        report["post_write_validation"] = post_write_validate(
            args.jy_draftc,
            timeline_draft_content,
            run_dir,
            current_dec,
            args.gap_threshold_us,
        )
    else:
        report["applied"] = False

    write_json(run_dir / "caption_cleanup_report.json", report)
    write_preview(run_dir / "caption_cleanup_preview.txt", report)
    print(str(run_dir / "caption_cleanup_report.json"))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
