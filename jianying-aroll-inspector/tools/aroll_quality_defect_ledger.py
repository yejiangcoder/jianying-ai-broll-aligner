from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOL_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TOOL_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aroll_runtime_paths import get_aroll_quality_defect_ledger_dir


ROOT_CAUSE_FIX_MAP = {
    "dangling_prefix_or_suffix": {
        "mechanism": "final visible caption repair and dangling prefix/suffix gate",
        "files": ["src/aroll_v21/quality/final_visible_caption_repair.py"],
        "test_area": "tests/test_aroll_v21_final_visible_generic_qc_regressions.py",
        "assertion": "final visible captions should merge or drop dangling prefix/suffix fragments without sample-text exceptions",
    },
    "caption_boundary_split_error": {
        "mechanism": "caption boundary repair after final timeline compilation",
        "files": ["src/aroll_v21/quality/final_visible_caption_repair.py", "src/aroll_v21/render/subtitle_renderer.py"],
        "test_area": "tests/test_aroll_v21_final_visible_generic_qc_regressions.py",
        "assertion": "caption boundaries should not emit orphaned syntactic suffix/prefix captions",
    },
    "semantic_garbage_caption": {
        "mechanism": "final visible semantic quality gate",
        "files": ["src/aroll_v21/quality/quality_gate.py", "src/aroll_v21/decision/semantic_adjudication.py"],
        "test_area": "tests/test_aroll_v21_final_visible_generic_qc_regressions.py",
        "assertion": "semantically incoherent final visible captions should be blocked or routed to semantic adjudication",
    },
    "asr_text_error": {
        "mechanism": "ASR/native word confidence and final visible semantic quality gate",
        "files": ["src/aroll_v21/ingest/source_graph.py", "src/aroll_v21/quality/quality_gate.py"],
        "test_area": "tests/test_aroll_v21_final_visible_generic_qc_regressions.py",
        "assertion": "ASR-suspect visible captions should carry evidence and fail the final quality gate when unrepaired",
    },
    "cross_caption_semantic_containment": {
        "mechanism": "cross-caption semantic containment detection",
        "files": ["src/aroll_v21/quality/final_caption_visible_repeat.py", "src/aroll_v21/quality/final_visible_caption_repair.py"],
        "test_area": "tests/test_aroll_v21_final_visible_generic_qc_regressions.py",
        "assertion": "adjacent visible captions should be checked as a combined semantic window, not only as isolated captions",
    },
    "restart_repeat_not_removed": {
        "mechanism": "restart/repeat visible-caption convergence",
        "files": ["src/aroll_v21/quality/final_caption_visible_repeat.py", "src/aroll_v21/compiler/rough_cut_quality_normalizer.py"],
        "test_area": "tests/test_aroll_v21_final_visible_generic_qc_regressions.py",
        "assertion": "restart repeats should be removed when the second visible caption restarts and completes the first take",
    },
}


@dataclass(frozen=True)
class QCIssueInput:
    issue_no: str
    bad_visible_text: str
    root_cause: str = ""
    note: str = ""
    expected_visible_text: str = ""


def read_json_artifact(run_dir: Path, name: str) -> Any:
    path = run_dir / name
    if path.exists():
        return _read_json_path(path)
    gzip_path = path.with_suffix(path.suffix + ".gz")
    if gzip_path.exists():
        return _read_json_path(gzip_path)
    return None


def _read_json_path(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text("utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def normalize_visible_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("/", "").replace("\\", "").replace("|", "")
    return re.sub(r"[，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}<>《》-]", "", text).lower()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _row_text(row: Any) -> str:
    return str(_as_dict(row).get("text") or _as_dict(row).get("visible_text") or "")


def _row_id(row: Any, *keys: str) -> str:
    data = _as_dict(row)
    for key in keys:
        value = data.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def _row_int(row: Any, key: str) -> int | None:
    value = _as_dict(row).get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _word_ids_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        for word_id in row.get("word_ids") or []:
            text = str(word_id)
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        strings: list[str] = []
        for key, item in value.items():
            strings.append(str(key))
            strings.extend(_flatten_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(_flatten_strings(item))
        return strings
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    return []


def _payload_mentions(payload: Any, issue_text: str, word_ids: list[str], segment_ids: list[str]) -> bool:
    strings = _flatten_strings(payload)
    normalized_blob = normalize_visible_text("".join(strings))
    normalized_issue = normalize_visible_text(issue_text)
    if normalized_issue and normalized_issue in normalized_blob:
        return True
    raw_blob = "\n".join(strings)
    return any(item and item in raw_blob for item in [*word_ids, *segment_ids])


def _matched_rows(payloads: Any, issue_text: str, word_ids: list[str], segment_ids: list[str], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = [row for row in _as_list(payloads) if _payload_mentions(row, issue_text, word_ids, segment_ids)]
    return [_compact_payload(row) for row in rows[:limit]]


def _matched_rows_by_cluster(payloads: Any, cluster_ids: set[str], *, limit: int = 5) -> list[dict[str, Any]]:
    if not cluster_ids:
        return []
    rows = []
    for row in _as_list(payloads):
        data = _as_dict(row)
        if str(data.get("cluster_id") or "") in cluster_ids:
            rows.append(row)
    return [_compact_payload(row) for row in rows[:limit]]


def _compact_payload(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {"value": row}
    keep_keys = (
        "cluster_id",
        "issue_id",
        "issue_type",
        "type",
        "repeat_type",
        "cluster_type",
        "decision",
        "reason",
        "confidence",
        "requires_human_review",
        "text",
        "source_text",
        "candidate_text",
        "drop_text",
        "keep_text",
        "_decision_source",
        "_semantic_mode",
        "_blocker_code",
    )
    compact = {key: row.get(key) for key in keep_keys if key in row}
    for key in ("word_ids", "segment_ids", "caption_ids", "timeline_segment_ids"):
        if key in row:
            compact[key] = row.get(key)
    if not compact:
        strings = _flatten_strings(row)
        compact["excerpt"] = " ".join(strings[:20])[:500]
    return compact


def _load_artifacts(run_dir: Path) -> dict[str, Any]:
    names = (
        "run_summary.json",
        "artifact_manifest.json",
        "captions.json",
        "final_timeline.json",
        "final_edl.json",
        "source_graph.json",
        "semantic_request_payloads.json",
        "semantic_adjudication_report.json",
        "semantic_decisions.resolved.json",
        "deepseek_decisions.json",
        "quality_gate_report.json",
        "final_caption_visible_repeat_gate.json",
        "final_visible_caption_repair_report.json",
        "repeat_clusters.json",
        "decision_trace.json",
    )
    return {name: read_json_artifact(run_dir, name) for name in names}


def _find_caption_window(captions: list[Any], issue_text: str) -> dict[str, Any]:
    normalized_issue = normalize_visible_text(issue_text)
    if not normalized_issue or not captions:
        return {"match_found": False, "start_index": None, "end_index": None, "score": 0.0, "match_reason": "no_text_or_captions"}
    best: dict[str, Any] = {"match_found": False, "start_index": None, "end_index": None, "score": 0.0, "match_reason": "no_candidate"}
    max_window = min(4, len(captions))
    for start in range(len(captions)):
        for size in range(1, max_window + 1):
            end = start + size
            if end > len(captions):
                continue
            rows = [_as_dict(row) for row in captions[start:end]]
            window_text = "".join(str(row.get("text") or "") for row in rows)
            normalized_window = normalize_visible_text(window_text)
            if not normalized_window:
                continue
            if normalized_issue == normalized_window:
                score = 1000.0 + size
                reason = "exact_visible_window"
            elif normalized_issue in normalized_window or normalized_window in normalized_issue:
                score = 800.0 + min(len(normalized_issue), len(normalized_window)) - size
                reason = "substring_visible_window"
            else:
                score = _char_overlap_score(normalized_issue, normalized_window)
                reason = "character_overlap"
            if score > float(best["score"]):
                best = {
                    "match_found": score >= 0.45,
                    "start_index": start,
                    "end_index": end - 1,
                    "score": round(float(score), 4),
                    "match_reason": reason,
                    "matched_visible_text": window_text,
                }
    return best


def _char_overlap_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_chars = set(left)
    right_chars = set(right)
    overlap = len(left_chars & right_chars)
    return overlap / max(len(left_chars), len(right_chars))


def _caption_context(captions: list[Any], match: dict[str, Any]) -> dict[str, Any]:
    if not match.get("match_found"):
        return {
            "caption_segment_id": "",
            "caption_ids": [],
            "caption_start_us": None,
            "caption_end_us": None,
            "neighbor_prev_caption": "",
            "neighbor_next_caption": "",
            "timeline_segment_ids": [],
            "caption_word_ids": [],
            "matched_caption_text": "",
        }
    start = int(match["start_index"])
    end = int(match["end_index"])
    rows = [_as_dict(row) for row in captions[start : end + 1]]
    caption_ids = [_row_id(row, "caption_id", "id") for row in rows]
    starts = [_row_int(row, "target_start_us") for row in rows]
    ends = [_row_int(row, "target_end_us") for row in rows]
    timeline_segment_ids: list[str] = []
    caption_word_ids: list[str] = []
    for row in rows:
        for segment_id in row.get("timeline_segment_ids") or []:
            if str(segment_id) not in timeline_segment_ids:
                timeline_segment_ids.append(str(segment_id))
        for word_id in row.get("word_ids") or []:
            if str(word_id) not in caption_word_ids:
                caption_word_ids.append(str(word_id))
    prev_caption = _row_text(captions[start - 1]) if start > 0 else ""
    next_caption = _row_text(captions[end + 1]) if end + 1 < len(captions) else ""
    return {
        "caption_segment_id": caption_ids[0] if caption_ids else "",
        "caption_ids": [item for item in caption_ids if item],
        "caption_start_us": min([item for item in starts if item is not None], default=None),
        "caption_end_us": max([item for item in ends if item is not None], default=None),
        "neighbor_prev_caption": prev_caption,
        "neighbor_next_caption": next_caption,
        "timeline_segment_ids": timeline_segment_ids,
        "caption_word_ids": caption_word_ids,
        "matched_caption_text": "".join(_row_text(row) for row in rows),
    }


def _timeline_context(final_timeline: list[Any], timeline_segment_ids: list[str], issue_text: str) -> dict[str, Any]:
    rows_by_id = {_row_id(row, "segment_id", "id"): _as_dict(row) for row in final_timeline}
    rows = [rows_by_id[segment_id] for segment_id in timeline_segment_ids if segment_id in rows_by_id]
    if not rows:
        match = _find_caption_window(final_timeline, issue_text)
        if match.get("match_found"):
            rows = [_as_dict(row) for row in final_timeline[int(match["start_index"]) : int(match["end_index"]) + 1]]
    if not rows:
        return {
            "final_timeline_segment_id": "",
            "final_timeline_segment_ids": [],
            "video_segment_id": "",
            "source_media": "",
            "source_start_us": None,
            "source_end_us": None,
            "word_ids": [],
            "final_timeline_text": "",
        }
    segment_ids = [_row_id(row, "segment_id", "id") for row in rows]
    starts = [_row_int(row, "source_start_us") for row in rows]
    ends = [_row_int(row, "source_end_us") for row in rows]
    source_media = _row_id(rows[0], "source_material_id", "material_id")
    source_segment_id = _row_id(rows[0], "source_segment_id")
    return {
        "final_timeline_segment_id": segment_ids[0] if segment_ids else "",
        "final_timeline_segment_ids": [item for item in segment_ids if item],
        "video_segment_id": source_segment_id,
        "source_media": source_media,
        "source_start_us": min([item for item in starts if item is not None], default=None),
        "source_end_us": max([item for item in ends if item is not None], default=None),
        "word_ids": _word_ids_from_rows(rows),
        "final_timeline_text": "".join(_row_text(row) for row in rows),
    }


def _native_words(source_graph: Any, word_ids: list[str]) -> dict[str, Any]:
    graph = _as_dict(source_graph)
    words = [_as_dict(row) for row in _as_list(graph.get("words"))]
    by_id = {str(row.get("word_id") or ""): row for row in words}
    native_rows = [by_id[word_id] for word_id in word_ids if word_id in by_id]
    return {
        "native_words_text": "".join(str(row.get("text") or row.get("word_text") or "") for row in native_rows),
        "native_word_rows": [
            {
                "word_id": str(row.get("word_id") or ""),
                "text": str(row.get("text") or row.get("word_text") or ""),
                "source_start_us": row.get("source_start_us"),
                "source_end_us": row.get("source_end_us"),
                "confidence": row.get("confidence"),
            }
            for row in native_rows
        ],
    }


def _gate_summary(report: Any) -> dict[str, Any]:
    data = _as_dict(report)
    keys = (
        "gate_passed",
        "quality_gate_passed",
        "final_caption_visible_repeat_gate_passed",
        "validator_report_ok",
        "blocker_codes",
        "blocker_count",
        "dangling_prefix_suffix_count",
        "semantic_garbage_or_asr_suspect_count",
        "cross_caption_semantic_containment_count",
        "restart_repeat_visible_count",
        "fatal_semantic_issue_count",
    )
    return {key: data.get(key) for key in keys if key in data}


def _why_gate_passed(artifacts: dict[str, Any], issue_text: str, word_ids: list[str], segment_ids: list[str]) -> list[str]:
    reasons: list[str] = []
    quality = _as_dict(artifacts.get("quality_gate_report.json"))
    final_repeat = _as_dict(artifacts.get("final_caption_visible_repeat_gate.json"))
    repair = _as_dict(artifacts.get("final_visible_caption_repair_report.json"))
    if quality:
        matched = _payload_mentions(quality, issue_text, word_ids, segment_ids)
        if bool(quality.get("quality_gate_passed", quality.get("gate_passed", True))) and not matched:
            reasons.append("quality_gate_passed_without_issue_candidate_match")
        for key in (
            "dangling_prefix_suffix_count",
            "semantic_garbage_or_asr_suspect_count",
            "cross_caption_semantic_containment_count",
            "restart_repeat_visible_count",
        ):
            if key in quality:
                reasons.append(f"{key}={quality.get(key)}")
    if final_repeat:
        matched = _payload_mentions(final_repeat, issue_text, word_ids, segment_ids)
        if bool(final_repeat.get("gate_passed", final_repeat.get("final_caption_visible_repeat_gate_passed", True))) and not matched:
            reasons.append("final_caption_visible_repeat_gate_passed_without_issue_candidate_match")
    if repair:
        if repair.get("final_visible_repair_success") is True:
            reasons.append("final_visible_repair_report_marked_success")
        if repair.get("final_visible_repair_unresolved") in (0, [], None):
            reasons.append("final_visible_repair_unresolved_empty")
    if not reasons:
        reasons.append("no_gate_evidence_available_or_gate_did_not_pass")
    return reasons


def _suggestion_for_issue(issue: QCIssueInput, evidence: dict[str, Any]) -> dict[str, Any]:
    normalized_root = str(issue.root_cause or "").strip()
    config = ROOT_CAUSE_FIX_MAP.get(normalized_root, {})
    issue_slug = _slugify(issue.bad_visible_text or issue.issue_no)
    test_area = str(config.get("test_area") or "tests/test_aroll_v21_final_visible_generic_qc_regressions.py")
    mechanism = str(config.get("mechanism") or "final visible caption quality gate")
    assertion = str(config.get("assertion") or "final visible captions should block or repair this defect class without sample-text exceptions")
    return {
        "issue_no": issue.issue_no,
        "root_cause": normalized_root or "unclassified",
        "general_mechanism_broken": mechanism,
        "files_to_inspect": list(config.get("files") or ["src/aroll_v21/quality/quality_gate.py"]),
        "test_file": test_area,
        "test_name": f"test_qc_defect_{issue.issue_no}_{issue_slug}",
        "fixture_strategy": "Build a minimal source_graph/final_timeline/captions fixture from the ledger evidence; avoid raw video and sample-specific literals in production code.",
        "assertion": assertion,
        "evidence_keys": {
            "caption_ids": evidence.get("caption_ids") or [],
            "final_timeline_segment_ids": evidence.get("final_timeline_segment_ids") or [],
            "word_ids": evidence.get("word_ids") or [],
        },
    }


def _slugify(value: str) -> str:
    normalized = normalize_visible_text(value)
    if not normalized:
        return "unknown"
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", normalized)
    ascii_slug = ascii_slug.strip("_")
    return ascii_slug[:48] or "cjk_visible_caption"


def build_ledger(
    *,
    run_dir: Path,
    issues: list[QCIssueInput],
    out_root: Path | None = None,
    case_id: str = "",
    qc_source: str = "manual_qc",
    draft_label: str = "",
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    artifacts = _load_artifacts(run_dir)
    run_summary = _as_dict(artifacts.get("run_summary.json"))
    captions = _as_list(artifacts.get("captions.json"))
    final_timeline = _as_list(artifacts.get("final_timeline.json"))
    configured_out_root = get_aroll_quality_defect_ledger_dir()
    out_root = (out_root or configured_out_root).resolve()
    case_id = case_id or f"{run_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = out_root / case_id
    created_at = datetime.now(timezone.utc).isoformat()
    ledger_issues: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for issue in issues:
        match = _find_caption_window(captions, issue.bad_visible_text)
        caption = _caption_context(captions, match)
        timeline = _timeline_context(final_timeline, caption["timeline_segment_ids"], issue.bad_visible_text)
        word_ids = list(timeline.get("word_ids") or caption.get("caption_word_ids") or [])
        segment_ids = list(timeline.get("final_timeline_segment_ids") or caption.get("timeline_segment_ids") or [])
        native = _native_words(artifacts.get("source_graph.json"), word_ids)
        semantic_rows = _matched_rows(artifacts.get("semantic_request_payloads.json"), issue.bad_visible_text, word_ids, segment_ids)
        semantic_cluster_ids = {str(row.get("cluster_id") or "") for row in semantic_rows if str(row.get("cluster_id") or "")}
        deepseek_rows = _matched_rows(artifacts.get("deepseek_decisions.json"), issue.bad_visible_text, word_ids, segment_ids)
        if not deepseek_rows:
            deepseek_rows = _matched_rows_by_cluster(artifacts.get("deepseek_decisions.json"), semantic_cluster_ids)
        resolved_rows = _matched_rows(artifacts.get("semantic_decisions.resolved.json"), issue.bad_visible_text, word_ids, segment_ids)
        if not resolved_rows:
            resolved_rows = _matched_rows_by_cluster(artifacts.get("semantic_decisions.resolved.json"), semantic_cluster_ids)
        repeat_gate_rows = _matched_rows([artifacts.get("final_caption_visible_repeat_gate.json")], issue.bad_visible_text, word_ids, segment_ids)
        quality_gate_rows = _matched_rows([artifacts.get("quality_gate_report.json")], issue.bad_visible_text, word_ids, segment_ids)
        repair_rows = _matched_rows([artifacts.get("final_visible_caption_repair_report.json")], issue.bad_visible_text, word_ids, segment_ids)
        evidence = {
            "issue_no": issue.issue_no,
            "bad_visible_text": issue.bad_visible_text,
            "expected_visible_text": issue.expected_visible_text,
            "note": issue.note,
            "root_cause": issue.root_cause,
            "caption_match": match,
            **caption,
            **timeline,
            **native,
            "entered_semantic_request_payloads": bool(semantic_rows),
            "semantic_request_payload_matches": semantic_rows,
            "entered_deepseek": bool(deepseek_rows),
            "deepseek_decision_matches": deepseek_rows,
            "resolved_semantic_decision_matches": resolved_rows,
            "entered_final_caption_visible_repeat_gate": bool(repeat_gate_rows),
            "final_caption_visible_repeat_gate_summary": _gate_summary(artifacts.get("final_caption_visible_repeat_gate.json")),
            "final_caption_visible_repeat_gate_matches": repeat_gate_rows,
            "entered_quality_gate_report": bool(quality_gate_rows),
            "quality_gate_summary": _gate_summary(artifacts.get("quality_gate_report.json")),
            "quality_gate_matches": quality_gate_rows,
            "entered_final_visible_repair_report": bool(repair_rows),
            "final_visible_repair_summary": _gate_summary(artifacts.get("final_visible_caption_repair_report.json")),
            "final_visible_repair_matches": repair_rows,
            "why_gate_passed": _why_gate_passed(artifacts, issue.bad_visible_text, word_ids, segment_ids),
        }
        suggestion = _suggestion_for_issue(issue, evidence)
        evidence["suggested_regression_test"] = suggestion
        ledger_issues.append(evidence)
        suggestions.append(suggestion)
    ledger = {
        "schema_version": 1,
        "case_id": case_id,
        "created_at": created_at,
        "qc_source": qc_source,
        "draft_label": draft_label,
        "source_run": {
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "status": run_summary.get("status"),
            "write_status": run_summary.get("write_status"),
            "ready_for_user_manual_qc": run_summary.get("READY_FOR_USER_MANUAL_QC") or run_summary.get("ready_for_user_manual_qc"),
            "blocker_codes": run_summary.get("blocker_codes") or [],
            "report_profile": run_summary.get("report_profile"),
        },
        "runtime_config": {
            "default_quality_defect_ledger_dir": str(configured_out_root),
            "effective_quality_defect_ledger_dir": str(out_root),
            "override_used": out_root != configured_out_root.resolve(),
        },
        "artifact_availability": {name: artifacts.get(name) is not None for name in sorted(artifacts)},
        "issue_count": len(ledger_issues),
        "issues": ledger_issues,
        "suggested_regression_tests": suggestions,
        "outputs": {
            "json": str(out_dir / "defect_ledger.json"),
            "markdown": str(out_dir / "defect_ledger.md"),
        },
    }
    write_json(out_dir / "defect_ledger.json", ledger)
    (out_dir / "defect_ledger.md").write_text(render_markdown(ledger), "utf-8")
    return ledger


def render_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        f"# A-Roll Quality Defect Ledger: {ledger['case_id']}",
        "",
        f"- created_at: `{ledger['created_at']}`",
        f"- run_dir: `{ledger['source_run']['run_dir']}`",
        f"- status: `{ledger['source_run'].get('status')}`",
        f"- write_status: `{ledger['source_run'].get('write_status')}`",
        f"- issue_count: {ledger['issue_count']}",
        "",
        "## Issues",
        "",
        "| issue | bad visible text | root cause | caption ids | timeline ids | semantic | deepseek | gate pass reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for issue in ledger["issues"]:
        lines.append(
            "| {issue_no} | {bad} | {root} | {captions} | {segments} | {semantic} | {deepseek} | {reason} |".format(
                issue_no=_md(issue.get("issue_no")),
                bad=_md(issue.get("bad_visible_text")),
                root=_md(issue.get("root_cause") or "unclassified"),
                captions=_md(", ".join(issue.get("caption_ids") or [])),
                segments=_md(", ".join(issue.get("final_timeline_segment_ids") or [])),
                semantic="yes" if issue.get("entered_semantic_request_payloads") else "no",
                deepseek="yes" if issue.get("entered_deepseek") else "no",
                reason=_md("; ".join(issue.get("why_gate_passed") or [])),
            )
        )
    lines.extend(["", "## Evidence Detail", ""])
    for issue in ledger["issues"]:
        lines.extend(
            [
                f"### Issue {issue.get('issue_no')}",
                "",
                f"- bad_visible_text: `{issue.get('bad_visible_text')}`",
                f"- expected_visible_text: `{issue.get('expected_visible_text') or ''}`",
                f"- matched_caption_text: `{issue.get('matched_caption_text') or ''}`",
                f"- neighbor_prev_caption: `{issue.get('neighbor_prev_caption') or ''}`",
                f"- neighbor_next_caption: `{issue.get('neighbor_next_caption') or ''}`",
                f"- caption_start_us: `{issue.get('caption_start_us')}`",
                f"- caption_end_us: `{issue.get('caption_end_us')}`",
                f"- source_media: `{issue.get('source_media') or ''}`",
                f"- source_start_us: `{issue.get('source_start_us')}`",
                f"- source_end_us: `{issue.get('source_end_us')}`",
                f"- word_ids: `{', '.join(issue.get('word_ids') or [])}`",
                f"- native_words_text: `{issue.get('native_words_text') or ''}`",
                f"- why_gate_passed: `{'; '.join(issue.get('why_gate_passed') or [])}`",
                "",
                "**Suggested Regression Test**",
                "",
                f"- file: `{issue['suggested_regression_test']['test_file']}`",
                f"- test: `{issue['suggested_regression_test']['test_name']}`",
                f"- mechanism: `{issue['suggested_regression_test']['general_mechanism_broken']}`",
                f"- assertion: {issue['suggested_regression_test']['assertion']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _validate_parallel_issue_args(issue_texts: list[str], named_values: list[tuple[str, list[str]]]) -> None:
    issue_count = len(issue_texts)
    for arg_name, values in named_values:
        if values and len(values) != issue_count:
            raise ValueError(
                f"{arg_name} count ({len(values)}) must match --issue-text count ({issue_count}); "
                "use --issues-json for sparse per-issue metadata."
            )


def load_issues(path: Path | None, issue_texts: list[str], root_causes: list[str], notes: list[str], expected_texts: list[str]) -> list[QCIssueInput]:
    issues: list[QCIssueInput] = []
    if path is not None:
        payload = _read_json_path(path)
        rows = payload.get("issues") if isinstance(payload, dict) else payload
        for index, row in enumerate(_as_list(rows), start=1):
            data = _as_dict(row)
            issues.append(
                QCIssueInput(
                    issue_no=str(data.get("issue_no") or index),
                    bad_visible_text=str(data.get("bad_visible_text") or data.get("text") or ""),
                    root_cause=str(data.get("root_cause") or ""),
                    note=str(data.get("note") or ""),
                    expected_visible_text=str(data.get("expected_visible_text") or ""),
                )
            )
    _validate_parallel_issue_args(
        issue_texts,
        [
            ("--root-cause", root_causes),
            ("--note", notes),
            ("--expected-visible-text", expected_texts),
        ],
    )
    start_index = len(issues) + 1
    for offset, text in enumerate(issue_texts):
        index = start_index + offset
        issues.append(
            QCIssueInput(
                issue_no=str(index),
                bad_visible_text=text,
                root_cause=root_causes[offset] if offset < len(root_causes) else "",
                note=notes[offset] if offset < len(notes) else "",
                expected_visible_text=expected_texts[offset] if offset < len(expected_texts) else "",
            )
        )
    issues = [issue for issue in issues if issue.bad_visible_text.strip()]
    if not issues:
        raise ValueError("No QC issues provided. Use --issue-text or --issues-json.")
    return issues


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register manual QC defects and extract A-Roll V21 run evidence into a ledger.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--issues-json", type=Path)
    parser.add_argument("--issue-text", action="append", default=[], help="Bad visible caption text from manual QC. Can be repeated.")
    parser.add_argument("--root-cause", action="append", default=[], help="Root cause for the matching --issue-text. Can be repeated.")
    parser.add_argument("--note", action="append", default=[], help="Operator note for the matching --issue-text. Can be repeated.")
    parser.add_argument("--expected-visible-text", action="append", default=[], help="Expected corrected visible text. Can be repeated.")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--qc-source", default="manual_qc")
    parser.add_argument("--draft-label", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    issues = load_issues(args.issues_json, args.issue_text, args.root_cause, args.note, args.expected_visible_text)
    ledger = build_ledger(
        run_dir=args.run_dir,
        issues=issues,
        out_root=args.out_root,
        case_id=args.case_id,
        qc_source=args.qc_source,
        draft_label=args.draft_label,
    )
    print(ledger["outputs"]["json"])
    print(ledger["outputs"]["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
