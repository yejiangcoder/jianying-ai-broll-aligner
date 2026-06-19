from __future__ import annotations

from pathlib import Path
from typing import Any


class RootMirrorDetectionError(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(str(report.get("root_mirror_error") or "ROOT_MIRROR_DETECTION_FAILED"))


def root_mirror_detection_report(
    draft_dir: Path,
    timeline_id: str,
    *,
    root_mirror_error: str = "",
    reason: str = "",
) -> dict[str, Any]:
    draft_root = Path(draft_dir)
    timeline_text = str(timeline_id or "").strip()
    timeline_dir = draft_root / "Timelines" / timeline_text if timeline_text else draft_root / "Timelines"
    root_draft_content = draft_root / "draft_content.json"
    root_template = draft_root / "template-2.tmp"
    timeline_draft_content = timeline_dir / "draft_content.json"
    timeline_template = timeline_dir / "template-2.tmp"

    root_exists = root_draft_content.exists()
    mirror_exists = root_template.exists()
    timeline_exists = timeline_draft_content.exists() and timeline_template.exists()
    draft_content_match = False
    template_match = False
    detected_error = str(root_mirror_error or "")
    detected_reason = str(reason or "")

    if not detected_error:
        if not draft_root.exists():
            detected_error = "ROOT_MIRROR_DRAFT_DIR_MISSING"
            detected_reason = "draft directory does not exist"
        elif not timeline_text:
            detected_error = "ROOT_MIRROR_TIMELINE_ID_MISSING"
            detected_reason = "active timeline id is missing"
        elif not timeline_dir.exists() or not timeline_exists:
            detected_error = "ROOT_MIRROR_TIMELINE_FILES_MISSING"
            detected_reason = "active timeline draft_content/template files are missing"
        elif root_exists and mirror_exists:
            draft_content_match = _same_file_bytes(root_draft_content, timeline_draft_content)
            template_match = _same_file_bytes(root_template, timeline_template)
            if draft_content_match and template_match:
                detected_reason = "root mirror files match active timeline files"
            else:
                detected_error = "ROOT_MIRROR_FILES_DO_NOT_MATCH_ACTIVE_TIMELINE"
                detected_reason = "root mirror files exist but do not match active timeline files"
        elif root_exists or mirror_exists:
            detected_error = "ROOT_MIRROR_PARTIAL_FILES_PRESENT"
            detected_reason = "only one root mirror file exists"
        else:
            detected_reason = "root mirror files are absent"
    elif not detected_reason:
        detected_reason = "root mirror detection function failed"

    root_mirror_match = bool(root_exists and mirror_exists and timeline_exists and draft_content_match and template_match)
    return {
        "root_mirror_error": detected_error,
        "expected_root_path": str(root_draft_content),
        "timeline_path": str(timeline_draft_content),
        "mirror_path": str(root_template),
        "root_template_path": str(root_template),
        "timeline_template_path": str(timeline_template),
        "root_exists": root_exists,
        "timeline_exists": timeline_exists,
        "mirror_exists": mirror_exists,
        "root_draft_content_match": draft_content_match,
        "root_template_match": template_match,
        "root_mirror_match": root_mirror_match,
        "reason": detected_reason,
    }


def root_mirrors_timeline_id(draft_dir: Path, _jy_draftc: Path | None, _run_dir: Path | None, timeline_id: str) -> bool:
    report = root_mirror_detection_report(Path(draft_dir), str(timeline_id or ""))
    if report["root_mirror_error"]:
        raise RootMirrorDetectionError(report)
    return bool(report["root_mirror_match"])


def root_mirror_report_from_exception(exc: Exception, draft_dir: Path, timeline_id: str) -> dict[str, Any]:
    report = getattr(exc, "report", None)
    if isinstance(report, dict):
        return dict(report)
    return root_mirror_detection_report(
        Path(draft_dir),
        str(timeline_id or ""),
        root_mirror_error=str(exc),
        reason="root mirror detection function failed",
    )


def _same_file_bytes(left: Path, right: Path) -> bool:
    try:
        if not left.exists() or not right.exists():
            return False
        if left.stat().st_size != right.stat().st_size:
            return False
        return left.read_bytes() == right.read_bytes()
    except OSError:
        return False
