from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jy_bridge import decrypt  # noqa: E402


AUTOMATION_MARKERS = (
    "v21_",
    "aroll_v21",
    "generated_caption",
    "source_segment_template",
    "write_video_row",
    "resolved_template_map",
    "actual_audio_coverage",
)

TARGET_FILE_NAMES = ("draft_content.json", "draft_content.json.bak", "template-2.tmp")
BACKUP_STAMP_RE = re.compile(r"^(\d{14})_")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def backup_stamp(path: Path) -> str:
    match = BACKUP_STAMP_RE.match(path.name)
    return match.group(1) if match else ""


def json_blob(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(data)


def normalize_media_path(path_value: str) -> str:
    return str(path_value or "").replace("\\", "/").rstrip("/")


def media_path_under_root(path_value: str, root_value: str) -> bool:
    path = normalize_media_path(path_value).casefold()
    root = normalize_media_path(root_value).casefold()
    if not path or not root:
        return False
    return path == root or path.startswith(root + "/")


def summarize_media(data: Any, expected_source_media_root: str = "") -> dict[str, Any]:
    videos = []
    if isinstance(data, dict):
        materials = data.get("materials")
        if isinstance(materials, dict) and isinstance(materials.get("videos"), list):
            videos = [row for row in materials.get("videos") or [] if isinstance(row, dict)]

    material_paths: list[str] = []
    material_names: list[str] = []
    seen_paths: set[str] = set()
    seen_names: set[str] = set()
    for row in videos:
        path = str(row.get("path") or "")
        name = str(row.get("material_name") or row.get("name") or "")
        path_key = normalize_media_path(path).casefold()
        name_key = name.casefold()
        if path and path_key not in seen_paths:
            material_paths.append(path)
            seen_paths.add(path_key)
        if name and name_key not in seen_names:
            material_names.append(name)
            seen_names.add(name_key)

    expected_root = normalize_media_path(expected_source_media_root)
    source_media_root_ok: bool | None = None
    source_media_root_mismatches: list[str] = []
    source_media_root_missing_path_count = 0
    if expected_root:
        source_media_root_missing_path_count = sum(1 for row in videos if not str(row.get("path") or ""))
        source_media_root_mismatches = [
            path for path in material_paths if not media_path_under_root(path, expected_root)
        ]
        source_media_root_ok = bool(material_paths) and source_media_root_missing_path_count == 0 and not source_media_root_mismatches

    return {
        "video_material_count": len(videos),
        "video_material_paths": material_paths,
        "video_material_names": material_names,
        "expected_source_media_root": expected_root,
        "source_media_root_ok": source_media_root_ok,
        "source_media_root_mismatches": source_media_root_mismatches,
        "source_media_root_missing_path_count": source_media_root_missing_path_count,
    }


def summarize_tracks(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "valid_draft_content": False,
            "track_count": 0,
            "video_segments": 0,
            "text_segments": 0,
        }

    tracks = data.get("tracks")
    if not isinstance(tracks, list):
        return {
            "valid_draft_content": False,
            "track_count": 0,
            "video_segments": 0,
            "text_segments": 0,
        }

    video_segments = 0
    text_segments = 0
    for track in tracks:
        if not isinstance(track, dict):
            continue
        segments = track.get("segments")
        if not isinstance(segments, list):
            continue
        track_type = str(track.get("type") or "").lower()
        if track_type == "video":
            video_segments += len(segments)
        elif track_type == "text":
            text_segments += len(segments)

    return {
        "valid_draft_content": True,
        "track_count": len(tracks),
        "video_segments": video_segments,
        "text_segments": text_segments,
    }


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if root and path.is_relative_to(root) else path.name,
        "name": path.name,
        "length": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "last_write_time_epoch": stat.st_mtime,
        "last_write_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
        "backup_timestamp": backup_stamp(path),
    }


def decrypt_candidate(
    jy_draftc: Path,
    path: Path,
    output_dir: Path,
    keep_decrypted: bool,
    expected_source_media_root: str = "",
) -> dict[str, Any]:
    record = file_record(path)
    record.update(
        {
            "sha256": sha256_path(path),
            "decrypt_ok": False,
            "json_ok": False,
            "valid_draft_content": False,
            "clean_like": False,
            "dirty_like": False,
            "automation_marker_count": 0,
            "automation_markers": {},
            "track_count": 0,
            "video_segments": 0,
            "text_segments": 0,
            "video_material_count": 0,
            "video_material_paths": [],
            "video_material_names": [],
            "expected_source_media_root": normalize_media_path(expected_source_media_root),
            "source_media_root_ok": None if not expected_source_media_root else False,
            "source_media_root_mismatches": [],
            "source_media_root_missing_path_count": 0,
            "decrypted_path": "",
            "error": "",
        }
    )

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name)
    dec_path = output_dir / "decrypted_candidates" / f"{safe_name}.{record['sha256'][:12]}.dec.json"
    dec_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        decrypt(jy_draftc, path, dec_path)
        record["decrypt_ok"] = True
        data = json.loads(dec_path.read_text("utf-8"))
        record["json_ok"] = True
        record.update(summarize_tracks(data))
        record.update(summarize_media(data, expected_source_media_root))
        blob = json_blob(data)
        markers = {marker: blob.count(marker) for marker in AUTOMATION_MARKERS}
        marker_count = sum(markers.values())
        record["automation_markers"] = markers
        record["automation_marker_count"] = marker_count
        source_media_root_ok = True
        if expected_source_media_root:
            source_media_root_ok = bool(record.get("source_media_root_ok"))
        record["clean_like"] = bool(
            record["valid_draft_content"] and record["video_segments"] > 0 and marker_count == 0 and source_media_root_ok
        )
        record["dirty_like"] = bool(record["valid_draft_content"] and marker_count > 0)
        if keep_decrypted:
            record["decrypted_path"] = str(dec_path)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if not keep_decrypted:
            try:
                dec_path.unlink()
            except FileNotFoundError:
                pass

    return record


def candidate_order(record: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(record.get("backup_timestamp") or ""),
        int(record.get("mtime_ns") or 0),
        int(record.get("length") or 0),
        str(record.get("path") or ""),
    )


def candidate_epoch(record: dict[str, Any]) -> float:
    return float(record.get("last_write_time_epoch") or 0.0)


def collect_backup_files(backup_root: Path) -> list[Path]:
    if not backup_root.exists():
        return []
    return sorted((path for path in backup_root.rglob("*") if path.is_file()), key=lambda path: str(path).lower())


def collect_active_targets(draft_dir: Path) -> list[Path]:
    targets: list[Path] = []
    for name in TARGET_FILE_NAMES:
        targets.append(draft_dir / name)

    timelines_root = draft_dir / "Timelines"
    if timelines_root.exists():
        for timeline_dir in sorted((p for p in timelines_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            if (timeline_dir / "draft_content.json").exists() or (timeline_dir / "template-2.tmp").exists():
                for name in TARGET_FILE_NAMES:
                    targets.append(timeline_dir / name)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in targets:
        key = str(path).lower()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def active_target_record(
    jy_draftc: Path,
    path: Path,
    output_dir: Path,
    keep_decrypted: bool,
    expected_source_media_root: str = "",
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "sha256": "",
            "length": 0,
            "decrypt_ok": False,
            "valid_draft_content": False,
            "video_segments": 0,
            "text_segments": 0,
            "automation_marker_count": 0,
            "video_material_count": 0,
            "video_material_paths": [],
            "video_material_names": [],
            "expected_source_media_root": normalize_media_path(expected_source_media_root),
            "source_media_root_ok": None if not expected_source_media_root else False,
            "source_media_root_mismatches": [],
            "source_media_root_missing_path_count": 0,
        }
    record = decrypt_candidate(jy_draftc, path, output_dir, keep_decrypted, expected_source_media_root)
    record["exists"] = True
    return record


def select_baseline(
    candidates: list[dict[str, Any]],
    active_targets: list[dict[str, Any]],
    baseline_path: Path | None,
    jy_draftc: Path,
    output_dir: Path,
    keep_decrypted: bool,
    selection_mode: str,
    initial_cluster_seconds: int,
    expected_source_media_root: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    warnings: list[str] = []

    if baseline_path is not None:
        selected = decrypt_candidate(jy_draftc, baseline_path, output_dir, keep_decrypted, expected_source_media_root)
        selected["selection_reason"] = "explicit_baseline_path"
        if not selected.get("clean_like"):
            warnings.append("explicit baseline is not clean_like; use only when intentionally restoring this exact file")
        if expected_source_media_root and not selected.get("source_media_root_ok"):
            warnings.append("explicit baseline source media paths do not match expected_source_media_root")
        return selected, {
            "strategy": "explicit_baseline_path",
            "confidence": "high" if selected.get("clean_like") else "low",
            "warnings": warnings,
            "first_dirty_backup": None,
        }

    clean_candidates = [candidate for candidate in candidates if candidate.get("clean_like")]
    dirty_candidates = [candidate for candidate in candidates if candidate.get("dirty_like")]
    active_dirty = any(target.get("dirty_like") for target in active_targets)

    if not clean_candidates:
        return None, {
            "strategy": "latest_clean_before_first_dirty",
            "confidence": "none",
            "warnings": ["no decryptable clean backup candidate found"],
            "first_dirty_backup": min(dirty_candidates, key=candidate_order) if dirty_candidates else None,
        }

    first_dirty = min(dirty_candidates, key=candidate_order) if dirty_candidates else None

    if selection_mode == "auto":
        selection_mode = "latest-clean-before-dirty" if dirty_candidates else "initial-clean-cluster"

    if selection_mode == "initial-clean-cluster":
        first_clean = min(clean_candidates, key=candidate_order)
        first_epoch = candidate_epoch(first_clean)
        cluster = [
            candidate
            for candidate in clean_candidates
            if candidate_epoch(candidate) <= first_epoch + max(0, initial_cluster_seconds)
        ]
        if not cluster:
            cluster = [first_clean]
        selected = max(cluster, key=candidate_order)
        selected["selection_reason"] = "latest_clean_in_initial_cluster"
        later_clean_count = sum(1 for candidate in clean_candidates if candidate_order(candidate) > candidate_order(selected))
        if later_clean_count:
            warnings.append(
                f"{later_clean_count} later clean-like backup(s) ignored to avoid selecting manual edits as baseline"
            )
        confidence = "high" if (first_dirty or active_dirty or later_clean_count) else "medium"
        return selected, {
            "strategy": "initial_clean_cluster",
            "confidence": confidence,
            "warnings": warnings,
            "first_dirty_backup": first_dirty,
            "selected_order_cutoff": candidate_order(selected),
            "later_clean_candidate_count": later_clean_count,
        }

    if selection_mode != "latest-clean-before-dirty":
        warnings.append(f"unknown selection_mode={selection_mode}; falling back to initial_clean_cluster")
        return select_baseline(
            candidates=candidates,
            active_targets=active_targets,
            baseline_path=None,
            jy_draftc=jy_draftc,
            output_dir=output_dir,
            keep_decrypted=keep_decrypted,
            selection_mode="initial-clean-cluster",
            initial_cluster_seconds=initial_cluster_seconds,
            expected_source_media_root=expected_source_media_root,
        )

    if first_dirty is not None:
        first_dirty_key = candidate_order(first_dirty)
        eligible = [candidate for candidate in clean_candidates if candidate_order(candidate) < first_dirty_key]
        if not eligible:
            eligible = clean_candidates
            warnings.append("no clean backup before first dirty backup; selected latest clean backup instead")
    else:
        eligible = clean_candidates
        warnings.append("no dirty backup marker found; selected latest clean backup")

    selected = max(eligible, key=candidate_order)
    selected["selection_reason"] = "latest_clean_before_first_dirty" if first_dirty else "latest_clean_no_dirty_marker"

    same_stamp = [
        candidate
        for candidate in clean_candidates
        if candidate.get("backup_timestamp") and candidate.get("backup_timestamp") == selected.get("backup_timestamp")
    ]
    hashes_at_stamp = {candidate.get("sha256") for candidate in same_stamp}
    if len(hashes_at_stamp) > 1:
        warnings.append("multiple clean hashes share selected backup timestamp; inspect report before applying")

    confidence = "high" if first_dirty or active_dirty else "medium"
    if len(hashes_at_stamp) > 1:
        confidence = "low"

    return selected, {
        "strategy": "latest_clean_before_first_dirty",
        "confidence": confidence,
        "warnings": warnings,
        "first_dirty_backup": first_dirty,
        "selected_order_cutoff": candidate_order(selected),
        "later_clean_candidate_count": sum(
            1 for candidate in clean_candidates if candidate_order(candidate) > candidate_order(selected)
        ),
    }


def collect_quarantine_backup_paths(
    backup_root: Path,
    all_backup_files: list[dict[str, Any]],
    first_dirty_backup: dict[str, Any] | None,
    selected_candidate: dict[str, Any] | None,
) -> list[str]:
    paths: list[str] = []
    if selected_candidate is not None:
        cutoff = candidate_order(selected_candidate)
        for record in all_backup_files:
            if candidate_order(record) > cutoff:
                paths.append(str(record["path"]))
    elif first_dirty_backup is not None:
        cutoff = candidate_order(first_dirty_backup)
        for record in all_backup_files:
            if candidate_order(record) >= cutoff:
                paths.append(str(record["path"]))

    manifest = backup_root / "timeline_backup_manifest.json"
    if manifest.exists():
        paths.append(str(manifest))

    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        key = path.lower()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    draft_dir = Path(args.draft_dir).resolve()
    jy_draftc = Path(args.jy_draftc).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    backup_root = draft_dir / ".backup"
    backup_files = collect_backup_files(backup_root)
    all_backup_records = [file_record(path, backup_root) for path in backup_files]
    candidate_paths = [path for path in backup_files if path.suffix.lower() == ".bak"]
    expected_source_media_root = normalize_media_path(args.expected_source_media_root)
    candidates = [
        decrypt_candidate(jy_draftc, path, output_dir, args.keep_decrypted, expected_source_media_root)
        for path in candidate_paths
    ]

    target_paths = collect_active_targets(draft_dir)
    active_targets = [
        active_target_record(jy_draftc, path, output_dir, args.keep_decrypted, expected_source_media_root)
        for path in target_paths
    ]

    selected, selection = select_baseline(
        candidates=candidates,
        active_targets=active_targets,
        baseline_path=Path(args.baseline_path).resolve() if args.baseline_path else None,
        jy_draftc=jy_draftc,
        output_dir=output_dir,
        keep_decrypted=args.keep_decrypted,
        selection_mode=args.selection_mode,
        initial_cluster_seconds=args.initial_cluster_seconds,
        expected_source_media_root=expected_source_media_root,
    )

    quarantine_paths = collect_quarantine_backup_paths(
        backup_root=backup_root,
        all_backup_files=all_backup_records,
        first_dirty_backup=selection.get("first_dirty_backup"),
        selected_candidate=selected,
    )

    selected_hash = str(selected.get("sha256") if selected else "")
    active_hash_match_count = sum(1 for target in active_targets if target.get("sha256") == selected_hash and selected_hash)
    active_dirty_count = sum(1 for target in active_targets if target.get("dirty_like"))

    report = {
        "draft_dir": str(draft_dir),
        "backup_root": str(backup_root),
        "jy_draftc": str(jy_draftc),
        "output_dir": str(output_dir),
        "expected_source_media_root": expected_source_media_root,
        "selected_candidate": selected,
        "selection": selection,
        "active_targets": active_targets,
        "candidate_count": len(candidates),
        "clean_candidate_count": sum(1 for candidate in candidates if candidate.get("clean_like")),
        "dirty_candidate_count": sum(1 for candidate in candidates if candidate.get("dirty_like")),
        "active_dirty_target_count": active_dirty_count,
        "active_hash_match_selected_count": active_hash_match_count,
        "all_active_targets_match_selected": bool(selected_hash and active_hash_match_count == len(active_targets)),
        "quarantine_backup_paths": quarantine_paths,
        "quarantine_backup_count": len(quarantine_paths),
        "candidates": sorted(candidates, key=candidate_order),
        "all_backup_files": all_backup_records,
    }

    report_path = output_dir / "rollback_candidate_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Jianying draft backups and select a clean rollback baseline.")
    parser.add_argument("--draft-dir", required=True)
    parser.add_argument("--jy-draftc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-path", default="")
    parser.add_argument("--keep-decrypted", action="store_true")
    parser.add_argument("--expected-source-media-root", default="")
    parser.add_argument(
        "--selection-mode",
        choices=("auto", "initial-clean-cluster", "latest-clean-before-dirty"),
        default="auto",
    )
    parser.add_argument("--initial-cluster-seconds", type=int, default=600)
    args = parser.parse_args()

    report = build_report(args)
    summary = {
        "report_path": report["report_path"],
        "selected_path": report["selected_candidate"]["path"] if report.get("selected_candidate") else "",
        "selected_sha256": report["selected_candidate"]["sha256"] if report.get("selected_candidate") else "",
        "confidence": report["selection"]["confidence"],
        "warnings": report["selection"]["warnings"],
        "clean_candidate_count": report["clean_candidate_count"],
        "dirty_candidate_count": report["dirty_candidate_count"],
        "active_dirty_target_count": report["active_dirty_target_count"],
        "quarantine_backup_count": report["quarantine_backup_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
