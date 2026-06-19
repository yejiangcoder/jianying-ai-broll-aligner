from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TOOL_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TOOL_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aroll_runtime_paths import get_aroll_audits_dir, get_runtime_root

DEFAULT_RUNTIME_ROOT = get_runtime_root()
DEFAULT_AUDIT_ROOT = get_aroll_audits_dir()

MOVE_DIR_MAP = {
    "runtime": ("aroll_v21_uat_runs",),
    "release": ("packages", "release"),
    "dev_snapshot": ("packages", "dev_snapshot"),
    "logs": ("logs",),
    "exports": ("exports",),
    "temp": ("temp",),
    "cache": ("cache",),
    "reports": ("aroll_v21_audits",),
    "audio_vad": ("aroll_v21_test_outputs", "audio_vad"),
    "post_inspect_runtime": ("aroll_v21_uat_runs", "post_inspect_runtime"),
    "baseline_backup": ("aroll_v21_backups", "baseline_backup"),
    "backup": ("aroll_v21_backups", "backup"),
    "restore_check": ("aroll_v21_backups", "restore_check"),
}

KEEP_ROOT_DIRS = {"src", "tests", "profiles", "config", "docs", "tools"}
KEEP_ROOT_FILES = {
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "README_UAT.md",
}
IGNORE_DIR_NAMES = {"__pycache__", ".pytest_cache"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}
MEDIA_SUFFIXES = {".mp4", ".mov", ".wav", ".mp3", ".png", ".jpg", ".jpeg", ".webp"}
DRAFT_ARTIFACT_NAMES = {"template-2.tmp"}


@dataclass
class SizeStats:
    path: str
    file_count: int
    size_bytes: int
    size_human: str


def fmt_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def all_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return [item for item in path.rglob("*") if item.is_file()]


def path_stats(path: Path) -> SizeStats:
    files = all_files(path)
    size = sum(item.stat().st_size for item in files if item.exists())
    return SizeStats(str(path), len(files), size, fmt_size(size))


def rel(path: Path) -> str:
    return path.relative_to(TOOL_ROOT).as_posix()


def is_draft_artifact(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in DRAFT_ARTIFACT_NAMES
        or name.startswith("draft_content")
        or name.endswith(".s16le")
        or name.endswith(".tmp")
        or name.endswith(".dec.json")
        or name.endswith(".enc.json")
    )


def classify_path(path: Path, runtime_root: Path, skip_drafts: bool = False) -> dict[str, Any]:
    relative = rel(path)
    parts = path.relative_to(TOOL_ROOT).parts
    top = parts[0] if parts else path.name
    size = path_stats(path) if path.is_dir() else SizeStats(str(path), 1, path.stat().st_size, fmt_size(path.stat().st_size))
    row = {
        "path": relative,
        "absolute_path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size_bytes": size.size_bytes,
        "size_human": size.size_human,
        "file_count": size.file_count,
        "category": "",
        "target_path": "",
        "reason": "",
    }
    if top in KEEP_ROOT_DIRS or path.name in KEEP_ROOT_FILES or path.name.startswith("run_"):
        row["category"] = "KEEP_IN_REPO"
        row["reason"] = "source_or_documentation"
        return row
    if top in MOVE_DIR_MAP:
        row["category"] = "MOVE_TO_EXTERNAL_RUNTIME"
        row["target_path"] = str(runtime_root.joinpath(*MOVE_DIR_MAP[top]))
        row["reason"] = "generated_runtime_or_package_output"
        return row
    if any(part in IGNORE_DIR_NAMES for part in parts) or path.suffix.lower() in IGNORE_SUFFIXES:
        row["category"] = "IGNORE_ONLY"
        row["reason"] = "cache_or_compiled_python"
        return row
    if is_draft_artifact(path):
        row["category"] = "REVIEW_BEFORE_MOVE"
        row["target_path"] = "" if skip_drafts else str(runtime_root / "drafts" / "draft_backups")
        row["reason"] = "draft_or_decrypted_artifact_requires_manual_review"
        return row
    if path.suffix.lower() in MEDIA_SUFFIXES:
        row["category"] = "REVIEW_BEFORE_MOVE"
        row["target_path"] = str(runtime_root / "exports")
        row["reason"] = "media_artifact_requires_manual_review"
        return row
    row["category"] = "REVIEW_BEFORE_MOVE"
    row["reason"] = "unclassified_project_root_item"
    return row


def scan_project(runtime_root: Path, skip_drafts: bool = False) -> dict[str, Any]:
    direct_items = sorted([item for item in TOOL_ROOT.iterdir() if item.name not in {".git"}], key=lambda p: p.name.lower())
    categorized = [classify_path(item, runtime_root, skip_drafts=skip_drafts) for item in direct_items]
    files = all_files(TOOL_ROOT)
    large_files = [
        {
            "path": rel(path),
            "absolute_path": str(path),
            "size_bytes": path.stat().st_size,
            "size_human": fmt_size(path.stat().st_size),
        }
        for path in files
        if path.stat().st_size > 1_048_576
    ]
    draft_like = [
        {
            "path": rel(path),
            "absolute_path": str(path),
            "size_bytes": path.stat().st_size,
            "size_human": fmt_size(path.stat().st_size),
        }
        for path in files
        if is_draft_artifact(path)
    ]
    stats = {
        "total": asdict(path_stats(TOOL_ROOT)),
        "src": asdict(path_stats(TOOL_ROOT / "src")),
        "tests": asdict(path_stats(TOOL_ROOT / "tests")),
        "runtime": asdict(path_stats(TOOL_ROOT / "runtime")),
        "release": asdict(path_stats(TOOL_ROOT / "release")),
        "dev_snapshot": asdict(path_stats(TOOL_ROOT / "dev_snapshot")),
        "__pycache__": {
            "file_count": len([path for path in files if "__pycache__" in path.parts or path.suffix.lower() in IGNORE_SUFFIXES]),
            "size_bytes": sum(path.stat().st_size for path in files if "__pycache__" in path.parts or path.suffix.lower() in IGNORE_SUFFIXES),
        },
        "draft_artifacts": {
            "file_count": len(draft_like),
            "size_bytes": sum(item["size_bytes"] for item in draft_like),
        },
    }
    stats["__pycache__"]["size_human"] = fmt_size(int(stats["__pycache__"]["size_bytes"]))
    stats["draft_artifacts"]["size_human"] = fmt_size(int(stats["draft_artifacts"]["size_bytes"]))
    categories: dict[str, list[dict[str, Any]]] = {
        "KEEP_IN_REPO": [],
        "MOVE_TO_EXTERNAL_RUNTIME": [],
        "IGNORE_ONLY": [],
        "REVIEW_BEFORE_MOVE": [],
    }
    for item in categorized:
        categories[item["category"]].append(item)
    return {
        "tool_root": str(TOOL_ROOT),
        "runtime_root": str(runtime_root),
        "stats": stats,
        "categories": categories,
        "large_files": sorted(large_files, key=lambda row: row["size_bytes"], reverse=True),
        "suspected_draft_artifacts": sorted(draft_like, key=lambda row: row["size_bytes"], reverse=True),
    }


def migration_plan(scan: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    mappings = []
    for item in scan["categories"]["MOVE_TO_EXTERNAL_RUNTIME"]:
        mappings.append(
            {
                "source": item["path"],
                "source_absolute": item["absolute_path"],
                "target": item["target_path"],
                "size_bytes": item["size_bytes"],
                "size_human": item["size_human"],
                "file_count": item["file_count"],
            }
        )
    return {
        "runtime_root": str(runtime_root),
        "directory_layout": {
            "aroll_v21_uat_runs": str(runtime_root / "aroll_v21_uat_runs"),
            "aroll_v21_audits": str(runtime_root / "aroll_v21_audits"),
            "aroll_v21_test_outputs": str(runtime_root / "aroll_v21_test_outputs"),
            "aroll_v21_backups": str(runtime_root / "aroll_v21_backups"),
            "temp": str(runtime_root / "temp"),
            "cache": str(runtime_root / "cache"),
            "broll/design_runs": str(runtime_root / "broll" / "design_runs"),
            "broll/material_index": str(runtime_root / "broll" / "material_index"),
            "broll/downloaded_materials": str(runtime_root / "broll" / "downloaded_materials"),
            "ai_images/batches": str(runtime_root / "ai_images" / "batches"),
            "ai_images/manifests": str(runtime_root / "ai_images" / "manifests"),
            "drafts/real_drafts": str(runtime_root / "drafts" / "real_drafts"),
            "drafts/draft_backups": str(runtime_root / "drafts" / "draft_backups"),
            "exports": str(runtime_root / "exports"),
            "logs": str(runtime_root / "logs"),
            "packages/release": str(runtime_root / "packages" / "release"),
            "packages/dev_snapshot": str(runtime_root / "packages" / "dev_snapshot"),
        },
        "mappings": mappings,
        "destructive_action_default": False,
        "requires_confirmation": True,
    }


def dry_run_report(plan: dict[str, Any], scan: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    move_candidates = plan["mappings"]
    total_size = sum(int(item["size_bytes"]) for item in move_candidates)
    return {
        "move_candidates": move_candidates,
        "ignored_candidates": scan["categories"]["IGNORE_ONLY"],
        "large_files": scan["large_files"],
        "risk_items": scan["categories"]["REVIEW_BEFORE_MOVE"],
        "total_size_to_move": fmt_size(total_size),
        "total_size_to_move_bytes": total_size,
        "runtime_root": str(runtime_root),
        "requires_confirmation": True,
    }


def write_scan_markdown(path: Path, scan: dict[str, Any]) -> None:
    stats = scan["stats"]
    lines = [
        "# Project Tree Scan Report",
        "",
        f"- tool_root: `{scan['tool_root']}`",
        f"- runtime_root: `{scan['runtime_root']}`",
        f"- total_files: {stats['total']['file_count']}",
        f"- total_size: {stats['total']['size_human']}",
        f"- src: {stats['src']['file_count']} files / {stats['src']['size_human']}",
        f"- tests: {stats['tests']['file_count']} files / {stats['tests']['size_human']}",
        f"- runtime: {stats['runtime']['file_count']} files / {stats['runtime']['size_human']}",
        f"- release: {stats['release']['file_count']} files / {stats['release']['size_human']}",
        f"- dev_snapshot: {stats['dev_snapshot']['file_count']} files / {stats['dev_snapshot']['size_human']}",
        f"- pycache/pyc: {stats['__pycache__']['file_count']} files / {stats['__pycache__']['size_human']}",
        f"- draft artifacts: {stats['draft_artifacts']['file_count']} files / {stats['draft_artifacts']['size_human']}",
        "",
        "## Categories",
    ]
    for category, rows in scan["categories"].items():
        lines.extend(["", f"### {category}", ""])
        if not rows:
            lines.append("- none")
            continue
        for row in rows:
            target = f" -> `{row['target_path']}`" if row.get("target_path") else ""
            lines.append(f"- `{row['path']}` | {row['size_human']} | {row['reason']}{target}")
    lines.extend(["", "## Files > 1MB", ""])
    for row in scan["large_files"][:200]:
        lines.append(f"- `{row['path']}` | {row['size_human']}")
    lines.extend(["", "## Suspected Draft / Decrypted Artifacts", ""])
    for row in scan["suspected_draft_artifacts"][:200]:
        lines.append(f"- `{row['path']}` | {row['size_human']}")
    path.write_text("\n".join(lines) + "\n", "utf-8")


def write_plan_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines = ["# Runtime Migration Plan", "", f"- runtime_root: `{plan['runtime_root']}`", "- mode: dry-run mapping only", ""]
    lines.append("## Target Layout")
    for key, value in plan["directory_layout"].items():
        lines.append(f"- `{key}` -> `{value}`")
    lines.extend(["", "## Project Path Mapping", ""])
    for row in plan["mappings"]:
        lines.append(f"- `{row['source']}` -> `{row['target']}` | {row['size_human']} | {row['file_count']} files")
    path.write_text("\n".join(lines) + "\n", "utf-8")


def write_dry_run_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Migration Dry Run Report",
        "",
        f"- runtime_root: `{report['runtime_root']}`",
        f"- move_candidate_count: {len(report['move_candidates'])}",
        f"- total_size_to_move: {report['total_size_to_move']}",
        f"- requires_confirmation: {str(report['requires_confirmation']).lower()}",
        "",
        "## Move Candidates",
    ]
    for row in report["move_candidates"]:
        lines.append(f"- `{row['source']}` -> `{row['target']}` | {row['size_human']}")
    lines.extend(["", "## Risk Items"])
    for row in report["risk_items"]:
        lines.append(f"- `{row['path']}` | {row['size_human']} | {row['reason']}")
    path.write_text("\n".join(lines) + "\n", "utf-8")


def write_layering_report(path_md: Path, path_json: Path, scan: dict[str, Any], plan: dict[str, Any], dry: dict[str, Any]) -> dict[str, Any]:
    report = {
        "scan_summary": scan["stats"],
        "target_runtime_root": plan["runtime_root"],
        "new_files_expected": [
            "AGENTS.md",
            "docs/PROJECT_LAYOUT.md",
            "docs/RUNTIME_POLICY.md",
            "docs/CODEX_ROLES.md",
            "docs/IDEA_SETUP.md",
            "config/runtime_paths.example.yaml",
            ".env.example",
            ".gitignore",
            "tools/migrate_runtime.py",
            "src/aroll_runtime_paths.py",
        ],
        "gitignore_updates": [
            "runtime/",
            "release/",
            "dev_snapshot/",
            "__pycache__/",
            "*.s16le",
            "*.tmp",
            "draft_content*.json",
            "template-2.tmp",
            "*.mp4",
            "*.png",
            "*.jpg",
        ],
        "migration_mappings": plan["mappings"],
        "risk_items": dry["risk_items"],
        "rollback_strategy": [
            "No files are deleted by dry-run.",
            "Execute mode can copy first before move.",
            "Do not move real Jianying drafts.",
            "If moved output is wrong, copy back from external runtime path to original project path.",
        ],
        "idea_setup": (
            f"Open only {TOOL_ROOT.as_posix()}. Exclude runtime/release/dev_snapshot. "
            f"Do not add {Path(plan['runtime_root']).as_posix()} as content root."
        ),
        "roles": {
            "idea_codex": "precise code edits only; no runtime scan; no real draft writes",
            "desktop_codex": "migration, long tasks, UAT, package generation, reports; default dry-run",
        },
        "ready_for_execute_migration": len(dry["move_candidates"]) > 0,
        "blockers": [],
    }
    write_json(path_json, report)
    lines = [
        "# Project Layering Report",
        "",
        f"- target_runtime_root: `{report['target_runtime_root']}`",
        f"- ready_for_execute_migration: {str(report['ready_for_execute_migration']).lower()}",
        f"- move_candidate_count: {len(dry['move_candidates'])}",
        f"- total_size_to_move: {dry['total_size_to_move']}",
        "",
        "## Current Scan Summary",
        f"- total: {scan['stats']['total']['file_count']} files / {scan['stats']['total']['size_human']}",
        f"- runtime: {scan['stats']['runtime']['file_count']} files / {scan['stats']['runtime']['size_human']}",
        f"- src: {scan['stats']['src']['file_count']} files / {scan['stats']['src']['size_human']}",
        "",
        "## IDEA Rule",
        report["idea_setup"],
        "",
        "## Runtime Migration Dry-run",
    ]
    for row in plan["mappings"]:
        lines.append(f"- `{row['source']}` -> `{row['target']}` | {row['size_human']}")
    lines.extend(["", "## Risk Items"])
    for row in dry["risk_items"]:
        lines.append(f"- `{row['path']}` | {row['size_human']} | {row['reason']}")
    path_md.write_text("\n".join(lines) + "\n", "utf-8")
    return report


def execute_plan(plan: dict[str, Any], copy_first: bool, confirm_execute: bool) -> dict[str, Any]:
    if not confirm_execute:
        raise RuntimeError("EXECUTE_REQUIRES_EXPLICIT_CONFIRMATION")
    actions = []
    for row in plan["mappings"]:
        src = Path(row["source_absolute"])
        dst = Path(row["target"])
        if not src.exists():
            actions.append(row | {"status": "missing_source"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
        actions.append(
            row
            | {
                "status": "copied_source_preserved",
                "copy_first": bool(copy_first),
                "delete_source": False,
                "note": "Source is intentionally preserved; remove it only after manual verification.",
            }
        )
    return {"actions": actions}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run runtime migration planner for jianying-aroll-inspector.")
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--skip-drafts", action="store_true")
    parser.add_argument("--copy-first", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-execute", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_AUDIT_ROOT)
    args = parser.parse_args()

    runtime_root = args.runtime_root
    scan = scan_project(runtime_root, skip_drafts=args.skip_drafts)
    plan = migration_plan(scan, runtime_root)
    dry = dry_run_report(plan, scan, runtime_root)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "project_tree_scan_report.json", scan)
    write_scan_markdown(report_dir / "project_tree_scan_report.md", scan)
    write_json(report_dir / "runtime_migration_plan.json", plan)
    write_plan_markdown(report_dir / "runtime_migration_plan.md", plan)
    write_json(report_dir / "migration_dry_run_report.json", dry)
    write_dry_run_markdown(report_dir / "migration_dry_run_report.md", dry)
    write_layering_report(
        report_dir / "project_layering_report.md",
        report_dir / "project_layering_report.json",
        scan,
        plan,
        dry,
    )

    if args.execute:
        execution = execute_plan(plan, copy_first=args.copy_first, confirm_execute=args.confirm_execute)
        write_json(report_dir / "migration_execute_report.json", execution)
        print("status=executed")
    else:
        print("status=dry_run")
    print(f"runtime_root={runtime_root}")
    print(f"move_candidate_count={len(dry['move_candidates'])}")
    print(f"total_size_to_move={dry['total_size_to_move']}")
    print(f"project_tree_scan_report={report_dir / 'project_tree_scan_report.md'}")
    print(f"runtime_migration_plan={report_dir / 'runtime_migration_plan.md'}")
    print(f"migration_dry_run_report={report_dir / 'migration_dry_run_report.md'}")
    print(f"project_layering_report={report_dir / 'project_layering_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
