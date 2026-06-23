from __future__ import annotations

import os
from pathlib import Path


DEFAULT_RUNTIME_ROOT = Path(r"D:\auto_clip_runtime\image_aligner")


def get_runtime_root() -> Path:
    return Path(os.environ.get("IMAGE_ALIGNER_RUNTIME_DIR", DEFAULT_RUNTIME_ROOT))


def get_runs_dir() -> Path:
    return get_runtime_root() / "runs"


def get_logs_dir() -> Path:
    return get_runtime_root() / "logs"


def get_reports_dir() -> Path:
    return get_runtime_root() / "reports"


def ensure_runtime_dirs() -> None:
    for path in [get_runs_dir(), get_logs_dir(), get_reports_dir()]:
        path.mkdir(parents=True, exist_ok=True)
