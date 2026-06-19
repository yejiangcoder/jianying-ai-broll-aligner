from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any


TOOL_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = TOOL_ROOT / "config"
LOCAL_CONFIG = CONFIG_DIR / "runtime_paths.local.yaml"
EXAMPLE_CONFIG = CONFIG_DIR / "runtime_paths.example.yaml"
DEFAULT_EXTERNAL_RUNTIME_ROOT = Path.home() / ".auto_clip_runtime"


def _clean_value(value: str) -> str:
    value = value.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text("utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _clean_value(value)
    return root


def _nested(config: dict[str, Any], *keys: str) -> str:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return str(current or "").strip()


def _config() -> dict[str, Any]:
    if LOCAL_CONFIG.exists():
        return _load_simple_yaml(LOCAL_CONFIG)
    if EXAMPLE_CONFIG.exists():
        return _load_simple_yaml(EXAMPLE_CONFIG)
    return {}


def _path_from_env_or_config(env_key: str, config_keys: tuple[str, ...], fallback: Path) -> Path:
    env_value = os.environ.get(env_key, "").strip()
    if env_value:
        return Path(env_value)
    config = _config()
    config_value = _nested(config, *config_keys)
    if config_value:
        path = Path(config_value)
        return path if path.is_absolute() else TOOL_ROOT / path
    warnings.warn(
        f"{env_key} and config path {'.'.join(config_keys)} are not configured; falling back to {fallback}",
        RuntimeWarning,
        stacklevel=2,
    )
    return fallback


def get_runtime_root() -> Path:
    return _path_from_env_or_config("AUTO_CLIP_RUNTIME_DIR", ("runtime_root",), DEFAULT_EXTERNAL_RUNTIME_ROOT)


def get_aroll_runs_dir() -> Path:
    return _path_from_env_or_config("AUTO_CLIP_AROLL_RUNS_DIR", ("aroll", "runs_dir"), get_runtime_root() / "aroll_v21_uat_runs")


def get_aroll_audits_dir() -> Path:
    return _path_from_env_or_config("AUTO_CLIP_AROLL_AUDITS_DIR", ("aroll", "audits_dir"), get_runtime_root() / "aroll_v21_audits")


def get_aroll_test_outputs_dir() -> Path:
    return _path_from_env_or_config("AUTO_CLIP_AROLL_TEST_OUTPUTS_DIR", ("aroll", "test_outputs_dir"), get_runtime_root() / "aroll_v21_test_outputs")


def get_aligner_root() -> Path:
    return _path_from_env_or_config("JY_ALIGNER_ROOT", ("jianying", "aligner_root"), TOOL_ROOT.parent / "jianying-ai-image-aligner")


def get_deepseek_config_path() -> Path:
    return _path_from_env_or_config("DEEPSEEK_CONFIG_PATH", ("deepseek", "config_path"), CONFIG_DIR / "deepseek.yaml")


def get_release_dir() -> Path:
    return _path_from_env_or_config("AUTO_CLIP_RELEASE_DIR", ("packages", "release_dir"), get_runtime_root() / "packages" / "release")


def get_dev_snapshot_dir() -> Path:
    return _path_from_env_or_config("AUTO_CLIP_DEV_SNAPSHOT_DIR", ("packages", "dev_snapshot_dir"), get_runtime_root() / "packages" / "dev_snapshot")


def get_logs_dir() -> Path:
    return _path_from_env_or_config("AUTO_CLIP_LOGS_DIR", ("logs_dir",), get_runtime_root() / "logs")


def get_temp_dir() -> Path:
    return _path_from_env_or_config("AUTO_CLIP_TEMP_DIR", ("temp_dir",), get_runtime_root() / "temp")
