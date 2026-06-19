from __future__ import annotations

import importlib
import json
import os
import re
import sys
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType
from typing import Any

from aroll_runtime_paths import get_aligner_root, get_aroll_runs_dir

DEFAULT_ALIGNER_ROOT = get_aligner_root()


def _aligner_root() -> Path:
    return get_aligner_root()


ALIGNER_ROOT = _aligner_root()
ALIGNER_SRC = ALIGNER_ROOT / "src"
AI_TRACK_NAME = os.environ.get("JY_AI_TRACK_NAME", "AI_BROLL")
DEFAULT_JY_DRAFTC = Path(os.environ.get("JY_DRAFTC") or os.environ.get("JY_DRAFTC_EXE") or "JianyingPro")
DEFAULT_RUNTIME = get_aroll_runs_dir()

_BRIDGE_MODULE: ModuleType | None = None


def _refresh_aligner_paths() -> tuple[Path, Path]:
    root = _aligner_root()
    return root, root / "src"


def _load_bridge() -> ModuleType:
    global _BRIDGE_MODULE, ALIGNER_ROOT, ALIGNER_SRC
    if _BRIDGE_MODULE is not None:
        return _BRIDGE_MODULE

    ALIGNER_ROOT, ALIGNER_SRC = _refresh_aligner_paths()
    if not str(ALIGNER_ROOT).strip():
        raise RuntimeError("JY_ALIGNER_ROOT_NOT_CONFIGURED")
    if not ALIGNER_SRC.exists():
        raise RuntimeError(f"JY_ALIGNER_SRC_NOT_FOUND:{ALIGNER_SRC}")

    src_text = str(ALIGNER_SRC)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    try:
        _BRIDGE_MODULE = importlib.import_module("direct_draft_broll_writer")
    except ModuleNotFoundError as exc:
        if exc.name == "direct_draft_broll_writer":
            raise RuntimeError(f"JY_ALIGNER_SRC_NOT_FOUND:{ALIGNER_SRC}") from exc
        raise
    return _BRIDGE_MODULE


def _bridge_attr(name: str) -> Any:
    module = _load_bridge()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise RuntimeError(f"JY_BRIDGE_SYMBOL_NOT_FOUND:{name}") from exc


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def norm_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def dice_score(left: Any, right: Any) -> float:
    left_text = norm_text(left)
    right_text = norm_text(right)
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    if len(left_text) == 1 or len(right_text) == 1:
        return 1.0 if left_text == right_text else 0.0
    left_bigrams = {left_text[index : index + 2] for index in range(len(left_text) - 1)}
    right_bigrams = {right_text[index : index + 2] for index in range(len(right_text) - 1)}
    if not left_bigrams and not right_bigrams:
        return 1.0
    return 2 * len(left_bigrams & right_bigrams) / max(1, len(left_bigrams) + len(right_bigrams))


def text_score(left: Any, right: Any) -> float:
    return SequenceMatcher(None, norm_text(left), norm_text(right)).ratio()


def guid() -> str:
    return str(uuid.uuid4()).upper()


def decrypt(*args: Any, **kwargs: Any) -> Any:
    return _bridge_attr("decrypt")(*args, **kwargs)


def encrypt(*args: Any, **kwargs: Any) -> Any:
    return _bridge_attr("encrypt")(*args, **kwargs)


def resolve_timeline_id(*args: Any, **kwargs: Any) -> Any:
    return _bridge_attr("resolve_timeline_id")(*args, **kwargs)


def root_mirrors_timeline_id(*args: Any, **kwargs: Any) -> Any:
    from aroll_root_mirror import root_mirrors_timeline_id as detect_root_mirror

    return detect_root_mirror(*args, **kwargs)


def assert_all_project_timeline_files_match_folder_ids(*args: Any, **kwargs: Any) -> Any:
    return _bridge_attr("assert_all_project_timeline_files_match_folder_ids")(*args, **kwargs)


def assert_layout_has_no_duplicate_timeline_ids(*args: Any, **kwargs: Any) -> Any:
    return _bridge_attr("assert_layout_has_no_duplicate_timeline_ids")(*args, **kwargs)


def assert_timeline_content_id(*args: Any, **kwargs: Any) -> Any:
    return _bridge_attr("assert_timeline_content_id")(*args, **kwargs)


__all__ = [
    "ALIGNER_ROOT",
    "ALIGNER_SRC",
    "AI_TRACK_NAME",
    "DEFAULT_JY_DRAFTC",
    "DEFAULT_RUNTIME",
    "assert_all_project_timeline_files_match_folder_ids",
    "assert_layout_has_no_duplicate_timeline_ids",
    "assert_timeline_content_id",
    "decrypt",
    "dice_score",
    "encrypt",
    "guid",
    "norm_text",
    "read_json",
    "resolve_timeline_id",
    "root_mirrors_timeline_id",
    "text_score",
    "write_json",
]
