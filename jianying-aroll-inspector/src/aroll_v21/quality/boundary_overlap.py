from __future__ import annotations

from aroll_text_normalize import normalize_text


LABEL_INTRODUCERS = ("叫做", "称为", "称作", "叫", "算作", "属于", "是")
DEFINITIONAL_CONNECTORS = ("等于", "就是", "指的是", "意味着", "是", "叫做", "叫")
ATTRIBUTIVE_LABEL_CONTEXT_MARKERS = ("般的", "式的", "型的", "感的", "的")
EXPLANATORY_PREFIXES = ("什么叫", "什么是", "何为", "所谓")


def boundary_suffix_prefix_overlap(left_text: str, right_text: str, *, max_size: int | None = None) -> str:
    left = normalize_text(str(left_text or ""))
    right = normalize_text(str(right_text or ""))
    limit = min(len(left), len(right), max_size if max_size is not None else min(len(left), len(right)))
    for size in range(limit, 1, -1):
        candidate = left[-size:]
        if right.startswith(candidate):
            return candidate
    return ""


def is_semantic_label_reuse_boundary(left_text: str, right_text: str, overlap_text: str) -> bool:
    overlap = normalize_text(str(overlap_text or ""))
    if len(overlap) < 2:
        return False
    left = normalize_text(str(left_text or ""))
    right = normalize_text(str(right_text or ""))
    if not left.endswith(overlap) or not right.startswith(overlap):
        return False
    left_prefix = left[: -len(overlap)]
    right_suffix = right[len(overlap) :]
    if not any(right_suffix.startswith(marker) for marker in DEFINITIONAL_CONNECTORS):
        return False
    if any(left_prefix.endswith(marker) for marker in LABEL_INTRODUCERS):
        return True
    return _has_attributive_label_context(left_prefix)


def _has_attributive_label_context(left_prefix: str) -> bool:
    if len(left_prefix) < 3:
        return False
    return any(left_prefix.endswith(marker) for marker in ATTRIBUTIVE_LABEL_CONTEXT_MARKERS)


def is_explanatory_term_reuse(shorter_text: str, longer_text: str) -> bool:
    shorter = normalize_text(str(shorter_text or ""))
    longer = normalize_text(str(longer_text or ""))
    if len(shorter) < 2 or shorter not in longer:
        return False
    return any(longer.startswith(f"{prefix}{shorter}") for prefix in EXPLANATORY_PREFIXES)
