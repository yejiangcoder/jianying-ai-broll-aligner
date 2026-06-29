from __future__ import annotations

from aroll_text_normalize import normalize_text


LABEL_INTRODUCERS = ("叫做", "称为", "称作", "叫", "算作", "属于", "是")
DEFINITIONAL_CONNECTORS = ("等于", "就是", "指的是", "意味着", "是", "叫做", "叫")
ATTRIBUTIVE_LABEL_CONTEXT_MARKERS = ("般的", "式的", "型的", "感的", "的")
EXPLANATORY_PREFIXES = ("什么叫", "什么是", "何为", "所谓")
LIGHT_FILLER_PREFIXES = ("哈", "啊", "嗯", "呃", "诶", "哎")
PROGRESSIVE_SEMANTIC_MARKERS = ("更", "也", "还", "再", "又", "并且", "而且", "同时", "不仅", "越")
PARALLEL_ENUMERATION_BLOCKED_SHARED_PREFIXES = ("但其实", "其实", "就是", "然后", "但是", "所以", "因为", "但", "如果", "假如", "要是")
OPEN_PREDICATE_BRIDGES = (
    "是为了",
    "为了",
    "必须具备",
    "必须是",
    "不能是",
    "需要",
)
ENUMERATION_CONTEXT_MARKERS = (
    "以下几个",
    "以下几种",
    "分别是",
    "包括",
    "例如",
    "比如",
    "第一",
    "第二",
    "第三",
    "最后一步",
)


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


def is_parallel_progressive_semantic_expansion(left_text: str, right_text: str, shared_text: str) -> bool:
    shared = normalize_text(str(shared_text or ""))
    if len(shared) < 3 or _cjk_char_count(shared) < 2:
        return False
    left = _strip_light_filler_prefix(normalize_text(str(left_text or "")))
    right = _strip_light_filler_prefix(normalize_text(str(right_text or "")))
    if not left or not right or left == right:
        return False
    left_index = left.find(shared)
    right_index = right.find(shared)
    if left_index < 0 or right_index < 0:
        return False
    if max(left_index, right_index) > 2:
        return False
    left_tail = left[left_index + len(shared) :]
    right_tail = right[right_index + len(shared) :]
    if not left_tail or not right_tail or left_tail == right_tail:
        return False
    if _semantic_tail_len(left_tail) < 2 or _semantic_tail_len(right_tail) < 2:
        return False
    if left.endswith(shared) and right.startswith(shared):
        return False
    if any(marker in shared for marker in PROGRESSIVE_SEMANTIC_MARKERS):
        return True
    return _looks_like_parallel_enumeration(shared, left_tail, right_tail)


def is_open_predicate_bridge(text: str) -> bool:
    normalized = normalize_text(str(text or ""))
    if not normalized:
        return False
    if normalized in OPEN_PREDICATE_BRIDGES:
        return True
    return any(normalized.endswith(marker) for marker in OPEN_PREDICATE_BRIDGES)


def is_enumeration_slot_continuation(
    text: str,
    next_text: str,
    shared_text: str,
    *,
    previous_text: str = "",
) -> bool:
    current = normalize_text(str(text or ""))
    following = normalize_text(str(next_text or ""))
    shared = normalize_text(str(shared_text or ""))
    previous = normalize_text(str(previous_text or ""))
    if not current or not following:
        return False
    if previous and any(marker in previous for marker in ENUMERATION_CONTEXT_MARKERS):
        return True
    if len(shared) >= 2 and f"那么{shared}" in following and not following.startswith(shared):
        return True
    return False


def _strip_light_filler_prefix(text: str) -> str:
    for prefix in sorted(LIGHT_FILLER_PREFIXES, key=len, reverse=True):
        if text.startswith(prefix) and len(text) > len(prefix) + 1:
            return text[len(prefix) :]
    return text


def _semantic_tail_len(text: str) -> int:
    return sum(1 for char in text if "\u3400" <= char <= "\u9fff" or char.isalnum())


def _looks_like_parallel_enumeration(shared: str, left_tail: str, right_tail: str) -> bool:
    if any(shared.startswith(prefix) or prefix.startswith(shared) for prefix in PARALLEL_ENUMERATION_BLOCKED_SHARED_PREFIXES):
        return False
    if left_tail.startswith(right_tail) or right_tail.startswith(left_tail):
        return False
    if left_tail.endswith(("的", "地", "得", "了", "着", "过")) or right_tail.endswith(("的", "地", "得", "了", "着", "过")):
        return False
    return True


def _cjk_char_count(text: str) -> int:
    return sum(1 for char in str(text or "") if "\u3400" <= char <= "\u9fff")
