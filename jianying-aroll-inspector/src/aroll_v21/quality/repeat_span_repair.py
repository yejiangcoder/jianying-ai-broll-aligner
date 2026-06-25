from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from aroll_text_normalize import normalize_text


SELF_REPAIR_MIN_COMMON_PREFIX_CHARS = 4
SELF_REPAIR_MIN_SIMILARITY = 0.58
SELF_REPAIR_AMBIGUOUS_SIMILARITY = 0.52

_SENTENCE_FINAL_PARTICLES = set("\u4e86\u5427\u5417\u5462\u554a\u5440\u54e6\u54c8\u5457\u5566\u561b")
_FRAGMENT_TAIL_PARTICLES = set("\u7684\u5f97\u5730\u4e4b\u5728\u4ece\u5bf9\u628a\u88ab\u5c06\u8ba9\u4f7f\u8ddf\u548c\u4e0e\u6216\u53ca\u4ee5\u4e3a\u4e8e\u5230")
_OPEN_FILLER_SUFFIXES = ("那个", "这个", "就是", "然后", "那么", "所以")
_ENUMERATION_BAD_SUFFIX_STARTS = (
    "的",
    "地",
    "得",
    "了",
    "着",
    "过",
    "是",
    "在",
    "从",
    "对",
    "把",
    "被",
    "让",
    "使",
    "需要",
    "可以",
    "应该",
    "就是",
    "然后",
    "那么",
    "所以",
    "但是",
)
_ENUMERATION_BAD_SUFFIX_ENDS = (
    "的",
    "地",
    "得",
    "在",
    "从",
    "对",
    "把",
    "被",
    "让",
    "使",
)


def recommended_drop_indices(row: dict[str, Any], cluster: dict[str, Any] | None = None) -> list[int]:
    cluster = cluster or {}
    drop_index = int(row.get("drop_index") or row.get("recommended_drop_index") or cluster.get("recommended_drop_index") or 0)
    return [drop_index] if drop_index > 0 else []


def contained_repeat_drop_side(left_text: str, right_text: str) -> str:
    left = normalize_text(left_text)
    right = normalize_text(right_text)
    if not left or not right or left == right:
        return ""
    if left in right:
        return "drop_left"
    if right in left:
        return "drop_right"
    return ""


def longest_suffix_prefix_overlap(left_tokens: list[str], right_tokens: list[str]) -> int:
    max_size = min(len(left_tokens), len(right_tokens))
    for size in range(max_size, 0, -1):
        if left_tokens[-size:] == right_tokens[:size]:
            return size
    no_overlap = 0
    return no_overlap


def self_repair_aborted_phrase_candidate(left_text: str, right_text: str) -> dict[str, Any] | None:
    """Detect a short abandoned start immediately followed by a completed restart."""

    no_candidate: dict[str, Any] | None = None
    left = normalize_text(left_text)
    right = normalize_text(right_text)
    if not left or not right or left == right:
        return no_candidate
    if len(left) < 4 or len(right) < 5 or len(left) > 16:
        return no_candidate
    if left in right or right in left:
        return no_candidate

    prefix_len = _common_prefix_len(left, right)
    if prefix_len < SELF_REPAIR_MIN_COMMON_PREFIX_CHARS:
        return no_candidate
    if parallel_enumeration_candidate(left_text, right_text) is not None:
        return no_candidate
    left_suffix = left[prefix_len:]
    right_suffix = right[prefix_len:]
    if not left_suffix or not right_suffix:
        return no_candidate
    if len(left_suffix) > 8 or len(right_suffix) > 10:
        return no_candidate
    if left[-1] in _SENTENCE_FINAL_PARTICLES:
        return no_candidate

    ratio = SequenceMatcher(None, left, right).ratio()
    shared_suffix_chars = set(left_suffix) & set(right_suffix)
    has_restart_tail_evidence = bool(shared_suffix_chars)
    if left[-1] in _FRAGMENT_TAIL_PARTICLES:
        has_restart_tail_evidence = True
    if left_suffix in _OPEN_FILLER_SUFFIXES:
        has_restart_tail_evidence = True

    deterministic = ratio >= SELF_REPAIR_MIN_SIMILARITY and has_restart_tail_evidence
    ambiguous = ratio >= SELF_REPAIR_AMBIGUOUS_SIMILARITY
    if not deterministic and not ambiguous:
        return no_candidate

    return {
        "reason": "self_repair_aborted_phrase",
        "left_text": left_text,
        "right_text": right_text,
        "left_normalized_text": left,
        "right_normalized_text": right,
        "common_prefix": left[:prefix_len],
        "common_prefix_chars": prefix_len,
        "left_restart_suffix": left_suffix,
        "right_restart_suffix": right_suffix,
        "similarity": round(ratio, 6),
        "shared_restart_suffix_chars": sorted(shared_suffix_chars),
        "deterministic_drop_left": deterministic,
        "requires_semantic_adjudication": not deterministic,
        "suggested_decision": "drop_left_keep_right" if deterministic else "semantic_adjudication_required",
    }


def parallel_enumeration_candidate(left_text: str, right_text: str) -> dict[str, Any] | None:
    """Detect adjacent parallel list items that reuse a predicate but keep different objects."""

    no_candidate: dict[str, Any] | None = None
    left = normalize_text(left_text)
    right = normalize_text(right_text)
    if not left or not right or left == right:
        return no_candidate
    if left in right or right in left:
        return no_candidate
    prefix_len = _common_prefix_len(left, right)
    if _cjk_char_count(left[:prefix_len]) < SELF_REPAIR_MIN_COMMON_PREFIX_CHARS:
        return no_candidate
    if not left[:prefix_len].endswith("的"):
        return no_candidate
    left_suffix = left[prefix_len:]
    right_suffix = right[prefix_len:]
    if not _looks_like_parallel_item_suffix(left_suffix) or not _looks_like_parallel_item_suffix(right_suffix):
        return no_candidate
    left_chars = _cjk_chars(left_suffix)
    right_chars = _cjk_chars(right_suffix)
    if not left_chars or not right_chars:
        return no_candidate
    overlap = set(left_chars) & set(right_chars)
    overlap_ratio = len(overlap) / max(1, min(len(set(left_chars)), len(set(right_chars))))
    if overlap_ratio >= 0.5:
        return no_candidate
    return {
        "reason": "parallel_enumeration",
        "left_text": left_text,
        "right_text": right_text,
        "left_normalized_text": left,
        "right_normalized_text": right,
        "common_prefix": left[:prefix_len],
        "common_prefix_chars": prefix_len,
        "left_item_text": left_suffix,
        "right_item_text": right_suffix,
    }


def _looks_like_parallel_item_suffix(text: str) -> bool:
    if not text:
        return False
    cjk_count = _cjk_char_count(text)
    if cjk_count < 2 or cjk_count > 10:
        return False
    if any(text.startswith(prefix) for prefix in _ENUMERATION_BAD_SUFFIX_STARTS):
        return False
    if any(text.endswith(suffix) for suffix in _ENUMERATION_BAD_SUFFIX_ENDS):
        return False
    if text in _OPEN_FILLER_SUFFIXES:
        return False
    return True


def _cjk_chars(text: str) -> list[str]:
    return [char for char in text if "\u4e00" <= char <= "\u9fff"]


def _cjk_char_count(text: str) -> int:
    return len(_cjk_chars(text))


def _common_prefix_len(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def dropped_span_report(decision_trace: list[dict[str, Any]]) -> dict[str, Any]:
    dropped_cluster_ids: set[str] = set()
    dropped_segment_indices: set[int] = set()
    clusters_per_segment: dict[int, set[str]] = {}
    for row in decision_trace:
        if not isinstance(row, dict) or row.get("route") != "final_target_repeat" or not row.get("applied"):
            continue
        cluster_id = str(row.get("cluster_id") or "")
        if cluster_id:
            dropped_cluster_ids.add(cluster_id)
        for value in list(row.get("dropped_segment_indices") or []) + list(row.get("dropped_indices") or []):
            index = int(value or 0)
            if index <= 0:
                continue
            dropped_segment_indices.add(index)
            if cluster_id:
                clusters_per_segment.setdefault(index, set()).add(cluster_id)
        if not row.get("dropped_segment_indices") and int(row.get("drop_index") or 0) > 0:
            index = int(row.get("drop_index") or 0)
            dropped_segment_indices.add(index)
            if cluster_id:
                clusters_per_segment.setdefault(index, set()).add(cluster_id)
    return {
        "dropped_cluster_ids": sorted(dropped_cluster_ids),
        "dropped_segment_indices": sorted(dropped_segment_indices),
        "dropped_cluster_count": len(dropped_cluster_ids),
        "dropped_segment_count": len(dropped_segment_indices),
        "clusters_per_dropped_segment": {
            str(index): sorted(cluster_ids)
            for index, cluster_ids in sorted(clusters_per_segment.items())
        },
    }
