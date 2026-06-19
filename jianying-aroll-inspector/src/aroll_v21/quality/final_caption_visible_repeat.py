from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from aroll_adjacent_modifier_semantic_redundancy_gate import detect_adjacent_modifier_semantic_redundancy
from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CaptionRenderUnit
from aroll_v21.quality.repeat_span_repair import longest_suffix_prefix_overlap, self_repair_aborted_phrase_candidate


NGRAM_SIZE = 4
PREFIX_SUFFIX_MIN_OVERLAP = 3
NEAR_DUPLICATE_RATIO = 0.9
DANGLING_ASPECT_PREFIXES = ("了", "着", "过")
DANGLING_DE_EXCEPTIONS = ("的确", "的话")
DANGLING_ASPECT_EXCEPTIONS = ("了解", "了不起", "过去", "过程", "过来", "过渡", "着陆")
NEGATIVE_RESTART_PREFIX = "不"
NEGATIVE_PREDICATE_MODAL_PREFIXES = ("可", "能", "会", "敢", "受", "被")
PARTIAL_RESTART_MIN_CHARS = 2
PARTIAL_RESTART_MAX_DROP_CHARS = 6
PARTIAL_RESTART_MAX_COMPLETED_CHARS = 10
PARTIAL_RESTART_LEFT_CONTEXT_CHARS = 5
FINAL_VISIBLE_RECHECK_DECISIONS = [
    "drop_bad_fragment",
    "trim_repeated_prefix",
    "keep_if_coherent",
    "requires_human_review",
]


def build_final_caption_visible_repeat_gate(captions: list[CaptionRenderUnit]) -> dict[str, Any]:
    ordered = sorted(captions, key=lambda row: (int(row.target_start_us), int(row.target_end_us), str(row.caption_id)))
    containment_candidates = _containment_candidates(ordered)
    containment_pairs = _candidate_pairs(containment_candidates)
    prefix_suffix_candidates = _prefix_suffix_candidates(ordered, containment_pairs)
    excluded_pairs = _candidate_pairs([*containment_candidates, *prefix_suffix_candidates])
    ngram_candidates = _ngram_candidates(ordered, excluded_pairs)
    near_duplicate_candidates = _near_duplicate_candidates(ordered, containment_candidates, prefix_suffix_candidates)
    modifier_redundancy_candidates = _modifier_redundancy_candidates(ordered)
    self_repair_candidates = _self_repair_aborted_phrase_candidates(ordered)
    dangling_candidates = _dangling_prefix_suffix_candidates(ordered)
    semantic_suspect_candidates = _semantic_garbage_or_asr_suspect_candidates(ordered)
    cross_caption_containment_candidates = _cross_caption_semantic_containment_candidates(ordered)
    restart_repeat_candidates = [
        *_restart_repeat_visible_candidates(ordered),
        *_negative_predicate_restart_candidates(ordered),
        *_partial_phrase_restart_candidates(ordered),
    ]
    visible_repeat_candidates = [
        *containment_candidates,
        *prefix_suffix_candidates,
        *ngram_candidates,
        *near_duplicate_candidates,
        *cross_caption_containment_candidates,
        *restart_repeat_candidates,
    ]
    final_visible_quality_candidates = [
        *visible_repeat_candidates,
        *dangling_candidates,
        *semantic_suspect_candidates,
    ]
    blocker_codes: list[str] = []
    if final_visible_quality_candidates:
        blocker_codes.append("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED")
    if dangling_candidates:
        blocker_codes.append("V21_FINAL_VISIBLE_DANGLING_PREFIX_SUFFIX")
    if semantic_suspect_candidates:
        blocker_codes.append("V21_FINAL_VISIBLE_SEMANTIC_GARBAGE_OR_ASR_SUSPECT")
    if cross_caption_containment_candidates:
        blocker_codes.append("V21_FINAL_VISIBLE_CROSS_CAPTION_SEMANTIC_CONTAINMENT")
    if restart_repeat_candidates:
        blocker_codes.append("V21_FINAL_VISIBLE_RESTART_REPEAT")
    if modifier_redundancy_candidates:
        blocker_codes.append("V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED")
    if self_repair_candidates:
        blocker_codes.append("V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED")
    return {
        "gate_passed": not blocker_codes,
        "blocker_codes": blocker_codes,
        "visible_repeat_candidate_count": len(visible_repeat_candidates),
        "containment_repeat_count": len(containment_candidates),
        "prefix_suffix_overlap_count": len(prefix_suffix_candidates),
        "ngram_repeat_count": len(ngram_candidates),
        "near_duplicate_visible_caption_count": len(near_duplicate_candidates),
        "modifier_redundancy_residual_count": len(modifier_redundancy_candidates),
        "self_repair_aborted_phrase_count": len(self_repair_candidates),
        "dangling_prefix_suffix_count": len(dangling_candidates),
        "semantic_garbage_or_asr_suspect_count": len(semantic_suspect_candidates),
        "cross_caption_semantic_containment_count": len(cross_caption_containment_candidates),
        "restart_repeat_visible_count": len(restart_repeat_candidates),
        "visible_repeat_candidates": visible_repeat_candidates,
        "containment_repeat_candidates": containment_candidates,
        "prefix_suffix_overlap_candidates": prefix_suffix_candidates,
        "ngram_repeat_candidates": ngram_candidates,
        "near_duplicate_visible_caption_candidates": near_duplicate_candidates,
        "modifier_redundancy_residual_candidates": modifier_redundancy_candidates,
        "self_repair_aborted_phrase_candidates": self_repair_candidates,
        "dangling_prefix_suffix_candidates": dangling_candidates,
        "semantic_garbage_or_asr_suspect_candidates": semantic_suspect_candidates,
        "cross_caption_semantic_containment_candidates": cross_caption_containment_candidates,
        "restart_repeat_visible_candidates": restart_repeat_candidates,
        "final_caption_visible_repeat_gate_enabled": True,
        "ngram_size": NGRAM_SIZE,
        "prefix_suffix_min_overlap": PREFIX_SUFFIX_MIN_OVERLAP,
        "near_duplicate_ratio": NEAR_DUPLICATE_RATIO,
        "final_visible_recheck_allowed_decisions": FINAL_VISIBLE_RECHECK_DECISIONS,
    }


def _containment_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(captions):
        left_text = normalize_text(left.text)
        if not left_text:
            continue
        for right_index in range(left_index + 1, len(captions)):
            right = captions[right_index]
            right_text = normalize_text(right.text)
            if not right_text:
                continue
            if left_text == right_text or (len(left_text) >= 2 and left_text in right_text) or (len(right_text) >= 2 and right_text in left_text):
                candidates.append(
                    _candidate(
                        "containment_repeat",
                        left,
                        right,
                        overlap_text=left_text if len(left_text) <= len(right_text) else right_text,
                        score=1.0,
                    )
                )
    return candidates


def _prefix_suffix_candidates(captions: list[CaptionRenderUnit], excluded_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left, right in zip(captions, captions[1:]):
        if (left.caption_id, right.caption_id) in excluded_pairs:
            continue
        left_chars = list(normalize_text(left.text))
        right_chars = list(normalize_text(right.text))
        overlap = longest_suffix_prefix_overlap(left_chars, right_chars)
        if overlap >= PREFIX_SUFFIX_MIN_OVERLAP:
            candidates.append(
                _candidate(
                    "prefix_suffix_overlap",
                    left,
                    right,
                    overlap_text="".join(left_chars[-overlap:]),
                    score=overlap,
                )
            )
    return candidates


def _ngram_candidates(captions: list[CaptionRenderUnit], excluded_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    seen: dict[str, tuple[CaptionRenderUnit, int]] = {}
    candidates: list[dict[str, Any]] = []
    emitted: set[tuple[str, str, str]] = set()
    for caption_index, caption in enumerate(captions):
        text = normalize_text(caption.text)
        for ngram in _ngrams(text, NGRAM_SIZE):
            previous_row = seen.get(ngram)
            if previous_row is None:
                seen[ngram] = (caption, caption_index)
                continue
            previous, previous_index = previous_row
            seen[ngram] = (caption, caption_index)
            if (previous.caption_id, caption.caption_id) in excluded_pairs:
                continue
            if not _ngram_repeat_is_blocking(previous, caption, ngram, previous_index, caption_index):
                continue
            key = (previous.caption_id, caption.caption_id, ngram)
            if key in emitted:
                continue
            emitted.add(key)
            candidates.append(
                _candidate(
                    "ngram_repeat",
                    previous,
                    caption,
                    overlap_text=ngram,
                    score=NGRAM_SIZE,
                )
            )
    return candidates


def _ngram_repeat_is_blocking(
    left: CaptionRenderUnit,
    right: CaptionRenderUnit,
    ngram: str,
    left_index: int,
    right_index: int,
) -> bool:
    left_text = normalize_text(left.text)
    right_text = normalize_text(right.text)
    shorter = min(len(left_text), len(right_text))
    if not shorter:
        return False
    coverage = len(ngram) / shorter
    if right_index - left_index == 1:
        return coverage >= 0.75 or _ngram_touches_caption_boundary(left_text, ngram) or _ngram_touches_caption_boundary(right_text, ngram)
    return coverage >= 0.75


def _ngram_touches_caption_boundary(text: str, ngram: str) -> bool:
    if not text or not ngram:
        return False
    return text.startswith(ngram) or text.endswith(ngram)


def _near_duplicate_candidates(
    captions: list[CaptionRenderUnit],
    containment_candidates: list[dict[str, Any]],
    prefix_suffix_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    excluded_pairs = _candidate_pairs([*containment_candidates, *prefix_suffix_candidates])
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(captions):
        left_text = normalize_text(left.text)
        if len(left_text) < NGRAM_SIZE:
            continue
        for right_index in range(left_index + 1, len(captions)):
            right = captions[right_index]
            right_text = normalize_text(right.text)
            if len(right_text) < NGRAM_SIZE:
                continue
            pair = (left.caption_id, right.caption_id)
            if pair in excluded_pairs:
                continue
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= NEAR_DUPLICATE_RATIO:
                candidates.append(
                    _candidate(
                        "near_duplicate_visible_caption",
                        left,
                        right,
                        overlap_text="",
                        score=round(ratio, 6),
                    )
                )
    return candidates


def _modifier_redundancy_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    rows = [
        {
            "fragment_id": caption.caption_id,
            "fragment_text": caption.text,
            "text": caption.text,
        }
        for caption in captions
    ]
    candidates: list[dict[str, Any]] = []
    for row in detect_adjacent_modifier_semantic_redundancy(rows):
        severity = str(row.get("severity") or "fatal")
        if severity not in {"fatal", "high"}:
            continue
        row_index = int(row.get("row_index") or 0)
        if not (1 <= row_index <= len(captions)):
            continue
        caption = captions[row_index - 1]
        related_caption = captions[int(row.get("next_row_index") or row_index) - 1] if int(row.get("next_row_index") or 0) in range(1, len(captions) + 1) else caption
        candidate = _candidate(
            "fatal_modifier_redundancy_residual",
            caption,
            related_caption,
            overlap_text=str(row.get("phrase") or ""),
            score=1.0,
        )
        candidate.update(
            {
                "type": str(row.get("type") or "adjacent_modifier_semantic_redundancy"),
                "severity": severity,
                "scope": str(row.get("scope") or ""),
                "phrase": str(row.get("phrase") or ""),
                "modifiers": [
                    str(row.get("left_modifier") or ""),
                    str(row.get("right_modifier") or ""),
                ],
                "head": str(row.get("head_text") or ""),
                "requires_semantic_adjudication": True,
                "suggested_decision": "drop_redundant_modifier",
            }
        )
        candidates.append(candidate)
    return candidates


def _self_repair_aborted_phrase_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left, right in zip(captions, captions[1:]):
        row = self_repair_aborted_phrase_candidate(left.text, right.text)
        if row is None:
            continue
        candidate = _candidate(
            "self_repair_aborted_phrase_residual",
            left,
            right,
            overlap_text=str(row.get("common_prefix") or ""),
            score=float(row.get("similarity") or 0.0),
        )
        candidate.update(row)
        candidates.append(candidate)
    return candidates


def _dangling_prefix_suffix_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for caption in captions:
        text = normalize_text(caption.text)
        reason = _dangling_prefix_suffix_reason(text)
        if not reason:
            continue
        candidate = _candidate(reason, caption, caption, overlap_text=text[: min(len(text), 4)], score=1.0)
        candidate.update(
            {
                "type": "dangling_prefix_or_suffix",
                "severity": "fatal",
                "caption_boundary_split_error": True,
            }
        )
        candidates.append(candidate)
    return candidates


def _dangling_prefix_suffix_reason(text: str) -> str:
    if not text:
        return ""
    if text.startswith("的是"):
        return "dangling_de_shi_prefix"
    if text.startswith("的") and not text.startswith(DANGLING_DE_EXCEPTIONS):
        return "dangling_de_prefix"
    if text.startswith(DANGLING_ASPECT_PREFIXES) and not text.startswith(DANGLING_ASPECT_EXCEPTIONS):
        return "dangling_aspect_suffix_caption"
    return ""


def _semantic_garbage_or_asr_suspect_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for caption in captions:
        text = normalize_text(caption.text)
        row = _asr_restart_suspect(text)
        if not row:
            continue
        candidate = _candidate(
            "semantic_garbage_or_asr_suspect",
            caption,
            caption,
            overlap_text=str(row.get("overlap_text") or ""),
            score=float(row.get("score") or 1.0),
        )
        candidate.update(
            {
                "type": "visible_asr_restart_residual",
                "severity": "high",
                "semantic_quality_recheck_required": True,
                "allowed_recheck_decisions": list(FINAL_VISIBLE_RECHECK_DECISIONS),
                **row,
            }
        )
        candidates.append(candidate)
    return candidates


def _asr_restart_suspect(text: str) -> dict[str, Any] | None:
    if len(text) < 4:
        empty_result: dict[str, Any] | None = None
        return empty_result
    for unit_len in range(1, min(3, len(text) // 2) + 1):
        unit = text[:unit_len]
        if not re.fullmatch(r"[\u4e00-\u9fff]+", unit):
            continue
        if text[unit_len:].startswith("就" + unit):
            return {
                "pattern": "repeated_prefix_around_jiu",
                "repeated_prefix": unit,
                "overlap_text": unit + "就" + unit,
                "score": round((unit_len * 2 + 1) / max(1, len(text)), 6),
            }
    empty_result: dict[str, Any] | None = None
    return empty_result


def _cross_caption_semantic_containment_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    emitted: set[tuple[str, tuple[str, ...]]] = set()
    for index, caption in enumerate(captions):
        left_text = normalize_text(caption.text)
        if len(left_text) < PREFIX_SUFFIX_MIN_OVERLAP + 1:
            continue
        for window_size in (2, 3):
            window = captions[index + 1 : index + 1 + window_size]
            if len(window) < 2:
                continue
            combined = normalize_text("".join(row.text for row in window))
            if not combined:
                continue
            overlap_text = left_text if left_text in combined else ""
            if not overlap_text:
                overlap = _longest_common_substring(left_text, combined)
                overlap_text = overlap if len(overlap) >= max(PREFIX_SUFFIX_MIN_OVERLAP + 1, min(8, len(left_text))) else ""
            if not overlap_text:
                continue
            key = (caption.caption_id, tuple(row.caption_id for row in window))
            if key in emitted:
                continue
            emitted.add(key)
            candidate = _candidate(
                "cross_caption_semantic_containment",
                caption,
                window[-1],
                overlap_text=overlap_text,
                score=round(len(overlap_text) / max(1, len(left_text)), 6),
            )
            candidate.update(
                {
                    "type": "cross_caption_semantic_containment",
                    "severity": "high",
                    "window_caption_ids": [row.caption_id for row in window],
                    "window_text": "".join(row.text for row in window),
                }
            )
            candidates.append(candidate)
    return candidates


def _restart_repeat_visible_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    emitted: set[tuple[str, tuple[str, ...], str]] = set()
    for index, caption in enumerate(captions):
        left_text = normalize_text(caption.text)
        internal_restart = _internal_restart_repeat(left_text)
        if internal_restart is not None:
            candidate = _candidate(
                "internal_restart_repeat_visible",
                caption,
                caption,
                overlap_text=str(internal_restart.get("overlap_text") or ""),
                score=float(internal_restart.get("score") or 1.0),
            )
            candidate.update(
                {
                    "type": "internal_restart_repeat_visible",
                    "severity": "high",
                    "window_caption_ids": [caption.caption_id],
                    "window_text": caption.text,
                    **internal_restart,
                }
            )
            candidates.append(candidate)
        if len(left_text) < PREFIX_SUFFIX_MIN_OVERLAP + 1:
            continue
        for window_size in (1, 2):
            window = captions[index + 1 : index + 1 + window_size]
            if not window:
                continue
            combined = normalize_text("".join(row.text for row in window))
            overlap_text = ""
            if left_text in combined and not combined.startswith(left_text):
                overlap_text = left_text
            else:
                overlap = _longest_common_substring(left_text, combined)
                if len(overlap) >= max(PREFIX_SUFFIX_MIN_OVERLAP + 1, min(8, len(left_text))):
                    overlap_text = overlap
            if not overlap_text:
                continue
            key = (caption.caption_id, tuple(row.caption_id for row in window), overlap_text)
            if key in emitted:
                continue
            emitted.add(key)
            candidate = _candidate(
                "restart_repeat_visible",
                caption,
                window[-1],
                overlap_text=overlap_text,
                score=round(len(overlap_text) / max(1, len(left_text)), 6),
            )
            candidate.update(
                {
                    "type": "restart_repeat_visible",
                    "severity": "high",
                    "window_caption_ids": [row.caption_id for row in window],
                    "window_text": "".join(row.text for row in window),
                }
            )
            candidates.append(candidate)
    return candidates


def _negative_predicate_restart_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    emitted: set[tuple[str, tuple[str, ...], str]] = set()
    for index, caption in enumerate(captions):
        windows = [(caption,)]
        if index + 1 < len(captions):
            windows.append((caption, captions[index + 1]))
        for window in windows:
            texts = [normalize_text(row.text) for row in window]
            combined = "".join(texts)
            if len(combined) < 5:
                continue
            boundary = len(texts[0])
            for row in _negative_predicate_restart_matches(combined):
                start = int(row.get("start") or 0)
                end = int(row.get("end") or 0)
                if len(window) > 1 and not (start < boundary < end):
                    continue
                key = (window[0].caption_id, tuple(item.caption_id for item in window), str(row.get("drop_text") or ""))
                if key in emitted:
                    continue
                emitted.add(key)
                related = window[-1]
                candidate = _candidate(
                    "negative_predicate_restart_visible",
                    window[0],
                    related,
                    overlap_text=str(row.get("overlap_text") or ""),
                    score=float(row.get("score") or 1.0),
                )
                candidate.update(
                    {
                        "type": "negative_predicate_restart_visible",
                        "severity": "high",
                        "window_caption_ids": [item.caption_id for item in window],
                        "window_text": "".join(item.text for item in window),
                        **row,
                    }
                )
                candidates.append(candidate)
    return candidates


def _partial_phrase_restart_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    emitted: set[tuple[str, str, str]] = set()
    for left, right in zip(captions, captions[1:]):
        left_text = normalize_text(left.text)
        right_text = normalize_text(right.text)
        if not left_text or len(right_text) < PARTIAL_RESTART_MIN_CHARS + 1:
            continue
        if left_text in right_text or right_text in left_text:
            continue
        combined = f"{left_text}{right_text}"
        boundary = len(left_text)
        search_start = max(0, boundary - PARTIAL_RESTART_LEFT_CONTEXT_CHARS)
        for row in _partial_phrase_restart_matches(combined, boundary, search_start):
            drop_text = str(row.get("drop_text") or "")
            key = (left.caption_id, right.caption_id, drop_text)
            if key in emitted:
                continue
            emitted.add(key)
            candidate = _candidate(
                "partial_phrase_restart_visible",
                left,
                right,
                overlap_text=str(row.get("overlap_text") or ""),
                score=float(row.get("score") or 1.0),
            )
            candidate.update(
                {
                    "type": "partial_phrase_restart_visible",
                    "severity": "high",
                    "window_caption_ids": [left.caption_id, right.caption_id],
                    "window_text": f"{left.text}{right.text}",
                    **row,
                }
            )
            candidates.append(candidate)
    return candidates


def _partial_phrase_restart_matches(text: str, boundary: int, search_start: int = 0) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if len(text) < PARTIAL_RESTART_MIN_CHARS * 2 + 1:
        return matches
    max_start = min(boundary, len(text) - PARTIAL_RESTART_MIN_CHARS * 2)
    for start in range(max(0, search_start), max_start + 1):
        for drop_len in range(PARTIAL_RESTART_MIN_CHARS, PARTIAL_RESTART_MAX_DROP_CHARS + 1):
            pivot = start + drop_len
            if not (start < boundary < pivot):
                continue
            if pivot >= len(text):
                continue
            drop_text = text[start:pivot]
            if not _plain_cjk(drop_text):
                continue
            max_completed_len = min(PARTIAL_RESTART_MAX_COMPLETED_CHARS, len(text) - pivot)
            for completed_len in range(drop_len + 1, max_completed_len + 1):
                completed_text = text[pivot : pivot + completed_len]
                if completed_text.startswith(drop_text) and _plain_cjk(completed_text):
                    matches.append(
                        {
                            "pattern": "partial_phrase_restart",
                            "start": start,
                            "pivot": pivot,
                            "end": pivot + completed_len,
                            "drop_text": drop_text,
                            "completed_text": completed_text,
                            "overlap_text": text[start : pivot + completed_len],
                            "score": round(len(drop_text) / max(1, completed_len), 6),
                        }
                    )
                    break
    return matches


def _negative_predicate_restart_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for start in range(0, max(0, len(text) - 4)):
        if text[start] != NEGATIVE_RESTART_PREFIX:
            continue
        second = text.find(NEGATIVE_RESTART_PREFIX, start + 2)
        if second < 0 or second - start > 5:
            continue
        left = text[start + 1 : second]
        if not left:
            continue
        right_limit = min(len(text), second + 1 + 6)
        for end in range(second + 2, right_limit + 1):
            right = text[second + 1 : end]
            if _negative_predicate_restart_relation(left, right):
                drop_text = text[start:second]
                overlap_text = text[start:end]
                matches.append(
                    {
                        "pattern": "negative_predicate_restart",
                        "start": start,
                        "end": end,
                        "left_predicate": left,
                        "right_predicate": right,
                        "drop_text": drop_text,
                        "overlap_text": overlap_text,
                        "score": round(len(drop_text) / max(1, len(overlap_text)), 6),
                    }
                )
                break
    return matches


def _negative_predicate_restart_relation(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    left_core = _strip_negative_predicate_modal(left)
    right_core = _strip_negative_predicate_modal(right)
    if not left_core or not right_core:
        return False
    if left_core == right_core:
        return True
    if len(left_core) == 1 and right_core.startswith(left_core):
        return True
    if len(right_core) == 1 and left_core.startswith(right_core):
        return True
    return left_core in right_core or right_core in left_core


def _strip_negative_predicate_modal(text: str) -> str:
    current = str(text or "")
    while current.startswith(NEGATIVE_PREDICATE_MODAL_PREFIXES) and len(current) > 1:
        current = current[1:]
    return current


def _internal_restart_repeat(text: str) -> dict[str, Any] | None:
    if len(text) < 5:
        no_restart: dict[str, Any] | None = None
        return no_restart
    for pivot in ("是", "有", "要", "会", "能", "敢"):
        first_pivot = text.find(pivot)
        if first_pivot <= 0:
            continue
        second_pivot = text.find(pivot, first_pivot + 1)
        if second_pivot <= first_pivot + 1 or second_pivot > 8:
            continue
        aborted_prefix = text[:first_pivot]
        restarted_prefix = text[first_pivot + len(pivot) : second_pivot]
        if not aborted_prefix or not restarted_prefix:
            continue
        if not restarted_prefix.startswith(aborted_prefix):
            continue
        if len(restarted_prefix) <= len(aborted_prefix):
            continue
        drop_text = text[: first_pivot + len(pivot)]
        return {
            "pattern": "internal_pivot_restart",
            "pivot": pivot,
            "aborted_prefix": aborted_prefix,
            "restarted_prefix": restarted_prefix,
            "drop_text": drop_text,
            "overlap_text": text[: second_pivot + len(pivot)],
            "score": round(len(drop_text) / max(1, len(text)), 6),
        }
    no_restart: dict[str, Any] | None = None
    return no_restart


def _longest_common_substring(left: str, right: str) -> str:
    if not left or not right:
        return ""
    best = ""
    for left_index in range(len(left)):
        for right_index in range(len(right)):
            offset = 0
            while left_index + offset < len(left) and right_index + offset < len(right) and left[left_index + offset] == right[right_index + offset]:
                offset += 1
            if offset > len(best):
                best = left[left_index : left_index + offset]
    return best


def _plain_cjk(text: str) -> bool:
    return bool(text) and bool(re.fullmatch(r"[\u4e00-\u9fff]+", text))


def _candidate_pairs(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row.get("caption_id") or ""), str(row.get("related_caption_id") or ""))
        for row in rows
    }


def _ngrams(text: str, size: int) -> list[str]:
    if len(text) < size:
        empty: list[str] = []
        return empty
    return [text[index : index + size] for index in range(0, len(text) - size + 1)]


def _candidate(
    reason: str,
    caption: CaptionRenderUnit,
    related: CaptionRenderUnit,
    *,
    overlap_text: str,
    score: float | int,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "caption_id": caption.caption_id,
        "related_caption_id": related.caption_id,
        "target_start_us": int(caption.target_start_us),
        "target_end_us": int(caption.target_end_us),
        "duration_us": int(caption.target_end_us) - int(caption.target_start_us),
        "text": caption.text,
        "related_target_start_us": int(related.target_start_us),
        "related_target_end_us": int(related.target_end_us),
        "related_text": related.text,
        "overlap_text": overlap_text,
        "score": score,
        "caption_word_ids": list(caption.word_ids),
        "related_word_ids": list(related.word_ids),
    }
