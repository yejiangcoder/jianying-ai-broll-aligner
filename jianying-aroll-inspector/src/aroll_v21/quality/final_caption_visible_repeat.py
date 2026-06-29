from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from aroll_adjacent_modifier_semantic_redundancy_gate import detect_adjacent_modifier_semantic_redundancy
from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CaptionRenderUnit
from aroll_v21.quality.final_caption_visible import (
    FinalCaptionVisibleDetectorSet,
    build_final_caption_visible_gate_report,
    build_final_caption_visible_policy,
    build_final_caption_visible_repair_signal,
    classify_final_caption_visible_evidence,
    detect_final_caption_visible_evidence,
)
from aroll_v21.quality.final_semantic_integrity import build_final_semantic_integrity_candidates
from aroll_v21.quality.boundary_overlap import (
    is_explanatory_term_reuse,
    is_parallel_progressive_semantic_expansion,
    is_semantic_label_reuse_boundary,
)
from aroll_v21.quality.repeat_span_repair import longest_suffix_prefix_overlap, self_repair_aborted_phrase_candidate


NGRAM_SIZE = 4
PREFIX_SUFFIX_MIN_OVERLAP = 3
NEAR_DUPLICATE_RATIO = 0.9
DANGLING_ASPECT_PREFIXES = ("了", "着", "过")
DANGLING_DE_EXCEPTIONS = ("的确", "的话")
DANGLING_ASPECT_EXCEPTIONS = ("了解", "了不起", "过去", "过程", "过来", "过渡", "着陆")
DANGLING_PRONOUN_MODAL_PRONOUNS = ("你", "他", "我", "咱", "人家", "自己")
DANGLING_PRONOUN_MODAL_TAILS = ("只", "就", "还", "都", "也", "才", "会", "能", "敢", "要", "把", "给", "在", "被", "让")
DANGLING_PRONOUN_OPEN_TAIL_ENDS = ("在", "把", "给", "被", "让", "向", "对", "跟", "和", "去")
DANGLING_WEAK_PRONOUN_FRAGMENT_PREFIXES = ("就", "但", "但你", "就是")
NEGATIVE_RESTART_PREFIX = "不"
NEGATIVE_PREDICATE_MODAL_PREFIXES = ("可", "能", "会", "敢", "受", "被")
PARTIAL_RESTART_MIN_CHARS = 2
PARTIAL_RESTART_MAX_DROP_CHARS = 6
PARTIAL_RESTART_MAX_COMPLETED_CHARS = 10
PARTIAL_RESTART_LEFT_CONTEXT_CHARS = 8
PARTIAL_RESTART_MIN_SHARED_PREFIX_CHARS = 4
PARTIAL_RESTART_MAX_TAIL_MISMATCH_CHARS = 2
PARTIAL_RESTART_MIN_TAIL_MISMATCH_COMPLETION_CHARS = 2
PARTIAL_RESTART_TAIL_MISMATCH_ALLOWED_TAILS = ("去", "来", "上", "下", "里", "中", "内", "外", "了", "的")
SHORT_FRAGMENT_RESTART_MIN_OVERLAP_CHARS = 3
SHORT_FRAGMENT_RESTART_MAX_LEFT_CHARS = 8
SHORT_FRAGMENT_RESTART_LOOKAHEAD = 3
SHORT_FRAGMENT_RESTART_MIN_COVERAGE = 0.6
WEAK_RESTART_PREFIXES = ("但其实", "其实", "就是", "然后", "但是", "所以", "因为", "就", "但")
FRAGMENT_TAILS = ("全是", "就是", "是", "的", "在", "把", "给", "去", "就")
INTERNAL_PREFIX_RESTART_MAX_PHRASE_CHARS = 10
INTERNAL_PREFIX_RESTART_MAX_GAP_CHARS = 4
INTERNAL_PREFIX_RESTART_MAX_DROP_CHARS = 14
INTERNAL_PREFIX_RESTART_LEAD_PREFIXES = ("", "就", "呃", "啊", "嗯")
INTERNAL_PREFIX_RESTART_MAX_LEAD_CHARS = 2
INTERNAL_PREFIX_RESTART_MAX_FUZZY_GAP_CHARS = 2
INTERNAL_PREFIX_RESTART_PRONOUN_LEADS = ("我", "你", "他", "她", "它", "咱", "我们", "你们", "他们", "她们")
INTERNAL_PREFIX_RESTART_LEAD_FOLLOWERS = ("会", "要", "能", "敢", "在", "把", "给", "被", "让", "是", "就", "还", "都")
ABANDONED_CLAUSE_RESTART_MIN_PHRASE_CHARS = 2
ABANDONED_CLAUSE_RESTART_MAX_PHRASE_CHARS = 10
ABANDONED_CLAUSE_RESTART_MAX_GAP_CHARS = 12
ABANDONED_CLAUSE_RESTART_MAX_DROP_CHARS = 24
ABANDONED_CLAUSE_RESTART_PARTICLES = ("呢", "啊", "哦", "喔", "呃", "嗯", "吗", "嘛", "吧")
REPEATED_DISCOURSE_OPENERS = ("但凡", "如果", "假如", "要是", "所以", "因为", "但是", "然后", "其实", "就是")
REPEATED_DISCOURSE_MIN_REMAINDER_CHARS = 3
REPEATED_DISCOURSE_MAX_SOURCE_GAP_US = 500_000
CONDITIONAL_DISCOURSE_OPENERS = ("但凡", "如果", "假如", "要是")
CONDITIONAL_QUANTIFIED_SUBJECT_STARTS = (
    "任何",
    "任意",
    "每个",
    "每一个",
    "所有",
    "一个",
    "一名",
    "这种",
    "这个",
    "那个",
    "这些",
    "那些",
    "凡是",
)
SHORT_ASCII_HEAD_CONTINUATION_PREFIXES = (
    "只是",
    "才是",
    "不是",
    "是",
    "也",
    "会",
    "能",
    "要",
    "追求",
    "阶段",
    "标准",
    "级",
    "的",
)
SUSPECT_REPEATED_NUMERALS = set("一二三四五六七八九十两")
SUSPECT_NUMERAL_CLASSIFIER_STARTS = set("个件节门块万千百元分毛次种类套张份杯瓶支台辆只条")
FINAL_VISIBLE_RECHECK_DECISIONS = [
    "drop_bad_fragment",
    "trim_repeated_prefix",
    "keep_if_coherent",
    "requires_human_review",
]


def build_final_caption_visible_repeat_gate(captions: list[CaptionRenderUnit]) -> dict[str, Any]:
    evidence = detect_final_caption_visible_evidence(captions, _final_caption_visible_detector_set())
    classification = classify_final_caption_visible_evidence(evidence)
    policy_decisions = build_final_caption_visible_policy(evidence, classification)
    repair_signal = build_final_caption_visible_repair_signal(policy_decisions)
    return build_final_caption_visible_gate_report(
        evidence=evidence,
        classification=classification,
        policy_decisions=policy_decisions,
        repair_signal=repair_signal,
        ngram_size=NGRAM_SIZE,
        prefix_suffix_min_overlap=PREFIX_SUFFIX_MIN_OVERLAP,
        near_duplicate_ratio=NEAR_DUPLICATE_RATIO,
        final_visible_recheck_decisions=FINAL_VISIBLE_RECHECK_DECISIONS,
    )


def _final_caption_visible_detector_set() -> FinalCaptionVisibleDetectorSet:
    return FinalCaptionVisibleDetectorSet(
        containment_candidates=_containment_candidates,
        candidate_pairs=_candidate_pairs,
        prefix_suffix_candidates=_prefix_suffix_candidates,
        ngram_candidates=_ngram_candidates,
        near_duplicate_candidates=_near_duplicate_candidates,
        modifier_redundancy_candidates=_modifier_redundancy_candidates,
        self_repair_candidates=_self_repair_aborted_phrase_candidates,
        dangling_candidates=_dangling_prefix_suffix_candidates,
        semantic_suspect_candidates=_semantic_garbage_or_asr_suspect_candidates,
        semantic_integrity_candidates=build_final_semantic_integrity_candidates,
        cross_caption_containment_candidates=_cross_caption_semantic_containment_candidates,
        restart_repeat_candidates=_raw_restart_repeat_candidates,
    )


def _raw_restart_repeat_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    return [
        *_restart_repeat_visible_candidates(captions),
        *_repeated_discourse_opener_candidates(captions),
        *_negative_predicate_restart_candidates(captions),
        *_partial_phrase_restart_candidates(captions),
        *_short_fragment_restart_candidates(captions),
    ]


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
            if is_explanatory_term_reuse(left_text, right_text) or is_explanatory_term_reuse(right_text, left_text):
                continue
            if left_text == right_text or (len(left_text) >= 2 and left_text in right_text) or (len(right_text) >= 2 and right_text in left_text):
                short_index = left_index if len(left_text) <= len(right_text) else right_index
                if _short_ascii_caption_heads_immediate_continuation(captions, short_index):
                    continue
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


def _short_ascii_caption_heads_immediate_continuation(captions: list[CaptionRenderUnit], index: int) -> bool:
    if index < 0 or index + 1 >= len(captions):
        return False
    caption = captions[index]
    text = normalize_text(caption.text)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]{0,7}", text or ""):
        return False
    next_caption = captions[index + 1]
    gap_us = int(next_caption.target_start_us) - int(caption.target_end_us)
    if gap_us < -80_000 or gap_us > 300_000:
        return False
    next_text = normalize_text(next_caption.text)
    if not next_text:
        return False
    return next_text.startswith(SHORT_ASCII_HEAD_CONTINUATION_PREFIXES)


def _prefix_suffix_candidates(captions: list[CaptionRenderUnit], excluded_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left, right in zip(captions, captions[1:]):
        if (left.caption_id, right.caption_id) in excluded_pairs:
            continue
        left_chars = list(normalize_text(left.text))
        right_chars = list(normalize_text(right.text))
        overlap = longest_suffix_prefix_overlap(left_chars, right_chars)
        if overlap >= PREFIX_SUFFIX_MIN_OVERLAP:
            overlap_text = "".join(left_chars[-overlap:])
            if is_semantic_label_reuse_boundary(left.text, right.text, overlap_text):
                continue
            candidates.append(
                _candidate(
                    "prefix_suffix_overlap",
                    left,
                    right,
                    overlap_text=overlap_text,
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
            if is_explanatory_term_reuse(previous.text, caption.text) or is_explanatory_term_reuse(caption.text, previous.text):
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
    if _dangling_pronoun_modal_suffix(text):
        return "dangling_pronoun_modal_suffix"
    if _dangling_weak_pronoun_fragment(text):
        return "dangling_weak_pronoun_fragment"
    return ""


def _dangling_pronoun_modal_suffix(text: str) -> str:
    if len(text) < 5:
        return ""
    for pronoun in DANGLING_PRONOUN_MODAL_PRONOUNS:
        for tail in DANGLING_PRONOUN_MODAL_TAILS:
            suffix = f"{pronoun}{tail}"
            if text.endswith(suffix) and len(text) > len(suffix) + 2:
                return suffix
    search_start = max(0, len(text) - 6)
    for pronoun in DANGLING_PRONOUN_MODAL_PRONOUNS:
        index = text.rfind(pronoun, search_start)
        if index < 0:
            continue
        suffix = text[index:]
        if 3 <= len(suffix) <= 5 and suffix.endswith(DANGLING_PRONOUN_OPEN_TAIL_ENDS) and len(text) > len(suffix) + 2:
            return suffix
    return ""


def _dangling_weak_pronoun_fragment(text: str) -> bool:
    if not 4 <= len(text) <= 6:
        return False
    if not any(text.startswith(prefix) for prefix in DANGLING_WEAK_PRONOUN_FRAGMENT_PREFIXES):
        return False
    return any(text.endswith(pronoun) for pronoun in DANGLING_PRONOUN_MODAL_PRONOUNS)


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
    intra_token = _intra_token_numeral_duplicate_suspect(text)
    if intra_token:
        return intra_token
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


def _intra_token_numeral_duplicate_suspect(text: str) -> dict[str, Any] | None:
    normalized = normalize_text(text)
    if len(normalized) < 3:
        no_suspect: dict[str, Any] | None = None
        return no_suspect
    first, second, third = normalized[0], normalized[1], normalized[2]
    if first != second:
        no_suspect: dict[str, Any] | None = None
        return no_suspect
    if first not in SUSPECT_REPEATED_NUMERALS or third not in SUSPECT_NUMERAL_CLASSIFIER_STARTS:
        no_suspect: dict[str, Any] | None = None
        return no_suspect
    return {
        "pattern": "intra_token_repeated_numeral_before_classifier",
        "overlap_text": normalized[:3],
        "score": 1.0,
        "requires_token_level_audio_review": True,
    }


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
            if is_explanatory_term_reuse(left_text, combined) or is_explanatory_term_reuse(combined, left_text):
                continue
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
        internal_prefix_restart = _internal_prefix_restart_repeat(left_text)
        if internal_prefix_restart is not None:
            candidate = _candidate(
                "internal_prefix_restart_repeat_visible",
                caption,
                caption,
                overlap_text=str(internal_prefix_restart.get("overlap_text") or ""),
                score=float(internal_prefix_restart.get("score") or 1.0),
            )
            candidate.update(
                {
                    "type": "internal_prefix_restart_repeat_visible",
                    "severity": "high",
                    "window_caption_ids": [caption.caption_id],
                    "window_text": caption.text,
                    **internal_prefix_restart,
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
            if is_explanatory_term_reuse(left_text, combined) or is_explanatory_term_reuse(combined, left_text):
                continue
            overlap_text = ""
            if left_text in combined and not combined.startswith(left_text):
                overlap_text = left_text
            else:
                overlap = _longest_common_substring(left_text, combined)
                if len(overlap) >= max(PREFIX_SUFFIX_MIN_OVERLAP + 1, min(8, len(left_text))):
                    overlap_text = overlap
            if not overlap_text:
                continue
            if is_parallel_progressive_semantic_expansion(left_text, combined, overlap_text):
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


def _repeated_discourse_opener_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left, right in zip(captions, captions[1:]):
        left_text = normalize_text(left.text)
        right_text = normalize_text(right.text)
        if not left_text or not right_text:
            continue
        opener = _shared_discourse_opener(left_text, right_text)
        if not opener:
            continue
        if _is_parallel_condition_discourse_opener(left_text, right_text, opener):
            continue
        if len(left_text) - len(opener) < REPEATED_DISCOURSE_MIN_REMAINDER_CHARS:
            continue
        if len(right_text) - len(opener) < REPEATED_DISCOURSE_MIN_REMAINDER_CHARS:
            continue
        if not _caption_source_gap_within(left, right, REPEATED_DISCOURSE_MAX_SOURCE_GAP_US):
            continue
        candidate = _candidate(
            "adjacent_repeated_discourse_opener_visible",
            right,
            left,
            overlap_text=opener,
            score=round(len(opener) / max(1, min(len(left_text), len(right_text))), 6),
        )
        candidate.update(
            {
                "type": "adjacent_repeated_discourse_opener_visible",
                "severity": "high",
                "pattern": "repeated_discourse_opener",
                "drop_text": opener,
                "window_caption_ids": [left.caption_id, right.caption_id],
                "window_text": f"{left.text}{right.text}",
            }
        )
        candidates.append(candidate)
    return candidates


def _shared_discourse_opener(left_text: str, right_text: str) -> str:
    for opener in REPEATED_DISCOURSE_OPENERS:
        if left_text.startswith(opener) and right_text.startswith(opener):
            return opener
    return ""


def _is_parallel_condition_discourse_opener(left_text: str, right_text: str, opener: str) -> bool:
    if opener not in CONDITIONAL_DISCOURSE_OPENERS:
        return False
    left_remainder = normalize_text(left_text[len(opener) :])
    right_remainder = normalize_text(right_text[len(opener) :])
    if not left_remainder or not right_remainder:
        return False
    left_quantified = any(left_remainder.startswith(prefix) for prefix in CONDITIONAL_QUANTIFIED_SUBJECT_STARTS)
    right_quantified = any(right_remainder.startswith(prefix) for prefix in CONDITIONAL_QUANTIFIED_SUBJECT_STARTS)
    if not (left_quantified or right_quantified):
        return False
    if left_remainder.startswith(right_remainder) or right_remainder.startswith(left_remainder):
        return False
    return True


def _caption_source_gap_within(left: CaptionRenderUnit, right: CaptionRenderUnit, max_gap_us: int) -> bool:
    left_end = int(left.spoken_source_end_us or 0)
    right_start = int(right.spoken_source_start_us or 0)
    if left_end <= 0 or right_start <= 0:
        target_gap_us = int(right.target_start_us) - int(left.target_end_us)
        return -80_000 <= target_gap_us <= max_gap_us
    return -80_000 <= right_start - left_end <= max_gap_us


def _short_fragment_restart_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    emitted: set[tuple[str, str]] = set()
    for index, caption in enumerate(captions):
        left_text = normalize_text(caption.text)
        if len(left_text) < SHORT_FRAGMENT_RESTART_MIN_OVERLAP_CHARS:
            continue
        for window_size in range(1, SHORT_FRAGMENT_RESTART_LOOKAHEAD + 1):
            window = captions[index + 1 : index + 1 + window_size]
            if not window:
                continue
            combined = normalize_text("".join(row.text for row in window))
            if is_explanatory_term_reuse(left_text, combined) or is_explanatory_term_reuse(combined, left_text):
                continue
            row = _short_fragment_restart_match(left_text, combined)
            if row is None:
                continue
            overlap_text = str(row.get("overlap_text") or "")
            if is_parallel_progressive_semantic_expansion(left_text, combined, overlap_text):
                continue
            key = (caption.caption_id, overlap_text)
            if key in emitted:
                continue
            emitted.add(key)
            candidate = _candidate(
                "short_fragment_restart_visible",
                caption,
                window[-1],
                overlap_text=overlap_text,
                score=float(row.get("score") or 1.0),
            )
            candidate.update(
                {
                    "type": "short_fragment_restart_visible",
                    "severity": "high",
                    "window_caption_ids": [item.caption_id for item in window],
                    "window_text": "".join(item.text for item in window),
                    **row,
                }
            )
            candidates.append(candidate)
            break
    return candidates


def _short_fragment_restart_match(left_text: str, combined_right_text: str) -> dict[str, Any] | None:
    no_match: dict[str, Any] | None = None
    if not left_text or not combined_right_text:
        return no_match
    if left_text == combined_right_text:
        return no_match

    overlap_row = _longest_common_substring_row(left_text, combined_right_text)
    overlap_text = str(overlap_row.get("text") or "")
    if len(overlap_text) < SHORT_FRAGMENT_RESTART_MIN_OVERLAP_CHARS or not _plain_cjk(overlap_text):
        return no_match

    left_start = int(overlap_row.get("left_start") or 0)
    right_start = int(overlap_row.get("right_start") or 0)
    touches_left_boundary = left_start == 0 or left_start + len(overlap_text) == len(left_text)
    touches_right_boundary = right_start == 0 or right_start + len(overlap_text) == len(combined_right_text)
    coverage = len(overlap_text) / max(1, len(left_text))
    left_is_short = len(left_text) <= SHORT_FRAGMENT_RESTART_MAX_LEFT_CHARS
    weak_prefix = _weak_restart_prefix(left_text)
    right_weak_prefix = _weak_restart_prefix(combined_right_text)
    tail_fragment = _fragment_tail(left_text)

    whole_short_fragment_repeated = left_is_short and left_text in combined_right_text and not combined_right_text.startswith(left_text)
    short_high_coverage_restart = (
        left_is_short
        and coverage >= SHORT_FRAGMENT_RESTART_MIN_COVERAGE
        and (touches_left_boundary or touches_right_boundary)
        and bool(weak_prefix or tail_fragment or _contains_copula_restart(left_text))
    )
    weak_prefix_restart = (
        bool(weak_prefix)
        and weak_prefix == right_weak_prefix
        and len(weak_prefix) >= SHORT_FRAGMENT_RESTART_MIN_OVERLAP_CHARS
        and left_text != combined_right_text
    )
    tail_overlap_restart = left_is_short and bool(tail_fragment) and coverage >= SHORT_FRAGMENT_RESTART_MIN_COVERAGE and (
        touches_left_boundary or touches_right_boundary or len(overlap_text) >= 4
    )
    if not (whole_short_fragment_repeated or short_high_coverage_restart or weak_prefix_restart or tail_overlap_restart):
        return no_match

    return {
        "pattern": "short_fragment_restart",
        "overlap_text": overlap_text,
        "drop_text": left_text,
        "left_start": left_start,
        "right_start": right_start,
        "coverage": round(coverage, 6),
        "score": round(coverage, 6),
        "weak_restart_prefix": weak_prefix,
        "fragment_tail": tail_fragment,
    }


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
    matches.extend(_partial_phrase_tail_mismatch_restart_matches(text, boundary, search_start))
    return matches


def _partial_phrase_tail_mismatch_restart_matches(text: str, boundary: int, search_start: int = 0) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if boundary <= 0 or boundary >= len(text):
        return matches
    right_text = text[boundary:]
    if not _plain_cjk(right_text):
        return matches
    max_start = min(boundary - PARTIAL_RESTART_MIN_SHARED_PREFIX_CHARS, boundary)
    for start in range(max(0, search_start), max_start + 1):
        left_suffix = text[start:boundary]
        if not _plain_cjk(left_suffix):
            continue
        max_shared = min(len(left_suffix) - 1, len(right_text), PARTIAL_RESTART_MAX_DROP_CHARS)
        for shared_len in range(max_shared, PARTIAL_RESTART_MIN_SHARED_PREFIX_CHARS - 1, -1):
            shared_prefix = left_suffix[:shared_len]
            left_tail = left_suffix[shared_len:]
            if not left_tail or len(left_tail) > PARTIAL_RESTART_MAX_TAIL_MISMATCH_CHARS:
                continue
            if left_tail not in PARTIAL_RESTART_TAIL_MISMATCH_ALLOWED_TAILS:
                continue
            if not right_text.startswith(shared_prefix):
                continue
            right_completion = right_text[shared_len:]
            if _cjk_char_count(right_completion) < PARTIAL_RESTART_MIN_TAIL_MISMATCH_COMPLETION_CHARS:
                continue
            if right_completion.startswith(left_tail):
                continue
            completed_len = min(len(right_text), shared_len + max(PARTIAL_RESTART_MIN_TAIL_MISMATCH_COMPLETION_CHARS, len(left_tail) + 1))
            matches.append(
                {
                    "pattern": "partial_phrase_restart_tail_mismatch",
                    "start": start,
                    "pivot": boundary,
                    "end": boundary + completed_len,
                    "drop_text": left_suffix,
                    "completed_text": right_text[:completed_len],
                    "overlap_text": left_suffix + right_text[:completed_len],
                    "shared_prefix": shared_prefix,
                    "left_tail": left_tail,
                    "score": round(len(shared_prefix) / max(1, len(left_suffix)), 6),
                }
            )
            break
    return matches


def _negative_predicate_restart_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    matches.extend(_negative_filler_restart_matches(text))
    matches.extend(_strengthened_negative_restart_matches(text))
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


def _negative_filler_restart_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for match in re.finditer(r"不([\u4e00-\u9fff]{1,2})不([\u4e00-\u9fff]{2,8})", text):
        filler = match.group(1)
        right = match.group(2)
        if filler not in {"就", "是", "会", "能", "敢", "要", "可"}:
            continue
        previous_char = text[match.start() - 1 : match.start()] if match.start() > 0 else ""
        if previous_char == filler and filler in {"会", "能", "敢", "要"}:
            continue
        drop_text = text[match.start() : match.start(2) - 1]
        overlap_text = text[match.start() : match.end()]
        matches.append(
            {
                "pattern": "negative_predicate_restart",
                "start": match.start(),
                "end": match.end(),
                "left_predicate": filler,
                "right_predicate": right,
                "drop_text": drop_text,
                "overlap_text": overlap_text,
                "score": round(len(drop_text) / max(1, len(overlap_text)), 6),
            }
        )
    return matches


def _strengthened_negative_restart_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    pattern = re.compile(r"((?:[\u4e00-\u9fff]{0,2})不(?:会|能|敢|要|想|肯))((?:绝不|不)[\u4e00-\u9fff]{1,8})")
    for match in pattern.finditer(text):
        drop_text = match.group(1)
        if len(drop_text) < 2 or len(drop_text) > 8:
            continue
        if drop_text.count(NEGATIVE_RESTART_PREFIX) > 1:
            continue
        negative_index = drop_text.rfind(NEGATIVE_RESTART_PREFIX)
        if negative_index > 0 and drop_text[negative_index - 1] in {"会", "能", "敢", "要"}:
            continue
        overlap_text = match.group(0)
        matches.append(
            {
                "pattern": "negative_predicate_restart",
                "start": match.start(1),
                "end": match.end(2),
                "left_predicate": drop_text,
                "right_predicate": match.group(2),
                "drop_text": drop_text,
                "overlap_text": overlap_text,
                "score": round(len(drop_text) / max(1, len(overlap_text)), 6),
            }
        )
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


def _internal_prefix_restart_repeat(text: str) -> dict[str, Any] | None:
    if len(text) < SHORT_FRAGMENT_RESTART_MIN_OVERLAP_CHARS * 2 + 1:
        no_restart: dict[str, Any] | None = None
        return no_restart
    max_phrase_len = min(INTERNAL_PREFIX_RESTART_MAX_PHRASE_CHARS, len(text) // 2)
    best: dict[str, Any] | None = None
    for phrase_len in range(SHORT_FRAGMENT_RESTART_MIN_OVERLAP_CHARS, max_phrase_len + 1):
        for first_start in range(0, max(0, len(text) - phrase_len * 2) + 1):
            phrase = text[first_start : first_start + phrase_len]
            if not _plain_cjk(phrase):
                continue
            lead_prefix = text[:first_start]
            if lead_prefix not in INTERNAL_PREFIX_RESTART_LEAD_PREFIXES:
                continue
            second_start = text.find(phrase, first_start + phrase_len)
            if second_start < 0:
                continue
            gap = text[first_start + phrase_len : second_start]
            if len(gap) > INTERNAL_PREFIX_RESTART_MAX_GAP_CHARS:
                continue
            restart_start = _internal_prefix_restart_start_before_second(text, first_start, phrase_len, second_start)
            drop_text = text[:restart_start]
            if not drop_text or len(drop_text) > INTERNAL_PREFIX_RESTART_MAX_DROP_CHARS:
                continue
            if best is None or phrase_len > int(best.get("phrase_chars") or 0):
                best = {
                    "pattern": "internal_prefix_restart",
                    "restart_phrase": phrase,
                    "overlap_text": text[first_start : second_start + phrase_len],
                    "drop_text": drop_text,
                    "first_start": first_start,
                    "restart_start": restart_start,
                    "second_start": second_start,
                    "gap_text": gap,
                    "phrase_chars": phrase_len,
                    "score": round(phrase_len / max(1, len(drop_text)), 6),
                }
    fuzzy = _fuzzy_internal_prefix_restart_repeat(text)
    if fuzzy is not None and (
        best is None or int(fuzzy.get("phrase_chars") or 0) > int(best.get("phrase_chars") or 0)
    ):
        return fuzzy
    abandoned = _abandoned_clause_restart_repeat(text)
    if abandoned is not None and (
        best is None or int(abandoned.get("phrase_chars") or 0) >= int(best.get("phrase_chars") or 0)
    ):
        return abandoned
    return best


def _internal_prefix_restart_start_before_second(
    text: str,
    first_start: int,
    phrase_len: int,
    second_start: int,
) -> int:
    gap = text[first_start + phrase_len : second_start]
    if not gap:
        return second_start
    lead = _restart_lead_suffix(gap)
    if not lead:
        return second_start
    return second_start - len(lead)


def _restart_lead_suffix(gap: str) -> str:
    for lead in sorted(INTERNAL_PREFIX_RESTART_PRONOUN_LEADS, key=len, reverse=True):
        start = gap.rfind(lead)
        if start < 0:
            continue
        suffix = gap[start:]
        if len(suffix) <= len(lead):
            continue
        if suffix[len(lead) :].startswith(INTERNAL_PREFIX_RESTART_LEAD_FOLLOWERS):
            return suffix
    return ""


def _abandoned_clause_restart_repeat(text: str) -> dict[str, Any] | None:
    if len(text) < ABANDONED_CLAUSE_RESTART_MIN_PHRASE_CHARS * 2 + 1:
        no_restart: dict[str, Any] | None = None
        return no_restart
    best: dict[str, Any] | None = None
    max_phrase_len = min(ABANDONED_CLAUSE_RESTART_MAX_PHRASE_CHARS, len(text) // 2)
    for phrase_len in range(ABANDONED_CLAUSE_RESTART_MIN_PHRASE_CHARS, max_phrase_len + 1):
        for first_start in range(0, max(0, len(text) - phrase_len * 2) + 1):
            phrase = text[first_start : first_start + phrase_len]
            if not _plain_cjk(phrase):
                continue
            second_start = text.find(phrase, first_start + phrase_len + 1)
            if second_start < 0:
                continue
            gap = text[first_start + phrase_len : second_start]
            if not gap or len(gap) > ABANDONED_CLAUSE_RESTART_MAX_GAP_CHARS or not _plain_cjk(gap):
                continue
            drop_text = text[first_start:second_start]
            if phrase.startswith(NEGATIVE_RESTART_PREFIX):
                continue
            if len(drop_text) > ABANDONED_CLAUSE_RESTART_MAX_DROP_CHARS:
                continue
            after_second = text[second_start + phrase_len :]
            min_remainder_chars = 2 if phrase_len <= 2 else REPEATED_DISCOURSE_MIN_REMAINDER_CHARS
            if _cjk_char_count(after_second) < min_remainder_chars:
                continue
            if not _abandoned_clause_restart_signal(gap, after_second, phrase_len):
                continue
            if best is None or (
                phrase_len,
                len(drop_text),
            ) > (
                int(best.get("phrase_chars") or 0),
                int(best.get("drop_chars") or 0),
            ):
                best = {
                    "pattern": "abandoned_clause_restart",
                    "restart_phrase": phrase,
                    "overlap_text": text[first_start : second_start + phrase_len],
                    "drop_text": drop_text,
                    "first_start": first_start,
                    "second_start": second_start,
                    "gap_text": gap,
                    "phrase_chars": phrase_len,
                    "drop_chars": len(drop_text),
                    "score": round(phrase_len / max(1, len(drop_text)), 6),
                }
    return best


def _abandoned_clause_restart_signal(gap: str, after_second: str, phrase_len: int) -> bool:
    if gap.endswith(ABANDONED_CLAUSE_RESTART_PARTICLES):
        return True
    if phrase_len <= 2 and gap[-1:] and after_second.startswith(gap[-1]):
        return True
    if len(gap) >= 2 and after_second.startswith(gap[-2:]):
        return True
    return False


def _fuzzy_internal_prefix_restart_repeat(text: str) -> dict[str, Any] | None:
    if len(text) < SHORT_FRAGMENT_RESTART_MIN_OVERLAP_CHARS * 2 + 1:
        no_restart: dict[str, Any] | None = None
        return no_restart
    max_phrase_len = min(INTERNAL_PREFIX_RESTART_MAX_PHRASE_CHARS, len(text) // 2)
    best: dict[str, Any] | None = None
    for phrase_len in range(SHORT_FRAGMENT_RESTART_MIN_OVERLAP_CHARS, max_phrase_len + 1):
        max_first_start = min(INTERNAL_PREFIX_RESTART_MAX_LEAD_CHARS, max(0, len(text) - phrase_len * 2))
        for first_start in range(0, max_first_start + 1):
            phrase = text[first_start : first_start + phrase_len]
            if not _plain_cjk(phrase):
                continue
            second_start = text.find(phrase, first_start + phrase_len + 1)
            if second_start < 0:
                continue
            lead_prefix = text[:first_start]
            gap = text[first_start + phrase_len : second_start]
            if len(gap) > INTERNAL_PREFIX_RESTART_MAX_FUZZY_GAP_CHARS:
                continue
            if gap and not _plain_cjk(gap):
                continue
            restart_start = second_start - len(gap) if gap else second_start
            drop_text = text[:restart_start]
            if not drop_text or len(drop_text) > INTERNAL_PREFIX_RESTART_MAX_DROP_CHARS:
                continue
            after_second = text[second_start + phrase_len :]
            if _cjk_char_count(after_second) < REPEATED_DISCOURSE_MIN_REMAINDER_CHARS:
                continue
            if best is None or phrase_len > int(best.get("phrase_chars") or 0):
                best = {
                    "pattern": "internal_prefix_restart",
                    "restart_phrase": phrase,
                    "overlap_text": text[first_start : second_start + phrase_len],
                    "drop_text": drop_text,
                    "first_start": first_start,
                    "restart_start": restart_start,
                    "second_start": second_start,
                    "gap_text": gap,
                    "lead_prefix": lead_prefix,
                    "phrase_chars": phrase_len,
                    "score": round(phrase_len / max(1, len(drop_text)), 6),
                }
    return best


def _weak_restart_prefix(text: str) -> str:
    for prefix in WEAK_RESTART_PREFIXES:
        if text.startswith(prefix):
            return prefix
    return ""


def _fragment_tail(text: str) -> str:
    for tail in FRAGMENT_TAILS:
        if text.endswith(tail):
            return tail
    return ""


def _contains_copula_restart(text: str) -> bool:
    return text.count("是") >= 2 or text.startswith("就是")


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


def _longest_common_substring_row(left: str, right: str) -> dict[str, Any]:
    if not left or not right:
        return {"text": "", "left_start": 0, "right_start": 0}
    best = ""
    best_left = 0
    best_right = 0
    for left_index in range(len(left)):
        for right_index in range(len(right)):
            offset = 0
            while left_index + offset < len(left) and right_index + offset < len(right) and left[left_index + offset] == right[right_index + offset]:
                offset += 1
            if offset > len(best):
                best = left[left_index : left_index + offset]
                best_left = left_index
                best_right = right_index
    return {"text": best, "left_start": best_left, "right_start": best_right}


def _plain_cjk(text: str) -> bool:
    return bool(text) and bool(re.fullmatch(r"[\u4e00-\u9fff]+", text))


def _cjk_char_count(text: str) -> int:
    return sum(1 for char in str(text or "") if "\u4e00" <= char <= "\u9fff")


def _candidate_pairs(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row.get("caption_id") or ""), str(row.get("related_caption_id") or ""))
        for row in rows
    }


def _candidates_by_reason(
    rows: list[dict[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("reason") or "") == reason]


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
