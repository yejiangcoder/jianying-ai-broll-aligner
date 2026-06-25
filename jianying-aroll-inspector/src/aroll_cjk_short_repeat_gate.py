from __future__ import annotations

import re
from typing import Any


MAX_OVERLAP_CHARS = 8
MIN_FATAL_OVERLAP_CHARS = 2
MIN_WARNING_OVERLAP_CHARS = 1
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TEXT_SPLIT_RE = re.compile(r"[\s,，.。!！?？;；:：、]+")
_PRONOUNS = set("\u6211\u4f60\u4ed6\u5979\u5b83\u8fd9\u90a3")
_CONNECTORS = set("\u5c31\u4f1a\u53c8\u4e5f\u8fd8\u518d\u90fd\u624d\u8981\u60f3\u80fd\u8be5")
_SINGLE_CHAR_REPEATABLE = _PRONOUNS | _CONNECTORS
_CJK_NUMERAL_CHARS = set("\u96f6\u3007\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u4ebf\u4e24\u51e0\u534a0123456789")
_REDUPLICATION_MODIFIER_SUFFIXES = ("的", "地", "得")
_MAX_PROTECTED_MODIFIER_REDUPLICATION_CHARS = 4
_REDUPLICATION_AMOUNT_UNITS = set("\u4e2a\u53ea\u4ef6\u53f0\u90e8\u676f\u74f6\u4efd\u6b21\u5757\u5143\u6bdb\u89d2\u5206\u94b1\u4e07\u5343\u767e\u5341\u4ebf\u5e74\u6708\u5929\u5c0f\u65f6\u5206\u949f\u516c\u91cc\u7c73\u65a4\u514b\u5c81\u5e73")
_LABEL_INTRODUCERS = ("叫做", "称为", "称作", "叫", "算作", "属于", "是")
_DEFINITIONAL_CONNECTORS = ("等于", "就是", "指的是", "意味着", "是", "叫做", "叫")
_ATTRIBUTIVE_LABEL_CONTEXT_MARKERS = ("般的", "式的", "型的", "感的", "的")


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("fragment_text") or row.get("text") or row.get("subtitle_text") or "")


def _cjk_only(text: str) -> str:
    return "".join(ch for ch in str(text or "") if _CJK_RE.match(ch))


def _cjk_parts(text: str) -> list[str]:
    return [part for part in (_cjk_only(raw) for raw in _TEXT_SPLIT_RE.split(str(text or ""))) if part]


def _best_suffix_prefix_overlap(left: str, right: str, *, min_chars: int = MIN_FATAL_OVERLAP_CHARS) -> tuple[int, str]:
    max_len = min(MAX_OVERLAP_CHARS, len(left), len(right))
    for size in range(max_len, min_chars - 1, -1):
        phrase = left[-size:]
        if phrase and right.startswith(phrase):
            return size, phrase
    return 0, ""


def _overlap_severity(phrase: str, left: str, _right: str) -> str:
    if _is_numeral_headed_short_phrase(phrase):
        return "warning"
    if len(phrase) >= MIN_FATAL_OVERLAP_CHARS:
        return "fatal"
    if len(phrase) == 1 and phrase in _CONNECTORS and len(left) <= 4:
        return "fatal"
    return "warning"


def _confidence_for_severity(severity: str) -> str:
    return "high" if severity == "fatal" else "medium"


def _is_a_not_a_context(normalized: str, start: int, end: int) -> bool:
    for unit_len in (1, 2):
        min_pattern_start = max(0, start - (unit_len + 1))
        max_pattern_start = min(start + 1, len(normalized) - (unit_len * 2))
        for pattern_start in range(min_pattern_start, max_pattern_start + 1):
            left_start = pattern_start
            not_index = pattern_start + unit_len
            right_start = not_index + 1
            right_end = right_start + unit_len
            if right_end > len(normalized):
                continue
            left = normalized[left_start:not_index]
            right = normalized[right_start:right_end]
            if not left or normalized[not_index] != "\u4e0d" or left != right:
                continue
            if start < right_end and end > left_start:
                return True
    return False


def _is_numeral_headed_short_phrase(phrase: str) -> bool:
    if len(phrase) < 2 or len(phrase) > 4:
        return False
    if phrase[0] not in _CJK_NUMERAL_CHARS:
        return False
    return all(_CJK_RE.match(ch) or ch.isdigit() for ch in phrase[1:])


def _adjacent_phrase_repeat_classification(normalized: str, start: int, size: int, phrase: str) -> tuple[str, str]:
    end = start + (size * 2)
    if _is_a_not_a_context(normalized, start, end):
        return "warning", "A-not-A CJK question structure; not treated as stutter"
    if _is_numeral_headed_short_phrase(phrase):
        return "warning", "numeral-headed short phrase reduplication; not treated as high-confidence stutter"
    if _is_modifier_reduplication(normalized, start, size, phrase):
        return "warning", "modifier reduplication before 的/地/得; not treated as high-confidence stutter"
    return "fatal", "adjacent identical CJK ngram in final subtitle text"


def classify_adjacent_cjk_ngram_repeat(normalized: str, start: int, size: int, phrase: str) -> tuple[str, str]:
    return _adjacent_phrase_repeat_classification(normalized, start, size, phrase)


def _is_modifier_reduplication(normalized: str, start: int, size: int, phrase: str) -> bool:
    if not phrase or size <= 0 or size > _MAX_PROTECTED_MODIFIER_REDUPLICATION_CHARS:
        return False
    if not all(_CJK_RE.match(char) for char in phrase):
        return False
    end = start + (size * 2)
    if start < 0 or end > len(normalized):
        return False
    if normalized[start:end] != phrase + phrase:
        return False
    if end >= len(normalized):
        return False
    following = normalized[end : end + 8]
    if following.startswith(_REDUPLICATION_MODIFIER_SUFFIXES):
        return True
    if len(phrase) == 1 and phrase in _SINGLE_CHAR_REPEATABLE:
        return False
    return _looks_like_quantity_or_amount_follow(following)


def _looks_like_quantity_or_amount_follow(following: str) -> bool:
    if not following or following[0] not in _CJK_NUMERAL_CHARS:
        return False
    tail = following[1:8]
    if not tail:
        return False
    return any(char in _REDUPLICATION_AMOUNT_UNITS for char in tail)


def is_semantic_label_reuse_boundary(left_text: str, right_text: str, overlap_text: str) -> bool:
    overlap = _cjk_only(str(overlap_text or ""))
    if len(overlap) < 2:
        return False
    left = _cjk_only(str(left_text or ""))
    right = _cjk_only(str(right_text or ""))
    if not left.endswith(overlap) or not right.startswith(overlap):
        return False
    left_prefix = left[: -len(overlap)]
    right_suffix = right[len(overlap) :]
    if not any(right_suffix.startswith(marker) for marker in _DEFINITIONAL_CONNECTORS):
        return False
    if any(left_prefix.endswith(marker) for marker in _LABEL_INTRODUCERS):
        return True
    return _has_attributive_label_context(left_prefix)


def _has_attributive_label_context(left_prefix: str) -> bool:
    if len(left_prefix) < 3:
        return False
    return any(left_prefix.endswith(marker) for marker in _ATTRIBUTIVE_LABEL_CONTEXT_MARKERS)


def _candidate(
    *,
    issue_type: str,
    candidate_type: str,
    scope: str,
    phrase: str,
    severity: str = "fatal",
    text: str = "",
    left_text: str = "",
    right_text: str = "",
    row_index: int | None = None,
    next_row_index: int | None = None,
    overlap_chars: int | None = None,
    span: dict[str, Any] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    effective_span = span or {"row_index": row_index, "next_row_index": next_row_index}
    return {
        "type": candidate_type,
        "issue_type": issue_type,
        "severity": severity,
        "confidence": _confidence_for_severity(severity),
        "scope": scope,
        "phrase": phrase,
        "overlap": phrase,
        "text": text,
        "prev_text": left_text,
        "next_text": right_text,
        "left_text": left_text,
        "right_text": right_text,
        "row_index": row_index,
        "next_row_index": next_row_index,
        "overlap_chars": overlap_chars if overlap_chars is not None else len(phrase),
        "span": effective_span,
        "reason": reason,
    }


def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int | None, int | None]] = set()
    for row in candidates:
        key = (
            str(row.get("type") or row.get("issue_type") or ""),
            str(row.get("scope") or ""),
            str(row.get("overlap") or row.get("phrase") or ""),
            row.get("row_index"),
            row.get("next_row_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _restart_matches(normalized: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for start in range(0, max(0, len(normalized) - 2)):
        first = normalized[start]
        connector = normalized[start + 1]
        restarted = normalized[start + 2]
        if first in _PRONOUNS and connector in _CONNECTORS and restarted == first:
            matches.append((start, normalized[start : start + 3]))
    return matches


def _detect_intra_text(text: str, row_index: int) -> list[dict[str, Any]]:
    normalized = _cjk_only(text)
    candidates: list[dict[str, Any]] = []
    if not normalized:
        return candidates

    for size in range(1, min(MAX_OVERLAP_CHARS, len(normalized) // 2) + 1):
        for start in range(0, len(normalized) - (size * 2) + 1):
            phrase = normalized[start : start + size]
            if size == 1 and phrase not in _SINGLE_CHAR_REPEATABLE:
                continue
            if phrase == normalized[start + size : start + (size * 2)]:
                severity, reason = _adjacent_phrase_repeat_classification(normalized, start, size, phrase)
                candidates.append(
                    _candidate(
                        issue_type="cjk_adjacent_phrase_repeat",
                        candidate_type="intra_subtitle_ngram_repeat",
                        scope="intra_subtitle",
                        phrase=phrase,
                        severity=severity,
                        text=text,
                        row_index=row_index,
                        overlap_chars=size,
                        span={"row_index": row_index, "start_char": start, "end_char": start + (size * 2)},
                        reason=reason,
                    )
                )

    for start, phrase in _restart_matches(normalized):
        candidates.append(
            _candidate(
                issue_type="cjk_pronoun_connector_restart",
                candidate_type="restart_disfluency",
                scope="intra_subtitle",
                phrase=phrase,
                severity="fatal",
                text=text,
                row_index=row_index,
                overlap_chars=1,
                span={"row_index": row_index, "start_char": start, "end_char": start + 3},
                reason="CJK pronoun plus short connector restarts with the same pronoun",
            )
        )

    parts = _cjk_parts(text)
    for part_index, (left, right) in enumerate(zip(parts, parts[1:]), start=1):
        overlap_size, phrase = _best_suffix_prefix_overlap(left, right, min_chars=MIN_WARNING_OVERLAP_CHARS)
        if overlap_size:
            if is_semantic_label_reuse_boundary(left, right, phrase):
                continue
            severity = _overlap_severity(phrase, left, right)
            candidates.append(
                _candidate(
                    issue_type="cjk_adjacent_clause_overlap",
                    candidate_type="boundary_prefix_containment",
                    scope="intra_subtitle_clause_boundary",
                    phrase=phrase,
                    severity=severity,
                    text=text,
                    left_text=left,
                    right_text=right,
                    row_index=row_index,
                    overlap_chars=overlap_size,
                    span={"row_index": row_index, "left_part_index": part_index, "right_part_index": part_index + 1},
                    reason="normalized suffix of the previous clause is contained in the next clause prefix",
                )
            )
    return candidates


def _detect_boundary_restart(
    *,
    left: str,
    right: str,
    left_text: str,
    right_text: str,
    row_index: int,
) -> list[dict[str, Any]]:
    combined = left + right
    boundary_index = len(left)
    candidates: list[dict[str, Any]] = []
    for start, phrase in _restart_matches(combined):
        end = start + 3
        if start < boundary_index < end:
            candidates.append(
                _candidate(
                    issue_type="cjk_pronoun_connector_restart",
                    candidate_type="restart_disfluency",
                    scope="subtitle_boundary",
                    phrase=phrase,
                    severity="fatal",
                    left_text=left_text,
                    right_text=right_text,
                    row_index=row_index,
                    next_row_index=row_index + 1,
                    overlap_chars=1,
                    span={
                        "row_index": row_index,
                        "next_row_index": row_index + 1,
                        "boundary_char_index": boundary_index,
                        "start_char": start,
                        "end_char": end,
                    },
                    reason="CJK pronoun plus short connector restart crosses adjacent subtitle boundary",
                )
            )
    return candidates


def detect_cjk_short_repeats(display_subtitle_plan: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    rows = list(display_subtitle_plan or [])
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        candidates.extend(_detect_intra_text(_row_text(row), index))

    for index, (left_row, right_row) in enumerate(zip(rows, rows[1:]), start=1):
        left_text = _row_text(left_row)
        right_text = _row_text(right_row)
        left = _cjk_only(left_text)
        right = _cjk_only(right_text)
        if not left or not right:
            continue
        overlap_size, phrase = _best_suffix_prefix_overlap(left, right, min_chars=MIN_WARNING_OVERLAP_CHARS)
        if overlap_size:
            if is_semantic_label_reuse_boundary(left_text, right_text, phrase):
                continue
            severity = _overlap_severity(phrase, left, right)
            candidates.append(
                _candidate(
                    issue_type="cjk_adjacent_subtitle_boundary_overlap",
                    candidate_type="boundary_prefix_containment",
                    scope="subtitle_boundary",
                    phrase=phrase,
                    severity=severity,
                    left_text=left_text,
                    right_text=right_text,
                    row_index=index,
                    next_row_index=index + 1,
                    overlap_chars=overlap_size,
                    span={"row_index": index, "next_row_index": index + 1},
                    reason="normalized suffix of the previous final subtitle is contained in the next subtitle prefix",
                )
            )
        candidates.extend(
            _detect_boundary_restart(
                left=left,
                right=right,
                left_text=left_text,
                right_text=right_text,
                row_index=index,
            )
        )

    candidates.sort(
        key=lambda row: (
            str(row.get("severity") or "") != "fatal",
            -int(row.get("overlap_chars") or 0),
            str(row.get("scope") or ""),
            int(row.get("row_index") or 0),
            str(row.get("overlap") or row.get("phrase") or ""),
        )
    )
    return _dedupe(candidates)
