from __future__ import annotations

import re
from typing import Any


_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_MODIFIER_STACK_RE = re.compile(rf"([{_CJK}]{{1,2}})的([{_CJK}]{{1,2}})的([{_CJK}]{{1,6}})")
_CJK_RE = re.compile(rf"[{_CJK}]")
_NOMINAL_SCOPE_HEADS = ("男人", "女人", "角色", "主体", "对象", "人")
_EMPHASIS_ADVERBS = ("确确实实", "真的", "确实", "其实", "明显", "的确")
_PRONOUN_OBJECT_TAILS = frozenset("我你他她它咱")
_RELATIONAL_LEFT_MODIFIERS = frozenset(
    {
        "该有",
        "应有",
        "已有",
        "就有",
        "现有",
        "原有",
        "固有",
        "自有",
        "所有",
    }
)
_ACTION_INTENT_LEFT_MODIFIERS = frozenset(
    {
        "要做",
        "想做",
        "会做",
        "能做",
        "该做",
        "所做",
    }
)
_PROTECTED_BOUNDARY_BIGRAMS = frozenset(
    {
        "本该",
        "应该",
        "咬合",
        "配合",
        "适合",
        "符合",
        "结合",
        "契合",
        "融合",
        "吻合",
        "贴合",
        "整合",
    }
)


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("fragment_text") or row.get("text") or row.get("subtitle_text") or "")


def _cjk_with_offsets(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(str(text or "")):
        if _CJK_RE.match(char):
            chars.append(char)
            offsets.append(index)
    return "".join(chars), offsets


def _candidate(
    *,
    match: re.Match[str],
    scope: str,
    text: str = "",
    left_text: str = "",
    right_text: str = "",
    row_index: int,
    next_row_index: int | None = None,
    span: dict[str, Any],
    severity: str = "fatal",
) -> dict[str, Any]:
    left_modifier, right_modifier, head = match.groups()
    return {
        "type": "adjacent_modifier_semantic_redundancy",
        "issue_type": "adjacent_modifier_semantic_redundancy",
        "severity": severity,
        "confidence": "medium",
        "scope": scope,
        "text": text,
        "prev_text": left_text,
        "next_text": right_text,
        "left_text": left_text,
        "right_text": right_text,
        "phrase": match.group(0),
        "left_modifier": left_modifier,
        "right_modifier": right_modifier,
        "head_text": head,
        "row_index": row_index,
        "next_row_index": next_row_index,
        "span": span,
        "requires_self_review": True,
        "review_options": ["keep_left", "keep_right", "keep_both", "merge", "block_write"],
        "reason": "adjacent short CJK modifiers before the same head require semantic redundancy review",
    }


def _is_nominal_scope_emphasis(normalized: str, match: re.Match[str]) -> bool:
    left_modifier, _right_modifier, _head = match.groups()
    after_left_modifier = normalized[match.start() + len(left_modifier) + 1 :]
    for noun in _NOMINAL_SCOPE_HEADS:
        if not after_left_modifier.startswith(noun):
            continue
        after_noun = after_left_modifier[len(noun) :]
        if any(after_noun.startswith(adverb) for adverb in _EMPHASIS_ADVERBS):
            return True
    return False


def _starts_inside_protected_morpheme(normalized: str, match: re.Match[str]) -> bool:
    if match.start() <= 0:
        return False
    return normalized[match.start() - 1 : match.start() + 1] in _PROTECTED_BOUNDARY_BIGRAMS


def _is_relational_or_object_fragment(match: re.Match[str]) -> bool:
    left_modifier, _right_modifier, _head = match.groups()
    if left_modifier in _RELATIONAL_LEFT_MODIFIERS:
        return True
    if left_modifier in _ACTION_INTENT_LEFT_MODIFIERS:
        return True
    if len(left_modifier) == 2 and left_modifier[-1] in _PRONOUN_OBJECT_TAILS:
        return True
    return False


def _should_skip_modifier_stack(normalized: str, match: re.Match[str]) -> bool:
    return (
        _is_nominal_scope_emphasis(normalized, match)
        or _starts_inside_protected_morpheme(normalized, match)
        or _is_relational_or_object_fragment(match)
    )


def detect_adjacent_modifier_semantic_redundancy(display_subtitle_plan: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    rows = list(display_subtitle_plan or [])
    for row_index, row in enumerate(rows, start=1):
        text = _row_text(row)
        normalized, offsets = _cjk_with_offsets(text)
        for match in _MODIFIER_STACK_RE.finditer(normalized):
            if _should_skip_modifier_stack(normalized, match):
                continue
            start_char = offsets[match.start()] if offsets else match.start()
            end_char = (offsets[match.end() - 1] + 1) if offsets and match.end() > match.start() else match.end()
            candidates.append(
                _candidate(
                    match=match,
                    scope="intra_subtitle",
                    text=text,
                        row_index=row_index,
                        severity="fatal",
                        span={
                        "row_index": row_index,
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                )
            )
    for row_index, (left_row, right_row) in enumerate(zip(rows, rows[1:]), start=1):
        left_text = _row_text(left_row)
        right_text = _row_text(right_row)
        left_norm, _left_offsets = _cjk_with_offsets(left_text)
        right_norm, _right_offsets = _cjk_with_offsets(right_text)
        if not left_norm or not right_norm:
            continue
        combined = left_norm + right_norm
        boundary = len(left_norm)
        for match in _MODIFIER_STACK_RE.finditer(combined):
            if match.start() < boundary < match.end():
                if _should_skip_modifier_stack(combined, match):
                    continue
                candidates.append(
                    _candidate(
                        match=match,
                        scope="subtitle_boundary",
                        left_text=left_text,
                        right_text=right_text,
                        row_index=row_index,
                        next_row_index=row_index + 1,
                        severity="warning",
                        span={
                            "row_index": row_index,
                            "next_row_index": row_index + 1,
                            "boundary_char_index": boundary,
                            "start_char": match.start(),
                            "end_char": match.end(),
                        },
                    )
                )
    return candidates
