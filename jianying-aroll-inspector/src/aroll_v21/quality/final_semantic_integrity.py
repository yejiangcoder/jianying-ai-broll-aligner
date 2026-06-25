from __future__ import annotations

import re
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CaptionRenderUnit


PURE_VOCALIZATION_CHARS = frozenset("嗯呃额啊哦噢喔哎唉诶咳")
INTERJECTION_CHARS = frozenset("啊呀哎唉诶嗯呃额哦噢喔哈")
OPEN_CLAUSE_TAILS = (
    "是",
    "为",
    "把",
    "被",
    "给",
    "让",
    "使",
    "在",
    "从",
    "向",
    "对",
    "和",
    "与",
    "及",
    "或",
    "但",
    "而",
    "并",
)
HARD_OPEN_CLAUSE_TAILS = (
    "是",
    "为",
    "把",
    "被",
    "给",
    "让",
    "使",
    "和",
    "与",
    "及",
    "或",
    "但",
    "而",
    "并",
)
OPEN_COORDINATORS = ("以及", "并且", "和", "与", "及", "或")
RESTART_LEADING_CONNECTORS = ("然后", "那么", "那", "接着", "结果", "于是")
DANGLING_DISCOURSE_CONNECTORS = tuple(
    sorted(
        {
            "反而",
            "但是",
            "可是",
            "然而",
            "不过",
            "然后",
            "所以",
            "因为",
            "虽然",
            "如果",
            "而且",
            "并且",
            "甚至",
            "于是",
            "那么",
            "要是",
            "既然",
            "否则",
        },
        key=len,
        reverse=True,
    )
)
DISCOURSE_PRONOUN_TAIL_PRONOUNS = (
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "她们",
    "自己",
    "人家",
)
STANDALONE_CLASSIFIERS = frozenset("个件条张节款台辆本套部名位份颗枚")
OPEN_SINGLE_CHAR_TAILS = frozenset("大大小高低贵贱好差新旧多少美丑强弱冷热")
NOMINAL_PHRASE_HEAD_TAILS = frozenset("人心事物钱脸手脚头口眼车房课话路门店片图文号单")
IMMEDIATE_CONTEXT_GAP_US = 450_000
COMPLETE_DISCOURSE_TAILS = ("就是", "老是", "总是", "还是", "也是", "不是")
PREDICATE_CONTINUATION_PREFIXES = (
    "自己",
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "她们",
    "为了",
    "用来",
    "砸",
    "花",
    "给",
    "被",
    "把",
    "在",
    "由",
    "让",
    "能",
    "可以",
    "要",
    "去",
    "发",
    "做",
    "当",
)
PASSIVE_CONTINUATION_BLOCKED_PREFIXES = tuple(
    sorted(
        {
            *DANGLING_DISCOURSE_CONNECTORS,
            *RESTART_LEADING_CONNECTORS,
            "啊",
            "哦",
            "噢",
            "喔",
            "嗯",
            "呃",
            "不是",
            "没有",
            "后面",
            "下面",
            "接下来",
        },
        key=len,
        reverse=True,
    )
)
PASSIVE_CONTINUATION_VERB_HINTS = (
    "吃",
    "打",
    "压",
    "埋",
    "淹",
    "吞",
    "冲",
    "撕",
    "踢",
    "榨",
    "拿",
    "看",
    "判",
    "审",
    "识别",
    "发现",
    "证明",
    "消耗",
    "掏空",
)
NOMINAL_PREDICATE_BLOCKED_PREFIXES = tuple(
    sorted(
        {
            *DANGLING_DISCOURSE_CONNECTORS,
            *RESTART_LEADING_CONNECTORS,
            "了",
            "着",
            "过",
            "的",
            "地",
            "得",
            "吗",
            "嘛",
            "吧",
            "啊",
            "哦",
            "噢",
            "喔",
            "嗯",
            "呃",
            "后面",
            "后续",
            "下面",
            "之后",
            "随后",
            "接下来",
            "下一句",
        },
        key=len,
        reverse=True,
    )
)
DEVICE_PROMPT_MARKERS = (
    "请",
    "模式",
    "配置",
    "连接",
    "断开",
    "重启",
    "开机",
    "关机",
    "电源",
    "网络",
    "蓝牙",
    "配对",
    "设备",
)
DEVICE_PROMPT_ACTIONS = ("重启", "断电", "连接", "配对", "开机", "关机", "检查", "切换")


def build_final_semantic_integrity_candidates(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    ordered = sorted(captions, key=lambda row: (int(row.target_start_us), int(row.target_end_us), str(row.caption_id)))
    candidates: list[dict[str, Any]] = []
    for index, caption in enumerate(ordered):
        text = normalize_text(caption.text)
        if not text:
            continue
        previous = ordered[index - 1] if index > 0 else None
        next_caption = ordered[index + 1] if index + 1 < len(ordered) else None
        for row in (
            _opening_vocalization_candidate(caption, index, text),
            _repeated_interjection_candidate(caption, text),
            _device_prompt_candidate(caption, text),
            _short_open_clause_candidate(caption, next_caption, text),
            _previous_complete_prefix_retry_candidate(caption, previous, text),
            _standalone_classifier_head_candidate(caption, previous, text),
            _incomplete_lexical_tail_candidate(caption, next_caption, text),
            _open_coordination_tail_candidate(caption, next_caption, text),
            _local_open_recurrence_candidate(caption, previous, next_caption, text),
        ):
            if row is not None:
                candidates.append(row)
    return _dedupe(candidates)


def semantic_integrity_reason_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        reason = str(candidate.get("reason") or "semantic_integrity")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _opening_vocalization_candidate(caption: CaptionRenderUnit, index: int, text: str) -> dict[str, Any] | None:
    if index != 0:
        return None
    if not _pure_vocalization(text):
        return None
    duration_us = int(caption.target_end_us) - int(caption.target_start_us)
    if duration_us < 250_000:
        return None
    return _candidate(
        "opening_vocalization_residual",
        caption,
        caption,
        overlap_text=text,
        text=text,
        action_hint="drop_fragment",
        evidence={"duration_us": duration_us, "position": "opening"},
    )


def _repeated_interjection_candidate(caption: CaptionRenderUnit, text: str) -> dict[str, Any] | None:
    match = re.search(rf"([{''.join(INTERJECTION_CHARS)}])\1+", text)
    if not match:
        return None
    return _candidate(
        "repeated_interjection_residual",
        caption,
        caption,
        overlap_text=match.group(0),
        text=text,
        action_hint="trim_duplicate_interjection",
        evidence={"interjection": match.group(1), "repeat_text": match.group(0)},
    )


def _device_prompt_candidate(caption: CaptionRenderUnit, text: str) -> dict[str, Any] | None:
    marker_count = sum(1 for marker in DEVICE_PROMPT_MARKERS if marker in text)
    has_action = any(action in text for action in DEVICE_PROMPT_ACTIONS)
    if marker_count < 2 or not has_action:
        return None
    if _human_speech_anchor_score(text) >= 2:
        return None
    return _candidate(
        "non_primary_device_prompt_residual",
        caption,
        caption,
        overlap_text=text[: min(8, len(text))],
        text=text,
        action_hint="drop_fragment",
        evidence={"device_prompt_marker_count": marker_count, "has_device_action": has_action},
    )


def _short_open_clause_candidate(
    caption: CaptionRenderUnit,
    next_caption: CaptionRenderUnit | None,
    text: str,
) -> dict[str, Any] | None:
    if len(text) > 6:
        return None
    if not text.endswith(HARD_OPEN_CLAUSE_TAILS):
        return None
    if _pure_vocalization(text):
        return None
    if _has_immediate_predicate_continuation(caption, next_caption, text):
        return None
    return _candidate(
        "short_abandoned_open_clause",
        caption,
        caption,
        overlap_text=text,
        text=text,
        action_hint="drop_fragment",
        evidence={"tail": text[-1], "length": len(text)},
    )


def _standalone_classifier_head_candidate(
    caption: CaptionRenderUnit,
    previous: CaptionRenderUnit | None,
    text: str,
) -> dict[str, Any] | None:
    if len(text) < 3 or len(text) > 10:
        return None
    if text[0] not in STANDALONE_CLASSIFIERS:
        return None
    if text.startswith(("个人", "个位", "个体")):
        return None
    if _has_immediate_previous_context(caption, previous):
        return None
    return _candidate(
        "dependent_classifier_head_fragment",
        caption,
        caption,
        overlap_text=text[: min(4, len(text))],
        text=text,
        action_hint="semantic_recheck",
        evidence={"classifier": text[0]},
    )


def _previous_complete_prefix_retry_candidate(
    caption: CaptionRenderUnit,
    previous: CaptionRenderUnit | None,
    text: str,
) -> dict[str, Any] | None:
    if previous is None:
        return None
    connector, retry_text = _strip_restart_connector(text)
    if not connector or len(retry_text) < 4:
        return None
    previous_text = normalize_text(previous.text)
    if len(previous_text) < len(retry_text) + 1:
        return None
    if not previous_text.startswith(retry_text):
        return None
    if _pure_vocalization(previous_text) or _pure_vocalization(retry_text):
        return None
    return _candidate(
        "previous_complete_prefix_retry",
        caption,
        previous,
        overlap_text=text,
        text=text,
        action_hint="drop_retried_prefix_caption",
        evidence={"connector": connector, "retry_text": retry_text, "previous_text": previous.text},
    )


def _incomplete_lexical_tail_candidate(
    caption: CaptionRenderUnit,
    next_caption: CaptionRenderUnit | None,
    text: str,
) -> dict[str, Any] | None:
    if len(text) < 4:
        return None
    tail = text[-1]
    if not _is_cjk(tail):
        return None
    dangling_connector = dangling_discourse_connector_suffix(text)
    if dangling_connector:
        return _candidate(
            "dangling_discourse_connector_tail",
            caption,
            caption,
            overlap_text=dangling_connector,
            text=text,
            action_hint="trim_dangling_connector_tail",
            evidence={"connector": dangling_connector, "tail_context": text[-min(len(text), len(dangling_connector) + 2) :]},
        )
    dangling_connector_pronoun = dangling_discourse_pronoun_suffix(text)
    if dangling_connector_pronoun and not _has_immediate_predicate_continuation(caption, next_caption, text):
        return _candidate(
            "dangling_discourse_pronoun_tail",
            caption,
            caption,
            overlap_text=dangling_connector_pronoun,
            text=text,
            action_hint="trim_dangling_discourse_pronoun_tail",
            evidence={
                "connector_pronoun_tail": dangling_connector_pronoun,
                "tail_context": text[-min(len(text), len(dangling_connector_pronoun) + 2) :],
            },
        )
    if text[-2:] in {"可以", "所以", "因为", "但是", "然后"}:
        return None
    if text[-1] in HARD_OPEN_CLAUSE_TAILS:
        if _has_immediate_predicate_continuation(caption, next_caption, text):
            return None
        return _candidate(
            "incomplete_lexical_tail",
            caption,
            caption,
            overlap_text=tail,
            text=text,
            action_hint="semantic_recheck",
            evidence={"tail": tail, "tail_context": text[-3:]},
        )
    if tail in OPEN_SINGLE_CHAR_TAILS and _contains_amount_or_price(text[:-1]):
        return _candidate(
            "single_char_false_start_tail",
            caption,
            caption,
            overlap_text=tail,
            text=text,
            action_hint="semantic_recheck",
            evidence={"tail": tail, "amount_context": True},
        )
    if (
        re.search(r"[的地得][\u3400-\u9fff]$", text)
        and tail not in NOMINAL_PHRASE_HEAD_TAILS
        and _contains_amount_or_price(text)
    ):
        return _candidate(
            "truncated_nominal_prefix_tail",
            caption,
            caption,
            overlap_text=tail,
            text=text,
            action_hint="semantic_recheck",
            evidence={"tail": tail, "tail_context": text[-3:]},
        )
    return None


def _open_coordination_tail_candidate(
    caption: CaptionRenderUnit,
    next_caption: CaptionRenderUnit | None,
    text: str,
) -> dict[str, Any] | None:
    if next_caption is None:
        return None
    next_text = normalize_text(next_caption.text)
    if not next_text:
        return None
    for connector in OPEN_COORDINATORS:
        pos = text.rfind(connector)
        if pos < 0:
            continue
        tail = text[pos + len(connector) :]
        if not (1 <= len(tail) <= 2 and all(_is_cjk(char) for char in tail)):
            continue
        completed = _completed_coordination_tail(connector, tail, next_text)
        if not completed:
            continue
        return _candidate(
            "open_coordination_tail",
            caption,
            next_caption,
            overlap_text=f"{connector}{tail}",
            text=text,
            action_hint="drop_or_merge_with_completed_next",
            evidence={
                "connector": connector,
                "tail": tail,
                "completed_tail": completed,
                "next_text": next_caption.text,
            },
        )
    return None


def _local_open_recurrence_candidate(
    caption: CaptionRenderUnit,
    previous: CaptionRenderUnit | None,
    next_caption: CaptionRenderUnit | None,
    text: str,
) -> dict[str, Any] | None:
    if not (_has_open_tail(text) or _ends_with_short_value_fragment(text)):
        return None
    if _has_open_tail(text) and _has_immediate_predicate_continuation(caption, next_caption, text):
        return None
    for related in (previous, next_caption):
        if related is None:
            continue
        related_text = normalize_text(related.text)
        if not related_text:
            continue
        shared = _longest_common_substring(text, related_text)
        if len(shared) >= 4 and (shared in text[-max(len(shared), 8) :] or shared in related_text[: max(len(shared), 8)]):
            return _candidate(
                "local_recurrence_with_open_tail",
                caption,
                related,
                overlap_text=shared,
                text=text,
                action_hint="semantic_recheck",
                evidence={"related_text": related.text, "shared_text": shared},
            )
    return None


def _candidate(
    reason: str,
    caption: CaptionRenderUnit,
    related: CaptionRenderUnit,
    *,
    overlap_text: str,
    text: str,
    action_hint: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reason": reason,
        "type": "final_semantic_integrity_residual",
        "issue_type": reason,
        "severity": "high",
        "caption_id": caption.caption_id,
        "related_caption_id": related.caption_id,
        "target_start_us": int(caption.target_start_us),
        "target_end_us": int(caption.target_end_us),
        "related_target_start_us": int(related.target_start_us),
        "related_target_end_us": int(related.target_end_us),
        "overlap_text": overlap_text,
        "text": text,
        "semantic_integrity_recheck_required": True,
        "suggested_action": action_hint,
        "evidence": dict(evidence),
    }


def _pure_vocalization(text: str) -> bool:
    return bool(text) and len(text) <= 4 and all(char in PURE_VOCALIZATION_CHARS for char in text)


def _human_speech_anchor_score(text: str) -> int:
    anchors = ("我", "你", "他", "她", "们", "钱", "资源", "生活", "时候", "因为", "所以", "但是", "其实")
    return sum(1 for anchor in anchors if anchor in text)


def _completed_coordination_tail(connector: str, tail: str, next_text: str) -> str:
    prefix = connector + tail
    start = next_text.find(prefix)
    if start < 0:
        return ""
    end = start + len(prefix)
    while end < len(next_text) and _is_cjk(next_text[end]) and end - start < len(prefix) + 4:
        end += 1
    completed = next_text[start:end]
    return completed if len(completed) >= len(prefix) + 1 else ""


def _has_open_tail(text: str) -> bool:
    return bool(text) and text.endswith(HARD_OPEN_CLAUSE_TAILS)


def _has_immediate_previous_context(caption: CaptionRenderUnit, previous: CaptionRenderUnit | None) -> bool:
    if previous is None:
        return False
    previous_text = normalize_text(previous.text)
    if not previous_text or _pure_vocalization(previous_text):
        return False
    gap_us = int(caption.target_start_us) - int(previous.target_end_us)
    return 0 <= gap_us <= IMMEDIATE_CONTEXT_GAP_US


def _has_immediate_predicate_continuation(
    caption: CaptionRenderUnit,
    next_caption: CaptionRenderUnit | None,
    text: str,
) -> bool:
    if next_caption is None:
        return False
    next_text = normalize_text(next_caption.text)
    if not next_text or _pure_vocalization(next_text):
        return False
    gap_us = int(next_caption.target_start_us) - int(caption.target_end_us)
    if gap_us < 0 or gap_us > IMMEDIATE_CONTEXT_GAP_US:
        return False
    if text.endswith(COMPLETE_DISCOURSE_TAILS):
        return True
    if text[-1:] == "给":
        return next_text.startswith(PREDICATE_CONTINUATION_PREFIXES)
    if text[-1:] == "被":
        return _looks_like_passive_predicate_continuation(next_text)
    if text[-1:] == "是":
        return next_text.startswith(PREDICATE_CONTINUATION_PREFIXES) or _looks_like_nominal_predicate_continuation(next_text)
    return False


def _looks_like_passive_predicate_continuation(next_text: str) -> bool:
    if len(next_text) < 2:
        return False
    if next_text.startswith(PASSIVE_CONTINUATION_BLOCKED_PREFIXES):
        return False
    if not _is_cjk(next_text[0]):
        return False
    early = next_text[:8]
    return any(hint in early for hint in PASSIVE_CONTINUATION_VERB_HINTS)


def _looks_like_nominal_predicate_continuation(next_text: str) -> bool:
    if len(next_text) < 2:
        return False
    if not _is_cjk(next_text[0]):
        return False
    if next_text.startswith(NOMINAL_PREDICATE_BLOCKED_PREFIXES):
        return False
    return "的" in next_text[:8]


def _ends_with_short_value_fragment(text: str) -> bool:
    for connector in OPEN_COORDINATORS:
        pos = text.rfind(connector)
        if pos >= 0 and len(text) - pos - len(connector) == 1:
            return True
    return False


def _contains_amount_or_price(text: str) -> bool:
    return bool(re.search(r"\d", text)) or any(unit in text for unit in ("块", "元", "万元", "千元", "百元"))


def dangling_discourse_connector_suffix(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    for connector in DANGLING_DISCOURSE_CONNECTORS:
        if normalized.endswith(connector) and len(normalized) > len(connector) + 1:
            return connector
    return ""


def dangling_discourse_pronoun_suffix(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    for connector in DANGLING_DISCOURSE_CONNECTORS:
        for pronoun in DISCOURSE_PRONOUN_TAIL_PRONOUNS:
            suffix = connector + pronoun
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                return suffix
    return ""


def _strip_restart_connector(text: str) -> tuple[str, str]:
    normalized = normalize_text(text)
    for connector in RESTART_LEADING_CONNECTORS:
        if normalized.startswith(connector) and len(normalized) > len(connector) + 1:
            return connector, normalized[len(connector) :]
    return "", normalized


def _longest_common_substring(left: str, right: str) -> str:
    best = ""
    for start in range(len(left)):
        for end in range(start + 1, len(left) + 1):
            value = left[start:end]
            if len(value) > len(best) and value in right:
                best = value
    return best


def _is_cjk(char: str) -> bool:
    return len(char) == 1 and "\u3400" <= char <= "\u9fff"


def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (
            str(candidate.get("caption_id") or ""),
            str(candidate.get("related_caption_id") or ""),
            str(candidate.get("reason") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped
