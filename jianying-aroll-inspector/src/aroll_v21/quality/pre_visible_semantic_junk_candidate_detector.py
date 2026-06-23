from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit
from aroll_v21.quality.boundary_overlap import is_explanatory_term_reuse, is_semantic_label_reuse_boundary


SemanticJunkType = Literal[
    "aborted_restart",
    "adjacent_suffix_semantic_recurrence",
    "adjacent_reordered_semantic_restart",
    "partial_previous_tail",
    "low_information_fragment",
    "cross_caption_restart",
    "asr_malformed_phrase",
    "prefix_restart",
    "standalone_topic_prefix_restart",
]
SemanticJunkAction = Literal["keep", "drop_fragment", "merge_with_next", "trim_prefix", "human_review"]


MAX_ABORTED_RESTART_CHARS = 8
MAX_ABORTED_RESTART_DURATION_US = 1_400_000
MIN_REPLACEMENT_CHARS = 6
MIN_SHARED_CORE_CHARS = 3
MIN_HIGH_CONFIDENCE = 0.95
MAX_CONTEXT_CAPTIONS = 2
MAX_ADJACENT_TARGET_GAP_US = 1_200_000
MIN_SUFFIX_RESTART_SHARED_CHARS = 6
MIN_SUFFIX_RESTART_RIGHT_COVERAGE = 0.45
MAX_STANDALONE_TOPIC_CHARS = 3
MAX_REORDERED_RESTART_LEFT_CHARS = 14
MIN_REORDERED_RESTART_RIGHT_EXTRA_CHARS = 2
MIN_REORDERED_RESTART_SHARED_CHARS = 3
MIN_REORDERED_RESTART_PREFIX_OVERLAP = 0.55

ABORTED_RESTART_TAILS = (
    "第一",
    "第二",
    "第三",
    "首先",
    "其次",
    "然后",
    "所以",
    "那么",
    "就是",
    "如果",
    "因为",
    "比如",
    "那个",
    "这个",
)
DISCOURSE_OPENERS = ("那么", "然后", "所以", "但是", "因为", "就是", "其实", "如果")
SPOKEN_RESTART_OPENERS = ("我说", "你说", "他说", "她说", "咱说", "就是说", "说白了", "讲白了")
ENUMERATION_PREFIXES = ("第一", "第二", "第三", "第四", "首先", "其次", "最后")
CONTRAST_PREFIXES = ("不是", "而是", "其实", "但是", "然而")
PROGRESSIVE_PREFIXES = ("不仅", "而且", "不是", "是", "更是")
GENERIC_ACTION_PIVOT_CHARS = frozenset("把被给让使去来进出入上下载拿看说讲做找买卖花投用抢丢扔放塞拉拖推砸剪删")
VAGUE_DIRECTIONAL_TAILS = (
    "去",
    "来",
    "里",
    "上",
    "下",
    "中",
    "内",
    "外",
    "进去",
    "进来",
    "出来",
    "过去",
    "上去",
    "下来",
)
SPECIFIC_CONTEXT_STARTS = ("这", "那", "某", "一", "个", "场", "环境", "地方", "里面", "里", "上", "下")


@dataclass(frozen=True)
class SemanticJunkCandidate:
    candidate_id: str
    type: SemanticJunkType
    target_caption_ids: list[str]
    target_segment_ids: list[str]
    target_word_ids: list[str]
    source_start_us: int
    source_end_us: int
    visible_text: str
    native_words_text: str
    previous_context: list[str]
    next_context: list[str]
    proposed_action: SemanticJunkAction
    local_confidence: float
    evidence: dict[str, Any]
    provider_required: bool
    safety_tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pre_visible_semantic_junk_candidate_report(
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    *,
    mode: str = "production",
) -> dict[str, Any]:
    candidates = detect_pre_visible_semantic_junk_candidates(captions, source_graph)
    actionable = [
        row
        for row in candidates
        if row.proposed_action == "drop_fragment"
        and row.local_confidence >= MIN_HIGH_CONFIDENCE
        and not row.provider_required
    ]
    protected_count = sum(1 for row in candidates if row.proposed_action == "keep")
    ambiguous_count = sum(1 for row in candidates if row.provider_required or row.proposed_action == "human_review")
    return {
        "detector_name": "pre_visible_semantic_junk_candidate_detector",
        "pre_visible_semantic_junk_enabled": True,
        "pre_visible_semantic_junk_candidate_detector_enabled": True,
        "pre_visible_semantic_junk_audit_only": True,
        "pre_visible_semantic_junk_timeline_mutation_allowed": False,
        "pre_visible_semantic_junk_mode": str(mode or "production"),
        "pre_visible_semantic_junk_candidate_count": len(candidates),
        "pre_visible_semantic_junk_high_confidence_candidate_count": len(actionable),
        "pre_visible_semantic_junk_protected_count": protected_count,
        "pre_visible_semantic_junk_ambiguous_count": ambiguous_count,
        "pre_visible_semantic_junk_candidates": [row.to_dict() for row in candidates],
        "pre_visible_semantic_junk_high_confidence_candidate_ids": [row.candidate_id for row in actionable],
        "pre_visible_semantic_junk_unresolved_candidate_ids": [
            row.candidate_id for row in candidates if row.provider_required or row.proposed_action == "human_review"
        ],
    }


def detect_pre_visible_semantic_junk_candidates(
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
) -> list[SemanticJunkCandidate]:
    ordered = sorted(captions, key=lambda row: (int(row.target_start_us), int(row.target_end_us), row.caption_id))
    candidates: list[SemanticJunkCandidate] = []
    for index, caption in enumerate(ordered):
        if index == 0:
            continue
        previous_rows = ordered[max(0, index - MAX_CONTEXT_CAPTIONS) : index]
        next_rows = ordered[index + 1 : min(len(ordered), index + 1 + MAX_CONTEXT_CAPTIONS)]
        builders = [
            _standalone_topic_prefix_restart_candidate,
            _adjacent_suffix_semantic_recurrence_candidate,
            _adjacent_reordered_semantic_restart_candidate,
            _aborted_restart_candidate,
        ]
        for builder in builders:
            candidate = builder(
                ordered,
                index,
                previous_rows=previous_rows,
                next_rows=next_rows,
                source_graph=source_graph,
            )
            if candidate is not None:
                candidates.append(candidate)
                break
    return candidates


def _aborted_restart_candidate(
    ordered: list[CaptionRenderUnit],
    index: int,
    *,
    previous_rows: list[CaptionRenderUnit],
    next_rows: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
) -> SemanticJunkCandidate | None:
    if index + 1 >= len(ordered):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    caption = ordered[index]
    next_caption = ordered[index + 1]
    text = normalize_text(caption.text)
    next_text = normalize_text(next_caption.text)
    if not _looks_like_short_fragment(caption, text):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if len(next_text) < MIN_REPLACEMENT_CHARS:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    shared = _longest_common_substring(text, next_text)
    if len(shared) < MIN_SHARED_CORE_CHARS:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if _is_protected_semantic_structure(ordered, index, shared):
        return _candidate(
            ordered,
            index,
            source_graph=source_graph,
            candidate_type="aborted_restart",
            proposed_action="keep",
            local_confidence=0.0,
            evidence={
                "reason": "protected_semantic_structure",
                "shared_core_text": shared,
                "next_caption_text": next_caption.text,
            },
            provider_required=False,
            safety_tags=["protected_semantic_structure"],
            previous_rows=previous_rows,
            next_rows=next_rows,
        )
    if not _fragment_has_open_tail(text) and not _next_restarts_fragment_core(text, next_text, shared):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    confidence = _aborted_restart_confidence(text, next_text, shared, caption)
    provider_required = confidence < MIN_HIGH_CONFIDENCE
    action: SemanticJunkAction = "drop_fragment" if confidence >= MIN_HIGH_CONFIDENCE else "human_review"
    return _candidate(
        ordered,
        index,
        source_graph=source_graph,
        candidate_type="aborted_restart",
        proposed_action=action,
        local_confidence=confidence,
        evidence={
            "reason": "short fragment is reopened by the next caption with a longer coherent continuation",
            "shared_core_text": shared,
            "shared_core_chars": len(shared),
            "fragment_chars": len(text),
            "next_caption_chars": len(next_text),
            "fragment_has_open_tail": _fragment_has_open_tail(text),
            "next_restarts_fragment_core": _next_restarts_fragment_core(text, next_text, shared),
            "next_caption_text": next_caption.text,
        },
        provider_required=provider_required,
        safety_tags=["no_rewrite", "drop_audio_and_caption_together"],
        previous_rows=previous_rows,
        next_rows=next_rows,
    )


def _standalone_topic_prefix_restart_candidate(
    ordered: list[CaptionRenderUnit],
    index: int,
    *,
    previous_rows: list[CaptionRenderUnit],
    next_rows: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
) -> SemanticJunkCandidate | None:
    if index + 1 >= len(ordered):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    caption = ordered[index]
    next_caption = ordered[index + 1]
    text = normalize_text(caption.text)
    next_text = normalize_text(next_caption.text)
    if not _looks_like_standalone_topic_fragment(caption, text):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if not _target_adjacent(caption, next_caption):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if len(next_text) < len(text) + 4:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    stripped_next = _strip_spoken_restart_opener(next_text)
    topic_index = stripped_next.find(text)
    if topic_index < 0 or topic_index > 2:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if _is_protected_semantic_structure(ordered, index, text):
        return _candidate(
            ordered,
            index,
            source_graph=source_graph,
            candidate_type="standalone_topic_prefix_restart",
            proposed_action="keep",
            local_confidence=0.0,
            evidence={
                "reason": "protected_semantic_structure",
                "shared_core_text": text,
                "next_caption_text": next_caption.text,
            },
            provider_required=False,
            safety_tags=["protected_semantic_structure"],
            previous_rows=previous_rows,
            next_rows=next_rows,
        )
    return _candidate(
        ordered,
        index,
        source_graph=source_graph,
        candidate_type="standalone_topic_prefix_restart",
        proposed_action="drop_fragment",
        local_confidence=0.97,
        evidence={
            "reason": "standalone short topic is immediately restated inside the next longer caption",
            "shared_core_text": text,
            "next_caption_text": next_caption.text,
            "stripped_next_caption_text": stripped_next,
            "topic_index_after_spoken_opener": topic_index,
            "target_gap_us": int(next_caption.target_start_us) - int(caption.target_end_us),
        },
        provider_required=False,
        safety_tags=["no_rewrite", "drop_audio_and_caption_together", "adjacent_standalone_topic_restart"],
        previous_rows=previous_rows,
        next_rows=next_rows,
    )


def _adjacent_suffix_semantic_recurrence_candidate(
    ordered: list[CaptionRenderUnit],
    index: int,
    *,
    previous_rows: list[CaptionRenderUnit],
    next_rows: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
) -> SemanticJunkCandidate | None:
    caption = ordered[index]
    previous_caption = ordered[index - 1] if index > 0 else None
    if previous_caption is None:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    previous_text = normalize_text(previous_caption.text)
    text = normalize_text(caption.text)
    if not previous_text or not text or len(text) >= len(previous_text):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if not _target_adjacent(previous_caption, caption):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    shared_suffix = _longest_common_suffix(previous_text, text)
    if _cjk_char_count(shared_suffix) < MIN_SUFFIX_RESTART_SHARED_CHARS:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    right_coverage = len(shared_suffix) / max(1, len(text))
    if right_coverage < MIN_SUFFIX_RESTART_RIGHT_COVERAGE:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if _is_protected_suffix_recurrence(previous_text, text):
        return _candidate(
            ordered,
            index,
            source_graph=source_graph,
            candidate_type="adjacent_suffix_semantic_recurrence",
            proposed_action="keep",
            local_confidence=0.0,
            evidence={
                "reason": "protected_semantic_structure",
                "shared_suffix_text": shared_suffix,
                "previous_caption_text": previous_caption.text,
            },
            provider_required=False,
            safety_tags=["protected_semantic_structure"],
            previous_rows=previous_rows,
            next_rows=next_rows,
        )
    return _candidate(
        ordered,
        index,
        source_graph=source_graph,
        candidate_type="adjacent_suffix_semantic_recurrence",
        proposed_action="drop_fragment",
        local_confidence=0.96,
        evidence={
            "reason": "adjacent shorter caption repeats the previous caption suffix without adding a new semantic branch",
            "shared_suffix_text": shared_suffix,
            "shared_suffix_chars": len(shared_suffix),
            "right_coverage": round(right_coverage, 6),
            "previous_caption_text": previous_caption.text,
            "target_gap_us": int(caption.target_start_us) - int(previous_caption.target_end_us),
        },
        provider_required=False,
        safety_tags=["no_rewrite", "drop_audio_and_caption_together", "adjacent_suffix_recurrence"],
        previous_rows=previous_rows,
        next_rows=next_rows,
    )


def _adjacent_reordered_semantic_restart_candidate(
    ordered: list[CaptionRenderUnit],
    index: int,
    *,
    previous_rows: list[CaptionRenderUnit],
    next_rows: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
) -> SemanticJunkCandidate | None:
    if index + 1 >= len(ordered):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    caption = ordered[index]
    next_caption = ordered[index + 1]
    text = normalize_text(caption.text)
    next_text = normalize_text(next_caption.text)
    if not text or not next_text or text == next_text:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if not _target_adjacent(caption, next_caption):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if len(text) > MAX_REORDERED_RESTART_LEFT_CHARS:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if len(next_text) < len(text) + MIN_REORDERED_RESTART_RIGHT_EXTRA_CHARS:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    shared_row = _longest_common_substring_detail(text, next_text)
    shared = str(shared_row.get("text") or "")
    if _cjk_char_count(shared) < MIN_REORDERED_RESTART_SHARED_CHARS:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if not _contains_action_pivot(shared):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    left_start = int(shared_row.get("left_start") or 0)
    right_start = int(shared_row.get("right_start") or 0)
    left_prefix = text[:left_start]
    right_prefix = next_text[:right_start]
    left_tail = text[left_start + len(shared) :]
    right_tail = next_text[right_start + len(shared) :]
    if not _looks_like_vague_reordered_restart_tail(left_tail, right_tail):
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    prefix_overlap = _char_overlap_ratio(left_prefix, right_prefix)
    if prefix_overlap < MIN_REORDERED_RESTART_PREFIX_OVERLAP:
        no_candidate: SemanticJunkCandidate | None = None
        return no_candidate
    if _has_contrastive_subject_markers(left_prefix, right_prefix):
        return _candidate(
            ordered,
            index,
            source_graph=source_graph,
            candidate_type="adjacent_reordered_semantic_restart",
            proposed_action="keep",
            local_confidence=0.0,
            evidence={
                "reason": "protected_contrastive_subject_structure",
                "shared_core_text": shared,
                "next_caption_text": next_caption.text,
                "left_prefix": left_prefix,
                "right_prefix": right_prefix,
            },
            provider_required=False,
            safety_tags=["protected_semantic_structure"],
            previous_rows=previous_rows,
            next_rows=next_rows,
        )
    if _is_protected_reordered_restart_structure(ordered, index, shared):
        return _candidate(
            ordered,
            index,
            source_graph=source_graph,
            candidate_type="adjacent_reordered_semantic_restart",
            proposed_action="keep",
            local_confidence=0.0,
            evidence={
                "reason": "protected_semantic_structure",
                "shared_core_text": shared,
                "next_caption_text": next_caption.text,
            },
            provider_required=False,
            safety_tags=["protected_semantic_structure"],
            previous_rows=previous_rows,
            next_rows=next_rows,
        )
    return _candidate(
        ordered,
        index,
        source_graph=source_graph,
        candidate_type="adjacent_reordered_semantic_restart",
        proposed_action="drop_fragment",
        local_confidence=0.96,
        evidence={
            "reason": "adjacent left caption restarts into the next longer caption with reordered prefix and a more specific continuation",
            "shared_core_text": shared,
            "left_prefix": left_prefix,
            "right_prefix": right_prefix,
            "left_tail": left_tail,
            "right_tail": right_tail,
            "prefix_overlap": round(prefix_overlap, 6),
            "next_caption_text": next_caption.text,
            "target_gap_us": int(next_caption.target_start_us) - int(caption.target_end_us),
        },
        provider_required=False,
        safety_tags=["no_rewrite", "drop_audio_and_caption_together", "adjacent_reordered_semantic_restart"],
        previous_rows=previous_rows,
        next_rows=next_rows,
    )


def _candidate(
    ordered: list[CaptionRenderUnit],
    index: int,
    *,
    source_graph: CanonicalSourceGraph,
    candidate_type: SemanticJunkType,
    proposed_action: SemanticJunkAction,
    local_confidence: float,
    evidence: dict[str, Any],
    provider_required: bool,
    safety_tags: list[str],
    previous_rows: list[CaptionRenderUnit],
    next_rows: list[CaptionRenderUnit],
) -> SemanticJunkCandidate:
    caption = ordered[index]
    word_text = _text_from_word_ids(caption.word_ids, source_graph) or str(caption.text or "")
    source_range = _caption_source_range(caption, source_graph) or (0, 0)
    return SemanticJunkCandidate(
        candidate_id=f"pre_visible_semantic_junk_candidate_{index + 1:06d}_{candidate_type}",
        type=candidate_type,
        target_caption_ids=[caption.caption_id],
        target_segment_ids=list(caption.timeline_segment_ids),
        target_word_ids=list(caption.word_ids),
        source_start_us=int(source_range[0]),
        source_end_us=int(source_range[1]),
        visible_text=str(caption.text or ""),
        native_words_text=word_text,
        previous_context=[str(row.text or "") for row in previous_rows],
        next_context=[str(row.text or "") for row in next_rows],
        proposed_action=proposed_action,
        local_confidence=round(float(local_confidence), 6),
        evidence=evidence,
        provider_required=provider_required,
        safety_tags=list(safety_tags),
    )


def _looks_like_short_fragment(caption: CaptionRenderUnit, text: str) -> bool:
    if not text or len(text) > MAX_ABORTED_RESTART_CHARS:
        return False
    if not any("\u4e00" <= char <= "\u9fff" for char in text):
        return False
    duration_us = int(caption.target_end_us) - int(caption.target_start_us)
    return duration_us <= 0 or duration_us <= MAX_ABORTED_RESTART_DURATION_US


def _looks_like_standalone_topic_fragment(caption: CaptionRenderUnit, text: str) -> bool:
    if not text or len(text) > MAX_STANDALONE_TOPIC_CHARS:
        return False
    if _cjk_char_count(text) != len(text) or len(text) < 2:
        return False
    duration_us = int(caption.target_end_us) - int(caption.target_start_us)
    return duration_us <= 0 or duration_us <= MAX_ABORTED_RESTART_DURATION_US


def _fragment_has_open_tail(text: str) -> bool:
    return any(text.endswith(tail) for tail in ABORTED_RESTART_TAILS)


def _next_restarts_fragment_core(text: str, next_text: str, shared: str) -> bool:
    if not shared or len(shared) < MIN_SHARED_CORE_CHARS:
        return False
    stripped_next = next_text
    for opener in DISCOURSE_OPENERS:
        if stripped_next.startswith(opener):
            stripped_next = stripped_next[len(opener) :]
            break
    if stripped_next.startswith(shared):
        return True
    return text.startswith(shared) and shared in stripped_next[: max(len(shared) + 4, 8)]


def _aborted_restart_confidence(text: str, next_text: str, shared: str, caption: CaptionRenderUnit) -> float:
    score = 0.52
    score += min(0.22, len(shared) / max(1, len(text)) * 0.22)
    if _fragment_has_open_tail(text):
        score += 0.14
    if _next_restarts_fragment_core(text, next_text, shared):
        score += 0.12
    if len(next_text) >= len(text) + 3:
        score += 0.08
    duration_us = int(caption.target_end_us) - int(caption.target_start_us)
    if duration_us > 0 and duration_us <= 900_000:
        score += 0.04
    return min(0.99, score)


def _target_adjacent(left: CaptionRenderUnit, right: CaptionRenderUnit) -> bool:
    gap_us = int(right.target_start_us) - int(left.target_end_us)
    return -80_000 <= gap_us <= MAX_ADJACENT_TARGET_GAP_US


def _strip_spoken_restart_opener(text: str) -> str:
    stripped = text
    for opener in SPOKEN_RESTART_OPENERS:
        if stripped.startswith(opener) and len(stripped) > len(opener) + 1:
            return stripped[len(opener) :]
    return stripped


def _is_protected_suffix_recurrence(previous_text: str, text: str) -> bool:
    if is_semantic_label_reuse_boundary(previous_text, text, _longest_common_suffix(previous_text, text)):
        return True
    if is_explanatory_term_reuse(previous_text, text) or is_explanatory_term_reuse(text, previous_text):
        return True
    if any(text.startswith(prefix) for prefix in ENUMERATION_PREFIXES + CONTRAST_PREFIXES + PROGRESSIVE_PREFIXES):
        return True
    return False


def _looks_like_vague_reordered_restart_tail(left_tail: str, right_tail: str) -> bool:
    if not left_tail or len(left_tail) > 2:
        return False
    if not right_tail or len(right_tail) < len(left_tail) + 2:
        return False
    if left_tail in VAGUE_DIRECTIONAL_TAILS:
        return True
    if any(right_tail.startswith(prefix) for prefix in SPECIFIC_CONTEXT_STARTS):
        return True
    return False


def _contains_action_pivot(text: str) -> bool:
    return any(char in GENERIC_ACTION_PIVOT_CHARS for char in text)


def _char_overlap_ratio(left: str, right: str) -> float:
    left_chars = {char for char in left if "\u4e00" <= char <= "\u9fff"}
    right_chars = {char for char in right if "\u4e00" <= char <= "\u9fff"}
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / max(1, min(len(left_chars), len(right_chars)))


def _has_contrastive_subject_markers(left: str, right: str) -> bool:
    if ("男" in left and "女" in right) or ("女" in left and "男" in right):
        return True
    if ("他" in left and "她" in right) or ("她" in left and "他" in right):
        return True
    return False


def _is_protected_reordered_restart_structure(ordered: list[CaptionRenderUnit], index: int, shared: str) -> bool:
    caption = ordered[index]
    next_caption = ordered[index + 1] if index + 1 < len(ordered) else None
    previous = ordered[index - 1] if index > 0 else None
    text = normalize_text(caption.text)
    next_text = normalize_text(next_caption.text if next_caption is not None else "")
    previous_text = normalize_text(previous.text if previous is not None else "")
    if not text or not next_text:
        return True
    if is_semantic_label_reuse_boundary(text, next_text, shared):
        return True
    if is_explanatory_term_reuse(text, next_text) or is_explanatory_term_reuse(next_text, text):
        return True
    if _looks_like_enumeration(text, previous_text, next_text):
        return True
    return False


def _is_protected_semantic_structure(ordered: list[CaptionRenderUnit], index: int, shared: str) -> bool:
    caption = ordered[index]
    next_caption = ordered[index + 1] if index + 1 < len(ordered) else None
    previous = ordered[index - 1] if index > 0 else None
    text = normalize_text(caption.text)
    next_text = normalize_text(next_caption.text if next_caption is not None else "")
    previous_text = normalize_text(previous.text if previous is not None else "")
    if not text or not next_text:
        return True
    overlap = text if next_text.startswith(text) else shared
    if is_semantic_label_reuse_boundary(text, next_text, overlap):
        return True
    if is_explanatory_term_reuse(text, next_text) or is_explanatory_term_reuse(next_text, text):
        return True
    if _looks_like_enumeration(text, previous_text, next_text):
        return True
    if _looks_like_contrast_or_progression(text, next_text):
        return True
    if _looks_like_parallel_scan(text, next_text, shared):
        return True
    return False


def _looks_like_enumeration(text: str, previous_text: str, next_text: str) -> bool:
    if any(text.startswith(prefix) for prefix in ENUMERATION_PREFIXES):
        return True
    if text.endswith("第一") and any(next_text.startswith(prefix) for prefix in ("第二", "其次")):
        return True
    if text.endswith("第二") and (previous_text.endswith("第一") or next_text.startswith("第三")):
        return True
    return False


def _looks_like_contrast_or_progression(text: str, next_text: str) -> bool:
    return any(text.startswith(prefix) for prefix in CONTRAST_PREFIXES) or any(
        next_text.startswith(prefix) for prefix in PROGRESSIVE_PREFIXES
    )


def _looks_like_parallel_scan(text: str, next_text: str, shared: str) -> bool:
    if len(text) < 5 or len(next_text) < 5 or len(shared) < MIN_SHARED_CORE_CHARS:
        return False
    left_index = text.find(shared)
    right_index = next_text.find(shared)
    if left_index < 0 or right_index < 0:
        return False
    left_tail = text[left_index + len(shared) :]
    right_tail = next_text[right_index + len(shared) :]
    if not left_tail or not right_tail:
        return False
    return abs(len(left_tail) - len(right_tail)) <= 3 and not _fragment_has_open_tail(text)


def _longest_common_substring(left: str, right: str) -> str:
    return str(_longest_common_substring_detail(left, right).get("text") or "")


def _longest_common_substring_detail(left: str, right: str) -> dict[str, Any]:
    best = ""
    best_left_start = 0
    best_right_start = 0
    for start in range(len(left)):
        for end in range(start + 1, len(left) + 1):
            candidate = left[start:end]
            if len(candidate) <= len(best):
                continue
            right_start = right.find(candidate)
            if right_start >= 0:
                best = candidate
                best_left_start = start
                best_right_start = right_start
    return {
        "text": best,
        "left_start": best_left_start,
        "right_start": best_right_start,
    }


def _longest_common_suffix(left: str, right: str) -> str:
    count = 0
    for left_char, right_char in zip(reversed(left), reversed(right)):
        if left_char != right_char:
            break
        count += 1
    if count <= 0:
        return ""
    return left[len(left) - count :]


def _cjk_char_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _caption_source_range(caption: CaptionRenderUnit, source_graph: CanonicalSourceGraph) -> tuple[int, int] | None:
    words_by_id = {word.word_id: word for word in source_graph.words}
    words = [words_by_id[word_id] for word_id in caption.word_ids if word_id in words_by_id]
    if not words:
        no_range: tuple[int, int] | None = None
        return no_range
    return (
        min(int(getattr(word, "source_start_us", 0) or 0) for word in words),
        max(int(getattr(word, "source_end_us", 0) or 0) for word in words),
    )


def _text_from_word_ids(word_ids: list[str], source_graph: CanonicalSourceGraph) -> str:
    words_by_id = {word.word_id: word for word in source_graph.words}
    return "".join(str(getattr(words_by_id[word_id], "text", "") or "") for word_id in word_ids if word_id in words_by_id)
