from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, FinalTimelineSegment
from aroll_v21.quality.final_visible_repair.proposal import TimelineRepairProposal
from aroll_v21.quality.final_visible_repair.timeline_utils import ordered_segments, text_from_word_ids


MIN_REPEATED_ISLAND_CJK_CHARS = 2
MAX_REPEATED_ISLAND_CJK_CHARS = 4
MAX_REPEATED_ISLAND_WORD_COUNT = 3
MIN_HIGH_CONFIDENCE_MIDDLE_CJK_CHARS = 2
MAX_HIGH_CONFIDENCE_MIDDLE_CJK_CHARS = 6
MIN_AFTER_SECOND_CJK_CHARS = 2
RESTART_PIVOT_SUFFIX_CHARS = frozenset("是有在要能会想把让给")
NEGATION_MIDDLE_TEXTS = frozenset({"不", "没", "没有", "别"})
LEADING_DISCOURSE_RESTART_PARTICLES = frozenset({"那", "这", "还", "就", "再", "又", "也", "都", "才"})
DEFINITION_OR_EMPHASIS_MARKERS = (
    "就是",
    "叫做",
    "称为",
    "指的是",
    "意味着",
    "等于",
)


@dataclass(frozen=True)
class RepeatedIslandCandidate:
    segment_id: str
    island_text: str
    first_word_ids: list[str]
    middle_word_ids: list[str]
    second_word_ids: list[str]
    after_second_word_ids: list[str]
    middle_text: str
    after_second_text: str
    island_char_count: int
    confidence: str
    risk_tags: list[str]
    reason: str

    def to_evidence(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "island_text": self.island_text,
            "first_word_ids": list(self.first_word_ids),
            "middle_word_ids": list(self.middle_word_ids),
            "second_word_ids": list(self.second_word_ids),
            "after_second_word_ids": list(self.after_second_word_ids),
            "middle_text": self.middle_text,
            "after_second_text": self.after_second_text,
            "island_char_count": int(self.island_char_count),
            "confidence": self.confidence,
            "risk_tags": list(self.risk_tags),
            "reason": self.reason,
        }


def detect_repeated_island_candidates(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> list[RepeatedIslandCandidate]:
    candidates: list[RepeatedIslandCandidate] = []
    for segment in ordered_segments(final_timeline):
        candidates.extend(_candidates_for_segment(segment, source_graph))
    return candidates


def build_repeated_island_proposals(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> list[TimelineRepairProposal]:
    proposals: list[TimelineRepairProposal] = []
    high_candidates = [
        candidate
        for candidate in detect_repeated_island_candidates(final_timeline, source_graph)
        if candidate.confidence == "high"
    ]
    for index, candidate in enumerate(high_candidates, start=1):
        target_word_ids = _proposal_target_word_ids(candidate, source_graph)
        target_text = text_from_word_ids(target_word_ids, source_graph)
        source_boundary_restart = len(target_word_ids) > len(candidate.first_word_ids)
        evidence = candidate.to_evidence()
        proposals.append(
            TimelineRepairProposal(
                proposal_id=f"repeated_island_{index:06d}_{candidate.segment_id}",
                issue_type="repeated_island",
                confidence=0.93,
                target_segment_id=candidate.segment_id,
                target_word_ids=target_word_ids,
                target_source_start_us=_word_source_start_us(candidate.first_word_ids, source_graph),
                target_source_end_us=_word_source_end_us(target_word_ids, source_graph),
                target_text=target_text,
                repair_action="internal_drop",
                risk_tags=[
                    "same_segment_repeated_island",
                    "whole_word_internal_drop",
                    *(["source_boundary_abandoned_middle_fragment"] if source_boundary_restart else []),
                ],
                evidence={
                    **evidence,
                    "proposal_policy": (
                        "drop_first_island_and_middle_fragment_keep_source_boundary_restart"
                        if source_boundary_restart
                        else "drop_first_island_keep_second_island_and_after_text"
                    ),
                },
            )
        )
    proposals.extend(_build_abandoned_middle_fragment_residual_proposals(final_timeline, source_graph, start_index=len(proposals) + 1))
    return proposals


def _candidates_for_segment(
    segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
) -> list[RepeatedIslandCandidate]:
    segment_word_ids = [str(word_id) for word_id in segment.word_ids if str(word_id)]
    if len(segment_word_ids) < 4:
        empty: list[RepeatedIslandCandidate] = []
        return empty
    words_by_id = {word.word_id: word for word in source_graph.words}
    if any(word_id not in words_by_id for word_id in segment_word_ids):
        empty: list[RepeatedIslandCandidate] = []
        return empty
    candidates: list[RepeatedIslandCandidate] = []
    seen_first_spans: set[tuple[int, int]] = set()
    max_width = min(MAX_REPEATED_ISLAND_WORD_COUNT, max(1, len(segment_word_ids) // 3))
    for width in range(max_width, 0, -1):
        for first_start in range(0, len(segment_word_ids) - (width * 2)):
            first_end = first_start + width
            if (first_start, first_end) in seen_first_spans:
                continue
            first_ids = segment_word_ids[first_start:first_end]
            island_text = text_from_word_ids(first_ids, source_graph)
            island_key = normalize_text(island_text)
            if not island_text:
                continue
            island_char_count = _cjk_char_count(island_text)
            for second_start in range(first_end + 1, len(segment_word_ids) - width + 1):
                second_end = second_start + width
                if second_end >= len(segment_word_ids):
                    continue
                second_ids = segment_word_ids[second_start:second_end]
                second_key = normalize_text(text_from_word_ids(second_ids, source_graph))
                if island_key != second_key:
                    continue
                middle_ids = segment_word_ids[first_end:second_start]
                after_ids = segment_word_ids[second_end:]
                middle_text = text_from_word_ids(middle_ids, source_graph)
                after_second_text = text_from_word_ids(after_ids, source_graph)
                if not middle_text or _cjk_char_count(after_second_text) < 1:
                    continue
                candidate = _classify_candidate(
                    segment_id=segment.segment_id,
                    island_text=island_text,
                    first_word_ids=first_ids,
                    middle_word_ids=middle_ids,
                    second_word_ids=second_ids,
                    after_second_word_ids=after_ids,
                    middle_text=middle_text,
                    after_second_text=after_second_text,
                    island_char_count=island_char_count,
                    starts_at_segment_start=first_start == 0,
                )
                candidates.append(candidate)
                seen_first_spans.add((first_start, first_end))
                break
    return _dedupe_candidates(candidates)


def _classify_candidate(
    *,
    segment_id: str,
    island_text: str,
    first_word_ids: list[str],
    middle_word_ids: list[str],
    second_word_ids: list[str],
    after_second_word_ids: list[str],
    middle_text: str,
    after_second_text: str,
    island_char_count: int,
    starts_at_segment_start: bool = False,
) -> RepeatedIslandCandidate:
    risk_tags = _risk_tags(
        island_text=island_text,
        middle_text=middle_text,
        after_second_text=after_second_text,
        island_char_count=island_char_count,
    )
    middle_cjk_count = _cjk_char_count(middle_text)
    after_cjk_count = _cjk_char_count(after_second_text)
    if "a_not_a_structure" in risk_tags:
        confidence = "low"
        reason = "protected_a_not_a_structure"
    elif _is_leading_discourse_particle_restart(
        island_text=island_text,
        middle_text=middle_text,
        after_second_text=after_second_text,
        starts_at_segment_start=starts_at_segment_start,
    ):
        confidence = "high"
        reason = "leading discourse particle A-B-A restart keeps the second particle pair and continuation"
    elif "single_char_island" in risk_tags:
        confidence = "low"
        reason = "single_character_repeat_is_not_safe_for_auto_repair"
    elif "definition_or_emphasis_structure" in risk_tags:
        confidence = "medium"
        reason = "definition_or_emphasis_like_repeat_requires_review"
    elif (
        MIN_REPEATED_ISLAND_CJK_CHARS <= island_char_count <= MAX_REPEATED_ISLAND_CJK_CHARS
        and middle_cjk_count >= MIN_HIGH_CONFIDENCE_MIDDLE_CJK_CHARS
        and middle_cjk_count <= MAX_HIGH_CONFIDENCE_MIDDLE_CJK_CHARS
        and after_cjk_count >= MIN_AFTER_SECOND_CJK_CHARS
        and _has_restart_pivot(island_text)
    ):
        confidence = "high"
        reason = "first short repeated island is followed by a fragment and the second island continues the expression"
    else:
        confidence = "medium"
        reason = "same_segment_repeated_island_requires_review"
    return RepeatedIslandCandidate(
        segment_id=segment_id,
        island_text=island_text,
        first_word_ids=list(first_word_ids),
        middle_word_ids=list(middle_word_ids),
        second_word_ids=list(second_word_ids),
        after_second_word_ids=list(after_second_word_ids),
        middle_text=middle_text,
        after_second_text=after_second_text,
        island_char_count=island_char_count,
        confidence=confidence,
        risk_tags=risk_tags,
        reason=reason,
    )


def _risk_tags(
    *,
    island_text: str,
    middle_text: str,
    after_second_text: str,
    island_char_count: int,
) -> list[str]:
    tags: list[str] = []
    if island_char_count <= 1:
        tags.append("single_char_island")
    if island_char_count > MAX_REPEATED_ISLAND_CJK_CHARS:
        tags.append("long_island")
    if middle_text in NEGATION_MIDDLE_TEXTS:
        tags.append("a_not_a_structure")
    if _contains_definition_marker(middle_text) or _contains_definition_marker(after_second_text):
        tags.append("definition_or_emphasis_structure")
    if not _has_restart_pivot(island_text):
        tags.append("no_restart_pivot")
    return tags


def _is_leading_discourse_particle_restart(
    *,
    island_text: str,
    middle_text: str,
    after_second_text: str,
    starts_at_segment_start: bool,
) -> bool:
    island = normalize_text(island_text)
    middle = normalize_text(middle_text)
    after = normalize_text(after_second_text)
    if not starts_at_segment_start:
        return False
    if _cjk_char_count(island) != 1 or _cjk_char_count(middle) != 1:
        return False
    if island not in LEADING_DISCOURSE_RESTART_PARTICLES or middle not in LEADING_DISCOURSE_RESTART_PARTICLES:
        return False
    if island == middle:
        return False
    if _cjk_char_count(after) < MIN_AFTER_SECOND_CJK_CHARS:
        return False
    if middle in NEGATION_MIDDLE_TEXTS:
        return False
    if _contains_definition_marker(after):
        return False
    return True


def _dedupe_candidates(candidates: list[RepeatedIslandCandidate]) -> list[RepeatedIslandCandidate]:
    deduped: list[RepeatedIslandCandidate] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    for candidate in sorted(
        candidates,
        key=lambda row: (
            confidence_order.get(row.confidence, 3),
            -int(row.island_char_count),
            row.segment_id,
            row.first_word_ids,
        ),
    ):
        key = (candidate.segment_id, tuple(candidate.first_word_ids), tuple(candidate.second_word_ids))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _proposal_target_word_ids(
    candidate: RepeatedIslandCandidate,
    source_graph: CanonicalSourceGraph,
) -> list[str]:
    if _middle_fragment_abandoned_by_source_boundary_restart(candidate, source_graph):
        return [*candidate.first_word_ids, *candidate.middle_word_ids]
    return list(candidate.first_word_ids)


def _middle_fragment_abandoned_by_source_boundary_restart(
    candidate: RepeatedIslandCandidate,
    source_graph: CanonicalSourceGraph,
) -> bool:
    if not candidate.middle_word_ids or not candidate.second_word_ids or not candidate.after_second_word_ids:
        return False
    words_by_id = {word.word_id: word for word in source_graph.words}
    left_ids = [*candidate.first_word_ids, *candidate.middle_word_ids]
    right_ids = [*candidate.second_word_ids, *candidate.after_second_word_ids]
    left_uids = [_subtitle_uid(words_by_id.get(word_id)) for word_id in left_ids]
    right_uids = [_subtitle_uid(words_by_id.get(word_id)) for word_id in right_ids]
    if any(not uid for uid in left_uids) or any(not uid for uid in right_uids):
        return False
    left_uid = left_uids[0]
    right_uid = right_uids[0]
    if left_uid == right_uid:
        return False
    return all(uid == left_uid for uid in left_uids) and all(uid == right_uid for uid in right_uids)


def _build_abandoned_middle_fragment_residual_proposals(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    *,
    start_index: int,
) -> list[TimelineRepairProposal]:
    proposals: list[TimelineRepairProposal] = []
    used_word_ids = {str(word_id) for segment in final_timeline for word_id in segment.word_ids if str(word_id)}
    words_by_id = {word.word_id: word for word in source_graph.words}
    source_index = {word.word_id: index for index, word in enumerate(source_graph.words)}
    for segment in ordered_segments(final_timeline):
        candidate = _abandoned_middle_fragment_residual_candidate(segment, source_graph, words_by_id, source_index, used_word_ids)
        if candidate is None:
            continue
        proposal_index = start_index + len(proposals)
        target_word_ids = list(candidate["target_word_ids"])
        proposals.append(
            TimelineRepairProposal(
                proposal_id=f"repeated_island_residual_{proposal_index:06d}_{segment.segment_id}",
                issue_type="repeated_island",
                confidence=0.91,
                target_segment_id=segment.segment_id,
                target_word_ids=target_word_ids,
                target_source_start_us=_word_source_start_us(target_word_ids, source_graph),
                target_source_end_us=_word_source_end_us(target_word_ids, source_graph),
                target_text=text_from_word_ids(target_word_ids, source_graph),
                repair_action="trim",
                risk_tags=["same_segment_repeated_island", "abandoned_middle_fragment_residual", "whole_word_prefix_trim"],
                evidence={
                    **candidate,
                    "proposal_policy": "trim_abandoned_middle_fragment_left_after_prior_restart_island_drop",
                },
            )
        )
    return proposals


def _abandoned_middle_fragment_residual_candidate(
    segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    words_by_id: dict[str, Any],
    source_index: dict[str, int],
    used_word_ids: set[str],
) -> dict[str, Any] | None:
    segment_word_ids = [str(word_id) for word_id in segment.word_ids if str(word_id)]
    if len(segment_word_ids) < 3:
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    first_word = words_by_id.get(segment_word_ids[0])
    if first_word is None:
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    first_source_index = source_index.get(segment_word_ids[0])
    if first_source_index is None or first_source_index <= 0:
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    previous_source_word = source_graph.words[first_source_index - 1]
    previous_word_id = str(getattr(previous_source_word, "word_id", "") or "")
    previous_text = normalize_text(str(getattr(previous_source_word, "text", "") or ""))
    first_uid = _subtitle_uid(first_word)
    if not previous_word_id or previous_word_id in used_word_ids or not previous_text or not first_uid:
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    if _subtitle_uid(previous_source_word) != first_uid or _cjk_char_count(previous_text) < MIN_REPEATED_ISLAND_CJK_CHARS:
        no_candidate: dict[str, Any] | None = None
        return no_candidate
    for restart_index in range(1, min(len(segment_word_ids), 6)):
        restart_word = words_by_id.get(segment_word_ids[restart_index])
        if restart_word is None:
            continue
        restart_uid = _subtitle_uid(restart_word)
        if not restart_uid or restart_uid == first_uid:
            continue
        restart_text = normalize_text(str(getattr(restart_word, "text", "") or ""))
        if restart_text != previous_text:
            continue
        prefix_ids = segment_word_ids[:restart_index]
        if not prefix_ids:
            continue
        if any(_subtitle_uid(words_by_id.get(word_id)) != first_uid for word_id in prefix_ids):
            continue
        after_ids = segment_word_ids[restart_index + 1 :]
        if not after_ids or _cjk_char_count(text_from_word_ids(after_ids, source_graph)) < MIN_AFTER_SECOND_CJK_CHARS:
            continue
        if _subtitle_uid(words_by_id.get(after_ids[0])) != restart_uid:
            continue
        prefix_text = text_from_word_ids(prefix_ids, source_graph)
        if _cjk_char_count(prefix_text) > MAX_HIGH_CONFIDENCE_MIDDLE_CJK_CHARS:
            continue
        return {
            "segment_id": segment.segment_id,
            "target_word_ids": prefix_ids,
            "dropped_restart_word_id": previous_word_id,
            "restart_word_id": segment_word_ids[restart_index],
            "restart_text": restart_text,
            "middle_text": prefix_text,
            "after_restart_text": text_from_word_ids(after_ids, source_graph),
            "confidence": "high",
            "reason": "leading middle fragment remains after the prior repeated island was already dropped before a source-boundary restart",
        }
    no_candidate: dict[str, Any] | None = None
    return no_candidate


def _subtitle_uid(word: Any) -> str:
    if word is None:
        return ""
    return str(getattr(word, "subtitle_uid", "") or "")


def _contains_definition_marker(text: str) -> bool:
    return any(marker in str(text or "") for marker in DEFINITION_OR_EMPHASIS_MARKERS)


def _has_restart_pivot(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return normalized[-1] in RESTART_PIVOT_SUFFIX_CHARS


def _word_source_start_us(word_ids: list[str], source_graph: CanonicalSourceGraph) -> int:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for word_id in word_ids:
        word = words_by_id.get(word_id)
        if word is not None:
            return int(word.source_start_us)
    return 0


def _word_source_end_us(word_ids: list[str], source_graph: CanonicalSourceGraph) -> int:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for word_id in reversed(word_ids):
        word = words_by_id.get(word_id)
        if word is not None:
            return int(word.source_end_us)
    return 0


def _cjk_char_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", str(text or "")))
