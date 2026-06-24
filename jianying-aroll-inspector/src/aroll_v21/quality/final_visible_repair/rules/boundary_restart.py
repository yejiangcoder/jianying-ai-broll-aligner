from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, FinalTimelineSegment
from aroll_v21.quality.final_visible_repair.proposal import TimelineRepairProposal
from aroll_v21.quality.final_visible_repair.timeline_utils import ordered_segments, text_from_word_ids


MIN_BOUNDARY_RESTART_OVERLAP_CJK_CHARS = 3
MAX_BOUNDARY_RESTART_TARGET_GAP_US = 300_000
MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS = 2
MAX_PARTIAL_BOUNDARY_RESTART_TAIL_CJK_CHARS = 3
MAX_SOURCE_TAIL_RESTART_GAP_US = 1_200_000
MAX_SOURCE_TAIL_RESTART_LOOKAHEAD_WORDS = 8
MAX_ELIDED_RESTART_SOURCE_GAP_US = 3_000_000
MIN_ELIDED_RESTART_TAIL_CJK_CHARS = 4
MAX_COMMAND_RESTART_SOURCE_GAP_US = 700_000
MAX_WHOLE_ABANDONED_RESTART_WORDS = 6
MAX_WHOLE_ABANDONED_RESTART_CJK_CHARS = 10
MIN_WHOLE_ABANDONED_RESTART_SHARED_CJK_CHARS = 3


@dataclass(frozen=True)
class BoundaryRestartCandidate:
    prev_segment_id: str
    next_segment_id: str
    overlap_text: str
    overlap_char_count: int
    prev_suffix_word_ids: list[str]
    next_prefix_word_ids: list[str]
    gap_us: int
    reason: str
    target_segment_id: str = ""
    target_word_ids: list[str] = field(default_factory=list)
    repair_action: str = "suffix_trim"
    proposal_policy: str = "trim_previous_suffix_keep_next_prefix"

    def to_evidence(self) -> dict[str, Any]:
        return {
            "prev_segment_id": self.prev_segment_id,
            "next_segment_id": self.next_segment_id,
            "overlap_text": self.overlap_text,
            "overlap_char_count": self.overlap_char_count,
            "prev_suffix_word_ids": list(self.prev_suffix_word_ids),
            "next_prefix_word_ids": list(self.next_prefix_word_ids),
            "gap_us": int(self.gap_us),
            "reason": self.reason,
        }


def detect_boundary_restart_candidates(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    *,
    max_target_gap_us: int = MAX_BOUNDARY_RESTART_TARGET_GAP_US,
) -> list[BoundaryRestartCandidate]:
    candidates: list[BoundaryRestartCandidate] = []
    segments = ordered_segments(final_timeline)
    used_word_ids = {str(word_id) for segment in segments for word_id in list(segment.word_ids)}
    for index in range(len(segments) - 1):
        previous = segments[index]
        next_segment = segments[index + 1]
        candidate = _candidate_for_pair(
            previous,
            next_segment,
            source_graph,
            used_word_ids=used_word_ids,
            max_target_gap_us=max_target_gap_us,
        )
        if candidate is not None:
            candidates.append(candidate)
    for segment in segments:
        candidate = _source_tail_restart_candidate(segment, source_graph, used_word_ids)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def build_boundary_restart_proposals(
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
) -> list[TimelineRepairProposal]:
    proposals: list[TimelineRepairProposal] = []
    for index, candidate in enumerate(detect_boundary_restart_candidates(final_timeline, source_graph), start=1):
        evidence = candidate.to_evidence()
        target_segment_id = candidate.target_segment_id or candidate.prev_segment_id
        target_word_ids = list(candidate.target_word_ids or candidate.prev_suffix_word_ids)
        proposals.append(
            TimelineRepairProposal(
                proposal_id=f"boundary_restart_{index:06d}_{candidate.prev_segment_id}_{candidate.next_segment_id}",
                issue_type="boundary_restart",
                confidence=0.95,
                target_segment_id=target_segment_id,
                target_word_ids=target_word_ids,
                target_source_start_us=_word_source_start_us(target_word_ids, source_graph),
                target_source_end_us=_word_source_end_us(target_word_ids, source_graph),
                target_text=candidate.overlap_text,
                repair_action=candidate.repair_action,
                risk_tags=["adjacent_boundary_restart", "whole_word_suffix_trim"],
                evidence={
                    **evidence,
                    "confidence": "high",
                    "proposal_policy": candidate.proposal_policy,
                    "target_segment_id": target_segment_id,
                    "target_word_ids": target_word_ids,
                },
            )
        )
    return proposals


def _candidate_for_pair(
    previous: FinalTimelineSegment,
    next_segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    *,
    used_word_ids: set[str],
    max_target_gap_us: int,
) -> BoundaryRestartCandidate | None:
    gap_us = int(next_segment.target_start_us) - int(previous.target_end_us)
    if gap_us < 0 or gap_us > max_target_gap_us:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    previous_word_ids = [str(word_id) for word_id in previous.word_ids if str(word_id)]
    next_word_ids = [str(word_id) for word_id in next_segment.word_ids if str(word_id)]
    if len(previous_word_ids) < 2 or len(next_word_ids) < 2:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    previous_text = normalize_text(text_from_word_ids(previous_word_ids, source_graph) or previous.text)
    next_text = normalize_text(text_from_word_ids(next_word_ids, source_graph) or next_segment.text)
    if not previous_text or not next_text:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    next_prefix_duplicate = _next_prefix_duplicate_after_elided_restart_candidate(
        previous,
        next_segment,
        source_graph,
        used_word_ids=used_word_ids,
        gap_us=gap_us,
        previous_text=previous_text,
        next_text=next_text,
        previous_word_ids=previous_word_ids,
        next_word_ids=next_word_ids,
    )
    if next_prefix_duplicate is not None:
        return next_prefix_duplicate
    command_restart = _command_shape_restart_candidate(
        previous,
        next_segment,
        source_graph,
        used_word_ids=used_word_ids,
        gap_us=gap_us,
        previous_word_ids=previous_word_ids,
        next_word_ids=next_word_ids,
    )
    if command_restart is not None:
        return command_restart
    max_word_count = min(len(previous_word_ids) - 1, len(next_word_ids) - 1)
    for count in range(max_word_count, 0, -1):
        previous_suffix_ids = previous_word_ids[-count:]
        next_prefix_ids = next_word_ids[:count]
        previous_suffix_text = normalize_text(text_from_word_ids(previous_suffix_ids, source_graph))
        next_prefix_text = normalize_text(text_from_word_ids(next_prefix_ids, source_graph))
        if not previous_suffix_text or previous_suffix_text != next_prefix_text:
            continue
        overlap_cjk_count = _cjk_char_count(previous_suffix_text)
        if overlap_cjk_count < MIN_BOUNDARY_RESTART_OVERLAP_CJK_CHARS:
            continue
        if not previous_text.endswith(previous_suffix_text) or not next_text.startswith(previous_suffix_text):
            continue
        next_remaining_text = next_text[len(previous_suffix_text) :]
        if _cjk_char_count(next_remaining_text) < 1:
            continue
        return BoundaryRestartCandidate(
            prev_segment_id=previous.segment_id,
            next_segment_id=next_segment.segment_id,
            overlap_text=previous_suffix_text,
            overlap_char_count=overlap_cjk_count,
            prev_suffix_word_ids=previous_suffix_ids,
            next_prefix_word_ids=next_prefix_ids,
            gap_us=gap_us,
            reason="previous suffix is restarted as next prefix and next segment continues with more complete expression",
        )
    partial = _partial_boundary_restart_candidate(
        previous,
        next_segment,
        source_graph,
        gap_us=gap_us,
        previous_text=previous_text,
        next_text=next_text,
        previous_word_ids=previous_word_ids,
    )
    if partial is not None:
        return partial
    no_candidate: BoundaryRestartCandidate | None = None
    return no_candidate


def _next_prefix_duplicate_after_elided_restart_candidate(
    previous: FinalTimelineSegment,
    next_segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    *,
    used_word_ids: set[str],
    gap_us: int,
    previous_text: str,
    next_text: str,
    previous_word_ids: list[str],
    next_word_ids: list[str],
) -> BoundaryRestartCandidate | None:
    if len(previous_word_ids) < 2 or len(next_word_ids) < 3:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    words_by_id = {word.word_id: word for word in source_graph.words}
    previous_last_word = words_by_id.get(previous_word_ids[-1])
    next_first_word = words_by_id.get(next_word_ids[0])
    if previous_last_word is None or next_first_word is None:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    previous_last_text = normalize_text(str(getattr(previous_last_word, "text", "") or ""))
    next_first_text = normalize_text(str(getattr(next_first_word, "text", "") or ""))
    if not previous_last_text or previous_last_text != next_first_text:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    if _cjk_char_count(next_text[len(next_first_text) :]) < 2:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    between_words = _unselected_words_between(previous_last_word, next_first_word, source_graph, used_word_ids)
    if not between_words:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    restart_window = normalize_text(
        "".join(str(getattr(word, "text", "") or "") for word in between_words) + next_first_text
    )
    if _cjk_char_count(restart_window) < MIN_ELIDED_RESTART_TAIL_CJK_CHARS:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    for tail_len in range(min(len(previous_text), len(restart_window), 14), MIN_ELIDED_RESTART_TAIL_CJK_CHARS - 1, -1):
        previous_tail = previous_text[-tail_len:]
        if _cjk_char_count(previous_tail) < MIN_ELIDED_RESTART_TAIL_CJK_CHARS:
            continue
        if previous_tail not in restart_window:
            continue
        return BoundaryRestartCandidate(
            prev_segment_id=previous.segment_id,
            next_segment_id=next_segment.segment_id,
            overlap_text=next_first_text,
            overlap_char_count=_cjk_char_count(next_first_text),
            prev_suffix_word_ids=[previous_word_ids[-1]],
            next_prefix_word_ids=[next_word_ids[0]],
            gap_us=gap_us,
            reason="next segment starts with the already-kept tail word after an elided source restart reconstructs the previous tail",
            target_segment_id=next_segment.segment_id,
            target_word_ids=[next_word_ids[0]],
            repair_action="trim_word_span",
            proposal_policy="trim_next_prefix_duplicate_after_elided_restart",
        )
    no_candidate: BoundaryRestartCandidate | None = None
    return no_candidate


def _command_shape_restart_candidate(
    previous: FinalTimelineSegment,
    next_segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    *,
    used_word_ids: set[str],
    gap_us: int,
    previous_word_ids: list[str],
    next_word_ids: list[str],
) -> BoundaryRestartCandidate | None:
    if len(previous_word_ids) < 3 or len(next_word_ids) < 4:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    words_by_id = {word.word_id: word for word in source_graph.words}
    previous_tail_words = [words_by_id.get(word_id) for word_id in previous_word_ids[-2:]]
    next_prefix_words = [words_by_id.get(word_id) for word_id in next_word_ids[:3]]
    if any(word is None for word in previous_tail_words + next_prefix_words):
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    previous_marker = normalize_text(str(getattr(previous_tail_words[0], "text", "") or ""))
    previous_object = normalize_text(str(getattr(previous_tail_words[1], "text", "") or ""))
    next_marker = normalize_text(str(getattr(next_prefix_words[0], "text", "") or ""))
    next_object = normalize_text(str(getattr(next_prefix_words[1], "text", "") or ""))
    next_echoed_marker = normalize_text(str(getattr(next_prefix_words[2], "text", "") or ""))
    if not previous_marker or not previous_object or not next_marker:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    if previous_object != next_object or previous_marker != next_echoed_marker:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    if previous_marker == next_marker:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    if _cjk_char_count(previous_marker) != 1 or _cjk_char_count(next_marker) != 1:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    between_words = _unselected_words_between(previous_tail_words[1], next_prefix_words[0], source_graph, used_word_ids)
    if len(between_words) != 1:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    between_text = normalize_text(str(getattr(between_words[0], "text", "") or ""))
    if between_text != next_marker:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    source_gap_us = int(getattr(next_prefix_words[0], "source_start_us", 0) or 0) - int(
        getattr(previous_tail_words[1], "source_end_us", 0) or 0
    )
    if source_gap_us < 0 or source_gap_us > MAX_COMMAND_RESTART_SOURCE_GAP_US:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    dropped_text = normalize_text("".join(str(getattr(word, "text", "") or "") for word in previous_tail_words))
    return BoundaryRestartCandidate(
        prev_segment_id=previous.segment_id,
        next_segment_id=next_segment.segment_id,
        overlap_text=dropped_text,
        overlap_char_count=_cjk_char_count(previous_object),
        prev_suffix_word_ids=previous_word_ids[-2:],
        next_prefix_word_ids=next_word_ids[:3],
        gap_us=gap_us,
        reason="previous command tail is abandoned before the next segment restarts the same object with an echoed marker",
        target_segment_id=previous.segment_id,
        target_word_ids=previous_word_ids[-2:],
        repair_action="suffix_trim",
        proposal_policy="trim_previous_command_tail_before_restarted_command_shape",
    )


def _partial_boundary_restart_candidate(
    previous: FinalTimelineSegment,
    next_segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    *,
    gap_us: int,
    previous_text: str,
    next_text: str,
    previous_word_ids: list[str],
) -> BoundaryRestartCandidate | None:
    tail_context = previous_text[-10:]
    for start in range(0, max(0, len(tail_context) - MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS)):
        suffix = tail_context[start:]
        if _cjk_char_count(suffix) < MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS + 1:
            continue
        max_shared = min(len(suffix) - 1, len(next_text) - 1)
        for shared_len in range(max_shared, MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS - 1, -1):
            shared = suffix[:shared_len]
            tail = suffix[shared_len:]
            if not shared or not tail:
                continue
            if not next_text.startswith(shared):
                continue
            if _cjk_char_count(shared) < MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS:
                continue
            if _cjk_char_count(shared) >= 4:
                continue
            if shared.startswith(("不", "没")):
                continue
            if _cjk_char_count(tail) > MAX_PARTIAL_BOUNDARY_RESTART_TAIL_CJK_CHARS:
                continue
            next_remainder = next_text[shared_len:]
            if _cjk_char_count(next_remainder) < 1:
                continue
            drop_ids = _trailing_word_ids_for_text(previous_word_ids, suffix, source_graph)
            if not drop_ids or len(drop_ids) >= len(previous_word_ids):
                continue
            if _starts_inside_local_reduplication(previous_word_ids, drop_ids, source_graph):
                continue
            if not _leading_word_prefix_matches(drop_ids, shared, source_graph):
                continue
            return BoundaryRestartCandidate(
                prev_segment_id=previous.segment_id,
                next_segment_id=next_segment.segment_id,
                overlap_text=suffix,
                overlap_char_count=_cjk_char_count(shared),
                prev_suffix_word_ids=drop_ids,
            next_prefix_word_ids=[],
                gap_us=gap_us,
                reason="previous suffix is an abandoned partial restart before the next segment completes the shared prefix",
            )
    no_candidate: BoundaryRestartCandidate | None = None
    return no_candidate


def _source_tail_restart_candidate(
    segment: FinalTimelineSegment,
    source_graph: CanonicalSourceGraph,
    used_word_ids: set[str],
) -> BoundaryRestartCandidate | None:
    segment_word_ids = [str(word_id) for word_id in segment.word_ids if str(word_id)]
    if len(segment_word_ids) < 2:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    segment_source_text = text_from_word_ids(segment_word_ids, source_graph)
    if not segment_source_text:
        segment_source_text = str(getattr(segment, "text", "") or "")
    segment_text = normalize_text(segment_source_text)
    words_by_id = {word.word_id: word for word in source_graph.words}
    last_word = words_by_id.get(segment_word_ids[-1])
    if last_word is None:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    lookahead_words = _following_unselected_words(last_word, source_graph, used_word_ids)
    if not lookahead_words:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    next_text = normalize_text("".join(str(getattr(word, "text", "") or "") for word in lookahead_words))
    if _cjk_char_count(next_text) < MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS + 1:
        no_candidate: BoundaryRestartCandidate | None = None
        return no_candidate
    tail_context = segment_text[-12:]
    for start in range(0, max(0, len(tail_context) - MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS)):
        suffix = tail_context[start:]
        max_shared = min(len(suffix) - 1, len(next_text) - 1)
        for shared_len in range(max_shared, MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS - 1, -1):
            shared = suffix[:shared_len]
            tail = suffix[shared_len:]
            if not shared or not tail or not next_text.startswith(shared):
                continue
            if _cjk_char_count(shared) < MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS:
                continue
            if _cjk_char_count(tail) > MAX_PARTIAL_BOUNDARY_RESTART_TAIL_CJK_CHARS:
                continue
            if _cjk_char_count(next_text[shared_len:]) < 2:
                continue
            drop_ids = _trailing_word_ids_for_text(segment_word_ids, suffix, source_graph)
            if not drop_ids:
                continue
            if len(drop_ids) >= len(segment_word_ids):
                if not _whole_segment_abandoned_source_restart_allowed(
                    segment_word_ids,
                    source_graph,
                    shared_text=shared,
                    tail_text=tail,
                    next_text=next_text,
                ):
                    continue
            if _starts_inside_local_reduplication(segment_word_ids, drop_ids, source_graph):
                continue
            if not _leading_word_prefix_matches(drop_ids, shared, source_graph):
                continue
            return BoundaryRestartCandidate(
                prev_segment_id=segment.segment_id,
                next_segment_id="",
                overlap_text=suffix,
                overlap_char_count=_cjk_char_count(shared),
                prev_suffix_word_ids=drop_ids,
                next_prefix_word_ids=[str(getattr(word, "word_id", "") or "") for word in lookahead_words],
                gap_us=int(getattr(lookahead_words[0], "source_start_us", 0) or 0) - int(getattr(last_word, "source_end_us", 0) or 0),
                reason="segment tail is an abandoned partial restart before the following source words complete the shared prefix",
            )
    no_candidate: BoundaryRestartCandidate | None = None
    return no_candidate


def _following_unselected_words(
    last_word: Any,
    source_graph: CanonicalSourceGraph,
    used_word_ids: set[str],
) -> list[Any]:
    last_end_us = int(getattr(last_word, "source_end_us", 0) or 0)
    source_material_id = str(getattr(last_word, "source_material_id", "") or "")
    rows = sorted(source_graph.words, key=lambda word: (int(getattr(word, "source_start_us", 0) or 0), int(getattr(word, "source_end_us", 0) or 0)))
    result: list[Any] = []
    previous_kept_end_us = last_end_us
    for word in rows:
        word_id = str(getattr(word, "word_id", "") or "")
        if not word_id or word_id in used_word_ids:
            continue
        word_start_us = int(getattr(word, "source_start_us", 0) or 0)
        if word_start_us < last_end_us:
            continue
        if word_start_us - previous_kept_end_us > MAX_SOURCE_TAIL_RESTART_GAP_US:
            break
        if source_material_id and str(getattr(word, "source_material_id", "") or "") not in {"", source_material_id}:
            continue
        result.append(word)
        previous_kept_end_us = int(getattr(word, "source_end_us", 0) or word_start_us)
        if len(result) >= MAX_SOURCE_TAIL_RESTART_LOOKAHEAD_WORDS:
            break
    return result


def _unselected_words_between(
    left_word: Any,
    right_word: Any,
    source_graph: CanonicalSourceGraph,
    used_word_ids: set[str],
) -> list[Any]:
    left_end_us = int(getattr(left_word, "source_end_us", 0) or 0)
    right_start_us = int(getattr(right_word, "source_start_us", 0) or 0)
    if right_start_us < left_end_us:
        empty: list[Any] = []
        return empty
    if right_start_us - left_end_us > MAX_ELIDED_RESTART_SOURCE_GAP_US:
        empty: list[Any] = []
        return empty
    source_material_id = str(getattr(left_word, "source_material_id", "") or "")
    rows = sorted(
        source_graph.words,
        key=lambda word: (
            int(getattr(word, "source_start_us", 0) or 0),
            int(getattr(word, "source_end_us", 0) or 0),
        ),
    )
    result: list[Any] = []
    for word in rows:
        word_id = str(getattr(word, "word_id", "") or "")
        if not word_id or word_id in used_word_ids:
            continue
        word_start_us = int(getattr(word, "source_start_us", 0) or 0)
        word_end_us = int(getattr(word, "source_end_us", 0) or 0)
        if word_start_us < left_end_us:
            continue
        if word_end_us > right_start_us:
            break
        if source_material_id and str(getattr(word, "source_material_id", "") or "") not in {"", source_material_id}:
            continue
        result.append(word)
    return result


def _leading_word_prefix_matches(
    word_ids: list[str],
    prefix_text: str,
    source_graph: CanonicalSourceGraph,
) -> bool:
    target = normalize_text(prefix_text)
    if not target:
        return False
    words_by_id = {word.word_id: word for word in source_graph.words}
    joined = ""
    for word_id in word_ids:
        word = words_by_id.get(word_id)
        if word is None:
            return False
        joined += normalize_text(str(getattr(word, "text", "") or ""))
        if joined == target:
            return True
        if len(joined) >= len(target):
            return False
    return False


def _starts_inside_local_reduplication(
    segment_word_ids: list[str],
    drop_word_ids: list[str],
    source_graph: CanonicalSourceGraph,
) -> bool:
    if not segment_word_ids or not drop_word_ids:
        return False
    try:
        first_drop_index = segment_word_ids.index(drop_word_ids[0])
    except ValueError:
        return False
    if first_drop_index <= 0:
        return False
    words_by_id = {word.word_id: word for word in source_graph.words}
    previous_word = words_by_id.get(segment_word_ids[first_drop_index - 1])
    first_drop_word = words_by_id.get(drop_word_ids[0])
    if previous_word is None or first_drop_word is None:
        return False
    previous_text = normalize_text(str(getattr(previous_word, "text", "") or ""))
    first_drop_text = normalize_text(str(getattr(first_drop_word, "text", "") or ""))
    return bool(previous_text and previous_text == first_drop_text)


def _whole_segment_abandoned_source_restart_allowed(
    segment_word_ids: list[str],
    source_graph: CanonicalSourceGraph,
    *,
    shared_text: str,
    tail_text: str,
    next_text: str,
) -> bool:
    if len(segment_word_ids) > MAX_WHOLE_ABANDONED_RESTART_WORDS:
        return False
    segment_text = normalize_text(text_from_word_ids(segment_word_ids, source_graph))
    if not segment_text or _cjk_char_count(segment_text) > MAX_WHOLE_ABANDONED_RESTART_CJK_CHARS:
        return False
    shared = normalize_text(shared_text)
    tail = normalize_text(tail_text)
    if _cjk_char_count(shared) < MIN_WHOLE_ABANDONED_RESTART_SHARED_CJK_CHARS:
        return False
    if _cjk_char_count(tail) < MIN_PARTIAL_BOUNDARY_RESTART_SHARED_CJK_CHARS:
        return False
    if _cjk_char_count(tail) > MAX_PARTIAL_BOUNDARY_RESTART_TAIL_CJK_CHARS:
        return False
    if not next_text.startswith(shared):
        return False
    if _cjk_char_count(next_text[len(shared) :]) < 2:
        return False
    return bool(shared and tail and tail.startswith(shared[0]))


def _trailing_word_ids_for_text(
    word_ids: list[str],
    suffix_text: str,
    source_graph: CanonicalSourceGraph,
) -> list[str]:
    target = normalize_text(suffix_text)
    if not target:
        empty: list[str] = []
        return empty
    selected: list[str] = []
    joined = ""
    words_by_id = {word.word_id: word for word in source_graph.words}
    for word_id in reversed(word_ids):
        word = words_by_id.get(word_id)
        if word is None:
            empty: list[str] = []
            return empty
        selected.insert(0, word_id)
        joined = normalize_text(str(getattr(word, "text", "") or "")) + joined
        if joined == target:
            return selected
        if not target.endswith(joined):
            empty: list[str] = []
            return empty
    empty: list[str] = []
    return empty


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
