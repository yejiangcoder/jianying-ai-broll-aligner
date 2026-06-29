from __future__ import annotations

from collections import Counter
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.boundary_overlap import (
    is_enumeration_slot_continuation,
    is_open_predicate_bridge,
    is_parallel_progressive_semantic_expansion,
)
from aroll_v21.quality.final_timeline_repair_intent import build_final_timeline_repair_intent_report
from aroll_v21.quality.repeat_span_repair import parallel_enumeration_candidate
from aroll_v21.quality.tiny_segment_classifier import classify_tiny_segment


CONNECTOR_RESTART_WORDS = {
    "所以",
    "但是",
    "然后",
    "因为",
    "就是",
    "其实",
    "可是",
    "如果",
    "那么",
}
SHORT_ISLAND_MAX_DURATION_US = 1_200_000
SHORT_ISLAND_MAX_CHARS = 4
SHORT_ISLAND_MIN_NEXT_CHARS = 5
SHORT_ISLAND_MAX_SOURCE_GAP_US = 1_500_000
SEMANTIC_RESTART_MAX_SAFE_DURATION_US = 900_000
SEMANTIC_RESTART_MAX_SAFE_GAP_US = 500_000
SEMANTIC_RESTART_MIN_OVERLAP_UNITS = 2
INTRA_SENTENCE_PAUSE_MAX_US = 250_000
INTRA_SENTENCE_PAUSE_MIN_SPLIT_SIDE_US = 650_000
PAUSE_BOUNDARY_FILLER_WORDS = {"啊", "呃", "嗯", "呐", "呢", "嘛", "吧", "咳"}
SENTENCE_FINAL_PARTICLES = {"了", "吧", "吗", "嘛", "呢", "啊", "呀", "啦", "哈"}
COMPLETE_SHORT_QUESTION_SUFFIXES = (
    "信不信",
    "是不是",
    "对不对",
    "行不行",
    "懂不懂",
    "知不知道",
    "有没有",
    "能不" + "能",
    "可不可以",
    "好不好",
    "要不要",
)
WEAK_OVERLAP_PARTICLES = {
    "的",
    "地",
    "得",
    "了",
    "着",
    "过",
    "是",
    "为",
    "把",
    "被",
    "给",
    "让",
    "使",
    "在",
    "和",
    "与",
    "及",
    "或",
    "个",
}
PHYSICAL_BLOCKING_CANDIDATE_TYPES = {
    "short_restart_residue_island",
    "expected_removed_text_still_visible",
    "intra_sentence_pause_gap_exceeds_limit",
    "dangling_word_before_connector",
    "video_segment_source_text_mismatch",
    "missing_requested_lead_handle",
}
BLOCKER_CODE_BY_CANDIDATE_TYPE = {
    "short_restart_residue_island": "V21_FINAL_TIMELINE_SHORT_RESTART_RESIDUE",
    "expected_removed_text_still_visible": "V21_FINAL_TIMELINE_EXPECTED_REMOVED_TEXT_STILL_VISIBLE",
    "intra_sentence_pause_gap_exceeds_limit": "V21_FINAL_TIMELINE_INTRA_SENTENCE_PAUSE_GAP_EXCEEDS_LIMIT",
    "dangling_word_before_connector": "V21_FINAL_TIMELINE_DANGLING_CONNECTOR_PREFIX",
    "video_segment_source_text_mismatch": "V21_FINAL_TIMELINE_SOURCE_TEXT_MISMATCH",
    "missing_requested_lead_handle": "V21_FINAL_TIMELINE_SAFE_CUT_HANDLE_MISSING",
    "caption_video_word_text_mismatch": "V21_FINAL_TIMELINE_CAPTION_SOURCE_TEXT_MISMATCH",
}


def build_final_timeline_quality_guard_report(
    *,
    source_graph: CanonicalSourceGraph,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
) -> dict[str, Any]:
    """Build read-only topology facts and risk candidates for the final timeline."""

    words_by_id = {str(word.word_id): word for word in source_graph.words}
    ordered_words = sorted(
        source_graph.words,
        key=lambda word: (
            int(getattr(word, "source_start_us", 0) or 0),
            int(getattr(word, "source_end_us", 0) or 0),
            str(getattr(word, "word_id", "") or ""),
        ),
    )
    selected_word_ids = {str(word_id) for segment in final_timeline for word_id in list(segment.word_ids or [])}
    ordered_segments = sorted(final_timeline, key=lambda row: (int(row.target_start_us), int(row.target_end_us), str(row.segment_id)))
    segment_facts = [
        _segment_fact(index, ordered_segments, words_by_id)
        for index, _segment in enumerate(ordered_segments)
    ]
    candidates: list[dict[str, Any]] = []
    candidates.extend(_short_restart_residue_candidates(ordered_segments, words_by_id))
    candidates.extend(
        _intra_sentence_pause_gap_candidates(
            ordered_segments,
            words_by_id,
            ordered_words=ordered_words,
            selected_word_ids=selected_word_ids,
        )
    )
    candidates.extend(
        _dangling_connector_candidates(
            ordered_segments,
            words_by_id,
            ordered_words=ordered_words,
            selected_word_ids=selected_word_ids,
        )
    )
    candidates.extend(_caption_video_text_mismatch_candidates(captions, ordered_segments, words_by_id))
    candidates.extend(_missing_lead_handle_candidates(ordered_segments, words_by_id))
    repair_intent_report = build_final_timeline_repair_intent_report(candidates)
    candidates.extend(
        _expected_removed_text_still_visible_candidates(
            repair_intent_report=repair_intent_report,
            segments=ordered_segments,
            captions=captions,
            words_by_id=words_by_id,
        )
    )
    blocking_candidates = _blocking_candidates(candidates)
    type_counts = Counter(str(row.get("type") or "") for row in candidates)
    severity_counts = Counter(str(row.get("severity") or "warning") for row in candidates)
    high_risk_count = sum(1 for row in candidates if row.get("severity") == "high")
    blocking_type_counts = Counter(str(row.get("type") or "") for row in blocking_candidates)
    blocker_codes = sorted(
        {
            BLOCKER_CODE_BY_CANDIDATE_TYPE.get(str(candidate.get("type") or ""), "V21_FINAL_TIMELINE_QUALITY_GUARD_FAILED")
            for candidate in blocking_candidates
        }
    )
    gate_passed = not blocking_candidates
    return {
        "report_name": "final_timeline_quality_guard",
        "report_only": True,
        "gate_passed": gate_passed,
        "write_gate_passed": gate_passed,
        "quality_guard_passed": gate_passed,
        "candidate_count": len(candidates),
        "high_risk_candidate_count": high_risk_count,
        "candidate_type_counts": dict(sorted(type_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "blocking_candidate_count": len(blocking_candidates),
        "physical_blocking_candidate_count": sum(
            1
            for row in blocking_candidates
            if str(row.get("type") or "") in PHYSICAL_BLOCKING_CANDIDATE_TYPES
        ),
        "caption_masking_candidate_count": sum(
            1
            for row in blocking_candidates
            if str(row.get("type") or "") == "caption_video_word_text_mismatch"
        ),
        "caption_source_text_mismatch_blocking_count": sum(
            1
            for row in blocking_candidates
            if str(row.get("type") or "") == "caption_video_word_text_mismatch"
        ),
        "blocking_candidate_type_counts": dict(sorted(blocking_type_counts.items())),
        "blocker_codes": blocker_codes,
        "blocking_candidates": blocking_candidates,
        "segment_fact_count": len(segment_facts),
        "segment_facts": segment_facts,
        "candidates": candidates,
        "repair_intent_report": repair_intent_report,
        "repair_intent_count": repair_intent_report["repair_intent_count"],
        "repair_intent_type_counts": repair_intent_report["repair_intent_type_counts"],
        "intra_sentence_pause_gap_threshold_us": INTRA_SENTENCE_PAUSE_MAX_US,
        "intra_sentence_pause_gap_candidate_count": type_counts.get("intra_sentence_pause_gap_exceeds_limit", 0),
        "intra_sentence_pause_gap_max_us": max(
            [
                int(row.get("gap_us") or 0)
                for row in candidates
                if str(row.get("type") or "") == "intra_sentence_pause_gap_exceeds_limit"
            ]
            or [0]
        ),
    }


def _blocking_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    physical: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_type = str(candidate.get("type") or "")
        if candidate_type not in PHYSICAL_BLOCKING_CANDIDATE_TYPES:
            continue
        if str(candidate.get("severity") or "") != "high":
            continue
        physical.append(candidate)
    caption_text_mismatches: list[dict[str, Any]] = []
    for candidate in candidates:
        if str(candidate.get("type") or "") != "caption_video_word_text_mismatch":
            continue
        if str(candidate.get("severity") or "") != "high":
            continue
        caption_text_mismatches.append(candidate)
    return [*_dedupe_candidates(physical), *_dedupe_candidates(caption_text_mismatches)]


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (
            candidate.get("type"),
            candidate.get("segment_id"),
            candidate.get("caption_id"),
            tuple(candidate.get("word_ids") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _segment_fact(
    index: int,
    segments: list[FinalTimelineSegment],
    words_by_id: dict[str, Any],
) -> dict[str, Any]:
    segment = segments[index]
    words = _words_for_segment(segment, words_by_id)
    previous = segments[index - 1] if index > 0 else None
    next_segment = segments[index + 1] if index + 1 < len(segments) else None
    source_text = _source_text(words)
    classification = classify_tiny_segment(segment)
    debug_hints = dict(segment.debug_hints or {})
    return {
        "segment_id": segment.segment_id,
        "index": index,
        "text": str(segment.text or ""),
        "source_word_text": source_text,
        "word_ids": list(segment.word_ids or []),
        "source_start_us": int(segment.source_start_us),
        "source_end_us": int(segment.source_end_us),
        "target_start_us": int(segment.target_start_us),
        "target_end_us": int(segment.target_end_us),
        "duration_us": max(0, int(segment.target_end_us) - int(segment.target_start_us)),
        "source_subtitle_indexes": _unique(
            int(getattr(word, "subtitle_index"))
            for word in words
            if getattr(word, "subtitle_index", None) is not None
        ),
        "first_word_text": str(getattr(words[0], "text", "") or "") if words else "",
        "last_word_text": str(getattr(words[-1], "text", "") or "") if words else "",
        "previous_segment_id": previous.segment_id if previous is not None else "",
        "next_segment_id": next_segment.segment_id if next_segment is not None else "",
        "previous_source_gap_us": (
            max(0, int(segment.source_start_us) - int(previous.source_end_us))
            if previous is not None
            else None
        ),
        "next_source_gap_us": (
            max(0, int(next_segment.source_start_us) - int(segment.source_end_us))
            if next_segment is not None
            else None
        ),
        "is_visual_gap_split": bool(debug_hints.get("visual_pacing_large_intra_segment_gap_split")),
        "is_semantic_bridge": bool(classification.semantic_bridge),
        "is_weak_filler": bool(classification.weak_filler),
        "lead_handle_us": int(segment.lead_handle_us or 0),
        "tail_handle_us": int(segment.tail_handle_us or 0),
        "safe_handle_requested_lead_us": int(debug_hints.get("safe_handle_requested_lead_us") or 0),
        "safe_handle_requested_tail_us": int(debug_hints.get("safe_handle_requested_tail_us") or 0),
    }


def _short_restart_residue_candidates(
    segments: list[FinalTimelineSegment],
    words_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(segments[:-1]):
        previous_segment = segments[index - 1] if index > 0 else None
        next_segment = segments[index + 1]
        words = _words_for_segment(segment, words_by_id)
        previous_words = _words_for_segment(previous_segment, words_by_id) if previous_segment is not None else []
        next_words = _words_for_segment(next_segment, words_by_id)
        text = normalize_text(_source_text(words))
        previous_text = normalize_text(_source_text(previous_words))
        next_text = normalize_text(_source_text(next_words))
        if not text or not next_text:
            continue
        duration_us = max(0, int(segment.target_end_us) - int(segment.target_start_us))
        classification = classify_tiny_segment(segment)
        debug_hints = dict(segment.debug_hints or {})
        is_visual_gap_split = bool(debug_hints.get("visual_pacing_large_intra_segment_gap_split"))
        is_bridge_like = bool(is_visual_gap_split or classification.semantic_bridge)
        if not is_bridge_like:
            continue
        if classification.weak_filler:
            continue
        if duration_us > SHORT_ISLAND_MAX_DURATION_US or len(text) > SHORT_ISLAND_MAX_CHARS:
            continue
        if len(next_text) < SHORT_ISLAND_MIN_NEXT_CHARS:
            continue
        source_gap_us = max(0, int(next_segment.source_start_us) - int(segment.source_end_us))
        if source_gap_us > SHORT_ISLAND_MAX_SOURCE_GAP_US:
            continue
        overlap = _meaningful_overlap_text(text, next_text)
        if not overlap:
            continue
        protected_structure = _protected_restart_residue_structure(
            text,
            next_text,
            overlap,
            previous_text=previous_text,
        )
        if protected_structure or _is_complete_short_utterance(text):
            continue
        semantic_overlap_units = _semantic_restart_overlap_units(words, next_words)
        semantic_safe_drop = _semantic_restart_residue_safe_drop(
            text=text,
            duration_us=duration_us,
            source_gap_us=source_gap_us,
            is_visual_gap_split=is_visual_gap_split,
            is_semantic_bridge=bool(classification.semantic_bridge),
            protected_structure=protected_structure,
            overlap_units=semantic_overlap_units,
        )
        candidates.append(
            {
                "type": "short_restart_residue_island",
                "severity": "high" if (is_visual_gap_split or semantic_safe_drop) else "warning",
                "segment_id": segment.segment_id,
                "related_segment_id": next_segment.segment_id,
                "text": segment.text,
                "source_word_text": text,
                "next_text": next_segment.text,
                "next_source_word_text": next_text,
                "word_ids": list(segment.word_ids or []),
                "source_gap_us": source_gap_us,
                "duration_us": duration_us,
                "overlap_text": overlap,
                "is_visual_gap_split": is_visual_gap_split,
                "is_semantic_bridge": bool(classification.semantic_bridge),
                "semantic_restart_residue_safe_drop": semantic_safe_drop,
                "semantic_restart_overlap_units": semantic_overlap_units,
                "protected_semantic_structure": protected_structure,
                "reason": "short bridge-like content island overlaps a longer following source expression",
            }
        )
    return candidates


def _semantic_restart_residue_safe_drop(
    *,
    text: str,
    duration_us: int,
    source_gap_us: int,
    is_visual_gap_split: bool,
    is_semantic_bridge: bool,
    protected_structure: bool,
    overlap_units: list[str],
) -> bool:
    if is_visual_gap_split:
        return True
    if not is_semantic_bridge:
        return False
    if protected_structure:
        return False
    if duration_us > SEMANTIC_RESTART_MAX_SAFE_DURATION_US:
        return False
    if source_gap_us > SEMANTIC_RESTART_MAX_SAFE_GAP_US:
        return False
    if _ends_with_sentence_final_particle(text):
        return False
    return len(overlap_units) >= SEMANTIC_RESTART_MIN_OVERLAP_UNITS


def _protected_restart_residue_structure(text: str, next_text: str, overlap: str, *, previous_text: str = "") -> bool:
    if is_open_predicate_bridge(text):
        return True
    if is_enumeration_slot_continuation(text, next_text, overlap, previous_text=previous_text):
        return True
    if parallel_enumeration_candidate(text, next_text) is not None:
        return True
    return is_parallel_progressive_semantic_expansion(text, next_text, overlap)


def _semantic_restart_overlap_units(words: list[Any], next_words: list[Any]) -> list[str]:
    next_text = normalize_text(_source_text(next_words))
    if not next_text:
        return list()
    fragments: list[str] = []
    tokens = [normalize_text(str(getattr(word, "text", "") or "")) for word in words]
    for token in tokens:
        if len(token) >= 2:
            fragments.append(token)
    for left, right in zip(tokens, tokens[1:]):
        joined = normalize_text(f"{left}{right}")
        if len(joined) >= 2:
            fragments.append(joined)
    result: list[str] = []
    for fragment in fragments:
        if fragment and fragment in next_text and fragment not in result:
            result.append(fragment)
    return result


def _ends_with_sentence_final_particle(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and normalized[-1] in SENTENCE_FINAL_PARTICLES


def _is_complete_short_utterance(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(normalized.endswith(suffix) for suffix in COMPLETE_SHORT_QUESTION_SUFFIXES)


def _dangling_connector_candidates(
    segments: list[FinalTimelineSegment],
    words_by_id: dict[str, Any],
    *,
    ordered_words: list[Any],
    selected_word_ids: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        words = _words_for_segment(segment, words_by_id)
        if len(words) < 2 or len(words) > 3:
            continue
        texts = [normalize_text(str(getattr(word, "text", "") or "")) for word in words]
        for connector_index in range(1, len(texts)):
            connector_text = texts[connector_index]
            if connector_text not in CONNECTOR_RESTART_WORDS:
                continue
            dangling_text = "".join(texts[:connector_index])
            if not dangling_text or len(dangling_text) > 2:
                continue
            next_segment = segments[index + 1] if index + 1 < len(segments) else None
            following_unselected = _unselected_words_after(
                words[connector_index],
                ordered_words=ordered_words,
                selected_word_ids=selected_word_ids,
                before_us=int(next_segment.source_start_us) if next_segment is not None else None,
            )
            candidates.append(
                {
                    "type": "dangling_word_before_connector",
                    "severity": "high",
                    "segment_id": segment.segment_id,
                    "related_segment_id": next_segment.segment_id if next_segment is not None else "",
                    "text": segment.text,
                    "source_word_text": _source_text(words),
                    "word_ids": list(segment.word_ids or []),
                    "dangling_word_ids": [str(getattr(word, "word_id", "") or "") for word in words[:connector_index]],
                    "connector_word_id": str(getattr(words[connector_index], "word_id", "") or ""),
                    "connector_text": connector_text,
                    "following_unselected_word_ids": [str(getattr(word, "word_id", "") or "") for word in following_unselected],
                    "following_unselected_text": _source_text(following_unselected),
                    "reason": "short residual word is glued before a discourse connector",
                }
            )
    return candidates


def _intra_sentence_pause_gap_candidates(
    segments: list[FinalTimelineSegment],
    words_by_id: dict[str, Any],
    *,
    ordered_words: list[Any],
    selected_word_ids: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for segment in segments:
        word_ids = _word_ids(segment.word_ids)
        words = [words_by_id[word_id] for word_id in word_ids if word_id in words_by_id]
        if len(words) < 2 or len(words) != len(word_ids):
            continue
        for index, (left_word, right_word) in enumerate(zip(words, words[1:])):
            gap_us = int(getattr(right_word, "source_start_us", 0) or 0) - int(getattr(left_word, "source_end_us", 0) or 0)
            if gap_us <= INTRA_SENTENCE_PAUSE_MAX_US:
                continue
            if str(getattr(left_word, "source_material_id", "") or "") != str(getattr(right_word, "source_material_id", "") or ""):
                continue
            if str(getattr(left_word, "source_segment_id", "") or "") != str(getattr(right_word, "source_segment_id", "") or ""):
                continue
            left_word_ids = word_ids[: index + 1]
            right_word_ids = word_ids[index + 1 :]
            left_words = [words_by_id[word_id] for word_id in left_word_ids]
            right_words = [words_by_id[word_id] for word_id in right_word_ids]
            left_duration_us = _word_run_duration_us(left_words)
            right_duration_us = _word_run_duration_us(right_words)
            unselected_gap_words = _unselected_words_between(
                left_word,
                right_word,
                ordered_words=ordered_words,
                selected_word_ids=selected_word_ids,
            )
            safety_reason = _intra_sentence_pause_split_safety_reason(
                left_word,
                right_word,
                left_words=left_words,
                right_words=right_words,
                left_duration_us=left_duration_us,
                right_duration_us=right_duration_us,
                unselected_gap_words=unselected_gap_words,
            )
            safe_split_available = not bool(safety_reason)
            candidates.append(
                {
                    "type": "intra_sentence_pause_gap_exceeds_limit",
                    "severity": "high" if safe_split_available else "warning",
                    "segment_id": segment.segment_id,
                    "text": segment.text,
                    "source_word_text": _source_text(words),
                    "word_ids": word_ids,
                    "left_word_id": str(getattr(left_word, "word_id", "") or ""),
                    "right_word_id": str(getattr(right_word, "word_id", "") or ""),
                    "left_word_text": str(getattr(left_word, "text", "") or ""),
                    "right_word_text": str(getattr(right_word, "text", "") or ""),
                    "left_word_ids": left_word_ids,
                    "right_word_ids": right_word_ids,
                    "split_after_word_id": str(getattr(left_word, "word_id", "") or ""),
                    "split_before_word_id": str(getattr(right_word, "word_id", "") or ""),
                    "gap_us": gap_us,
                    "max_allowed_gap_us": INTRA_SENTENCE_PAUSE_MAX_US,
                    "source_start_us": int(getattr(left_word, "source_end_us", 0) or 0),
                    "source_end_us": int(getattr(right_word, "source_start_us", 0) or 0),
                    "left_duration_us": left_duration_us,
                    "right_duration_us": right_duration_us,
                    "same_source_material": str(getattr(left_word, "source_material_id", "") or "")
                    == str(getattr(right_word, "source_material_id", "") or ""),
                    "same_source_segment": str(getattr(left_word, "source_segment_id", "") or "")
                    == str(getattr(right_word, "source_segment_id", "") or ""),
                    "same_subtitle_index": getattr(left_word, "subtitle_index", None)
                    == getattr(right_word, "subtitle_index", None),
                    "unselected_gap_word_ids": [str(getattr(word, "word_id", "") or "") for word in unselected_gap_words],
                    "unselected_gap_text": _source_text(unselected_gap_words),
                    "safe_split_available": safe_split_available,
                    "split_safety_reason": safety_reason,
                    "safe_cut_recompute_required": True,
                    "reason": "final timeline segment contains an over-limit internal source pause",
                }
            )
    return candidates


def _intra_sentence_pause_split_safety_reason(
    left_word: Any,
    right_word: Any,
    *,
    left_words: list[Any],
    right_words: list[Any],
    left_duration_us: int,
    right_duration_us: int,
    unselected_gap_words: list[Any],
) -> str:
    if str(getattr(left_word, "source_material_id", "") or "") != str(getattr(right_word, "source_material_id", "") or ""):
        return "cross_source_material"
    if str(getattr(left_word, "source_segment_id", "") or "") != str(getattr(right_word, "source_segment_id", "") or ""):
        return "cross_source_segment"
    if unselected_gap_words:
        return "unselected_words_inside_pause_gap"
    text_shape_reason = _pause_gap_text_shape_safety_reason(left_words, right_words)
    if text_shape_reason:
        return text_shape_reason
    if left_duration_us < INTRA_SENTENCE_PAUSE_MIN_SPLIT_SIDE_US:
        return "left_side_too_short"
    if right_duration_us < INTRA_SENTENCE_PAUSE_MIN_SPLIT_SIDE_US:
        return "right_side_too_short"
    return ""


def _pause_gap_text_shape_safety_reason(left_words: list[Any], right_words: list[Any]) -> str:
    left_text = normalize_text(_source_text(left_words))
    right_text = normalize_text(_source_text(right_words))
    if not left_text or not right_text:
        return "empty_split_side_text"
    if len(left_text) <= 2:
        return "left_side_short_text_island"
    if len(right_text) <= 2:
        return "right_side_short_text_island"
    if left_text in right_text or right_text in left_text:
        return "repeat_or_containment_boundary"
    left_last = normalize_text(str(getattr(left_words[-1], "text", "") or "")) if left_words else ""
    right_first = normalize_text(str(getattr(right_words[0], "text", "") or "")) if right_words else ""
    right_second = normalize_text(str(getattr(right_words[1], "text", "") or "")) if len(right_words) > 1 else ""
    if left_last in PAUSE_BOUNDARY_FILLER_WORDS or right_first in PAUSE_BOUNDARY_FILLER_WORDS:
        return "boundary_filler_word"
    if len(right_first) == 1 and right_second in CONNECTOR_RESTART_WORDS:
        return "single_word_intrusion_before_connector"
    return ""


def _word_run_duration_us(words: list[Any]) -> int:
    if not words:
        return 0
    return max(0, int(getattr(words[-1], "source_end_us", 0) or 0) - int(getattr(words[0], "source_start_us", 0) or 0))


def _caption_video_text_mismatch_candidates(
    captions: list[CaptionRenderUnit],
    segments: list[FinalTimelineSegment],
    words_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    segment_by_id = {segment.segment_id: segment for segment in segments}
    for segment in segments:
        source_text = _source_text(_words_for_segment(segment, words_by_id))
        if source_text and normalize_text(source_text) != normalize_text(segment.text):
            candidates.append(
                {
                    "type": "video_segment_source_text_mismatch",
                    "severity": "high",
                    "segment_id": segment.segment_id,
                    "text": segment.text,
                    "source_word_text": source_text,
                    "word_ids": list(segment.word_ids or []),
                    "reason": "final video segment text does not match its bound source words",
                }
            )
    for caption in captions:
        caption_words = [words_by_id[word_id] for word_id in _word_ids(caption.word_ids) if word_id in words_by_id]
        source_text = _source_text(caption_words)
        if not source_text or normalize_text(source_text) == normalize_text(caption.text):
            continue
        segment_ids = [str(segment_id) for segment_id in list(caption.timeline_segment_ids or []) if str(segment_id)]
        candidates.append(
            {
                "type": "caption_video_word_text_mismatch",
                "severity": "high",
                "caption_id": caption.caption_id,
                "segment_ids": segment_ids,
                "containing_video_segment_id": str(caption.containing_video_segment_id or ""),
                "caption_text": caption.text,
                "source_word_text": source_text,
                "video_segment_text": "".join(str(segment_by_id[segment_id].text) for segment_id in segment_ids if segment_id in segment_by_id),
                "word_ids": list(caption.word_ids or []),
                "reason": "visible caption text no longer matches the bound video source words",
            }
        )
    return candidates


def _missing_lead_handle_candidates(
    segments: list[FinalTimelineSegment],
    words_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        debug_hints = dict(segment.debug_hints or {})
        requested = int(debug_hints.get("safe_handle_requested_lead_us") or 0)
        applied = int(segment.lead_handle_us or 0)
        if requested <= 0 or applied > 0:
            continue
        previous = segments[index - 1] if index > 0 else None
        available = _available_lead_handle_us(segment, previous)
        words = _words_for_segment(segment, words_by_id)
        source_text = _source_text(words)
        first_text = normalize_text(str(getattr(words[0], "text", "") or "")) if words else ""
        has_connector = any(normalize_text(str(getattr(word, "text", "") or "")) in CONNECTOR_RESTART_WORDS for word in words)
        severity = "high" if available > 0 and (has_connector or first_text in CONNECTOR_RESTART_WORDS) else "warning"
        candidates.append(
            {
                "type": "missing_requested_lead_handle",
                "severity": severity,
                "segment_id": segment.segment_id,
                "text": segment.text,
                "source_word_text": source_text,
                "word_ids": list(segment.word_ids or []),
                "first_word_text": first_text,
                "contains_connector": has_connector,
                "lead_handle_us": applied,
                "safe_handle_requested_lead_us": requested,
                "available_lead_handle_us": available,
                "reason": "segment requested a leading safe handle but the final segment has none",
            }
        )
    return candidates


def _expected_removed_text_still_visible_candidates(
    *,
    repair_intent_report: dict[str, Any],
    segments: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    words_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    visible_rows = _visible_text_rows(segments=segments, captions=captions, words_by_id=words_by_id)
    seen: set[tuple[str, str, str, str]] = set()
    for intent in list(repair_intent_report.get("repair_intents") or []):
        if not isinstance(intent, dict):
            continue
        if str(intent.get("safety_level") or "") != "review_candidate":
            continue
        expected = normalize_text(str(intent.get("expected_removed_text") or ""))
        if not expected:
            continue
        for visible in visible_rows:
            if visible["normalized_text"] != expected:
                continue
            key = (
                str(intent.get("intent_id") or ""),
                str(visible.get("kind") or ""),
                str(visible.get("segment_id") or ""),
                str(visible.get("caption_id") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "type": "expected_removed_text_still_visible",
                    "severity": "high",
                    "segment_id": str(visible.get("segment_id") or intent.get("segment_id") or ""),
                    "caption_id": str(visible.get("caption_id") or ""),
                    "text": str(visible.get("text") or ""),
                    "source_word_text": str(visible.get("source_word_text") or ""),
                    "word_ids": list(visible.get("word_ids") or []),
                    "expected_removed_text": str(intent.get("expected_removed_text") or ""),
                    "expected_preserved_text": str(intent.get("expected_preserved_text") or ""),
                    "source_intent_id": str(intent.get("intent_id") or ""),
                    "source_intent_type": str(intent.get("intent_type") or ""),
                    "source_candidate_type": str(intent.get("source_candidate_type") or ""),
                    "source_candidate_severity": str(intent.get("source_candidate_severity") or ""),
                    "visible_surface": str(visible.get("kind") or ""),
                    "reason": "a review repair intent's expected_removed_text remains as a standalone final-visible unit",
                }
            )
    return candidates


def _visible_text_rows(
    *,
    segments: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    words_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        words = _words_for_segment(segment, words_by_id)
        source_text = _source_text(words)
        text = str(segment.text or source_text or "")
        normalized = normalize_text(source_text or text)
        if not normalized:
            continue
        rows.append(
            {
                "kind": "timeline_segment",
                "segment_id": segment.segment_id,
                "caption_id": "",
                "text": text,
                "source_word_text": source_text,
                "normalized_text": normalized,
                "word_ids": list(segment.word_ids or []),
            }
        )
    for caption in captions:
        word_ids = _word_ids(caption.word_ids)
        caption_words = [words_by_id[word_id] for word_id in word_ids if word_id in words_by_id]
        source_text = _source_text(caption_words)
        text = str(caption.text or source_text or "")
        normalized = normalize_text(source_text or text)
        if not normalized:
            continue
        rows.append(
            {
                "kind": "caption",
                "segment_id": str(caption.containing_video_segment_id or ""),
                "caption_id": caption.caption_id,
                "text": text,
                "source_word_text": source_text,
                "normalized_text": normalized,
                "word_ids": word_ids,
            }
        )
    return rows


def _available_lead_handle_us(segment: FinalTimelineSegment, previous: FinalTimelineSegment | None) -> int:
    source_start = int(segment.source_start_us)
    if previous is None:
        return max(0, source_start)
    previous_clip_end = int(previous.clip_source_end_us if previous.clip_source_end_us is not None else previous.source_end_us)
    return max(0, source_start - previous_clip_end)


def _unselected_words_after(
    word: Any,
    *,
    ordered_words: list[Any],
    selected_word_ids: set[str],
    before_us: int | None,
) -> list[Any]:
    word_end = int(getattr(word, "source_end_us", 0) or 0)
    result: list[Any] = []
    for candidate in ordered_words:
        word_id = str(getattr(candidate, "word_id", "") or "")
        if not word_id or word_id in selected_word_ids:
            continue
        start_us = int(getattr(candidate, "source_start_us", 0) or 0)
        end_us = int(getattr(candidate, "source_end_us", start_us) or start_us)
        if start_us < word_end:
            continue
        if before_us is not None and start_us >= int(before_us):
            break
        if end_us <= start_us:
            continue
        result.append(candidate)
    return result


def _unselected_words_between(
    left_word: Any,
    right_word: Any,
    *,
    ordered_words: list[Any],
    selected_word_ids: set[str],
) -> list[Any]:
    left_end = int(getattr(left_word, "source_end_us", 0) or 0)
    right_start = int(getattr(right_word, "source_start_us", 0) or 0)
    if right_start <= left_end:
        empty: list[Any] = []
        return empty
    result: list[Any] = []
    for candidate in ordered_words:
        word_id = str(getattr(candidate, "word_id", "") or "")
        if not word_id or word_id in selected_word_ids:
            continue
        start_us = int(getattr(candidate, "source_start_us", 0) or 0)
        end_us = int(getattr(candidate, "source_end_us", start_us) or start_us)
        if end_us <= left_end:
            continue
        if start_us >= right_start:
            break
        if end_us <= start_us:
            continue
        result.append(candidate)
    return result


def _words_for_segment(segment: FinalTimelineSegment, words_by_id: dict[str, Any]) -> list[Any]:
    return [words_by_id[word_id] for word_id in _word_ids(segment.word_ids) if word_id in words_by_id]


def _word_ids(values: Any) -> list[str]:
    return [str(value) for value in list(values or [])]


def _source_text(words: list[Any]) -> str:
    return "".join(str(getattr(word, "text", "") or "") for word in words)


def _meaningful_overlap_text(left: str, right: str) -> str:
    if left in right and _is_meaningful_overlap_fragment(left):
        return left
    for width in range(min(len(left), len(right), 4), 0, -1):
        for start in range(0, len(left) - width + 1):
            fragment = left[start : start + width]
            if fragment and fragment in right and _is_meaningful_overlap_fragment(fragment):
                return fragment
    return ""


def _is_meaningful_overlap_fragment(fragment: str) -> bool:
    normalized = normalize_text(fragment)
    if not normalized:
        return False
    if len(normalized) == 1 and normalized in WEAK_OVERLAP_PARTICLES:
        return False
    return True


def _unique(values: Any) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
