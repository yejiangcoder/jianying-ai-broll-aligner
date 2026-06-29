from __future__ import annotations

from collections import Counter
from typing import Any


def build_final_timeline_repair_intent_report(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert final timeline quality candidates into ordered, report-only repair intents."""

    lead_handle_by_segment = {
        str(candidate.get("segment_id") or ""): candidate
        for candidate in candidates
        if str(candidate.get("type") or "") == "missing_requested_lead_handle"
    }
    intents: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate_index, candidate in enumerate(candidates):
        for intent in _intents_for_candidate(
            candidate,
            candidate_index=candidate_index,
            lead_handle_by_segment=lead_handle_by_segment,
        ):
            key = _intent_key(intent)
            if key in seen:
                continue
            seen.add(key)
            intents.append({**intent, "intent_id": f"final_timeline_intent_{len(intents) + 1:06d}"})

    type_counts = Counter(str(intent.get("intent_type") or "") for intent in intents)
    safety_counts = Counter(str(intent.get("safety_level") or "review") for intent in intents)
    return {
        "report_name": "final_timeline_repair_intent",
        "report_only": True,
        "timeline_mutation_allowed": False,
        "repair_intent_count": len(intents),
        "repair_intent_type_counts": dict(sorted(type_counts.items())),
        "safety_level_counts": dict(sorted(safety_counts.items())),
        "source_topology_contract": {
            "source_words_are_authoritative": True,
            "edits_must_reference_word_ids": True,
            "caption_text_must_be_rendered_from_bound_words": True,
            "safe_cut_must_be_recomputed_after_source_word_span_changes": True,
        },
        "repair_intents": intents,
    }


def _intents_for_candidate(
    candidate: dict[str, Any],
    *,
    candidate_index: int,
    lead_handle_by_segment: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_type = str(candidate.get("type") or "")
    if candidate_type == "short_restart_residue_island":
        return [_drop_restart_residue_intent(candidate, candidate_index)]
    if candidate_type == "intra_sentence_pause_gap_exceeds_limit":
        return [_split_intra_sentence_pause_gap_intent(candidate, candidate_index)]
    if candidate_type == "dangling_word_before_connector":
        return [_trim_dangling_connector_intent(candidate, candidate_index, lead_handle_by_segment)]
    if candidate_type == "caption_video_word_text_mismatch":
        return [_rerender_caption_intent(candidate, candidate_index)]
    if candidate_type == "video_segment_source_text_mismatch":
        return [_refresh_segment_text_intent(candidate, candidate_index)]
    if candidate_type == "missing_requested_lead_handle":
        return [_recompute_lead_handle_intent(candidate, candidate_index)]
    return _no_repair_intents_for_unsupported_candidate()


def _no_repair_intents_for_unsupported_candidate() -> list[dict[str, Any]]:
    return list()


def _drop_restart_residue_intent(candidate: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    is_visual_gap_split = bool(candidate.get("is_visual_gap_split"))
    semantic_safe_drop = bool(candidate.get("semantic_restart_residue_safe_drop"))
    deterministic = is_visual_gap_split or semantic_safe_drop
    return _base_intent(
        candidate,
        candidate_index,
        intent_type="drop_restart_residue_segment",
        timeline_mutation="drop_whole_segment",
        segment_id=str(candidate.get("segment_id") or ""),
        related_segment_id=str(candidate.get("related_segment_id") or ""),
        word_ids=_string_list(candidate.get("word_ids")),
        drop_word_ids=_string_list(candidate.get("word_ids")),
        expected_removed_text=str(candidate.get("source_word_text") or ""),
        expected_preserved_text=str(candidate.get("next_source_word_text") or ""),
        is_visual_gap_split=is_visual_gap_split,
        is_semantic_bridge=bool(candidate.get("is_semantic_bridge")),
        semantic_restart_residue_safe_drop=semantic_safe_drop,
        semantic_restart_overlap_units=_string_list(candidate.get("semantic_restart_overlap_units")),
        protected_semantic_structure=bool(candidate.get("protected_semantic_structure")),
        safety_level="deterministic_candidate" if deterministic else "review_candidate",
        safe_cut_recompute_required=True,
        safety_checks=[
            "automatic drop requires visual-gap split evidence",
            "semantic bridge auto drop requires multi-overlap evidence and no protected parallel structure",
            "target segment is a whole short source-word island",
            "neighboring source expression remains present after removal",
            "captions must be rendered from the remaining final timeline",
        ],
    )


def _split_intra_sentence_pause_gap_intent(candidate: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    safe_split_available = bool(candidate.get("safe_split_available"))
    return _base_intent(
        candidate,
        candidate_index,
        intent_type="split_intra_sentence_pause_gap",
        timeline_mutation="split_segment_at_word_boundary",
        segment_id=str(candidate.get("segment_id") or ""),
        word_ids=_string_list(candidate.get("word_ids")),
        left_word_ids=_string_list(candidate.get("left_word_ids")),
        right_word_ids=_string_list(candidate.get("right_word_ids")),
        split_after_word_id=str(candidate.get("split_after_word_id") or ""),
        split_before_word_id=str(candidate.get("split_before_word_id") or ""),
        gap_us=int(candidate.get("gap_us") or 0),
        max_allowed_gap_us=int(candidate.get("max_allowed_gap_us") or 0),
        left_duration_us=int(candidate.get("left_duration_us") or 0),
        right_duration_us=int(candidate.get("right_duration_us") or 0),
        unselected_gap_word_ids=_string_list(candidate.get("unselected_gap_word_ids")),
        unselected_gap_text=str(candidate.get("unselected_gap_text") or ""),
        split_safety_reason=str(candidate.get("split_safety_reason") or ""),
        safety_level="deterministic_candidate" if safe_split_available else "review_candidate",
        safe_cut_recompute_required=True,
        safety_checks=[
            "split boundary must be between adjacent source words in one final segment",
            "no unselected source words may exist inside the removed pause gap",
            "both split sides must keep enough spoken duration",
            "safe cut handles must be recomputed after segment split",
            "captions must be rendered from the split final timeline",
        ],
    )


def _trim_dangling_connector_intent(
    candidate: dict[str, Any],
    candidate_index: int,
    lead_handle_by_segment: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    segment_id = str(candidate.get("segment_id") or "")
    lead_handle_candidate = lead_handle_by_segment.get(segment_id) or {}
    requested_lead = int(lead_handle_candidate.get("safe_handle_requested_lead_us") or 0)
    connector_word_id = str(candidate.get("connector_word_id") or "")
    return _base_intent(
        candidate,
        candidate_index,
        intent_type="trim_dangling_words_before_connector",
        timeline_mutation="trim_leading_word_ids_from_segment",
        segment_id=segment_id,
        related_segment_id=str(candidate.get("related_segment_id") or ""),
        word_ids=_string_list(candidate.get("word_ids")),
        drop_word_ids=_string_list(candidate.get("dangling_word_ids")),
        keep_anchor_word_ids=[connector_word_id] if connector_word_id else [],
        connector_text=str(candidate.get("connector_text") or ""),
        following_unselected_word_ids=_string_list(candidate.get("following_unselected_word_ids")),
        following_unselected_text=str(candidate.get("following_unselected_text") or ""),
        safety_level="deterministic_candidate",
        safe_cut_recompute_required=True,
        safe_cut_anchor_word_id=connector_word_id,
        requested_lead_handle_us=requested_lead,
        safety_checks=[
            "only leading dangling word ids may be removed",
            "connector word must stay bound to the segment",
            "lead handle must be recomputed from the connector anchor",
            "captions must be rendered from the post-trim source words",
        ],
    )


def _rerender_caption_intent(candidate: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    segment_ids = _string_list(candidate.get("segment_ids"))
    return _base_intent(
        candidate,
        candidate_index,
        intent_type="rerender_caption_from_source_words",
        timeline_mutation="none",
        segment_id=str(candidate.get("containing_video_segment_id") or (segment_ids[0] if segment_ids else "")),
        caption_id=str(candidate.get("caption_id") or ""),
        segment_ids=segment_ids,
        word_ids=_string_list(candidate.get("word_ids")),
        current_caption_text=str(candidate.get("caption_text") or ""),
        expected_caption_text=str(candidate.get("source_word_text") or ""),
        video_segment_text=str(candidate.get("video_segment_text") or ""),
        safety_level="report_only_candidate",
        safe_cut_recompute_required=False,
        safety_checks=[
            "caption text is derived from bound source word ids",
            "caption changes cannot mask a physical timeline source-word mismatch",
        ],
    )


def _refresh_segment_text_intent(candidate: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    return _base_intent(
        candidate,
        candidate_index,
        intent_type="refresh_segment_text_from_source_words",
        timeline_mutation="metadata_only",
        segment_id=str(candidate.get("segment_id") or ""),
        word_ids=_string_list(candidate.get("word_ids")),
        current_segment_text=str(candidate.get("text") or ""),
        expected_segment_text=str(candidate.get("source_word_text") or ""),
        safety_level="deterministic_candidate",
        safe_cut_recompute_required=False,
        safety_checks=[
            "segment text metadata is derived from bound source word ids",
            "metadata refresh does not change source or target timing",
        ],
    )


def _recompute_lead_handle_intent(candidate: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    return _base_intent(
        candidate,
        candidate_index,
        intent_type="recompute_missing_lead_handle",
        timeline_mutation="recompute_clip_source_bounds",
        segment_id=str(candidate.get("segment_id") or ""),
        word_ids=_string_list(candidate.get("word_ids")),
        first_word_text=str(candidate.get("first_word_text") or ""),
        contains_connector=bool(candidate.get("contains_connector")),
        lead_handle_us=int(candidate.get("lead_handle_us") or 0),
        requested_lead_handle_us=int(candidate.get("safe_handle_requested_lead_us") or 0),
        safety_level="deterministic_candidate" if candidate.get("severity") == "high" else "review_candidate",
        safe_cut_recompute_required=True,
        safety_checks=[
            "clip source bounds must stay inside the source window",
            "lead handle must not overlap the previous clip source window",
            "caption timing must stay bound to spoken source words",
        ],
    )


def _base_intent(
    candidate: dict[str, Any],
    candidate_index: int,
    *,
    intent_type: str,
    timeline_mutation: str,
    segment_id: str,
    safety_level: str,
    safe_cut_recompute_required: bool,
    safety_checks: list[str],
    **fields: Any,
) -> dict[str, Any]:
    return {
        "intent_type": intent_type,
        "source_candidate_index": candidate_index,
        "source_candidate_type": str(candidate.get("type") or ""),
        "source_candidate_severity": str(candidate.get("severity") or "warning"),
        "segment_id": segment_id,
        "timeline_mutation": timeline_mutation,
        "safe_cut_recompute_required": bool(safe_cut_recompute_required),
        "safety_level": safety_level,
        "safety_checks": safety_checks,
        "reason": str(candidate.get("reason") or ""),
        **fields,
    }


def _intent_key(intent: dict[str, Any]) -> tuple[Any, ...]:
    return (
        intent.get("intent_type"),
        intent.get("segment_id"),
        intent.get("caption_id"),
        tuple(intent.get("drop_word_ids") or []),
        tuple(intent.get("word_ids") or []),
    )


def _string_list(values: Any) -> list[str]:
    return [str(value) for value in list(values or [])]
