from __future__ import annotations

from collections import Counter
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CaptionRenderUnit
from aroll_v21.quality.boundary_overlap import (
    PROGRESSIVE_SEMANTIC_MARKERS,
    is_parallel_progressive_semantic_expansion,
)


ADJACENT_TARGET_GAP_US = 800_000
NEAR_TARGET_GAP_US = 3_000_000
SHORT_CONCEPT_MAX_CJK_CHARS = 3
REPEAT_SEMANTIC_STUTTER_RESTART = "stutter_restart"
REPEAT_SEMANTIC_PARALLEL_PROGRESSION = "parallel_progression"
REPEAT_SEMANTIC_EXPANSION = "semantic_expansion"
REPEAT_SEMANTIC_RHETORICAL_REPETITION = "rhetorical_repetition"
REPEAT_SEMANTIC_AMBIGUOUS_RESTART_OR_PROGRESSION = "ambiguous_restart_or_progression"
OPEN_FRAGMENT_TAILS = ("去", "把", "被", "给", "让", "对", "向", "跟", "和", "与", "在", "是", "的", "地", "得", "了", "着")
REPEAT_SEMANTIC_CLASS_DESCRIPTIONS = {
    REPEAT_SEMANTIC_STUTTER_RESTART: "aborted or duplicated speech restart that may be repaired by local safe-cut rules",
    REPEAT_SEMANTIC_PARALLEL_PROGRESSION: "parallel, enumerated, or progressive expression that must not be cut as a restart",
    REPEAT_SEMANTIC_EXPANSION: "term reuse, definition, or semantic expansion that is retained by policy",
    REPEAT_SEMANTIC_RHETORICAL_REPETITION: "rhetorical or distant recurrence that is reported but not treated as a stutter",
    REPEAT_SEMANTIC_AMBIGUOUS_RESTART_OR_PROGRESSION: "local recurrence whose stutter-vs-progression meaning needs semantic arbitration",
}


def classify_final_visible_repeat_candidates(
    captions: list[CaptionRenderUnit],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index_by_caption_id = {caption.caption_id: index for index, caption in enumerate(captions)}
    caption_by_id = {caption.caption_id: caption for caption in captions}
    classified: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        caption_id = str(row.get("caption_id") or "")
        related_caption_id = str(row.get("related_caption_id") or "")
        caption = caption_by_id.get(caption_id)
        related = caption_by_id.get(related_caption_id)
        distance_kind = _distance_kind(
            candidate=row,
            caption_index=index_by_caption_id.get(caption_id),
            related_index=index_by_caption_id.get(related_caption_id),
        )
        classification, severity, risk_tags, reason = _classification_for_candidate(
            row,
            distance_kind=distance_kind,
            caption=caption,
            related=related,
        )
        semantic_class, semantic_class_reason = _semantic_repeat_class_for_candidate(
            row,
            classification=classification,
            severity=severity,
            distance_kind=distance_kind,
        )
        needs_semantic_arbitration = semantic_class == REPEAT_SEMANTIC_AMBIGUOUS_RESTART_OR_PROGRESSION
        semantic_repeat_evidence = _semantic_repeat_evidence_for_candidate(
            row,
            caption=caption,
            related=related,
            classification=classification,
            needs_semantic_arbitration=needs_semantic_arbitration,
        )
        row.update(
            {
                "classification": classification,
                "repeat_kind": classification,
                "classified_as": classification,
                "semantic_classification": classification,
                "semantic_repeat_class": semantic_class,
                "semantic_repeat_class_reason": semantic_class_reason,
                "needs_semantic_arbitration": needs_semantic_arbitration,
                "semantic_repeat_evidence": semantic_repeat_evidence,
                "distance_kind": distance_kind,
                "severity": severity,
                "risk_tags": risk_tags,
                "classification_reason": reason,
                **semantic_repeat_evidence,
            }
        )
        classified.append(row)
    return classified


def blocking_repeat_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if str(candidate.get("severity") or "") in {"fatal", "high"}]


def warning_repeat_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if str(candidate.get("severity") or "") == "warning"]


def allowed_repeat_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if str(candidate.get("severity") or "") == "allow"]


def semantic_repeat_class_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(candidate.get("semantic_repeat_class") or REPEAT_SEMANTIC_AMBIGUOUS_RESTART_OR_PROGRESSION)
        for candidate in candidates
    )
    return dict(sorted(counts.items()))


def _classification_for_candidate(
    candidate: dict[str, Any],
    *,
    distance_kind: str,
    caption: CaptionRenderUnit | None,
    related: CaptionRenderUnit | None,
) -> tuple[str, str, list[str], str]:
    reason = str(candidate.get("reason") or candidate.get("type") or "")
    overlap_text = normalize_text(str(candidate.get("overlap_text") or ""))
    if distance_kind == "same_segment" and "restart" in reason:
        return "same_segment_restart", "fatal", ["same_segment", "local_restart"], "same segment restart remains blocking"
    if "restart" in reason:
        if distance_kind in {"adjacent", "near"}:
            return "local_restart", "fatal", [distance_kind, "restart"], "adjacent or near restart remains blocking"
        return "distant_restart_like_recurrence", "warning", ["distant"], "distant restart-like recurrence is not auto-fatal"
    if reason == "prefix_suffix_overlap":
        if distance_kind in {"adjacent", "near"}:
            return "local_boundary_overlap", "fatal", [distance_kind, "boundary_overlap"], "local suffix/prefix overlap remains blocking"
        return _nonlocal_reuse_classification(overlap_text, "distant_boundary_overlap")
    if reason == "cross_caption_semantic_containment":
        if distance_kind in {"adjacent", "near"}:
            return "local_cross_caption_containment", "fatal", [distance_kind, "cross_caption_window"], "local cross-caption containment remains blocking"
        return _nonlocal_reuse_classification(overlap_text, "distant_containment")
    if reason == "near_duplicate_visible_caption":
        if distance_kind in {"adjacent", "near"}:
            return "local_near_duplicate", "fatal", [distance_kind, "near_duplicate"], "local near duplicate remains blocking"
        return "distant_semantic_recurrence", "warning", ["distant"], "distant near-duplicate recurrence requires review but is not auto-fatal"
    if reason == "ngram_repeat":
        return _ngram_classification(candidate, distance_kind=distance_kind, caption=caption, related=related)
    if reason == "containment_repeat":
        return _containment_classification(candidate, distance_kind=distance_kind, caption=caption, related=related)
    if distance_kind in {"adjacent", "near"}:
        return "local_visible_repeat", "fatal", [distance_kind], "local visible repeat remains blocking"
    return _nonlocal_reuse_classification(overlap_text, "semantic_recurrence")


def _semantic_repeat_class_for_candidate(
    candidate: dict[str, Any],
    *,
    classification: str,
    severity: str,
    distance_kind: str,
) -> tuple[str, str]:
    reason = str(candidate.get("reason") or candidate.get("type") or "")
    if classification == "parallel_progressive_semantic_expansion":
        return REPEAT_SEMANTIC_PARALLEL_PROGRESSION, "classified as protected parallel/progressive semantic expansion"
    if classification == "short_concept_reuse":
        return REPEAT_SEMANTIC_EXPANSION, "short concept or address term recurrence is semantic reuse"
    if reason in {"containment_repeat", "cross_caption_semantic_containment"} and severity in {"allow", "warning"}:
        return REPEAT_SEMANTIC_EXPANSION, "non-blocking containment is treated as semantic expansion"
    if classification in {
        "same_segment_restart",
        "local_restart",
        "local_ngram_boundary_repeat",
        "local_boundary_overlap",
        "local_containment_restart",
    }:
        return REPEAT_SEMANTIC_STUTTER_RESTART, "local boundary or restart shape is treated as stutter restart"
    if "restart" in classification and severity in {"fatal", "high"}:
        return REPEAT_SEMANTIC_STUTTER_RESTART, "fatal restart-like classification is treated as stutter restart"
    if classification in {"local_semantic_recurrence", "local_visible_repeat"} and distance_kind in {"adjacent", "near"}:
        return (
            REPEAT_SEMANTIC_AMBIGUOUS_RESTART_OR_PROGRESSION,
            "local recurrence is not structurally decisive enough to choose stutter or progression",
        )
    if classification in {
        "distant_semantic_recurrence",
        "distant_boundary_overlap",
        "distant_containment",
        "distant_restart_like_recurrence",
        "local_near_duplicate",
        "local_exact_duplicate",
    }:
        return REPEAT_SEMANTIC_RHETORICAL_REPETITION, "recurrence is visible but not a deterministic stutter restart"
    if severity in {"fatal", "high"}:
        return REPEAT_SEMANTIC_STUTTER_RESTART, "blocking repeat remains in the local repair family"
    return (
        REPEAT_SEMANTIC_AMBIGUOUS_RESTART_OR_PROGRESSION,
        "candidate does not match a deterministic semantic repeat class",
    )


def _semantic_repeat_evidence_for_candidate(
    candidate: dict[str, Any],
    *,
    caption: CaptionRenderUnit | None,
    related: CaptionRenderUnit | None,
    classification: str,
    needs_semantic_arbitration: bool,
) -> dict[str, Any]:
    shared_text = normalize_text(str(candidate.get("overlap_text") or ""))
    left_text = _candidate_left_text(candidate, caption)
    right_text = _candidate_right_text(candidate, related)
    left_index = left_text.find(shared_text) if shared_text else -1
    right_index = right_text.find(shared_text) if shared_text else -1
    left_tail = left_text[left_index + len(shared_text) :] if left_index >= 0 else ""
    right_tail = right_text[right_index + len(shared_text) :] if right_index >= 0 else ""
    left_prefix = left_text[:left_index] if left_index >= 0 else ""
    right_prefix = right_text[:right_index] if right_index >= 0 else ""
    has_parallel_structure = is_parallel_progressive_semantic_expansion(left_text, right_text, shared_text)
    evidence = {
        "shared_text": shared_text,
        "left_tail": left_tail,
        "right_tail": right_tail,
        "is_boundary_restart": _is_boundary_restart_shape(shared_text, left_text, right_text),
        "has_progressive_marker": _has_progressive_marker(shared_text, left_text, right_text),
        "has_parallel_structure": has_parallel_structure,
        "source_gap_us": _source_gap_us(caption, related),
        "caption_gap_us": _target_gap_us(candidate),
        "left_is_fragment": _left_is_fragment(
            left_text,
            shared_text=shared_text,
            left_prefix=left_prefix,
            left_tail=left_tail,
        ),
        "right_completes_left": _right_completes_left(
            left_text,
            right_text,
            shared_text=shared_text,
            right_tail=right_tail,
        ),
        "semantic_classification": classification,
        "needs_semantic_arbitration": needs_semantic_arbitration,
    }
    if right_prefix:
        evidence["right_prefix"] = right_prefix
    if left_prefix:
        evidence["left_prefix"] = left_prefix
    return evidence


def _candidate_left_text(candidate: dict[str, Any], caption: CaptionRenderUnit | None) -> str:
    if caption is not None:
        return normalize_text(caption.text)
    return normalize_text(str(candidate.get("text") or ""))


def _candidate_right_text(candidate: dict[str, Any], related: CaptionRenderUnit | None) -> str:
    if candidate.get("window_text"):
        return normalize_text(str(candidate.get("window_text") or ""))
    if related is not None:
        return normalize_text(related.text)
    return normalize_text(str(candidate.get("related_text") or ""))


def _is_boundary_restart_shape(shared_text: str, left_text: str, right_text: str) -> bool:
    if not shared_text or not left_text or not right_text or left_text == right_text:
        return False
    return (
        left_text.endswith(shared_text)
        and right_text.startswith(shared_text)
    ) or (
        left_text == shared_text
        and right_text.startswith(shared_text)
        and len(right_text) > len(left_text)
    ) or (
        right_text == shared_text
        and left_text.startswith(shared_text)
        and len(left_text) > len(right_text)
    )


def _has_progressive_marker(shared_text: str, left_text: str, right_text: str) -> bool:
    left_head = left_text[:8]
    right_head = right_text[:8]
    return any(marker in shared_text or marker in left_head or marker in right_head for marker in PROGRESSIVE_SEMANTIC_MARKERS)


def _left_is_fragment(
    left_text: str,
    *,
    shared_text: str,
    left_prefix: str,
    left_tail: str,
) -> bool:
    if not left_text or not shared_text:
        return False
    if left_text == shared_text:
        return True
    if left_text.endswith(OPEN_FRAGMENT_TAILS):
        return True
    if not left_tail and left_text.endswith(shared_text) and len(left_prefix) <= 2:
        return True
    return False


def _right_completes_left(
    left_text: str,
    right_text: str,
    *,
    shared_text: str,
    right_tail: str,
) -> bool:
    if not left_text or not right_text or left_text == right_text:
        return False
    return (right_text.startswith(left_text) and len(right_text) > len(left_text)) or (
        bool(shared_text)
        and left_text == shared_text
        and right_text.startswith(shared_text)
        and bool(right_tail)
    )


def _source_gap_us(caption: CaptionRenderUnit | None, related: CaptionRenderUnit | None) -> int | None:
    gap_us: int | None = None
    if caption is None or related is None:
        return gap_us
    left_end = caption.spoken_source_end_us
    right_start = related.spoken_source_start_us
    if left_end is None or right_start is None:
        return gap_us
    return int(right_start) - int(left_end)


def _containment_classification(
    candidate: dict[str, Any],
    *,
    distance_kind: str,
    caption: CaptionRenderUnit | None,
    related: CaptionRenderUnit | None,
) -> tuple[str, str, list[str], str]:
    overlap_text = normalize_text(str(candidate.get("overlap_text") or ""))
    if distance_kind in {"adjacent", "near"} and _exact_visible_duplicate(caption=caption, related=related):
        return "local_exact_duplicate", "fatal", [distance_kind, "exact_duplicate"], "local exact visible duplicate remains blocking"
    if distance_kind in {"adjacent", "near"} and _boundary_containment_like_restart(candidate, caption=caption, related=related):
        return "local_containment_restart", "fatal", [distance_kind, "containment_restart"], "local containment touches a boundary and remains blocking"
    if distance_kind in {"adjacent", "near"}:
        return "local_semantic_recurrence", "warning", [distance_kind], "local containment without restart boundary is warning only"
    return _nonlocal_reuse_classification(overlap_text, "distant_containment")


def _exact_visible_duplicate(
    *,
    caption: CaptionRenderUnit | None,
    related: CaptionRenderUnit | None,
) -> bool:
    if caption is None or related is None:
        return False
    left_text = normalize_text(caption.text)
    right_text = normalize_text(related.text)
    return bool(left_text) and left_text == right_text


def _ngram_classification(
    candidate: dict[str, Any],
    *,
    distance_kind: str,
    caption: CaptionRenderUnit | None,
    related: CaptionRenderUnit | None,
) -> tuple[str, str, list[str], str]:
    overlap_text = normalize_text(str(candidate.get("overlap_text") or ""))
    if distance_kind in {"adjacent", "near"} and _ngram_forms_boundary_restart(overlap_text, caption=caption, related=related):
        return "local_ngram_boundary_repeat", "fatal", [distance_kind, "ngram_boundary"], "local ngram touches a caption boundary and remains blocking"
    if distance_kind in {"adjacent", "near"} and _ngram_forms_parallel_progressive_expansion(
        overlap_text,
        caption=caption,
        related=related,
    ):
        return (
            "parallel_progressive_semantic_expansion",
            "warning",
            [distance_kind, "progressive_expansion", "protected_semantic_structure"],
            "local shared ngram is a parallel or progressive semantic expansion, not a restart",
        )
    if distance_kind in {"adjacent", "near"}:
        return "local_semantic_recurrence", "warning", [distance_kind], "local shared ngram away from boundaries is warning only"
    return _nonlocal_reuse_classification(overlap_text, "distant_semantic_recurrence")


def _nonlocal_reuse_classification(
    overlap_text: str,
    default_classification: str,
) -> tuple[str, str, list[str], str]:
    cjk_count = _cjk_char_count(overlap_text)
    if cjk_count and cjk_count <= SHORT_CONCEPT_MAX_CJK_CHARS:
        return "short_concept_reuse", "allow", ["short_concept", "nonlocal_reuse"], "short concept or address term recurrence is allowed"
    return default_classification, "warning", ["nonlocal_reuse"], "distant recurrence is reported but not auto-fatal"


def _boundary_containment_like_restart(
    candidate: dict[str, Any],
    *,
    caption: CaptionRenderUnit | None,
    related: CaptionRenderUnit | None,
) -> bool:
    if caption is None or related is None:
        return False
    overlap_text = normalize_text(str(candidate.get("overlap_text") or ""))
    left_text = normalize_text(caption.text)
    right_text = normalize_text(related.text)
    if not overlap_text:
        return False
    return (
        left_text == overlap_text
        and right_text.startswith(overlap_text)
        and len(right_text) > len(overlap_text)
    ) or (
        right_text == overlap_text
        and left_text.startswith(overlap_text)
        and len(left_text) > len(overlap_text)
    )


def _ngram_forms_boundary_restart(
    overlap_text: str,
    *,
    caption: CaptionRenderUnit | None,
    related: CaptionRenderUnit | None,
) -> bool:
    if caption is None or related is None or not overlap_text:
        return False
    left_text = normalize_text(caption.text)
    right_text = normalize_text(related.text)
    return (
        left_text.endswith(overlap_text)
        and right_text.startswith(overlap_text)
        and len(left_text) > len(overlap_text)
        and len(right_text) > len(overlap_text)
    ) or (
        right_text.endswith(overlap_text)
        and left_text.startswith(overlap_text)
        and len(left_text) > len(overlap_text)
        and len(right_text) > len(overlap_text)
    )


def _ngram_forms_parallel_progressive_expansion(
    overlap_text: str,
    *,
    caption: CaptionRenderUnit | None,
    related: CaptionRenderUnit | None,
) -> bool:
    if caption is None or related is None or not overlap_text:
        return False
    return is_parallel_progressive_semantic_expansion(caption.text, related.text, overlap_text)


def _distance_kind(
    *,
    candidate: dict[str, Any],
    caption_index: int | None,
    related_index: int | None,
) -> str:
    if str(candidate.get("caption_id") or "") == str(candidate.get("related_caption_id") or ""):
        return "same_segment"
    if caption_index is None or related_index is None:
        return "distant"
    index_gap = abs(int(related_index) - int(caption_index))
    if _target_ranges_overlap(candidate):
        if index_gap <= 1:
            return "adjacent"
        if index_gap <= 2:
            return "near"
    target_gap_us = _target_gap_us(candidate)
    if index_gap == 1 and -80_000 <= target_gap_us <= ADJACENT_TARGET_GAP_US:
        return "adjacent"
    if 1 <= index_gap <= 2 and -80_000 <= target_gap_us <= NEAR_TARGET_GAP_US:
        return "near"
    return "distant"


def _target_ranges_overlap(candidate: dict[str, Any]) -> bool:
    left_start = int(candidate.get("target_start_us") or 0)
    left_end = int(candidate.get("target_end_us") or 0)
    right_start = int(candidate.get("related_target_start_us") or 0)
    right_end = int(candidate.get("related_target_end_us") or 0)
    return max(left_start, right_start) < min(left_end, right_end)


def _target_gap_us(candidate: dict[str, Any]) -> int:
    left_start = int(candidate.get("target_start_us") or 0)
    left_end = int(candidate.get("target_end_us") or 0)
    right_start = int(candidate.get("related_target_start_us") or 0)
    right_end = int(candidate.get("related_target_end_us") or 0)
    if right_start >= left_start:
        return right_start - left_end
    return left_start - right_end


def _cjk_char_count(text: str) -> int:
    return sum(1 for char in str(text or "") if "\u3400" <= char <= "\u9fff")
