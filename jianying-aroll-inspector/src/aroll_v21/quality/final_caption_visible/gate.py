from __future__ import annotations

from typing import Any

from aroll_v21.quality.final_caption_visible.policy import (
    blocking_policy_blocker_codes,
    policy_decision_counts,
)
from aroll_v21.quality.final_caption_visible.semantic_arbitration import (
    build_final_visible_repeat_semantic_arbitration_report,
)
from aroll_v21.quality.final_caption_visible.types import (
    FinalCaptionVisibleClassification,
    FinalCaptionVisibleEvidence,
    FinalCaptionVisiblePolicyDecision,
    FinalCaptionVisibleRepairSignal,
)
from aroll_v21.quality.final_semantic_integrity import semantic_integrity_reason_counts
from aroll_v21.quality.final_visible_repeat_classification import (
    REPEAT_SEMANTIC_CLASS_DESCRIPTIONS,
    semantic_repeat_class_counts,
)


def build_final_caption_visible_gate_report(
    *,
    evidence: FinalCaptionVisibleEvidence,
    classification: FinalCaptionVisibleClassification,
    policy_decisions: list[FinalCaptionVisiblePolicyDecision],
    repair_signal: FinalCaptionVisibleRepairSignal,
    ngram_size: int,
    prefix_suffix_min_overlap: int,
    near_duplicate_ratio: float,
    final_visible_recheck_decisions: list[str],
) -> dict[str, Any]:
    blocker_codes = blocking_policy_blocker_codes(policy_decisions)
    repair_signal_report = repair_signal.to_report()
    semantic_arbitration_report = build_final_visible_repeat_semantic_arbitration_report(
        classification.classified_repeat_candidates
    )
    return {
        "gate_passed": not blocker_codes,
        "blocker_codes": blocker_codes,
        "visible_repeat_candidate_count": len(classification.visible_repeat_candidates),
        "visible_repeat_fatal_candidate_count": len(classification.visible_repeat_candidates),
        "visible_repeat_warning_candidate_count": len(classification.repeat_warning_candidates),
        "visible_repeat_allow_candidate_count": len(classification.repeat_allowed_candidates),
        "repeat_classification_candidate_count": len(classification.classified_repeat_candidates),
        "repeat_classification_candidates": classification.classified_repeat_candidates,
        "repeat_semantic_class_counts": semantic_repeat_class_counts(classification.classified_repeat_candidates),
        "repeat_semantic_class_definitions": dict(REPEAT_SEMANTIC_CLASS_DESCRIPTIONS),
        "repeat_semantic_arbitration_candidate_count": int(semantic_arbitration_report["candidate_count"]),
        "repeat_semantic_arbitration_request_count": int(semantic_arbitration_report["request_count"]),
        "repeat_semantic_arbitration_candidate_ids": list(semantic_arbitration_report["candidate_ids"]),
        "repeat_semantic_arbitration_candidates": list(semantic_arbitration_report["candidates"]),
        "repeat_semantic_arbitration_request_payloads": list(semantic_arbitration_report["request_payloads"]),
        "repeat_semantic_arbitration_mode": str(semantic_arbitration_report["mode"]),
        "repeat_semantic_arbitration_provider": str(semantic_arbitration_report["provider"]),
        "visible_repeat_warning_candidates": classification.repeat_warning_candidates,
        "visible_repeat_allow_candidates": classification.repeat_allowed_candidates,
        "containment_repeat_count": len(classification.containment_candidates),
        "containment_repeat_raw_count": len(evidence.raw_containment_candidates),
        "prefix_suffix_overlap_count": len(classification.prefix_suffix_candidates),
        "ngram_repeat_count": len(classification.ngram_candidates),
        "ngram_repeat_raw_count": len(evidence.raw_ngram_candidates),
        "near_duplicate_visible_caption_count": len(classification.near_duplicate_candidates),
        "modifier_redundancy_residual_count": len(evidence.modifier_redundancy_candidates),
        "self_repair_aborted_phrase_count": len(evidence.self_repair_candidates),
        "dangling_prefix_suffix_count": len(evidence.dangling_candidates),
        "semantic_garbage_or_asr_suspect_count": len(evidence.semantic_suspect_candidates),
        "semantic_integrity_count": len(evidence.semantic_integrity_candidates),
        "semantic_integrity_reason_counts": semantic_integrity_reason_counts(evidence.semantic_integrity_candidates),
        "cross_caption_semantic_containment_count": len(classification.cross_caption_containment_candidates),
        "cross_caption_semantic_containment_raw_count": len(evidence.raw_cross_caption_containment_candidates),
        "restart_repeat_visible_count": len(classification.restart_repeat_candidates),
        "visible_repeat_candidates": classification.visible_repeat_candidates,
        "containment_repeat_candidates": classification.containment_candidates,
        "prefix_suffix_overlap_candidates": classification.prefix_suffix_candidates,
        "ngram_repeat_candidates": classification.ngram_candidates,
        "near_duplicate_visible_caption_candidates": classification.near_duplicate_candidates,
        "modifier_redundancy_residual_candidates": evidence.modifier_redundancy_candidates,
        "self_repair_aborted_phrase_candidates": evidence.self_repair_candidates,
        "dangling_prefix_suffix_candidates": evidence.dangling_candidates,
        "semantic_garbage_or_asr_suspect_candidates": evidence.semantic_suspect_candidates,
        "semantic_integrity_candidates": evidence.semantic_integrity_candidates,
        "cross_caption_semantic_containment_candidates": classification.cross_caption_containment_candidates,
        "restart_repeat_visible_candidates": classification.restart_repeat_candidates,
        "final_caption_visible_repeat_gate_enabled": True,
        "ngram_size": int(ngram_size),
        "prefix_suffix_min_overlap": int(prefix_suffix_min_overlap),
        "near_duplicate_ratio": float(near_duplicate_ratio),
        "final_visible_recheck_allowed_decisions": list(final_visible_recheck_decisions),
        "final_visible_policy_decisions": [decision.to_report() for decision in policy_decisions],
        "final_visible_policy_decision_counts": policy_decision_counts(policy_decisions),
        "final_visible_repair_signal_candidate_count": int(repair_signal_report["repairable_candidate_count"]),
        "final_visible_repair_signal_issue_types": list(repair_signal_report["repairable_issue_types"]),
        "final_visible_repair_signal_candidates": list(repair_signal_report["repairable_candidates"]),
    }
