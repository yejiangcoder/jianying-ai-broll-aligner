from __future__ import annotations

from typing import Any


AMBIGUOUS_REPEAT_ISSUE_TYPE = "ambiguous_repeat"
FINAL_VISIBLE_AMBIGUOUS_REPEAT_CLUSTER_TYPE = "final_visible_ambiguous_repeat"
SEMANTIC_REQUEST_PAYLOAD_TYPE = "semantic_decision_required"
SEMANTIC_ARBITRATION_ALLOWED_DECISIONS = [
    "keep_all",
    "drop_left",
    "drop_right",
    "requires_human_review",
    "no_decision",
]


def build_final_visible_repeat_semantic_arbitration_report(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    arbitration_candidates = [
        dict(candidate)
        for candidate in candidates
        if bool(candidate.get("needs_semantic_arbitration"))
    ]
    payloads = [
        _payload_from_candidate(candidate, index=index)
        for index, candidate in enumerate(arbitration_candidates, start=1)
    ]
    candidate_ids = [str(payload.get("cluster_id") or "") for payload in payloads if str(payload.get("cluster_id") or "")]
    return {
        "candidate_count": len(arbitration_candidates),
        "request_count": len(payloads),
        "candidate_ids": candidate_ids,
        "candidates": arbitration_candidates,
        "request_payloads": payloads,
        "mode": "request_only",
        "provider": "deepseek_v4_pro_compatible",
    }


def _payload_from_candidate(candidate: dict[str, Any], *, index: int) -> dict[str, Any]:
    issue_id = _issue_id(candidate, index=index)
    evidence = candidate.get("semantic_repeat_evidence") if isinstance(candidate.get("semantic_repeat_evidence"), dict) else {}
    left_text = str(candidate.get("text") or "")
    right_text = str(candidate.get("related_text") or "")
    candidate_caption_ids = [
        caption_id
        for caption_id in (
            str(candidate.get("caption_id") or ""),
            str(candidate.get("related_caption_id") or ""),
        )
        if caption_id
    ]
    word_ids = [
        str(word_id)
        for word_id in [
            *list(candidate.get("caption_word_ids") or []),
            *list(candidate.get("related_word_ids") or []),
        ]
        if str(word_id)
    ]
    return {
        "issue_id": issue_id,
        "cluster_id": issue_id,
        "issue_type": AMBIGUOUS_REPEAT_ISSUE_TYPE,
        "repeat_type": AMBIGUOUS_REPEAT_ISSUE_TYPE,
        "type": SEMANTIC_REQUEST_PAYLOAD_TYPE,
        "cluster_type": FINAL_VISIBLE_AMBIGUOUS_REPEAT_CLUSTER_TYPE,
        "severity": "medium",
        "warning_only": True,
        "candidate_caption_ids": candidate_caption_ids,
        "word_ids": word_ids,
        "target_start_us": int(candidate.get("target_start_us") or 0),
        "target_end_us": int(candidate.get("related_target_end_us") or candidate.get("target_end_us") or 0),
        "text": left_text,
        "left_text": left_text,
        "right_text": right_text,
        "local_context": {
            "cluster_id": issue_id,
            "type": "final_visible_repeat",
            "cluster_type": FINAL_VISIBLE_AMBIGUOUS_REPEAT_CLUSTER_TYPE,
            "candidate": dict(candidate),
            "semantic_repeat_evidence": dict(evidence),
        },
        "local_evidence": [
            {
                "evidence_id": f"{issue_id}_evidence",
                "evidence_type": "final_visible_repeat",
                "reason": str(candidate.get("classification_reason") or candidate.get("reason") or ""),
                "confidence": _candidate_confidence(candidate),
                "metadata": {
                    "candidate": dict(candidate),
                    "semantic_repeat_evidence": dict(evidence),
                },
            }
        ],
        "allowed_decisions": list(SEMANTIC_ARBITRATION_ALLOWED_DECISIONS),
        "recommended_action": "no_decision",
        "suggested_for_rough_cut": "no_decision",
        "why_local_policy_cannot_decide": (
            "adjacent or near visible repeat has overlap evidence but deterministic policy cannot distinguish "
            "stutter restart from parallel/progressive semantic expansion"
        ),
        "required_decision_schema": {
            "decision": " | ".join(SEMANTIC_ARBITRATION_ALLOWED_DECISIONS),
            "keep_unit_id": "",
            "drop_unit_ids": [],
            "reason": "",
            "confidence": 0.0,
            "requires_human_review": False,
        },
    }


def _issue_id(candidate: dict[str, Any], *, index: int) -> str:
    caption_id = _stable_id_part(str(candidate.get("caption_id") or "caption"))
    related_caption_id = _stable_id_part(str(candidate.get("related_caption_id") or "related"))
    reason = _stable_id_part(str(candidate.get("classification") or candidate.get("reason") or "repeat"))
    return f"final_visible_repeat_{index:04d}_{caption_id}_{related_caption_id}_{reason}"


def _stable_id_part(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value or ""))
    return cleaned.strip("_") or "unknown"


def _candidate_confidence(candidate: dict[str, Any]) -> float:
    score = float(candidate.get("score") or 0.0)
    if score <= 0:
        return 0.5
    return max(0.0, min(1.0, score))
