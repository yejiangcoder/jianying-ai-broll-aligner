from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from aroll_v21.decision.semantic_contracts import (
    SemanticAdjudicationDecision,
    SemanticAdjudicationDecisionType,
    SemanticAdjudicationMode,
    SemanticAdjudicationRequest,
    SemanticAdjudicationResult,
    SemanticRoutingDecision,
    SemanticIssueSeverity,
    SemanticIssueType,
    semantic_contract_to_dict,
)
from aroll_v21.ir.models import Blocker, FinalTimelineSegment, RepeatCluster


HIGH_RISK_SEVERITIES = {SemanticIssueSeverity.HIGH, SemanticIssueSeverity.FATAL}


@dataclass
class SemanticAdjudicationReportBuilder:
    mode: SemanticAdjudicationMode = SemanticAdjudicationMode.SEMANTIC_REQUESTS_ONLY
    provider_configured: bool = False
    provider_called_count: int = 0
    deterministic_baseline_refused_count: int = 0
    local_decision_count: int = 0
    provider_required_count: int = 0
    provider_skipped_count: int = 0
    provider_skipped_reasons: dict[str, int] = field(default_factory=dict)
    provider_error: str = ""
    semantic_decision_cache_used: bool = False
    deepseek_batch_enabled: bool = False
    deepseek_batch_request_count: int = 0
    deepseek_batch_attempt_count: int = 0
    deepseek_batch_retry_count: int = 0
    deepseek_batch_issue_count: int = 0
    deepseek_batch_resolved_count: int = 0
    deepseek_batch_unresolved_count: int = 0
    deepseek_batch_missing_issue_ids: list[str] = field(default_factory=list)
    deepseek_batch_unknown_issue_ids: list[str] = field(default_factory=list)
    deepseek_batch_error: str = ""
    deepseek_batch_chunk_count: int = 0
    deepseek_batch_chunk_sizes: list[int] = field(default_factory=list)
    deepseek_batch_request: dict[str, Any] = field(default_factory=dict)
    deepseek_batch_response: dict[str, Any] = field(default_factory=dict)
    deepseek_batch_error_payload: dict[str, Any] = field(default_factory=dict)
    commit_reused_semantic_cache: bool = False
    semantic_cache_input_hash: str = ""
    semantic_cache_issue_count: int = 0
    semantic_cache_resolved_count: int = 0
    semantic_cache_unresolved_count: int = 0
    routing_decisions: list[SemanticRoutingDecision] = field(default_factory=list)
    requests: list[SemanticAdjudicationRequest] = field(default_factory=list)
    decisions: list[SemanticAdjudicationDecision] = field(default_factory=list)
    results: list[SemanticAdjudicationResult] = field(default_factory=list)

    def add_route(self, route: SemanticRoutingDecision) -> None:
        if route.issue_id not in {item.issue_id for item in self.routing_decisions}:
            self.routing_decisions.append(route)
            if route.requires_provider:
                self.provider_required_count += 1
            else:
                self.provider_skipped_count += 1
                reason = route.provider_reason or route.local_action or "local_or_structural_issue"
                self.provider_skipped_reasons[reason] = int(self.provider_skipped_reasons.get(reason, 0)) + 1

    def add_local_decision(self) -> None:
        self.local_decision_count += 1

    def add_request(self, request: SemanticAdjudicationRequest) -> None:
        if request.issue_id not in {item.issue_id for item in self.requests}:
            self.requests.append(request)

    def add_decision(self, decision: SemanticAdjudicationDecision) -> None:
        self.decisions.append(decision)

    def add_result(self, result: SemanticAdjudicationResult) -> None:
        self.add_request(result.request)
        if result.decision is not None:
            self.add_decision(result.decision)
        self.results.append(result)
        if result.provider_called:
            self.provider_called_count += 1
        if result.deterministic_baseline_refused:
            self.deterministic_baseline_refused_count += 1

    def to_report(self, *, unresolved_issue_ids: Iterable[str] = (), blocker_codes: Iterable[str] = ()) -> dict[str, Any]:
        unresolved = {str(item) for item in unresolved_issue_ids if str(item)}
        blocker_code_set = {str(item) for item in blocker_codes if str(item)}
        fatal_unresolved = [
            request
            for request in self.requests
            if request.issue_id in unresolved and request.severity in HIGH_RISK_SEVERITIES
        ]
        gate_passed = not unresolved and not fatal_unresolved and not blocker_code_set
        return {
            "semantic_adjudication_gate_passed": gate_passed,
            "semantic_mode": self.mode.value,
            "semantic_request_count": len(self.requests),
            "semantic_request_unresolved_count": len(unresolved),
            "fatal_semantic_issue_count": len(fatal_unresolved),
            "deepseek_provider_configured": bool(self.provider_configured),
            "deepseek_provider_called_count": int(self.provider_called_count),
            "deepseek_provider_error": str(self.provider_error or ""),
            "deepseek_batch_enabled": bool(self.deepseek_batch_enabled),
            "deepseek_batch_request_count": int(self.deepseek_batch_request_count),
            "deepseek_batch_attempt_count": int(self.deepseek_batch_attempt_count),
            "deepseek_batch_retry_count": int(self.deepseek_batch_retry_count),
            "deepseek_batch_issue_count": int(self.deepseek_batch_issue_count),
            "deepseek_batch_resolved_count": int(self.deepseek_batch_resolved_count),
            "deepseek_batch_unresolved_count": int(self.deepseek_batch_unresolved_count),
            "deepseek_batch_missing_issue_ids": list(self.deepseek_batch_missing_issue_ids),
            "deepseek_batch_unknown_issue_ids": list(self.deepseek_batch_unknown_issue_ids),
            "deepseek_batch_error": str(self.deepseek_batch_error or ""),
            "deepseek_batch_chunk_count": int(self.deepseek_batch_chunk_count),
            "deepseek_batch_chunk_sizes": list(self.deepseek_batch_chunk_sizes),
            "deepseek_batch_request": dict(self.deepseek_batch_request),
            "deepseek_batch_response": dict(self.deepseek_batch_response),
            "deepseek_batch_error_payload": dict(self.deepseek_batch_error_payload),
            "deepseek_provider_skipped_count": int(self.provider_skipped_count),
            "deepseek_provider_skipped_reasons": dict(sorted(self.provider_skipped_reasons.items())),
            "semantic_decision_cache_used": bool(self.semantic_decision_cache_used),
            "commit_reused_semantic_cache": bool(self.commit_reused_semantic_cache),
            "semantic_cache_input_hash": str(self.semantic_cache_input_hash or ""),
            "semantic_cache_issue_count": int(self.semantic_cache_issue_count),
            "semantic_cache_resolved_count": int(self.semantic_cache_resolved_count),
            "semantic_cache_unresolved_count": int(self.semantic_cache_unresolved_count),
            "deterministic_baseline_refused_count": int(self.deterministic_baseline_refused_count),
            "semantic_auto_route_count": len(self.routing_decisions),
            "semantic_local_decision_count": int(self.local_decision_count),
            "semantic_provider_required_count": int(self.provider_required_count),
            "unresolved_issue_ids": sorted(unresolved),
            "blocker_codes": sorted(blocker_code_set),
            "routing_decisions": [semantic_contract_to_dict(route) for route in self.routing_decisions],
            "requests": [semantic_contract_to_dict(request) for request in self.requests],
            "decisions": [semantic_contract_to_dict(decision) for decision in self.decisions],
            "results": [semantic_contract_to_dict(result) for result in self.results],
        }


def normalize_semantic_mode(value: str | None) -> SemanticAdjudicationMode:
    raw = str(value or "").strip()
    if raw in {"", "default", "auto"}:
        return SemanticAdjudicationMode.AUTO
    if raw in {"semantic_requests_only", "semantic-requests-only"}:
        return SemanticAdjudicationMode.SEMANTIC_REQUESTS_ONLY
    if raw in {"deterministic_baseline", "deterministic-baseline"}:
        return SemanticAdjudicationMode.DETERMINISTIC_BASELINE
    if raw == "deepseek":
        return SemanticAdjudicationMode.DEEPSEEK
    if raw in {"fail_closed", "fail-closed"}:
        return SemanticAdjudicationMode.FAIL_CLOSED
    return SemanticAdjudicationMode.SEMANTIC_REQUESTS_ONLY


class SemanticIssueRouter:
    """Route semantic issues between local policy, provider adjudication, and fail-closed fallback."""

    PROVIDER_ISSUES = {
        SemanticIssueType.MODIFIER_REDUNDANCY,
        SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE,
        SemanticIssueType.NEAR_DUPLICATE_TAKE,
        SemanticIssueType.SEMANTIC_CONTAINMENT,
        SemanticIssueType.VISIBLE_CAPTION_REPEAT,
        SemanticIssueType.AMBIGUOUS_REPEAT,
    }

    def route_cluster(
        self,
        cluster: RepeatCluster,
        *,
        deterministic_action_available: bool,
        local_action: str = "",
    ) -> SemanticRoutingDecision:
        issue_type = issue_type_for_cluster(cluster)
        severity = severity_for_cluster(cluster)
        local_confidence = max((float(evidence.confidence or 0.0) for evidence in cluster.evidence), default=0.0)
        ambiguity_score = max(0.0, min(1.0, 1.0 - local_confidence))
        if deterministic_action_available:
            return SemanticRoutingDecision(
                issue_id=cluster.cluster_id,
                issue_type=issue_type,
                severity=severity,
                local_confidence=local_confidence,
                deterministic_action_available=True,
                local_action=local_action or _local_action_for_cluster(cluster),
                ambiguity_score=ambiguity_score,
                requires_provider=False,
                provider_reason="deterministic_action_available",
                fallback_policy="local",
            )
        requires_provider = self._requires_provider(cluster, issue_type, severity)
        fallback_policy = "blocker" if severity in HIGH_RISK_SEVERITIES else ("request_unresolved" if severity == SemanticIssueSeverity.MEDIUM else "warning")
        return SemanticRoutingDecision(
            issue_id=cluster.cluster_id,
            issue_type=issue_type,
            severity=severity,
            local_confidence=local_confidence,
            deterministic_action_available=False,
            local_action=local_action or _local_action_for_cluster(cluster),
            ambiguity_score=ambiguity_score,
            requires_provider=requires_provider,
            provider_reason=self._provider_reason(cluster, issue_type, severity) if requires_provider else "local_or_structural_issue",
            fallback_policy=fallback_policy,
        )

    def route_request(
        self,
        request: SemanticAdjudicationRequest,
        *,
        deterministic_action_available: bool = False,
        local_action: str = "",
    ) -> SemanticRoutingDecision:
        local_confidence = _severity_confidence(request.severity)
        ambiguity_score = max(0.0, min(1.0, 1.0 - local_confidence))
        if deterministic_action_available:
            return SemanticRoutingDecision(
                issue_id=request.issue_id,
                issue_type=request.issue_type,
                severity=request.severity,
                local_confidence=local_confidence,
                deterministic_action_available=True,
                local_action=local_action or request.recommended_action,
                ambiguity_score=ambiguity_score,
                requires_provider=False,
                provider_reason="deterministic_action_available",
                fallback_policy="local",
            )
        requires_provider = self._requires_provider_for_issue(request.issue_type, request.severity)
        fallback_policy = (
            "blocker"
            if request.severity in HIGH_RISK_SEVERITIES
            else ("request_unresolved" if request.severity == SemanticIssueSeverity.MEDIUM else "warning")
        )
        return SemanticRoutingDecision(
            issue_id=request.issue_id,
            issue_type=request.issue_type,
            severity=request.severity,
            local_confidence=local_confidence,
            deterministic_action_available=False,
            local_action=local_action or request.recommended_action,
            ambiguity_score=ambiguity_score,
            requires_provider=requires_provider,
            provider_reason=_provider_reason_for_issue(request.issue_type, request.severity) if requires_provider else "local_or_structural_issue",
            fallback_policy=fallback_policy,
        )

    def _requires_provider(
        self,
        cluster: RepeatCluster,
        issue_type: SemanticIssueType,
        severity: SemanticIssueSeverity,
    ) -> bool:
        if not any(evidence.requires_semantic_decision for evidence in cluster.evidence):
            return False
        return self._requires_provider_for_issue(issue_type, severity)

    def _requires_provider_for_issue(
        self,
        issue_type: SemanticIssueType,
        severity: SemanticIssueSeverity,
    ) -> bool:
        if issue_type in self.PROVIDER_ISSUES and severity in {SemanticIssueSeverity.MEDIUM, SemanticIssueSeverity.HIGH, SemanticIssueSeverity.FATAL}:
            return True
        if severity in HIGH_RISK_SEVERITIES:
            return True
        return False

    def _provider_reason(
        self,
        cluster: RepeatCluster,
        issue_type: SemanticIssueType,
        severity: SemanticIssueSeverity,
    ) -> str:
        if issue_type == SemanticIssueType.MODIFIER_REDUNDANCY:
            return "modifier_redundancy_local_repair_uncertain"
        if issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE:
            return "self_repair_aborted_phrase_ambiguous"
        if issue_type == SemanticIssueType.SEMANTIC_CONTAINMENT:
            return "semantic_containment_requires_judgment"
        if issue_type == SemanticIssueType.NEAR_DUPLICATE_TAKE:
            return "near_duplicate_take_keep_value_unclear"
        if issue_type == SemanticIssueType.VISIBLE_CAPTION_REPEAT:
            return "visible_caption_repeat_semantic_duplicate_unclear"
        if severity in HIGH_RISK_SEVERITIES:
            return "high_fatal_issue_without_deterministic_action"
        return str(cluster.local_recommendation or "semantic_issue_requires_provider")


def issue_type_for_cluster(cluster: RepeatCluster) -> SemanticIssueType:
    metadata = _cluster_candidate(cluster)
    issue_type = str(metadata.get("issue_type") or metadata.get("type") or "")
    if issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE.value:
        return SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE
    if cluster.repeat_type == "modifier_redundancy":
        return SemanticIssueType.MODIFIER_REDUNDANCY
    if cluster.local_recommendation in {"boundary_prefix_containment_requires_human_review", "boundary_prefix_containment_drop_left"}:
        return SemanticIssueType.SEMANTIC_CONTAINMENT
    if cluster.local_recommendation == "compiler_boundary_suffix_prefix_overlap_cleanup":
        return SemanticIssueType.PREFIX_SUFFIX_OVERLAP
    if cluster.repeat_type == "semantic_retry":
        return SemanticIssueType.AMBIGUOUS_REPEAT
    return SemanticIssueType.AMBIGUOUS_REPEAT


def severity_for_cluster(cluster: RepeatCluster) -> SemanticIssueSeverity:
    candidate = _cluster_candidate(cluster)
    raw = str(candidate.get("severity") or "").strip().lower()
    if raw in {item.value for item in SemanticIssueSeverity}:
        return SemanticIssueSeverity(raw)
    if cluster.repeat_type == "modifier_redundancy":
        return SemanticIssueSeverity.FATAL
    if str(candidate.get("issue_type") or candidate.get("type") or "") == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE.value:
        return SemanticIssueSeverity.MEDIUM
    confidence = max((float(evidence.confidence or 0.0) for evidence in cluster.evidence), default=0.0)
    if confidence >= 0.9:
        return SemanticIssueSeverity.HIGH
    if confidence >= 0.65:
        return SemanticIssueSeverity.MEDIUM
    return SemanticIssueSeverity.LOW


def semantic_blocker_code(issue_type: SemanticIssueType, severity: SemanticIssueSeverity) -> str:
    if issue_type == SemanticIssueType.MODIFIER_REDUNDANCY and severity in HIGH_RISK_SEVERITIES:
        return "V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED"
    if issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE:
        return "V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED"
    return "SEMANTIC_DECISION_NOT_PROVIDED"


def _local_action_for_cluster(cluster: RepeatCluster) -> str:
    recommendation = str(cluster.local_recommendation or "")
    if recommendation:
        return recommendation
    if cluster.repeat_type == "exact_repeat":
        return "keep_right_drop_left"
    if cluster.repeat_type == "hidden_audio_repeat":
        return "requires_unit_split"
    return "none"


def request_from_cluster(cluster: RepeatCluster, *, reason: str = "") -> SemanticAdjudicationRequest:
    issue_type = issue_type_for_cluster(cluster)
    severity = severity_for_cluster(cluster)
    word_ids = [word_id for unit in cluster.variants for word_id in unit.word_ids]
    source_starts = [int(unit.source_start_us) for unit in cluster.variants]
    source_ends = [int(unit.source_end_us) for unit in cluster.variants]
    texts = [str(unit.text or "") for unit in cluster.variants]
    local_context = {
        "cluster_id": cluster.cluster_id,
        "repeat_type": cluster.repeat_type,
        "local_recommendation": str(cluster.local_recommendation or ""),
        "variants": [
            {
                "unit_id": unit.unit_id,
                "text": unit.text,
                "word_ids": list(unit.word_ids),
                "source_start_us": int(unit.source_start_us),
                "source_end_us": int(unit.source_end_us),
            }
            for unit in cluster.variants
        ],
        "local_evidence": [
            {
                "evidence_id": evidence.evidence_id,
                "evidence_type": evidence.evidence_type,
                "reason": evidence.reason,
                "confidence": evidence.confidence,
                "metadata": evidence.metadata,
            }
            for evidence in cluster.evidence
        ],
    }
    candidate = _cluster_candidate(cluster)
    if candidate:
        local_context["candidate"] = candidate
    allowed = _allowed_decisions(issue_type, severity)
    return SemanticAdjudicationRequest(
        issue_id=cluster.cluster_id,
        issue_type=issue_type,
        severity=severity,
        candidate_segment_ids=[unit.unit_id for unit in cluster.variants],
        candidate_caption_ids=sorted({uid for unit in cluster.variants for uid in unit.subtitle_uids if uid}),
        word_ids=word_ids,
        source_start_us=min(source_starts, default=0),
        source_end_us=max(source_ends, default=0),
        target_start_us=0,
        target_end_us=0,
        text_before=texts[0] if texts else "",
        text_after=texts[-1] if len(texts) > 1 else "",
        local_context=local_context,
        recommended_action=_recommended_action(issue_type, cluster, candidate),
        why_local_policy_cannot_decide=reason or _why_local_policy_cannot_decide(issue_type, cluster, candidate),
        allowed_decisions=allowed,
    )


def request_from_self_repair_segments(
    *,
    issue_id: str,
    left: FinalTimelineSegment,
    right: FinalTimelineSegment,
    candidate: dict[str, Any],
    reason: str,
) -> SemanticAdjudicationRequest:
    return SemanticAdjudicationRequest(
        issue_id=issue_id,
        issue_type=SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE,
        severity=SemanticIssueSeverity.MEDIUM,
        candidate_segment_ids=[left.segment_id, right.segment_id],
        candidate_caption_ids=[],
        word_ids=[*left.word_ids, *right.word_ids],
        source_start_us=int(left.source_start_us),
        source_end_us=int(right.source_end_us),
        target_start_us=int(left.target_start_us),
        target_end_us=int(right.target_end_us),
        text_before=str(left.text or ""),
        text_after=str(right.text or ""),
        local_context={"candidate": dict(candidate), "left_segment_id": left.segment_id, "right_segment_id": right.segment_id},
        recommended_action=SemanticAdjudicationDecisionType.DROP_ABORTED.value,
        why_local_policy_cannot_decide=reason,
        allowed_decisions=[
            SemanticAdjudicationDecisionType.DROP_ABORTED.value,
            SemanticAdjudicationDecisionType.DROP_LEFT.value,
            SemanticAdjudicationDecisionType.DROP_RIGHT.value,
            SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW.value,
            SemanticAdjudicationDecisionType.NO_DECISION.value,
        ],
    )


def request_from_final_target_payload(payload: dict[str, Any]) -> SemanticAdjudicationRequest:
    issue_id = str(payload.get("issue_id") or payload.get("cluster_id") or "")
    cluster_type = str(payload.get("cluster_type") or payload.get("issue_type") or payload.get("type") or "")
    issue_type = _issue_type_from_final_target_cluster_type(cluster_type)
    severity = _severity_from_raw(str(payload.get("severity") or payload.get("confidence") or "medium"))
    candidates = [row for row in payload.get("candidates") or [] if isinstance(row, dict)]
    left = candidates[0] if candidates else {}
    right = candidates[1] if len(candidates) > 1 else {}
    candidate_ids = [
        str(row.get("candidate_id") or row.get("subtitle_uid") or "")
        for row in candidates
        if str(row.get("candidate_id") or row.get("subtitle_uid") or "")
    ]
    allowed_decisions = [str(item) for item in payload.get("allowed_decisions") or [] if str(item)]
    if not allowed_decisions:
        allowed_decisions = [
            SemanticAdjudicationDecisionType.KEEP_ALL.value,
            SemanticAdjudicationDecisionType.DROP_LEFT.value,
            SemanticAdjudicationDecisionType.DROP_RIGHT.value,
            SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW.value,
            SemanticAdjudicationDecisionType.NO_DECISION.value,
        ]
    return SemanticAdjudicationRequest(
        issue_id=issue_id,
        issue_type=issue_type,
        severity=severity,
        candidate_segment_ids=candidate_ids,
        candidate_caption_ids=candidate_ids,
        word_ids=[],
        text_before=str(left.get("text") or payload.get("left_text") or payload.get("text_before") or payload.get("text") or ""),
        text_after=str(right.get("text") or payload.get("right_text") or payload.get("text_after") or ""),
        local_context={
            "cluster_id": issue_id,
            "type": "final_target_repeat",
            "cluster_type": cluster_type,
            "candidates": candidates,
        },
        recommended_action=str(payload.get("recommended_action") or payload.get("suggested_for_rough_cut") or SemanticAdjudicationDecisionType.NO_DECISION.value),
        why_local_policy_cannot_decide=str(
            payload.get("why_local_policy_cannot_decide")
            or "final target repeat candidate requires semantic adjudication after final timeline compilation"
        ),
        allowed_decisions=allowed_decisions,
    )


def payload_from_request(request: SemanticAdjudicationRequest) -> dict[str, Any]:
    payload = semantic_contract_to_dict(request)
    payload["cluster_id"] = request.issue_id
    payload["repeat_type"] = request.issue_type.value
    payload["type"] = _legacy_payload_type(request)
    payload["text"] = request.text_before
    payload["left_text"] = request.text_before
    payload["right_text"] = request.text_after
    payload["allowed_decisions"] = list(request.allowed_decisions)
    payload["recommended_action"] = request.recommended_action
    payload["suggested_for_rough_cut"] = request.recommended_action
    payload["why_local_policy_cannot_decide"] = request.why_local_policy_cannot_decide
    local = request.local_context or {}
    if isinstance(local.get("variants"), list):
        payload["variants"] = list(local.get("variants") or [])
    if isinstance(local.get("local_evidence"), list):
        payload["local_evidence"] = list(local.get("local_evidence") or [])
    candidate = local.get("candidate") if isinstance(local.get("candidate"), dict) else {}
    if request.issue_type == SemanticIssueType.MODIFIER_REDUNDANCY:
        payload["suggested_for_rough_cut"] = "drop_redundant_modifier"
        payload.update(
            {
                "modifiers": list(candidate.get("modifiers") or []),
                "head": str(candidate.get("head") or ""),
                "fatal_modifier_redundancy_keep_all_allowed": request.severity not in HIGH_RISK_SEVERITIES,
                "required_decision_schema": {
                    "decision": "drop_redundant_modifier | requires_human_review"
                    if request.severity in HIGH_RISK_SEVERITIES
                    else "keep_all | drop_redundant_modifier | requires_human_review",
                    "keep_unit_id": "",
                    "drop_unit_ids": [],
                    "reason": "",
                    "confidence": 0.0,
                    "requires_human_review": False,
                },
            }
        )
    else:
        payload["required_decision_schema"] = {
            "decision": " | ".join(request.allowed_decisions),
            "keep_unit_id": "",
            "drop_unit_ids": [],
            "reason": "",
            "confidence": 0.0,
            "requires_human_review": False,
        }
    return payload


def legacy_row_from_adjudication_decision(
    decision: SemanticAdjudicationDecision,
    cluster: RepeatCluster,
) -> dict[str, Any]:
    issue_type = issue_type_for_cluster(cluster)
    severity = severity_for_cluster(cluster)
    allowed_decisions = set(request_from_cluster(cluster).allowed_decisions)
    if decision.decision.value not in allowed_decisions:
        return {
            "cluster_id": cluster.cluster_id,
            "_blocker_code": semantic_blocker_code(issue_type, severity),
            "_severity": "write_blocker",
            "_message": "semantic provider returned a decision not allowed for this issue",
            "_decision": decision.decision.value,
        }
    if decision.decision == SemanticAdjudicationDecisionType.KEEP_ALL and severity in HIGH_RISK_SEVERITIES:
        return {
            "cluster_id": cluster.cluster_id,
            "_blocker_code": semantic_blocker_code(issue_type, severity),
            "_severity": "write_blocker",
            "_message": "high/fatal semantic issue cannot be resolved with keep_all",
            "_decision": decision.decision.value,
        }
    if decision.decision in {SemanticAdjudicationDecisionType.NO_DECISION, SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW}:
        return {
            "cluster_id": cluster.cluster_id,
            "_blocker_code": semantic_blocker_code(issue_type, severity),
            "_severity": "write_blocker",
            "_message": decision.reason or "semantic provider did not make an actionable decision",
            "_decision": decision.decision.value,
        }
    if issue_type == SemanticIssueType.MODIFIER_REDUNDANCY and decision.decision in {
        SemanticAdjudicationDecisionType.DROP_RECOMMENDED,
        SemanticAdjudicationDecisionType.REPAIR_TEXT,
    }:
        candidate = _cluster_candidate(cluster)
        unit = cluster.variants[0] if cluster.variants else None
        drop_word_ids = decision.drop_word_ids or [str(item) for item in candidate.get("redundant_modifier_word_ids") or [] if str(item)]
        keep_word_ids = decision.keep_word_ids or [str(item) for item in candidate.get("keep_word_ids_after_drop") or [] if str(item)]
        return {
            "cluster_id": cluster.cluster_id,
            "_decision_kind": "unit_split",
            "split_id": f"semantic_adjudication_split_{cluster.cluster_id}",
            "unit_id": decision.unit_id or (unit.unit_id if unit is not None else ""),
            "drop_word_ids": drop_word_ids,
            "keep_word_ids": keep_word_ids,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "requires_human_review": decision.requires_human_review,
            "_decision_source": "deepseek_semantic_planner",
            "_semantic_json_decision": decision.decision.value,
        }
    if decision.decision in {
        SemanticAdjudicationDecisionType.DROP_LEFT,
        SemanticAdjudicationDecisionType.DROP_ABORTED,
    } and len(cluster.variants) >= 2:
        keep = cluster.variants[-1]
        return {
            "decision_id": f"semantic_adjudication_{cluster.cluster_id}",
            "cluster_id": cluster.cluster_id,
            "keep_unit_id": decision.keep_unit_id or keep.unit_id,
            "drop_unit_ids": decision.drop_unit_ids or [unit.unit_id for unit in cluster.variants[:-1]],
            "reason": decision.reason,
            "confidence": decision.confidence,
            "requires_human_review": decision.requires_human_review,
            "_decision_source": "deepseek_semantic_planner",
            "_semantic_json_decision": decision.decision.value,
        }
    if decision.decision == SemanticAdjudicationDecisionType.DROP_RIGHT and len(cluster.variants) >= 2:
        keep = cluster.variants[0]
        return {
            "decision_id": f"semantic_adjudication_{cluster.cluster_id}",
            "cluster_id": cluster.cluster_id,
            "keep_unit_id": decision.keep_unit_id or keep.unit_id,
            "drop_unit_ids": decision.drop_unit_ids or [unit.unit_id for unit in cluster.variants[1:]],
            "reason": decision.reason,
            "confidence": decision.confidence,
            "requires_human_review": decision.requires_human_review,
            "_decision_source": "deepseek_semantic_planner",
            "_semantic_json_decision": decision.decision.value,
        }
    keep = cluster.variants[0] if cluster.variants else None
    return {
        "decision_id": f"semantic_adjudication_{cluster.cluster_id}",
        "cluster_id": cluster.cluster_id,
        "keep_unit_id": decision.keep_unit_id or (keep.unit_id if keep is not None else ""),
        "drop_unit_ids": list(decision.drop_unit_ids),
        "reason": decision.reason,
        "confidence": decision.confidence,
        "requires_human_review": decision.requires_human_review,
        "_decision_source": "deepseek_semantic_planner",
        "_semantic_json_decision": decision.decision.value,
    }


def blocker_for_request(request: SemanticAdjudicationRequest, *, code: str | None = None, message: str = "") -> Blocker:
    return Blocker(
        code=code or semantic_blocker_code(request.issue_type, request.severity),
        message=message or "semantic adjudication request is unresolved",
        layer="decision",
        severity="write_blocker",
        context={
            "cluster_id": request.issue_id,
            "issue_id": request.issue_id,
            "issue_type": request.issue_type.value,
            "severity": request.severity.value,
            "requires_human_review": True,
            "allows_dry_run_discovery": True,
            "write_allowed": False,
        },
    )


def _cluster_candidate(cluster: RepeatCluster) -> dict[str, Any]:
    for evidence in cluster.evidence:
        metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
        candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
        if candidate:
            return dict(candidate)
    empty_candidate: dict[str, Any] = {}
    return empty_candidate


def _allowed_decisions(issue_type: SemanticIssueType, severity: SemanticIssueSeverity) -> list[str]:
    if issue_type == SemanticIssueType.MODIFIER_REDUNDANCY:
        values = [
            "drop_redundant_modifier",
            SemanticAdjudicationDecisionType.DROP_RECOMMENDED.value,
            SemanticAdjudicationDecisionType.REPAIR_TEXT.value,
            SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW.value,
            SemanticAdjudicationDecisionType.NO_DECISION.value,
        ]
        if severity not in HIGH_RISK_SEVERITIES:
            values.insert(0, SemanticAdjudicationDecisionType.KEEP_ALL.value)
        return values
    if issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE:
        return [
            SemanticAdjudicationDecisionType.DROP_ABORTED.value,
            SemanticAdjudicationDecisionType.DROP_LEFT.value,
            SemanticAdjudicationDecisionType.DROP_RIGHT.value,
            SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW.value,
            SemanticAdjudicationDecisionType.NO_DECISION.value,
        ]
    return [
        SemanticAdjudicationDecisionType.KEEP_ALL.value,
        SemanticAdjudicationDecisionType.DROP_LEFT.value,
        SemanticAdjudicationDecisionType.DROP_RIGHT.value,
        SemanticAdjudicationDecisionType.DROP_RECOMMENDED.value,
        SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW.value,
        SemanticAdjudicationDecisionType.NO_DECISION.value,
    ]


def _recommended_action(issue_type: SemanticIssueType, cluster: RepeatCluster, candidate: dict[str, Any]) -> str:
    if issue_type == SemanticIssueType.MODIFIER_REDUNDANCY:
        return SemanticAdjudicationDecisionType.REPAIR_TEXT.value
    if issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE:
        return SemanticAdjudicationDecisionType.DROP_ABORTED.value
    if cluster.local_recommendation in {"keep_right_drop_left", "boundary_prefix_containment_drop_left"}:
        return SemanticAdjudicationDecisionType.DROP_LEFT.value
    if candidate.get("suggested_decision"):
        return str(candidate.get("suggested_decision") or "")
    return SemanticAdjudicationDecisionType.NO_DECISION.value


def _why_local_policy_cannot_decide(issue_type: SemanticIssueType, cluster: RepeatCluster, candidate: dict[str, Any]) -> str:
    if issue_type == SemanticIssueType.MODIFIER_REDUNDANCY:
        if candidate.get("requires_human_review"):
            return "modifier redundancy could not be bound to safe whole-word repair"
        return "modifier redundancy is high/fatal and cannot be accepted by keep_all"
    if issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE:
        return "self-repair restart is ambiguous below deterministic drop threshold"
    return "local deterministic policy cannot safely decide this semantic issue"


def _legacy_payload_type(request: SemanticAdjudicationRequest) -> str:
    if request.issue_type == SemanticIssueType.MODIFIER_REDUNDANCY:
        return "single_variant_modifier_redundancy"
    if request.issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE:
        return "self_repair_aborted_phrase"
    return "semantic_decision_required"


def _issue_type_from_final_target_cluster_type(cluster_type: str) -> SemanticIssueType:
    raw = str(cluster_type or "").strip()
    if raw == "semantic_containment_take":
        return SemanticIssueType.SEMANTIC_CONTAINMENT
    if raw == SemanticIssueType.VISIBLE_CAPTION_REPEAT.value:
        return SemanticIssueType.VISIBLE_CAPTION_REPEAT
    if raw == SemanticIssueType.NEAR_DUPLICATE_TAKE.value:
        return SemanticIssueType.NEAR_DUPLICATE_TAKE
    return SemanticIssueType.AMBIGUOUS_REPEAT


def _severity_from_raw(raw: str) -> SemanticIssueSeverity:
    value = str(raw or "").strip().lower()
    if value in {item.value for item in SemanticIssueSeverity}:
        return SemanticIssueSeverity(value)
    if value == "high":
        return SemanticIssueSeverity.HIGH
    if value == "medium":
        return SemanticIssueSeverity.MEDIUM
    return SemanticIssueSeverity.LOW


def _severity_confidence(severity: SemanticIssueSeverity) -> float:
    if severity == SemanticIssueSeverity.FATAL:
        return 1.0
    if severity == SemanticIssueSeverity.HIGH:
        return 0.9
    if severity == SemanticIssueSeverity.MEDIUM:
        return 0.7
    return 0.4


def _provider_reason_for_issue(issue_type: SemanticIssueType, severity: SemanticIssueSeverity) -> str:
    if issue_type == SemanticIssueType.MODIFIER_REDUNDANCY:
        return "modifier_redundancy_local_repair_uncertain"
    if issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE:
        return "self_repair_aborted_phrase_ambiguous"
    if issue_type == SemanticIssueType.SEMANTIC_CONTAINMENT:
        return "semantic_containment_requires_judgment"
    if issue_type == SemanticIssueType.NEAR_DUPLICATE_TAKE:
        return "near_duplicate_take_keep_value_unclear"
    if issue_type == SemanticIssueType.VISIBLE_CAPTION_REPEAT:
        return "visible_caption_repeat_semantic_duplicate_unclear"
    if severity in HIGH_RISK_SEVERITIES:
        return "high_fatal_issue_without_deterministic_action"
    return "semantic_issue_requires_provider"
