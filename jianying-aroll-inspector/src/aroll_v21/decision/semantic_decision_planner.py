from __future__ import annotations

from typing import Any, Protocol

from aroll_text_normalize import normalize_text
from aroll_v21.decision.deepseek_semantic_planner import (
    BATCH_METADATA_FIELDS,
    DeepSeekSemanticPlannerAdapter,
    SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
    SEMANTIC_BATCH_PROVIDER_FAILED_CODE,
    SEMANTIC_BATCH_PROVIDER_MISSING_CODE,
    SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE,
)
from aroll_v21.decision.deterministic_baseline_policy import DeterministicBaselinePolicy
from aroll_v21.decision.semantic_adjudication import (
    HIGH_RISK_SEVERITIES,
    SemanticAdjudicationReportBuilder,
    SemanticIssueRouter,
    blocker_for_request,
    issue_type_for_cluster,
    normalize_semantic_mode,
    payload_from_request,
    request_from_cluster,
    semantic_blocker_code,
    severity_for_cluster,
)
from aroll_v21.decision.semantic_contracts import (
    SemanticAdjudicationDecision,
    SemanticAdjudicationDecisionType,
    SemanticAdjudicationMode,
    SemanticAdjudicationProvider,
    SemanticAdjudicationResult,
    SemanticIssueType,
    semantic_contract_to_dict,
)
from aroll_v21.ir.models import Blocker, DecisionPlan, RepeatCluster, TakeDecision, UnitSplitPlan


FORBIDDEN_DEEPSEEK_FIELDS = {
    "source_start_us",
    "source_end_us",
    "target_start_us",
    "target_end_us",
    "material_id",
    "source_material_id",
    "source_segment_id",
    "segment_id",
    "final_timeline",
    "final_edl",
    "edl",
    "draft_content",
}
SEMANTIC_PROVIDER_FAILURE_BLOCKER_CODES = {
    "SEMANTIC_DECISION_NOT_PROVIDED",
    "V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
    "V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED",
    "V21_SEMANTIC_ADJUDICATION_PROVIDER_MISSING",
    "V21_SEMANTIC_ADJUDICATION_PROVIDER_FAILED",
    SEMANTIC_BATCH_PROVIDER_MISSING_CODE,
    SEMANTIC_BATCH_PROVIDER_FAILED_CODE,
    SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
    SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE,
}

SEMANTIC_JSON_DECISIONS = {
    "keep_all",
    "drop_left",
    "drop_right",
    "keep_right_drop_left",
    "keep_left_drop_right",
    "keep_longest_drop_others",
    "drop_recommended",
    "drop_aborted",
    "drop_redundant_modifier",
    "repair_text",
    "apply_suggested_split",
    "requires_human_review",
    "no_decision",
}


def _unit_split_binding(
    cluster: RepeatCluster,
    *,
    existing_splits: list[UnitSplitPlan] | None = None,
) -> dict[str, Any]:
    if not cluster.variants:
        return _missing_split_binding("cluster_has_no_unit")
    unit = cluster.variants[0]
    drop_texts = _unit_split_drop_texts(cluster)
    reused = _reuse_existing_unit_split(unit.unit_id, drop_texts, existing_splits or [])
    if reused is not None:
        return reused
    metadata_binding = _metadata_unit_split_binding(cluster)
    if metadata_binding is not None:
        return metadata_binding
    token_binding = _word_token_unit_split_binding(cluster, drop_texts)
    if token_binding is not None:
        return token_binding
    failed_reason = _first_split_failed_reason(cluster) or (
        "word_token_binding_no_safe_whole_word_binding"
        if drop_texts
        else "drop_text_missing_for_unit_split_binding"
    )
    return _missing_split_binding(failed_reason, drop_texts=drop_texts)


def _missing_split_binding(failed_reason: str, *, drop_texts: list[str] | None = None) -> dict[str, Any]:
    return {
        "drop_word_ids": [],
        "keep_word_ids": [],
        "drop_text": (drop_texts or [""])[0] if drop_texts else "",
        "normalized_drop_text": normalize_text((drop_texts or [""])[0]) if drop_texts else "",
        "binding": "missing",
        "binding_source": "",
        "failed_reason": failed_reason,
    }


def _reuse_existing_unit_split(
    unit_id: str,
    drop_texts: list[str],
    existing_splits: list[UnitSplitPlan],
) -> dict[str, Any] | None:
    normalized_targets: list[str] = []
    for text in drop_texts:
        normalized = normalize_text(text)
        if normalized and normalized not in normalized_targets:
            normalized_targets.append(normalized)
    if not normalized_targets:
        return None
    for split in existing_splits:
        if split.unit_id != unit_id:
            continue
        split_texts = _split_plan_drop_texts(split)
        split_normalized = {normalize_text(text) for text in split_texts if normalize_text(text)}
        matched_drop_text = next((text for text in normalized_targets if text in split_normalized), "")
        if not matched_drop_text:
            continue
        return {
            "drop_word_ids": list(split.drop_word_ids),
            "keep_word_ids": list(split.keep_word_ids),
            "drop_text": matched_drop_text,
            "normalized_drop_text": matched_drop_text,
            "binding": "whole_word",
            "binding_source": "reuse_existing_split_decision",
            "reused_split_id": split.split_id,
            "failed_reason": "",
        }
    return None


def _split_plan_drop_texts(split: UnitSplitPlan) -> list[str]:
    metadata = split.metadata if isinstance(split.metadata, dict) else {}
    values: list[str] = []
    for key in ("drop_text", "normalized_drop_text"):
        value = str(metadata.get(key) or "")
        if value:
            values.append(value)
    for value in metadata.get("drop_texts") or []:
        if str(value):
            values.append(str(value))
    return values


def _metadata_unit_split_binding(cluster: RepeatCluster) -> dict[str, Any] | None:
    if not cluster.variants:
        return None
    unit = cluster.variants[0]
    for evidence in cluster.evidence:
        metadata = evidence.metadata or {}
        drop_word_ids = [str(word_id) for word_id in metadata.get("split_drop_word_ids") or [] if str(word_id)]
        keep_word_ids = [str(word_id) for word_id in metadata.get("split_keep_word_ids") or [] if str(word_id)]
        if not drop_word_ids:
            for span in metadata.get("spans") or []:
                if not isinstance(span, dict) or span.get("source") != "word_audio_sequence":
                    continue
                start = int(span.get("start_token") or 0)
                size = int(span.get("token_ngram_size") or 0)
                if size > 0:
                    drop_word_ids = unit.word_ids[start : start + size]
                    keep_word_ids = [word_id for word_id in unit.word_ids if word_id not in set(drop_word_ids)]
                    break
        if not _safe_unit_split_ids(unit, drop_word_ids, keep_word_ids):
            continue
        drop_text = str(metadata.get("split_drop_text") or metadata.get("drop_text") or "")
        normalized_drop_text = normalize_text(str(metadata.get("normalized_drop_text") or drop_text))
        return {
            "drop_word_ids": drop_word_ids,
            "keep_word_ids": keep_word_ids,
            "drop_text": drop_text,
            "normalized_drop_text": normalized_drop_text,
            "binding": "whole_word",
            "binding_source": str(metadata.get("split_binding_source") or "metadata_split_word_ids"),
            "failed_reason": "",
        }
    return None


def _word_token_unit_split_binding(cluster: RepeatCluster, drop_texts: list[str]) -> dict[str, Any] | None:
    if not cluster.variants:
        return None
    unit = cluster.variants[0]
    for evidence in cluster.evidence:
        tokens = _metadata_word_tokens(evidence.metadata or {}, unit)
        if not tokens:
            continue
        for drop_text in drop_texts:
            binding = _bind_drop_text_to_whole_word_tokens(unit, tokens, drop_text)
            if binding is not None:
                binding["binding_source"] = str(binding.get("binding_source") or "source_word_token_binding")
                return binding
    return None


def _metadata_word_tokens(metadata: dict[str, Any], unit: Any) -> list[dict[str, str]]:
    raw_tokens = metadata.get("word_tokens") or metadata.get("unit_word_tokens") or metadata.get("source_word_tokens") or []
    if not isinstance(raw_tokens, list):
        return []
    by_id: dict[str, str] = {}
    for row in raw_tokens:
        if not isinstance(row, dict):
            continue
        word_id = str(row.get("word_id") or "")
        text = normalize_text(str(row.get("text") or row.get("word_text") or ""))
        if word_id and text:
            by_id[word_id] = text
    ordered = [{"word_id": word_id, "text": by_id[word_id]} for word_id in unit.word_ids if word_id in by_id]
    if len(ordered) != len(unit.word_ids):
        return []
    return ordered


def _bind_drop_text_to_whole_word_tokens(
    unit: Any,
    tokens: list[dict[str, str]],
    drop_text: str,
) -> dict[str, Any] | None:
    target = normalize_text(drop_text)
    if not target:
        return None
    exact_spans = _exact_whole_word_spans(tokens, target)
    if not exact_spans:
        return None
    for span in exact_spans:
        if any(other["start"] >= span["end"] for other in exact_spans):
            return _split_binding_from_span(unit, span, drop_text, "exact_repeated_ngram")
        if _later_prefix_repeat_exists(tokens, span, target):
            return _split_binding_from_span(unit, span, drop_text, "short_phrase_before_longer_prefix_word")
    return None


def _exact_whole_word_spans(tokens: list[dict[str, str]], target: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for start in range(len(tokens)):
        joined = ""
        word_ids: list[str] = []
        for end in range(start, len(tokens)):
            joined += normalize_text(tokens[end]["text"])
            word_ids.append(str(tokens[end]["word_id"]))
            if joined == target:
                spans.append({"start": start, "end": end + 1, "word_ids": list(word_ids)})
                break
            if len(joined) >= len(target) or not target.startswith(joined):
                break
    return spans


def _later_prefix_repeat_exists(tokens: list[dict[str, str]], span: dict[str, Any], target: str) -> bool:
    for start in range(int(span["end"]), len(tokens)):
        joined = ""
        for end in range(start, len(tokens)):
            joined += normalize_text(tokens[end]["text"])
            if joined == target:
                return True
            if joined.startswith(target):
                return True
            if not target.startswith(joined):
                break
    return False


def _split_binding_from_span(unit: Any, span: dict[str, Any], drop_text: str, binding_source: str) -> dict[str, Any]:
    drop_word_ids = [str(word_id) for word_id in span.get("word_ids") or [] if str(word_id)]
    keep_word_ids = [word_id for word_id in unit.word_ids if word_id not in set(drop_word_ids)]
    if not _safe_unit_split_ids(unit, drop_word_ids, keep_word_ids):
        return _missing_split_binding("word_token_binding_produced_unsafe_word_ids", drop_texts=[drop_text])
    return {
        "drop_word_ids": drop_word_ids,
        "keep_word_ids": keep_word_ids,
        "drop_text": drop_text,
        "normalized_drop_text": normalize_text(drop_text),
        "binding": "whole_word",
        "binding_source": binding_source,
        "failed_reason": "",
    }


def _safe_unit_split_ids(unit: Any, drop_word_ids: list[str], keep_word_ids: list[str]) -> bool:
    unit_word_ids = set(unit.word_ids)
    drop_set = set(drop_word_ids)
    keep_set = set(keep_word_ids)
    return bool(drop_set and keep_set and drop_set < unit_word_ids and keep_set <= unit_word_ids and not (drop_set & keep_set))


def _unit_split_drop_texts(cluster: RepeatCluster) -> list[str]:
    values: list[str] = []
    for evidence in cluster.evidence:
        metadata = evidence.metadata or {}
        for key in ("split_drop_text", "drop_text", "normalized_drop_text"):
            value = str(metadata.get(key) or "")
            if value:
                values.append(value)
        candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
        for key in ("drop_text", "normalized_drop_text", "phrase", "overlap"):
            value = str(candidate.get(key) or "")
            if value:
                values.append(value)
        for span in metadata.get("spans") or []:
            if isinstance(span, dict):
                value = str(span.get("phrase") or span.get("overlap") or "")
                if value:
                    values.append(value)
    result: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in {normalize_text(item) for item in result}:
            result.append(value)
    return result


def _first_split_failed_reason(cluster: RepeatCluster) -> str:
    for evidence in cluster.evidence:
        metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
        reason = str(metadata.get("split_failed_reason") or metadata.get("failed_reason") or "")
        if reason:
            return reason
    return ""


class DeepSeekSemanticPlanner(Protocol):
    def decide(self, clusters: list[RepeatCluster]) -> list[dict[str, Any]]:
        """Return semantic decisions only: keep/drop unit ids, reason, confidence."""


class SemanticDecisionsJsonPlanner:
    """Manual/cloud semantic decision adapter with no physical editing authority."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows if isinstance(row, dict)]
        self.rows_by_cluster = {str(row.get("cluster_id") or ""): dict(row) for row in self.rows}

    def decide(self, clusters: list[RepeatCluster]) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        for cluster in clusters:
            raw = self.rows_by_cluster.get(cluster.cluster_id)
            if raw is None:
                decisions.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "_blocker_code": "SEMANTIC_DECISION_NOT_PROVIDED",
                        "_severity": "write_blocker",
                        "_message": "semantic decisions json does not cover this unresolved cluster",
                    }
                )
                continue
            forbidden = sorted(FORBIDDEN_DEEPSEEK_FIELDS & set(raw.keys()))
            if forbidden:
                decisions.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "_blocker_code": "SEMANTIC_DECISION_HAS_PHYSICAL_FIELDS",
                        "_message": "semantic decisions json contains forbidden physical timeline/material fields",
                        "_forbidden_fields": forbidden,
                    }
                )
                continue
            decision = str(raw.get("decision") or "").strip()
            if decision not in SEMANTIC_JSON_DECISIONS:
                decisions.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "_blocker_code": "SEMANTIC_DECISION_SCHEMA_INVALID",
                        "_message": "semantic decisions json uses an unsupported decision value",
                        "_decision": decision,
                    }
                )
                continue
            mapped = self._map_decision(cluster, raw, decision)
            decisions.append(mapped)
        return decisions

    def _map_decision(self, cluster: RepeatCluster, raw: dict[str, Any], decision: str) -> dict[str, Any]:
        if not cluster.variants:
            return {
                "cluster_id": cluster.cluster_id,
                "_blocker_code": "SEMANTIC_DECISION_SCHEMA_INVALID",
                "_message": "semantic cluster has no variants",
            }
        issue_type = issue_type_for_cluster(cluster)
        if issue_type == SemanticIssueType.SELF_REPAIR_ABORTED_PHRASE and decision == "keep_all":
            return {
                "cluster_id": cluster.cluster_id,
                "_blocker_code": semantic_blocker_code(issue_type, severity_for_cluster(cluster)),
                "_severity": "write_blocker",
                "_message": "self-repair aborted phrase cannot be accepted with keep_all",
                "_decision": decision,
            }
        if decision == "keep_all" and self._is_high_risk_semantic_issue(cluster):
            return {
                "cluster_id": cluster.cluster_id,
                "_blocker_code": semantic_blocker_code(issue_type_for_cluster(cluster), severity_for_cluster(cluster)),
                "_severity": "write_blocker",
                "_message": "high/fatal semantic issue cannot be accepted with keep_all",
                "_decision": decision,
            }
        if decision in {"requires_human_review", "no_decision"}:
            return {
                "cluster_id": cluster.cluster_id,
                "_blocker_code": semantic_blocker_code(issue_type_for_cluster(cluster), severity_for_cluster(cluster)),
                "_severity": "write_blocker",
                "_message": "semantic decision json did not provide an actionable repair/drop decision",
                "_decision": decision,
            }
        if decision == "drop_redundant_modifier":
            return self._map_modifier_split_decision(cluster, raw)
        if decision in {"repair_text", "drop_recommended"} and cluster.repeat_type == "modifier_redundancy":
            return self._map_modifier_split_decision(cluster, raw)
        if decision == "apply_suggested_split":
            return self._map_unit_split_decision(cluster, raw)
        keep = cluster.variants[0]
        drops: list[str] = []
        requires_human_review = bool(raw.get("requires_human_review")) or decision == "requires_human_review"
        if decision in {"drop_left", "keep_right_drop_left", "drop_aborted"}:
            if len(cluster.variants) < 2:
                return {
                    "cluster_id": cluster.cluster_id,
                    "_blocker_code": "SEMANTIC_DECISION_SCHEMA_INVALID",
                    "_message": "drop_left requires at least two variants",
                }
            keep = cluster.variants[-1]
            drops = [unit.unit_id for unit in cluster.variants[:-1]]
        elif decision in {"drop_right", "keep_left_drop_right"}:
            if len(cluster.variants) < 2:
                return {
                    "cluster_id": cluster.cluster_id,
                    "_blocker_code": "SEMANTIC_DECISION_SCHEMA_INVALID",
                    "_message": "drop_right requires at least two variants",
                }
            keep = cluster.variants[0]
            drops = [unit.unit_id for unit in cluster.variants[1:]]
        elif decision == "keep_longest_drop_others":
            keep = max(cluster.variants, key=lambda unit: len(unit.normalized_text or unit.text))
            drops = [unit.unit_id for unit in cluster.variants if unit.unit_id != keep.unit_id]
        return {
            "decision_id": str(raw.get("decision_id") or f"semantic_json_{cluster.cluster_id}"),
            "cluster_id": cluster.cluster_id,
            "keep_unit_id": keep.unit_id,
            "drop_unit_ids": drops,
            "reason": str(raw.get("reason") or decision),
            "confidence": float(raw.get("confidence") or 0.0),
            "requires_human_review": requires_human_review,
            "_decision_source": "semantic_decisions_json",
            "_semantic_json_decision": decision,
        }

    def _map_modifier_split_decision(self, cluster: RepeatCluster, raw: dict[str, Any]) -> dict[str, Any]:
        if cluster.repeat_type != "modifier_redundancy" or len(cluster.variants) != 1:
            return {
                "cluster_id": cluster.cluster_id,
                "_blocker_code": "SEMANTIC_DECISION_SCHEMA_INVALID",
                "_message": "drop_redundant_modifier applies only to single-variant modifier redundancy clusters",
            }
        unit = cluster.variants[0]
        candidate = self._modifier_candidate(cluster)
        drop_word_ids = [str(word_id) for word_id in candidate.get("redundant_modifier_word_ids") or [] if str(word_id)]
        keep_word_ids = [str(word_id) for word_id in candidate.get("keep_word_ids_after_drop") or [] if str(word_id)]
        unit_word_ids = set(unit.word_ids)
        if not drop_word_ids or not keep_word_ids or not set(drop_word_ids) < unit_word_ids or set(drop_word_ids) & set(keep_word_ids):
            return {
                "cluster_id": cluster.cluster_id,
                "_blocker_code": "MODIFIER_REDUNDANCY_WORD_BINDING_MISSING",
                "_message": "drop_redundant_modifier could not be bound to whole word ids",
            }
        return {
            "cluster_id": cluster.cluster_id,
            "_decision_kind": "unit_split",
            "split_id": str(raw.get("decision_id") or f"semantic_json_split_{cluster.cluster_id}"),
            "unit_id": unit.unit_id,
            "drop_word_ids": drop_word_ids,
            "keep_word_ids": keep_word_ids,
            "reason": str(raw.get("reason") or "drop redundant modifier before same head"),
            "confidence": float(raw.get("confidence") or 0.0),
            "requires_human_review": bool(raw.get("requires_human_review")),
            "_decision_source": "semantic_decisions_json",
            "_semantic_json_decision": "drop_redundant_modifier",
        }

    def _map_unit_split_decision(self, cluster: RepeatCluster, raw: dict[str, Any]) -> dict[str, Any]:
        if cluster.local_recommendation != "requires_unit_split" or len(cluster.variants) != 1:
            return {
                "cluster_id": cluster.cluster_id,
                "_blocker_code": "SEMANTIC_DECISION_SCHEMA_INVALID",
                "_message": "apply_suggested_split applies only to single-unit split review clusters",
            }
        unit = cluster.variants[0]
        split_ids = self._suggested_unit_split_word_ids(cluster)
        drop_word_ids = split_ids["drop_word_ids"]
        keep_word_ids = split_ids["keep_word_ids"]
        unit_word_ids = set(unit.word_ids)
        if not drop_word_ids or not keep_word_ids or not set(drop_word_ids) < unit_word_ids or set(drop_word_ids) & set(keep_word_ids):
            return {
                "cluster_id": cluster.cluster_id,
                "_blocker_code": "UNIT_SPLIT_WORD_BINDING_MISSING",
                "_message": "apply_suggested_split could not be bound to whole word ids",
            }
        return {
            "cluster_id": cluster.cluster_id,
            "_decision_kind": "unit_split",
            "split_id": str(raw.get("decision_id") or f"semantic_json_split_{cluster.cluster_id}"),
            "unit_id": unit.unit_id,
            "drop_word_ids": drop_word_ids,
            "keep_word_ids": keep_word_ids,
            "reason": str(raw.get("reason") or "apply suggested whole-word split"),
            "confidence": float(raw.get("confidence") or 0.0),
            "requires_human_review": bool(raw.get("requires_human_review")),
            "_decision_source": "semantic_decisions_json",
            "_semantic_json_decision": "apply_suggested_split",
        }

    def _suggested_unit_split_word_ids(self, cluster: RepeatCluster) -> dict[str, list[str]]:
        binding = _unit_split_binding(cluster)
        return {"drop_word_ids": list(binding["drop_word_ids"]), "keep_word_ids": list(binding["keep_word_ids"])}

    def _modifier_candidate(self, cluster: RepeatCluster) -> dict[str, Any]:
        for evidence in cluster.evidence:
            metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
            candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
            if candidate:
                return candidate
        return {}

    def _is_high_risk_semantic_issue(self, cluster: RepeatCluster) -> bool:
        return severity_for_cluster(cluster) in HIGH_RISK_SEVERITIES


class LocalPolicy:
    def decide(self, cluster: RepeatCluster) -> TakeDecision | UnitSplitPlan | Blocker:
        if cluster.repeat_type == "modifier_redundancy":
            return self._modifier_redundancy_decision(cluster)
        if cluster.local_recommendation in {"keep_right_drop_left", "boundary_prefix_containment_drop_left"} and len(cluster.variants) >= 2:
            keep = cluster.variants[-1]
            drops = [unit.unit_id for unit in cluster.variants[:-1]]
            is_boundary_prefix = cluster.local_recommendation == "boundary_prefix_containment_drop_left"
            return TakeDecision(
                decision_id=f"decision_{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                keep_unit_id=keep.unit_id,
                drop_unit_ids=drops,
                reason="right subtitle is complete prefix extension; drop left and keep right"
                if is_boundary_prefix
                else f"local policy resolved {cluster.repeat_type} by keeping the later complete unit",
                confidence=0.96 if is_boundary_prefix else 0.95,
                requires_human_review=False,
                source="local_policy",
            )
        if cluster.local_recommendation == "compiler_boundary_suffix_prefix_overlap_cleanup" and cluster.variants:
            return TakeDecision(
                decision_id=f"decision_{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                keep_unit_id=cluster.variants[0].unit_id,
                drop_unit_ids=[],
                reason="boundary suffix-prefix overlap is deferred to deterministic final timeline compiler cleanup",
                confidence=0.95,
                requires_human_review=False,
                source="local_policy",
            )
        if cluster.local_recommendation == "self_repair_drop_aborted" and len(cluster.variants) >= 2:
            keep = cluster.variants[-1]
            return TakeDecision(
                decision_id=f"decision_{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                keep_unit_id=keep.unit_id,
                drop_unit_ids=[unit.unit_id for unit in cluster.variants[:-1]],
                reason="high-confidence self-repair restart drops the incomplete aborted phrase",
                confidence=0.92,
                requires_human_review=False,
                source="local_policy",
            )
        if cluster.local_recommendation == "boundary_prefix_containment_requires_human_review":
            return Blocker(
                code="BOUNDARY_PREFIX_CONTAINMENT_REQUIRES_HUMAN_REVIEW",
                message="boundary prefix containment was detected but is not safe for automatic drop-left compilation",
                layer="decision",
                severity="write_blocker",
                context={"cluster_id": cluster.cluster_id, "unit_ids": [unit.unit_id for unit in cluster.variants]},
            )
        if cluster.local_recommendation == "requires_unit_split":
            return self._split_decision(cluster)
        return Blocker(
            code="SEMANTIC_DECISION_REQUIRED",
            message="cluster needs semantic decision before timeline compilation",
            layer="decision",
            context={"cluster_id": cluster.cluster_id, "repeat_type": cluster.repeat_type},
        )

    def can_resolve_without_semantic(self, cluster: RepeatCluster) -> bool:
        if cluster.local_recommendation == "self_repair_drop_aborted" and len(cluster.variants) >= 2:
            return True
        return isinstance(self._modifier_redundancy_decision(cluster), UnitSplitPlan)

    def _modifier_redundancy_decision(self, cluster: RepeatCluster) -> UnitSplitPlan | Blocker:
        if cluster.repeat_type != "modifier_redundancy" or len(cluster.variants) != 1:
            return Blocker(
                code="V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
                message="fatal modifier redundancy requires a single edit unit for deterministic repair",
                layer="decision",
                severity="write_blocker",
                context={"cluster_id": cluster.cluster_id, "repeat_type": cluster.repeat_type},
            )
        unit = cluster.variants[0]
        if unit.cut_policy == "unsafe":
            return Blocker(
                code="V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
                message="fatal modifier redundancy could not be repaired because the edit unit is unsafe to cut",
                layer="decision",
                severity="write_blocker",
                context={"cluster_id": cluster.cluster_id, "unit_id": unit.unit_id},
            )
        candidate = self._modifier_candidate(cluster)
        if str(candidate.get("severity") or "fatal") not in {"fatal", "high"}:
            return Blocker(
                code="SEMANTIC_DECISION_REQUIRED",
                message="modifier redundancy needs semantic decision before timeline compilation",
                layer="decision",
                severity="write_blocker",
                context={"cluster_id": cluster.cluster_id, "repeat_type": cluster.repeat_type},
            )
        drop_word_ids = [str(word_id) for word_id in candidate.get("redundant_modifier_word_ids") or [] if str(word_id)]
        keep_word_ids = [str(word_id) for word_id in candidate.get("keep_word_ids_after_drop") or [] if str(word_id)]
        unit_word_ids = set(unit.word_ids)
        if drop_word_ids and keep_word_ids and set(drop_word_ids) < unit_word_ids and not (set(drop_word_ids) & set(keep_word_ids)):
            return UnitSplitPlan(
                split_id=f"split_{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                unit_id=unit.unit_id,
                drop_word_ids=drop_word_ids,
                keep_word_ids=keep_word_ids,
                reason="drop redundant modifier before same head",
                source="local_policy",
                requires_human_review=False,
                metadata={
                    "repeat_type": "modifier_redundancy",
                    "candidate_type": str(candidate.get("type") or ""),
                    "suggested_decision": "drop_redundant_modifier",
                },
            )
        return Blocker(
            code="V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED",
            message="fatal modifier redundancy could not be bound to whole word ids for deterministic repair",
            layer="decision",
            severity="write_blocker",
            context={
                "cluster_id": cluster.cluster_id,
                "repeat_type": "modifier_redundancy",
                "unit_id": unit.unit_id,
            },
        )

    def _modifier_candidate(self, cluster: RepeatCluster) -> dict[str, Any]:
        for evidence in cluster.evidence:
            metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
            candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
            if candidate:
                return candidate
        return {}

    def _split_decision(self, cluster: RepeatCluster) -> UnitSplitPlan | Blocker:
        if len(cluster.variants) != 1:
            return Blocker(
                code="UNIT_SPLIT_REQUIRES_HUMAN_REVIEW",
                message="unit split candidate does not map to exactly one edit unit",
                layer="decision",
                context={"cluster_id": cluster.cluster_id, "repeat_type": cluster.repeat_type},
            )
        unit = cluster.variants[0]
        if unit.cut_policy == "unsafe":
            return Blocker(
                code="UNIT_SPLIT_UNSAFE_BOUNDARY",
                message="unit split is blocked because edit unit cut policy is unsafe",
                layer="decision",
                context={"cluster_id": cluster.cluster_id, "unit_id": unit.unit_id},
            )
        binding = _unit_split_binding(cluster)
        drop_word_ids = list(binding["drop_word_ids"])
        keep_word_ids = list(binding["keep_word_ids"])
        if _safe_unit_split_ids(unit, drop_word_ids, keep_word_ids):
            return UnitSplitPlan(
                split_id=f"split_{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                unit_id=unit.unit_id,
                drop_word_ids=drop_word_ids,
                keep_word_ids=keep_word_ids,
                reason="repeat evidence bound to safe whole-word split",
                source="local_policy",
                requires_human_review=False,
                metadata={
                    "repeat_type": cluster.repeat_type,
                    "drop_text": str(binding.get("drop_text") or ""),
                    "normalized_drop_text": str(binding.get("normalized_drop_text") or ""),
                    "binding_source": str(binding.get("binding_source") or ""),
                    "reused_split_id": str(binding.get("reused_split_id") or ""),
                },
            )
        return Blocker(
            code="UNIT_SPLIT_REQUIRES_HUMAN_REVIEW",
            message="repeat evidence requires unit split but no safe whole-word split span was available",
            layer="decision",
            context={
                "cluster_id": cluster.cluster_id,
                "repeat_type": cluster.repeat_type,
                "unit_ids": [unit.unit_id],
                "failed_reason": str(binding.get("failed_reason") or "whole_word_binding_missing"),
                "drop_text": str(binding.get("drop_text") or ""),
            },
        )


class SemanticDecisionPlanner:
    def __init__(
        self,
        local_policy: LocalPolicy | None = None,
        deepseek_planner: DeepSeekSemanticPlanner | None = None,
        *,
        semantic_mode: str = "auto",
        semantic_provider: SemanticAdjudicationProvider | None = None,
        issue_router: SemanticIssueRouter | None = None,
    ) -> None:
        self.local_policy = local_policy or LocalPolicy()
        self.issue_router = issue_router or SemanticIssueRouter()
        self.semantic_mode = normalize_semantic_mode(semantic_mode)
        if semantic_provider is not None and deepseek_planner is None:
            deepseek_planner = DeepSeekSemanticPlannerAdapter(semantic_provider)
        self.deepseek_planner = deepseek_planner
        self.semantic_provider_configured = semantic_provider is not None or bool(getattr(deepseek_planner, "deepseek_provider_configured", False))
        self.semantic_provider_config_source = str(
            getattr(semantic_provider, "config_source", "")
            or getattr(deepseek_planner, "deepseek_provider_config_source", "")
            or ""
        )
        self.semantic_decision_cache_used = bool(getattr(deepseek_planner, "semantic_decision_cache_used", False))

    def plan(self, clusters: list[RepeatCluster]) -> DecisionPlan:
        blockers: list[Blocker] = []
        decisions: list[TakeDecision] = []
        split_decisions: list[UnitSplitPlan] = []
        semantic_request_payloads: list[dict[str, Any]] = []
        decision_trace: list[dict[str, Any]] = []
        semantic_unresolved_cluster_ids: set[str] = set()
        adjudication_report = SemanticAdjudicationReportBuilder(
            mode=self.semantic_mode,
            provider_configured=self.semantic_provider_configured,
            semantic_decision_cache_used=self.semantic_decision_cache_used,
        )
        deterministic_available_by_cluster = {
            cluster.cluster_id: self.local_policy.can_resolve_without_semantic(cluster)
            for cluster in clusters
        }
        routing_by_cluster = {
            cluster.cluster_id: self.issue_router.route_cluster(
                cluster,
                deterministic_action_available=deterministic_available_by_cluster[cluster.cluster_id],
            )
            for cluster in clusters
        }
        if self.semantic_mode == SemanticAdjudicationMode.AUTO:
            for route in routing_by_cluster.values():
                adjudication_report.add_route(route)
        if self.semantic_mode == SemanticAdjudicationMode.AUTO:
            semantic_clusters = [cluster for cluster in clusters if routing_by_cluster[cluster.cluster_id].requires_provider]
        else:
            semantic_clusters = [
                cluster
                for cluster in clusters
                if any(item.requires_semantic_decision for item in cluster.evidence)
                and not deterministic_available_by_cluster[cluster.cluster_id]
            ]
        semantic_decisions = self._deepseek_decisions(
            semantic_clusters,
            blockers,
            semantic_request_payloads,
            decision_trace,
            semantic_unresolved_cluster_ids,
            adjudication_report,
        ) if semantic_clusters else {}

        for cluster in clusters:
            semantic_decision = semantic_decisions.get(cluster.cluster_id)
            if semantic_decision:
                if isinstance(semantic_decision, UnitSplitPlan):
                    split_decisions.append(semantic_decision)
                    decision_trace.append(
                        self._trace(
                            cluster,
                            route="deepseek_required",
                            output_decision=semantic_decision.split_id,
                            reason=semantic_decision.reason,
                        )
                    )
                else:
                    decisions.append(semantic_decision)
                    decision_trace.append(
                        self._trace(
                            cluster,
                            route="deepseek_required",
                            output_decision=semantic_decision.decision_id,
                            reason=semantic_decision.reason,
                        )
                    )
                continue
            if cluster.cluster_id in semantic_unresolved_cluster_ids:
                decision_trace.append(
                    self._trace(
                        cluster,
                        route="self_review",
                        blocker="UNRESOLVED_SEMANTIC_REVIEW_REQUIRED",
                        reason="semantic planner is not configured; unit is conservatively kept for dry-run discovery",
                    )
                )
                continue
            if cluster.local_recommendation == "requires_unit_split":
                recovered_split = self._unit_split_from_existing_reuse(cluster, split_decisions)
                if recovered_split is not None:
                    self._record_locally_resolved_semantic_issue(adjudication_report, cluster, recovered_split.split_id, recovered_split.reason)
                    adjudication_report.add_local_decision()
                    split_decisions.append(recovered_split)
                    decision_trace.append(
                        self._trace(
                            cluster,
                            route="split_generated",
                            output_decision=recovered_split.split_id,
                            reason=recovered_split.reason,
                        )
                    )
                    continue
            local = self.local_policy.decide(cluster)
            if isinstance(local, Blocker):
                if local.code == "UNIT_SPLIT_REQUIRES_HUMAN_REVIEW":
                    recovered_split = self._unit_split_from_existing_reuse(cluster, split_decisions)
                    if recovered_split is not None:
                        self._record_locally_resolved_semantic_issue(adjudication_report, cluster, recovered_split.split_id, recovered_split.reason)
                        adjudication_report.add_local_decision()
                        split_decisions.append(recovered_split)
                        decision_trace.append(
                            self._trace(
                                cluster,
                                route="split_generated",
                                output_decision=recovered_split.split_id,
                                reason=recovered_split.reason,
                            )
                        )
                        continue
                    semantic_decision = self._semantic_decision_for_unit_split_review(cluster, blockers, decision_trace)
                    if semantic_decision is not None:
                        if isinstance(semantic_decision, UnitSplitPlan):
                            split_decisions.append(semantic_decision)
                            decision_trace.append(
                                self._trace(
                                    cluster,
                                    route="deepseek_required",
                                    output_decision=semantic_decision.split_id,
                                    reason=semantic_decision.reason,
                                )
                            )
                        else:
                            decisions.append(semantic_decision)
                        continue
                    self._append_unit_split_semantic_request(semantic_request_payloads, cluster, local)
                blockers.append(local)
                route = "split_required" if local.code.startswith("UNIT_SPLIT") else "blocked"
                decision_trace.append(self._trace(cluster, route=route, blocker=local.code, reason=local.message))
            elif isinstance(local, UnitSplitPlan):
                self._record_locally_resolved_semantic_issue(adjudication_report, cluster, local.split_id, local.reason)
                adjudication_report.add_local_decision()
                split_decisions.append(local)
                decision_trace.append(self._trace(cluster, route="split_generated", output_decision=local.split_id, reason=local.reason))
            else:
                self._record_locally_resolved_semantic_issue(adjudication_report, cluster, local.decision_id, local.reason)
                adjudication_report.add_local_decision()
                decisions.append(local)
                if cluster.local_recommendation == "boundary_prefix_containment_drop_left":
                    route = "boundary_prefix_containment"
                elif cluster.local_recommendation == "self_repair_drop_aborted":
                    route = "self_repair_aborted_phrase"
                else:
                    route = "local_policy"
                decision_trace.append(self._trace(cluster, route=route, output_decision=local.decision_id, reason=local.reason))
        write_blocker_present = any(blocker.severity == "write_blocker" for blocker in blockers)
        human_review_required = bool(semantic_unresolved_cluster_ids) or write_blocker_present or any(decision.requires_human_review for decision in decisions)
        blocker_codes = [blocker.code for blocker in blockers if blocker.severity in {"fatal", "write_blocker"}]
        if self.deepseek_planner is not None:
            adjudication_report.commit_reused_semantic_cache = bool(getattr(self.deepseek_planner, "commit_reused_semantic_cache", False))
            adjudication_report.semantic_cache_input_hash = str(getattr(self.deepseek_planner, "semantic_cache_input_hash", "") or "")
            adjudication_report.semantic_cache_issue_count = int(getattr(self.deepseek_planner, "semantic_cache_issue_count", 0) or 0)
            adjudication_report.semantic_cache_resolved_count = int(getattr(self.deepseek_planner, "semantic_cache_resolved_count", 0) or 0)
            adjudication_report.semantic_cache_unresolved_count = int(getattr(self.deepseek_planner, "semantic_cache_unresolved_count", 0) or 0)
        final_adjudication_report = adjudication_report.to_report(
            unresolved_issue_ids=semantic_unresolved_cluster_ids,
            blocker_codes=blocker_codes,
        )
        final_adjudication_report["deepseek_provider_config_source"] = self.semantic_provider_config_source
        return DecisionPlan(
            decisions=decisions,
            split_decisions=split_decisions,
            blocked=any(blocker.severity == "fatal" for blocker in blockers),
            blockers=blockers,
            semantic_request_payloads=semantic_request_payloads,
            decision_trace=decision_trace,
            semantic_decision_rows=list(getattr(self.deepseek_planner, "rows", []) or []),
            semantic_adjudication_report=final_adjudication_report,
            semantic_unresolved_count=len(semantic_unresolved_cluster_ids),
            requires_human_review=human_review_required,
            write_allowed=not semantic_unresolved_cluster_ids and not human_review_required and not write_blocker_present,
            dry_run_continued_for_discovery=bool(semantic_unresolved_cluster_ids),
        )

    def _record_locally_resolved_semantic_issue(
        self,
        report: SemanticAdjudicationReportBuilder,
        cluster: RepeatCluster,
        decision_id: str,
        reason: str,
    ) -> None:
        if not any(evidence.requires_semantic_decision for evidence in cluster.evidence) and cluster.local_recommendation != "self_repair_drop_aborted":
            return
        request = request_from_cluster(cluster)
        report.add_result(
            SemanticAdjudicationResult(
                request=request,
                decision=SemanticAdjudicationDecision(
                    issue_id=cluster.cluster_id,
                    decision=SemanticAdjudicationDecisionType.DROP_ABORTED
                    if cluster.local_recommendation == "self_repair_drop_aborted"
                    else SemanticAdjudicationDecisionType.REPAIR_TEXT,
                    reason=reason,
                    confidence=0.92,
                    provider_name="local_policy",
                    metadata={"decision_id": decision_id},
                ),
                resolved=True,
                provider_configured=report.provider_configured,
                provider_called=False,
            )
        )

    def _semantic_decision_for_unit_split_review(
        self,
        cluster: RepeatCluster,
        blockers: list[Blocker],
        decision_trace: list[dict[str, Any]],
    ) -> TakeDecision | UnitSplitPlan | None:
        if self.deepseek_planner is None:
            return None
        if isinstance(self.deepseek_planner, DeepSeekSemanticPlannerAdapter):
            return None
        temp_blockers: list[Blocker] = []
        temp_unresolved: set[str] = set()
        temp_report = SemanticAdjudicationReportBuilder(
            mode=self.semantic_mode,
            provider_configured=self.semantic_provider_configured,
            semantic_decision_cache_used=self.semantic_decision_cache_used,
        )
        decisions = self._deepseek_decisions([cluster], temp_blockers, [], decision_trace, temp_unresolved, temp_report)
        decision = decisions.get(cluster.cluster_id)
        if decision is not None:
            return decision
        for blocker in temp_blockers:
            if blocker.code == "SEMANTIC_DECISION_NOT_PROVIDED":
                continue
            blockers.append(blocker)
        return None

    def _append_unit_split_semantic_request(
        self,
        semantic_request_payloads: list[dict[str, Any]],
        cluster: RepeatCluster,
        blocker: Blocker,
    ) -> None:
        existing = {str(payload.get("cluster_id") or "") for payload in semantic_request_payloads}
        if cluster.cluster_id in existing:
            return
        semantic_request_payloads.append(self._unit_split_semantic_request_payload(cluster, blocker))

    def _unit_split_from_existing_reuse(
        self,
        cluster: RepeatCluster,
        split_decisions: list[UnitSplitPlan],
    ) -> UnitSplitPlan | None:
        if len(cluster.variants) != 1:
            return None
        unit = cluster.variants[0]
        if unit.cut_policy == "unsafe":
            return None
        binding = _reuse_existing_unit_split(unit.unit_id, _unit_split_drop_texts(cluster), split_decisions)
        if binding is None:
            return None
        drop_word_ids = list(binding["drop_word_ids"])
        keep_word_ids = list(binding["keep_word_ids"])
        if not _safe_unit_split_ids(unit, drop_word_ids, keep_word_ids):
            return None
        return UnitSplitPlan(
            split_id=f"split_{cluster.cluster_id}",
            cluster_id=cluster.cluster_id,
            unit_id=unit.unit_id,
            drop_word_ids=drop_word_ids,
            keep_word_ids=keep_word_ids,
            reason="repeat evidence recovered safe whole-word split binding",
            source="local_policy",
            requires_human_review=False,
            metadata={
                "repeat_type": cluster.repeat_type,
                "drop_text": str(binding.get("drop_text") or ""),
                "normalized_drop_text": str(binding.get("normalized_drop_text") or ""),
                "binding_source": str(binding.get("binding_source") or ""),
                "reused_split_id": str(binding.get("reused_split_id") or ""),
            },
        )

    def _suggested_unit_split_word_ids(self, cluster: RepeatCluster) -> dict[str, list[str]]:
        binding = _unit_split_binding(cluster)
        return {"drop_word_ids": list(binding["drop_word_ids"]), "keep_word_ids": list(binding["keep_word_ids"])}

    def _deepseek_decisions(
        self,
        clusters: list[RepeatCluster],
        blockers: list[Blocker],
        semantic_request_payloads: list[dict[str, Any]],
        decision_trace: list[dict[str, Any]],
        semantic_unresolved_cluster_ids: set[str],
        adjudication_report: SemanticAdjudicationReportBuilder,
    ) -> dict[str, TakeDecision | UnitSplitPlan]:
        if self.deepseek_planner is None and self.semantic_mode == SemanticAdjudicationMode.DETERMINISTIC_BASELINE:
            raw_rows = self._deterministic_baseline_rows(clusters)
        elif self.deepseek_planner is None:
            for cluster in clusters:
                request = request_from_cluster(cluster)
                payload = payload_from_request(request)
                provider_missing_code = (
                    SEMANTIC_BATCH_PROVIDER_MISSING_CODE
                    if self.semantic_mode in {SemanticAdjudicationMode.AUTO, SemanticAdjudicationMode.DEEPSEEK}
                    else "SEMANTIC_DECISION_NOT_PROVIDED"
                )
                if request.severity.value == "low" and self.semantic_mode == SemanticAdjudicationMode.AUTO:
                    semantic_request_payloads.append(payload | {"warning_only": True})
                    adjudication_report.add_result(
                        SemanticAdjudicationResult(
                            request=request,
                            resolved=False,
                            provider_configured=False,
                            provider_called=False,
                            blocker_code="SEMANTIC_PROVIDER_SKIPPED_LOW_RISK",
                            message="provider unavailable for low-risk issue; emitted warning only",
                        )
                    )
                    decision_trace.append(
                        self._trace(
                            cluster,
                            route="semantic_warning",
                            blocker="SEMANTIC_PROVIDER_SKIPPED_LOW_RISK",
                            reason="low-risk semantic issue did not require provider fail-closed",
                        )
                    )
                    continue
                semantic_unresolved_cluster_ids.add(cluster.cluster_id)
                semantic_request_payloads.append(payload)
                blockers.append(
                    blocker_for_request(
                        request,
                        code=provider_missing_code,
                        message="semantic adjudication provider is not configured",
                    )
                )
                if provider_missing_code == SEMANTIC_BATCH_PROVIDER_MISSING_CODE:
                    blockers.append(
                        blocker_for_request(
                            request,
                            code="V21_SEMANTIC_ADJUDICATION_PROVIDER_MISSING",
                            message="semantic adjudication provider is not configured",
                        )
                    )
                legacy_missing = Blocker(
                    code="DEEPSEEK_SEMANTIC_PLANNER_NOT_CONFIGURED",
                    message="semantic cluster requires DeepSeek/local semantic planner",
                    layer="decision",
                    severity="write_blocker",
                    context={
                        "cluster_id": cluster.cluster_id,
                        "repeat_type": cluster.repeat_type,
                        "missing_config": "deepseek_semantic_planner",
                        "requires_human_review": True,
                        "allows_dry_run_discovery": True,
                        "write_allowed": False,
                    },
                )
                blockers.append(legacy_missing)
                issue_blocker_code = semantic_blocker_code(request.issue_type, request.severity)
                if issue_blocker_code not in {provider_missing_code, legacy_missing.code}:
                    blockers.append(blocker_for_request(request, code=issue_blocker_code))
                adjudication_report.add_result(
                    SemanticAdjudicationResult(
                        request=request,
                        resolved=False,
                        provider_configured=False,
                        provider_called=False,
                        blocker_code=provider_missing_code,
                        message="provider missing; high/fatal semantic issue fail-closed",
                    )
                )
                decision_trace.append(
                    self._trace(
                        cluster,
                        route="deepseek_required",
                        blocker=provider_missing_code,
                        reason="semantic planner is not configured; request payload emitted",
                    )
                )
                if provider_missing_code == SEMANTIC_BATCH_PROVIDER_MISSING_CODE:
                    decision_trace.append(
                        self._trace(
                            cluster,
                            route="deepseek_required",
                            blocker="V21_SEMANTIC_ADJUDICATION_PROVIDER_MISSING",
                            reason="semantic planner is not configured; legacy provider-missing trace alias",
                        )
                    )
            return {}
        else:
            raw_rows = self.deepseek_planner.decide(clusters)
        adjudication_report.provider_configured = self.semantic_provider_configured
        adjudication_report.provider_error = str(getattr(self.deepseek_planner, "deepseek_provider_error", "") or "")
        called_count = int(getattr(self.deepseek_planner, "provider_called_count", 0) or 0)
        adjudication_report.provider_called_count = called_count
        for field_name in BATCH_METADATA_FIELDS:
            value = getattr(self.deepseek_planner, field_name, None)
            if value is not None:
                setattr(adjudication_report, field_name, value)
        decision_rows = list(getattr(self.deepseek_planner, "decision_rows", []) or [])
        decision_by_issue = {
            str(row.get("issue_id") or ""): row
            for row in decision_rows
            if isinstance(row, dict)
        }
        decisions: dict[str, TakeDecision | UnitSplitPlan] = {}
        valid_unit_ids = {unit.unit_id for cluster in clusters for unit in cluster.variants}
        valid_word_ids = {word_id for cluster in clusters for unit in cluster.variants for word_id in unit.word_ids}
        valid_cluster_ids = {cluster.cluster_id for cluster in clusters}
        processed_cluster_ids: set[str] = set()
        for index, row in enumerate(raw_rows, start=1):
            explicit_blocker = str(row.get("_blocker_code") or "")
            cluster_id = str(row.get("cluster_id") or "")
            if cluster_id:
                processed_cluster_ids.add(cluster_id)
            if explicit_blocker:
                cluster = next((item for item in clusters if item.cluster_id == cluster_id), None)
                if explicit_blocker in SEMANTIC_PROVIDER_FAILURE_BLOCKER_CODES:
                    semantic_unresolved_cluster_ids.add(cluster_id)
                    if cluster is not None:
                        request = request_from_cluster(cluster)
                        adjudication_report.add_result(
                            SemanticAdjudicationResult(
                                request=request,
                                resolved=False,
                                provider_configured=self.semantic_provider_configured,
                                provider_called=called_count > 0,
                                deterministic_baseline_refused=bool(row.get("_deterministic_baseline_refused")),
                                blocker_code=explicit_blocker,
                                message=str(row.get("_message") or explicit_blocker),
                            )
                        )
                        adjudication_report.provider_called_count = called_count
                        existing_payload_ids = {str(payload.get("cluster_id") or "") for payload in semantic_request_payloads}
                        if cluster.cluster_id not in existing_payload_ids:
                            semantic_request_payloads.append(payload_from_request(request))
                blockers.append(
                    Blocker(
                        code=explicit_blocker,
                        message=str(row.get("_message") or "semantic decision row is not usable"),
                        layer="decision",
                        severity=str(row.get("_severity") or "fatal"),  # type: ignore[arg-type]
                        context={
                            "cluster_id": cluster_id,
                            "forbidden_fields": list(row.get("_forbidden_fields") or []),
                            "decision": str(row.get("_decision") or ""),
                        },
                    )
                )
                decision_trace.append(
                    self._trace(
                        cluster or clusters[0],
                        route="blocked" if explicit_blocker != "SEMANTIC_DECISION_NOT_PROVIDED" else "self_review",
                        blocker=explicit_blocker,
                        reason=str(row.get("_message") or explicit_blocker),
                    )
                )
                continue
            forbidden = sorted(FORBIDDEN_DEEPSEEK_FIELDS & set(row.keys()))
            if forbidden:
                blockers.append(
                    Blocker(
                        code="DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS",
                        message="DeepSeek decision attempted to control physical timeline/material fields",
                        layer="decision",
                        context={"cluster_id": cluster_id, "forbidden_fields": forbidden},
                    )
                )
                decision_trace.append(
                    self._trace(
                        next((cluster for cluster in clusters if cluster.cluster_id == cluster_id), clusters[0]),
                        route="blocked",
                        blocker="DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS",
                        reason="DeepSeek decision attempted physical control",
                    )
                )
                continue
            if row.get("_decision_kind") == "unit_split":
                unit_id = str(row.get("unit_id") or "")
                drop_word_ids = [str(item) for item in row.get("drop_word_ids") or [] if str(item)]
                keep_word_ids = [str(item) for item in row.get("keep_word_ids") or [] if str(item)]
                if (
                    cluster_id not in valid_cluster_ids
                    or unit_id not in valid_unit_ids
                    or any(word_id not in valid_word_ids for word_id in drop_word_ids + keep_word_ids)
                ):
                    blockers.append(
                        Blocker(
                            code="DEEPSEEK_DECISION_UNKNOWN_UNIT",
                            message="semantic split decision referenced unknown cluster, unit, or word id",
                            layer="decision",
                            context={
                                "cluster_id": cluster_id,
                                "unit_id": unit_id,
                                "drop_word_ids": drop_word_ids,
                                "keep_word_ids": keep_word_ids,
                            },
                        )
                    )
                    continue
                decisions[cluster_id] = UnitSplitPlan(
                    split_id=str(row.get("split_id") or f"deepseek_split_{index:06d}"),
                    cluster_id=cluster_id,
                    unit_id=unit_id,
                    drop_word_ids=drop_word_ids,
                    keep_word_ids=keep_word_ids,
                    reason=str(row.get("reason") or ""),
                    source=str(row.get("_decision_source") or "deepseek_semantic_planner"),  # type: ignore[arg-type]
                    requires_human_review=bool(row.get("requires_human_review")),
                    metadata={"semantic_json_decision": str(row.get("_semantic_json_decision") or "")},
                )
                cluster = next((item for item in clusters if item.cluster_id == cluster_id), None)
                if cluster is not None:
                    adjudication_report.add_result(
                        SemanticAdjudicationResult(
                            request=request_from_cluster(cluster),
                            decision=_adjudication_decision_from_row(row, decision_by_issue.get(cluster_id)),
                            resolved=True,
                            provider_configured=self.semantic_provider_configured,
                            provider_called=called_count > 0,
                        )
                    )
                continue
            keep_unit_id = str(row.get("keep_unit_id") or "")
            drop_unit_ids = [str(item) for item in row.get("drop_unit_ids") or [] if str(item)]
            if cluster_id not in valid_cluster_ids or keep_unit_id not in valid_unit_ids or any(unit_id not in valid_unit_ids for unit_id in drop_unit_ids):
                blockers.append(
                    Blocker(
                        code="DEEPSEEK_DECISION_UNKNOWN_UNIT",
                        message="DeepSeek decision referenced unknown cluster or unit id",
                        layer="decision",
                        context={"cluster_id": cluster_id, "keep_unit_id": keep_unit_id, "drop_unit_ids": drop_unit_ids},
                    )
                )
                continue
            source = str(row.get("_decision_source") or "deepseek_semantic_planner")
            if source not in {"deepseek_semantic_planner", "semantic_decisions_json", "deterministic_baseline"}:
                source = "deepseek_semantic_planner"
            decisions[cluster_id] = TakeDecision(
                decision_id=str(row.get("decision_id") or f"deepseek_decision_{index:06d}"),
                cluster_id=cluster_id,
                keep_unit_id=keep_unit_id,
                drop_unit_ids=drop_unit_ids,
                reason=str(row.get("reason") or ""),
                confidence=float(row.get("confidence") or 0.0),
                requires_human_review=bool(row.get("requires_human_review")),
                source=source,  # type: ignore[arg-type]
            )
            cluster = next((item for item in clusters if item.cluster_id == cluster_id), None)
            if cluster is not None:
                adjudication_report.add_result(
                    SemanticAdjudicationResult(
                        request=request_from_cluster(cluster),
                        decision=_adjudication_decision_from_row(row, decision_by_issue.get(cluster_id)),
                        resolved=True,
                            provider_configured=self.semantic_provider_configured,
                        provider_called=called_count > 0,
                    )
                )
        missing_cluster_ids = {cluster.cluster_id for cluster in clusters} - processed_cluster_ids
        for cluster in clusters:
            if cluster.cluster_id not in missing_cluster_ids:
                continue
            request = request_from_cluster(cluster)
            semantic_unresolved_cluster_ids.add(cluster.cluster_id)
            existing_payload_ids = {str(payload.get("cluster_id") or "") for payload in semantic_request_payloads}
            if cluster.cluster_id not in existing_payload_ids:
                semantic_request_payloads.append(payload_from_request(request))
            is_deterministic_baseline = any(
                str(row.get("_decision_source") or row.get("decision_source") or "") == "deterministic_baseline"
                or str(row.get("_semantic_mode") or row.get("semantic_mode") or "") in {"deterministic_baseline", "deterministic-baseline"}
                for row in raw_rows
                if isinstance(row, dict)
            ) or self.semantic_mode == SemanticAdjudicationMode.DETERMINISTIC_BASELINE
            blocker_code = semantic_blocker_code(request.issue_type, request.severity)
            blockers.append(
                blocker_for_request(
                    request,
                    code=blocker_code,
                    message="semantic planner did not return an actionable decision for this high-risk issue",
                )
            )
            adjudication_report.add_result(
                SemanticAdjudicationResult(
                    request=request,
                    resolved=False,
                    provider_configured=self.semantic_provider_configured,
                    provider_called=called_count > 0,
                    deterministic_baseline_refused=is_deterministic_baseline,
                    blocker_code=blocker_code,
                    message="planner returned no decision row",
                )
            )
        adjudication_report.provider_called_count = called_count
        return decisions

    def _deterministic_baseline_rows(self, clusters: list[RepeatCluster]) -> list[dict[str, Any]]:
        policy = DeterministicBaselinePolicy()
        rows: list[dict[str, Any]] = []
        for cluster in clusters:
            keep_unit_id = cluster.variants[0].unit_id if cluster.variants else ""
            row = policy.decision_for_missing_cluster(
                cluster.cluster_id,
                cluster_type=str(cluster.repeat_type or ""),
                context={
                    "keep_unit_id": keep_unit_id,
                    "drop_unit_ids": [],
                    "reason": "deterministic baseline keeps low-risk semantic speech units only",
                    "severity": severity_for_cluster(cluster).value,
                    "requires_semantic_decision": any(item.requires_semantic_decision for item in cluster.evidence),
                    "confidence": max((float(item.confidence or 0.0) for item in cluster.evidence), default=0.0),
                },
            )
            if row is not None:
                rows.append(row)
                continue
            rows.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "_blocker_code": semantic_blocker_code(issue_type_for_cluster(cluster), severity_for_cluster(cluster)),
                    "_severity": "write_blocker",
                    "_message": "deterministic baseline refused high-risk semantic issue",
                    "_decision_source": "deterministic_baseline",
                    "_semantic_mode": "deterministic_baseline",
                    "_deterministic_baseline_refused": True,
                }
            )
        return rows

    def _unit_split_semantic_request_payload(self, cluster: RepeatCluster, blocker: Blocker) -> dict[str, Any]:
        unit = cluster.variants[0] if cluster.variants else None
        split_binding = _unit_split_binding(cluster)
        evidence_rows = []
        drop_text = str(split_binding.get("drop_text") or "")
        for evidence in cluster.evidence:
            metadata = evidence.metadata or {}
            spans = []
            for span in metadata.get("spans") or []:
                if not isinstance(span, dict):
                    continue
                safe_span = {
                    key: value
                    for key, value in span.items()
                    if key not in {"source_start_us", "source_end_us", "target_start_us", "target_end_us", "material_id", "segment_id"}
                }
                spans.append(safe_span)
                if not drop_text:
                    drop_text = str(span.get("phrase") or "")
            candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
            if candidate and not drop_text:
                drop_text = str(candidate.get("phrase") or candidate.get("overlap") or "")
            evidence_rows.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "evidence_type": evidence.evidence_type,
                    "reason": evidence.reason,
                    "confidence": evidence.confidence,
                    "spans": spans[:10],
                }
            )
        return {
            "issue_id": cluster.cluster_id,
            "cluster_id": cluster.cluster_id,
            "issue_type": "ambiguous_repeat",
            "severity": "medium",
            "type": "unit_split_requires_human_review",
            "repeat_type": "unit_split",
            "source_repeat_type": cluster.repeat_type,
            "text": unit.text if unit is not None else "",
            "text_before": unit.text if unit is not None else "",
            "text_after": "",
            "candidate_segment_ids": [unit.unit_id] if unit is not None else [],
            "candidate_caption_ids": list(unit.subtitle_uids) if unit is not None else [],
            "word_ids": list(unit.word_ids) if unit is not None else [],
            "source_start_us": int(unit.source_start_us) if unit is not None else 0,
            "source_end_us": int(unit.source_end_us) if unit is not None else 0,
            "target_start_us": 0,
            "target_end_us": 0,
            "reason": blocker.code,
            "allowed_decisions": [
                "apply_suggested_split",
                "keep_all",
                "requires_human_review",
            ],
            "recommended_action": "apply_suggested_split",
            "why_local_policy_cannot_decide": "local policy could not bind a safe whole-word split automatically",
            "local_context": {"cluster_id": cluster.cluster_id, "repeat_type": cluster.repeat_type},
            "suggested_for_rough_cut": "apply_suggested_split",
            "split_summary": {
                "drop_text": drop_text,
                "keep_text": "",
                "result_text": "",
                "drop_word_ids": list(split_binding["drop_word_ids"]),
                "keep_word_ids": list(split_binding["keep_word_ids"]),
                "binding": str(split_binding.get("binding") or "missing"),
                "binding_source": str(split_binding.get("binding_source") or ""),
                "failed_reason": str(
                    split_binding.get("failed_reason")
                    or blocker.context.get("failed_reason")
                    or ""
                ),
            },
            "local_evidence": evidence_rows,
            "required_decision_schema": {
                "decision": "apply_suggested_split | keep_all | requires_human_review",
                "reason": "",
                "confidence": 0.0,
                "requires_human_review": False,
            },
        }

    def _modifier_candidate(self, cluster: RepeatCluster) -> dict[str, Any]:
        for evidence in cluster.evidence:
            metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
            candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
            if candidate:
                return candidate
        return {}

    def _trace(
        self,
        cluster: RepeatCluster,
        *,
        route: str,
        output_decision: str = "",
        blocker: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        row = {
            "cluster_id": cluster.cluster_id,
            "repeat_type": cluster.repeat_type,
            "evidence_source": ",".join(sorted({evidence.evidence_type for evidence in cluster.evidence})),
            "route": route,
            "input_units": [unit.unit_id for unit in cluster.variants],
            "output_decision": output_decision,
            "blocker": blocker,
            "reason": reason,
        }
        if route == "boundary_prefix_containment" and len(cluster.variants) >= 2:
            row.update(
                {
                    "left_text": cluster.variants[0].text,
                    "right_text": cluster.variants[-1].text,
                    "decision": "drop_left_keep_right",
                    "source": "local_policy",
                }
            )
        if route == "self_repair_aborted_phrase" and len(cluster.variants) >= 2:
            row.update(
                {
                    "left_text": cluster.variants[0].text,
                    "right_text": cluster.variants[-1].text,
                    "decision": "drop_left_keep_right",
                    "source": "local_policy",
                    "applied": True,
                }
            )
        return row


def _adjudication_decision_from_row(
    row: dict[str, Any],
    provider_row: dict[str, Any] | None = None,
) -> SemanticAdjudicationDecision:
    provider_row = provider_row or {}
    raw_decision = str(row.get("_semantic_json_decision") or provider_row.get("decision") or "keep_all")
    if raw_decision == "drop_redundant_modifier":
        raw_decision = SemanticAdjudicationDecisionType.REPAIR_TEXT.value
    if raw_decision not in {item.value for item in SemanticAdjudicationDecisionType}:
        raw_decision = SemanticAdjudicationDecisionType.KEEP_ALL.value
    return SemanticAdjudicationDecision(
        issue_id=str(row.get("cluster_id") or provider_row.get("issue_id") or ""),
        decision=SemanticAdjudicationDecisionType(raw_decision),
        reason=str(row.get("reason") or provider_row.get("reason") or ""),
        confidence=float(row.get("confidence") or provider_row.get("confidence") or 0.0),
        provider_name=str(provider_row.get("provider_name") or row.get("_decision_source") or "deepseek_semantic_planner"),
        keep_unit_id=str(row.get("keep_unit_id") or provider_row.get("keep_unit_id") or ""),
        drop_unit_ids=[str(item) for item in row.get("drop_unit_ids") or provider_row.get("drop_unit_ids") or [] if str(item)],
        unit_id=str(row.get("unit_id") or provider_row.get("unit_id") or ""),
        drop_word_ids=[str(item) for item in row.get("drop_word_ids") or provider_row.get("drop_word_ids") or [] if str(item)],
        keep_word_ids=[str(item) for item in row.get("keep_word_ids") or provider_row.get("keep_word_ids") or [] if str(item)],
        repair_text=str(provider_row.get("repair_text") or ""),
        requires_human_review=bool(row.get("requires_human_review") or provider_row.get("requires_human_review")),
        metadata={
            "legacy_row": {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "cluster_id",
                    "keep_unit_id",
                    "drop_unit_ids",
                    "unit_id",
                    "drop_word_ids",
                    "keep_word_ids",
                    "reason",
                    "confidence",
                    "requires_human_review",
                }
            }
        },
    )
