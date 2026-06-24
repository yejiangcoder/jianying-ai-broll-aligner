from __future__ import annotations

from typing import Any

from aroll_v21.decision.deepseek_request_builder import FORBIDDEN_DEEPSEEK_FIELDS
from aroll_v21.decision.semantic_adjudication import (
    HIGH_RISK_SEVERITIES,
    issue_type_for_cluster,
    semantic_blocker_code,
    severity_for_cluster,
)
from aroll_v21.decision.semantic_contracts import SemanticIssueType
from aroll_v21.decision.unit_split_binding import _unit_split_binding
from aroll_v21.ir.models import RepeatCluster

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


