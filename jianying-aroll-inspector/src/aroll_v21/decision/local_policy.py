from __future__ import annotations

from typing import Any

from aroll_v21.decision.unit_split_binding import _safe_unit_split_ids, _unit_split_binding
from aroll_v21.ir.models import Blocker, RepeatCluster, TakeDecision, UnitSplitPlan

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


