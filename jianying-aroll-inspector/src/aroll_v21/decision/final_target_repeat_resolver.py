from __future__ import annotations

from dataclasses import replace
from typing import Any

from aroll_take_clusterer import build_take_clusters
from aroll_text_normalize import normalize_text
from aroll_v21.decision.deterministic_baseline_policy import DeterministicBaselinePolicy
from aroll_v21.decision.semantic_decision_planner import FORBIDDEN_DEEPSEEK_FIELDS
from aroll_v21.ir.models import Blocker, DecisionPlan, FinalTimelineSegment


FINAL_TARGET_REPEAT_DECISIONS = {
    "keep_all",
    "drop_left",
    "drop_right",
    "keep_longest_drop_others",
    "drop_recommended",
    "requires_human_review",
}


NEAR_DUPLICATE_SIMILARITY_DROP_THRESHOLD = 0.90
MAX_FINAL_REPEAT_CONVERGENCE_ITERATIONS = 3


class FinalTargetRepeatResolver:
    """Compile-time resolver for final target repeat candidates.

    This runs before captions/materials/validators. It never repairs validator
    output and only performs whole-final-segment deletes.
    """

    def __init__(self, *, baseline_policy: DeterministicBaselinePolicy | None = None) -> None:
        self.baseline_policy = baseline_policy or DeterministicBaselinePolicy()

    def resolve(
        self,
        segments: list[FinalTimelineSegment],
        decision_plan: DecisionPlan,
    ) -> tuple[list[FinalTimelineSegment], list[Blocker]]:
        current = list(segments)
        blockers: list[Blocker] = []
        unresolved_ids: set[str] = set(decision_plan.final_target_repeat_unresolved_cluster_ids)
        initial_unresolved_ids = set(unresolved_ids)
        accepted_ids: set[str] = set(decision_plan.final_target_repeat_accepted_cluster_ids)
        resolved_ids: set[str] = set()
        convergence_iteration = 0
        while True:
            clusters = self._clusters(current)
            dropped_indices: set[int] = set()
            pending_drop_cluster_ids: set[str] = set()
            for cluster in clusters:
                cluster_id = self._cluster_id(cluster)
                exact_drop_indices = self._high_confidence_exact_repeat_drop_indices(cluster, current)
                if exact_drop_indices:
                    dropped_indices.update(index - 1 for index in exact_drop_indices)
                    pending_drop_cluster_ids.add(cluster_id)
                    kept_indices = sorted(self._cluster_subtitle_indices(cluster) - set(exact_drop_indices))
                    kept_index = kept_indices[0] if kept_indices else 0
                    decision_plan.decision_trace.append(
                        {
                            "route": "final_target_repeat",
                            "cluster_id": cluster_id,
                            "cluster_type": str(cluster.get("cluster_type") or ""),
                            "stage": "final_timeline_pre_emit",
                            "convergence_iteration": convergence_iteration + 1,
                            "decision": "auto_drop_high_confidence_exact_repeat",
                            "dropped_indices": exact_drop_indices,
                            "dropped_segment_indices": exact_drop_indices,
                            "kept_index": kept_index,
                            "text": self._cluster_exact_text(cluster),
                            "applied": True,
                            "reason": "high-confidence exact final target repeat",
                            "source": "local_policy",
                        }
                    )
                    continue
                baseline_decision = (
                    self.baseline_policy.decision_for_final_repeat_candidate(cluster)
                    if self.baseline_policy.is_enabled(decision_plan)
                    else None
                )
                if baseline_decision is not None and str(baseline_decision.get("decision") or "") == "drop_recommended":
                    drop_index = int(baseline_decision.get("drop_index") or baseline_decision.get("recommended_drop_index") or 0)
                    if 1 <= drop_index <= len(current):
                        dropped_indices.add(drop_index - 1)
                        pending_drop_cluster_ids.add(cluster_id)
                        decision_plan.decision_trace.append(
                            {
                                "route": "final_target_repeat",
                                "cluster_id": cluster_id,
                                "cluster_type": str(cluster.get("cluster_type") or ""),
                                "stage": "final_timeline_pre_emit",
                                "convergence_iteration": convergence_iteration + 1,
                                "decision": "drop_recommended",
                                "v21_resolution": str(baseline_decision.get("v21_resolution") or ""),
                                "recommended_drop_index": drop_index,
                                "drop_index": drop_index,
                                "dropped_cluster_ids": [cluster_id],
                                "dropped_segment_indices": [drop_index],
                                "applied": True,
                                "reason": str(baseline_decision.get("reason") or ""),
                                "source": "deterministic_baseline",
                                "decision_source": "deterministic_baseline",
                            }
                        )
                    continue
                if self._is_high_near_duplicate_auto_drop(cluster):
                    drop_index = int(cluster.get("recommended_drop_index") or 0)
                    if 1 <= drop_index <= len(current):
                        dropped_indices.add(drop_index - 1)
                        pending_drop_cluster_ids.add(cluster_id)
                        decision_plan.decision_trace.append(
                            {
                                "route": "final_target_repeat",
                                "cluster_id": cluster_id,
                                "cluster_type": str(cluster.get("cluster_type") or ""),
                                "stage": "final_timeline_pre_emit",
                                "convergence_iteration": convergence_iteration + 1,
                                "decision": "drop_recommended",
                                "recommended_drop_index": drop_index,
                                "drop_index": drop_index,
                                "dropped_cluster_ids": [cluster_id],
                                "dropped_segment_indices": [drop_index],
                                "applied": True,
                                "reason": "high-confidence duplicate take",
                                "source": "local_policy",
                                "decision_source": "local_policy",
                            }
                        )
                    continue

                if not self._is_semantic_final_target_candidate(cluster):
                    continue

                row = self._semantic_decision_row(decision_plan, cluster_id, cluster)
                if row is None:
                    if cluster_id not in unresolved_ids:
                        unresolved_ids.add(cluster_id)
                        self._append_semantic_request(decision_plan, cluster, cluster_id)
                        decision_plan.blockers.append(
                            Blocker(
                                code="FINAL_TARGET_REPEAT_SEMANTIC_DECISION_REQUIRED",
                                message="final target repeat candidate requires explicit semantic decision",
                                layer="decision",
                                severity="write_blocker",
                                context={
                                    "cluster_id": cluster_id,
                                    "cluster_type": str(cluster.get("cluster_type") or ""),
                                    "severity": str(cluster.get("confidence") or ""),
                                    "allows_dry_run_discovery": True,
                                },
                            )
                        )
                    continue

                forbidden = sorted(FORBIDDEN_DEEPSEEK_FIELDS & set(row.keys()))
                if forbidden:
                    blockers.append(
                        Blocker(
                            code="SEMANTIC_DECISION_HAS_PHYSICAL_FIELDS",
                            message="semantic decisions json contains forbidden physical timeline/material fields",
                            layer="decision",
                            context={"cluster_id": cluster_id, "forbidden_fields": forbidden},
                        )
                    )
                    continue

                decision = str(row.get("decision") or "").strip()
                if decision not in FINAL_TARGET_REPEAT_DECISIONS:
                    blockers.append(
                        Blocker(
                            code="SEMANTIC_DECISION_SCHEMA_INVALID",
                            message="semantic decisions json uses an unsupported final target repeat decision",
                            layer="decision",
                            context={"cluster_id": cluster_id, "decision": decision},
                        )
                    )
                    continue

                if decision == "keep_all":
                    if self._keep_all_disallowed_for_high_fatal_semantic_candidate(cluster):
                        if cluster_id not in unresolved_ids:
                            unresolved_ids.add(cluster_id)
                            self._append_semantic_request(decision_plan, cluster, cluster_id)
                            blocker = Blocker(
                                code="FINAL_TARGET_REPEAT_HIGH_FATAL_KEEP_ALL_REJECTED",
                                message="high/fatal final target semantic repeat cannot be accepted with keep_all",
                                layer="decision",
                                severity="write_blocker",
                                context={
                                    "cluster_id": cluster_id,
                                    "cluster_type": str(cluster.get("cluster_type") or ""),
                                    "severity": str(cluster.get("severity") or cluster.get("confidence") or ""),
                                    "decision": "keep_all",
                                },
                            )
                            decision_plan.blockers.append(blocker)
                            blockers.append(blocker)
                        decision_plan.decision_trace.append(
                            {
                                "route": "final_target_repeat",
                                "cluster_id": cluster_id,
                                "decision": "keep_all_rejected",
                                "applied": False,
                                "source": "SemanticDecisionsJson",
                                "validator_effect": "high_fatal_semantic_repeat_unresolved",
                                "reason": "high/fatal final target semantic repeat cannot be accepted with keep_all",
                            }
                        )
                        continue
                    accepted_ids.add(cluster_id)
                    unresolved_ids.discard(cluster_id)
                    resolved_ids.add(cluster_id)
                    decision_plan.decision_trace.append(
                        {
                            "route": "final_target_repeat",
                            "cluster_id": cluster_id,
                            "decision": "keep_all",
                            "applied": True,
                            "source": "SemanticDecisionsJson",
                            "validator_effect": "accepted_repeat_not_fatal",
                            "reason": str(row.get("reason") or "explicit semantic keep_all"),
                        }
                    )
                    matched_request_id = str(row.get("_matched_request_cluster_id") or "")
                    if matched_request_id:
                        resolved_ids.add(matched_request_id)
                    continue

                if decision == "requires_human_review" or bool(row.get("requires_human_review")):
                    if cluster_id not in unresolved_ids:
                        unresolved_ids.add(cluster_id)
                        self._append_semantic_request(decision_plan, cluster, cluster_id)
                        decision_plan.blockers.append(
                            Blocker(
                                code="FINAL_TARGET_REPEAT_REQUIRES_HUMAN_REVIEW",
                                message="final target repeat decision requires human review",
                                layer="decision",
                                severity="write_blocker",
                                context={"cluster_id": cluster_id},
                            )
                        )
                    continue

                semantic_drop_indices = self._drop_indices_for_decision(cluster, decision, row)
                for drop_index in semantic_drop_indices:
                    if 1 <= drop_index <= len(current):
                        dropped_indices.add(drop_index - 1)
                        pending_drop_cluster_ids.add(cluster_id)
                if semantic_drop_indices:
                    unresolved_ids.discard(cluster_id)
                    resolved_ids.add(cluster_id)
                    matched_request_id = str(row.get("_matched_request_cluster_id") or "")
                    if matched_request_id:
                        unresolved_ids.discard(matched_request_id)
                        resolved_ids.add(matched_request_id)
                    decision_plan.decision_trace.append(
                        {
                            "route": "final_target_repeat",
                            "cluster_id": cluster_id,
                            "convergence_iteration": convergence_iteration + 1,
                            "decision": decision,
                            "dropped_segment_indices": semantic_drop_indices,
                            "applied": True,
                            "source": "SemanticDecisionsJson",
                            "reason": str(row.get("reason") or decision),
                        }
                    )

            if dropped_indices:
                if convergence_iteration >= MAX_FINAL_REPEAT_CONVERGENCE_ITERATIONS:
                    for row in decision_plan.decision_trace:
                        if (
                            isinstance(row, dict)
                            and row.get("route") == "final_target_repeat"
                            and int(row.get("convergence_iteration") or 0) == convergence_iteration + 1
                        ):
                            row["applied"] = False
                            row["blocked_by"] = "V21_FINAL_REPEAT_UNRESOLVED_AFTER_CONVERGENCE"
                    unresolved_ids.update(pending_drop_cluster_ids)
                    blockers.append(
                        Blocker(
                            code="V21_FINAL_REPEAT_UNRESOLVED_AFTER_CONVERGENCE",
                            message="final target repeat convergence exceeded max iterations",
                            layer="decision",
                            severity="write_blocker",
                            context={
                                "max_iterations": MAX_FINAL_REPEAT_CONVERGENCE_ITERATIONS,
                                "pending_drop_cluster_ids": sorted(pending_drop_cluster_ids),
                                "pending_drop_segment_indices": sorted(index + 1 for index in dropped_indices),
                            },
                        )
                    )
                    break
                convergence_iteration += 1
                current = self._repack([segment for index, segment in enumerate(current) if index not in dropped_indices])
                continue
            break

        current_cluster_ids = {self._cluster_id(cluster) for cluster in self._clusters(current)}
        stale_unresolved_ids = {cluster_id for cluster_id in unresolved_ids if cluster_id not in current_cluster_ids}
        if stale_unresolved_ids:
            for cluster_id in sorted(stale_unresolved_ids):
                decision_plan.decision_trace.append(
                    {
                        "route": "final_target_repeat",
                        "cluster_id": cluster_id,
                        "decision": "resolved_stale_semantic_request",
                        "applied": True,
                        "source": "final_timeline_convergence",
                        "reason": "semantic repeat cluster no longer exists after final target repeat convergence",
                    }
                )
            resolved_ids.update(stale_unresolved_ids)

        if resolved_ids:
            self._clear_resolved_semantic_requests(decision_plan, resolved_ids)
            unresolved_ids.difference_update(resolved_ids)
        self._set_plan_list(decision_plan, "final_target_repeat_accepted_cluster_ids", sorted(accepted_ids))
        self._set_plan_list(decision_plan, "final_target_repeat_unresolved_cluster_ids", sorted(unresolved_ids))
        if unresolved_ids:
            newly_unresolved_ids = unresolved_ids - initial_unresolved_ids
            object.__setattr__(decision_plan, "semantic_unresolved_count", int(decision_plan.semantic_unresolved_count) + len(newly_unresolved_ids))
            object.__setattr__(decision_plan, "requires_human_review", True)
            object.__setattr__(decision_plan, "write_allowed", False)
            object.__setattr__(decision_plan, "dry_run_continued_for_discovery", True)
        self._recompute_semantic_state(decision_plan)
        return current, blockers

    def _high_confidence_exact_repeat_drop_indices(self, cluster: dict[str, Any], segments: list[FinalTimelineSegment]) -> list[int]:
        if str(cluster.get("confidence") or "") != "high":
            return []
        if bool(cluster.get("requires_llm")):
            return []
        indices = sorted(index for index in self._cluster_subtitle_indices(cluster) if 1 <= index <= len(segments))
        if len(indices) < 2:
            return []
        texts = [normalize_text(segments[index - 1].text) for index in indices]
        if not texts or any(not text for text in texts) or len(set(texts)) != 1:
            return []
        recommended = int(cluster.get("recommended_drop_index") or 0)
        if recommended in indices:
            return [recommended]
        keep_index = self._keep_index_for_exact_repeat(indices, segments)
        return [index for index in indices if index != keep_index]

    def _keep_index_for_exact_repeat(self, indices: list[int], segments: list[FinalTimelineSegment]) -> int:
        best_index = indices[0]
        best_score = (-1, -1, 0)
        for index in indices:
            segment = segments[index - 1]
            text_len = len(normalize_text(segment.text))
            duration = int(segment.source_end_us) - int(segment.source_start_us)
            score = (text_len, duration, -index)
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _cluster_subtitle_indices(self, cluster: dict[str, Any]) -> set[int]:
        indices: set[int] = set()
        for item in cluster.get("items") or []:
            if isinstance(item, dict):
                value = int(item.get("subtitle_index") or 0)
                if value > 0:
                    indices.add(value)
        for candidate in cluster.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            for value in candidate.get("subtitle_indices") or []:
                index = int(value or 0)
                if index > 0:
                    indices.add(index)
        return indices

    def _cluster_exact_text(self, cluster: dict[str, Any]) -> str:
        for item in cluster.get("items") or []:
            if isinstance(item, dict):
                text = normalize_text(str(item.get("text") or ""))
                if text:
                    return text
        for candidate in cluster.get("candidates") or []:
            if isinstance(candidate, dict):
                text = normalize_text(str(candidate.get("text") or candidate.get("norm_text") or ""))
                if text:
                    return text
        return ""

    def _clusters(self, segments: list[FinalTimelineSegment]) -> list[dict[str, Any]]:
        rows = [
            {
                "subtitle_uid": segment.segment_id,
                "subtitle_index": index,
                "subtitle_text": segment.text,
                "start_us": segment.target_start_us,
                "end_us": segment.target_end_us,
            }
            for index, segment in enumerate(segments, start=1)
            if str(segment.text or "").strip()
        ]
        clusters, _report = build_take_clusters(rows, [], window=5)
        return clusters

    def _is_high_near_duplicate_auto_drop(self, cluster: dict[str, Any]) -> bool:
        if str(cluster.get("cluster_type") or "") != "near_duplicate_take":
            return False
        if str(cluster.get("confidence") or "") != "high":
            return False
        if bool(cluster.get("requires_llm")):
            return False
        if int(cluster.get("recommended_drop_index") or 0) <= 0:
            return False
        if self._candidate_similarity(cluster) < NEAR_DUPLICATE_SIMILARITY_DROP_THRESHOLD:
            return False
        if str(cluster.get("recommended_drop_index") or "") and not self._is_recommended_drop_in_candidates(cluster):
            return False
        return True

    def _candidate_similarity(self, cluster: dict[str, Any]) -> float:
        similarity = cluster.get("similarity")
        if isinstance(similarity, (int, float)):
            return float(similarity)
        pairwise = [float(row.get("similarity") or 0.0) for row in cluster.get("pairwise_evidence") or [] if isinstance(row, dict)]
        if not pairwise:
            return 0.0
        return min(pairwise)

    def _is_recommended_drop_in_candidates(self, cluster: dict[str, Any]) -> bool:
        drop_index = int(cluster.get("recommended_drop_index") or 0)
        for candidate in list(cluster.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            indices = [int(item) for item in (candidate.get("subtitle_indices") or []) if int(item) > 0]
            if drop_index in indices:
                return True
        for item in list(cluster.get("items") or []):
            if not isinstance(item, dict):
                continue
            if int(item.get("subtitle_index") or 0) == drop_index:
                return True
        return False

    def _is_semantic_final_target_candidate(self, cluster: dict[str, Any]) -> bool:
        return (
            str(cluster.get("cluster_type") or "") in {
                "semantic_containment_take",
                "restart_take",
                "ambiguous_repeat",
                "near_duplicate_take",
            }
            and str(cluster.get("confidence") or "") in {"medium", "high", "fatal"}
            and bool(cluster.get("requires_llm"))
            and not cluster.get("recommended_drop_index")
        )

    def _keep_all_disallowed_for_high_fatal_semantic_candidate(self, cluster: dict[str, Any]) -> bool:
        if str(cluster.get("cluster_type") or "") not in {"semantic_containment_take", "visible_caption_repeat"}:
            return False
        severity = str(cluster.get("severity") or cluster.get("confidence") or "").strip().lower()
        return severity in {"high", "fatal"}

    def _pairwise_minimum(self, cluster: dict[str, Any], field: str, minimum: float) -> bool:
        rows = list(cluster.get("pairwise_evidence") or [])
        if not rows:
            return False
        return all(float(row.get(field) or 0.0) >= minimum for row in rows)

    def _cluster_id(self, cluster: dict[str, Any]) -> str:
        raw = str(cluster.get("cluster_id") or "")
        return raw if raw.startswith("final_target_repeat_") else f"final_target_repeat_{raw}"

    def _semantic_decision_row(self, decision_plan: DecisionPlan, cluster_id: str, cluster: dict[str, Any]) -> dict[str, Any] | None:
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in decision_plan.semantic_decision_rows:
            if not isinstance(row, dict):
                continue
            row_cluster_id = str(row.get("cluster_id") or row.get("issue_id") or "")
            if row_cluster_id:
                rows_by_id[row_cluster_id] = row
            if row_cluster_id == cluster_id:
                return row
        matched = self._semantic_decision_row_by_text_pair(decision_plan, cluster, rows_by_id)
        if matched is not None:
            return matched
        if self.baseline_policy.is_enabled(decision_plan):
            return self.baseline_policy.decision_for_missing_cluster(
                cluster_id,
                cluster_type=str(cluster.get("cluster_type") or "final_target_repeat"),
                context={
                    "reason": "deterministic baseline keeps low-risk final target repeat for sacrificial UAT",
                    "confidence_label": str(cluster.get("confidence") or ""),
                    "severity": str(cluster.get("confidence") or ""),
                    "requires_llm": bool(cluster.get("requires_llm")),
                },
            )
        return None

    def _semantic_decision_row_by_text_pair(
        self,
        decision_plan: DecisionPlan,
        cluster: dict[str, Any],
        rows_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        cluster_pair = self._cluster_text_pair(cluster)
        if cluster_pair is None:
            return None
        for payload in decision_plan.semantic_request_payloads:
            if not isinstance(payload, dict):
                continue
            payload_id = str(payload.get("cluster_id") or payload.get("issue_id") or "")
            if not payload_id:
                continue
            row = rows_by_id.get(payload_id)
            if row is None:
                continue
            if self._payload_text_pair(payload) != cluster_pair:
                continue
            matched = dict(row)
            matched["_matched_request_cluster_id"] = payload_id
            return matched
        return None

    def _cluster_text_pair(self, cluster: dict[str, Any]) -> tuple[str, str] | None:
        items = [item for item in list(cluster.get("items") or []) if isinstance(item, dict)]
        if len(items) < 2:
            return None
        left = normalize_text(str(items[0].get("text") or items[0].get("subtitle_text") or ""))
        right = normalize_text(str(items[1].get("text") or items[1].get("subtitle_text") or ""))
        if not left or not right:
            return None
        return left, right

    def _payload_text_pair(self, payload: dict[str, Any]) -> tuple[str, str] | None:
        candidates = [item for item in list(payload.get("candidates") or []) if isinstance(item, dict)]
        if len(candidates) >= 2:
            left = normalize_text(str(candidates[0].get("text") or ""))
            right = normalize_text(str(candidates[1].get("text") or ""))
        else:
            left = normalize_text(str(payload.get("left_text") or payload.get("text_before") or payload.get("text") or ""))
            right = normalize_text(str(payload.get("right_text") or payload.get("text_after") or ""))
        if not left or not right:
            return None
        return left, right

    def _append_semantic_request(self, decision_plan: DecisionPlan, cluster: dict[str, Any], cluster_id: str) -> None:
        existing = {str(row.get("cluster_id") or "") for row in decision_plan.semantic_request_payloads}
        if cluster_id in existing:
            return
        items = list(cluster.get("items") or [])
        cluster_type = str(cluster.get("cluster_type") or "")
        issue_type = "semantic_containment"
        if cluster_type == "visible_caption_repeat":
            issue_type = "visible_caption_repeat"
        elif cluster_type == "near_duplicate_take":
            issue_type = "near_duplicate_take"
        elif cluster_type in {"restart_take", "ambiguous_repeat"}:
            issue_type = "ambiguous_repeat"
        candidates = []
        for role, item in zip(("left", "right"), items[:2]):
            candidates.append(
                {
                    "role": role,
                    "text": str(item.get("text") or ""),
                    "subtitle_index": int(item.get("subtitle_index") or 0),
                    "candidate_id": str(item.get("subtitle_uid") or ""),
                }
            )
        decision_plan.semantic_request_payloads.append(
            {
                "cluster_id": cluster_id,
                "type": "final_target_repeat",
                "issue_id": cluster_id,
                "issue_type": issue_type,
                "cluster_type": cluster_type,
                "severity": str(cluster.get("confidence") or ""),
                "requires_llm": bool(cluster.get("requires_llm")),
                "provider_required": True,
                "text": str(candidates[0].get("text") or "") if candidates else "",
                "left_text": str(candidates[0].get("text") or "") if candidates else "",
                "right_text": str(candidates[1].get("text") or "") if len(candidates) > 1 else "",
                "candidates": candidates,
                "allowed_decisions": [
                    "keep_all",
                    "drop_left",
                    "drop_right",
                    "keep_longest_drop_others",
                    "requires_human_review",
                ],
                "recommended_action": "no_decision",
                "suggested_for_rough_cut": "no_decision",
                "why_local_policy_cannot_decide": "final target repeat candidate requires semantic adjudication after final timeline compilation",
                "required_decision_schema": {
                    "decision": "keep_all | drop_left | drop_right | keep_longest_drop_others | requires_human_review",
                    "reason": "",
                    "confidence": 0.0,
                    "requires_human_review": False,
                },
            }
        )

    def _drop_indices_for_decision(self, cluster: dict[str, Any], decision: str, row: dict[str, Any] | None = None) -> list[int]:
        row = row or {}
        if decision == "drop_recommended":
            drop_index = int(row.get("drop_index") or row.get("recommended_drop_index") or cluster.get("recommended_drop_index") or 0)
            return [drop_index] if drop_index > 0 else []
        items = list(cluster.get("items") or [])
        if len(items) < 2:
            return []
        indices = [int(item.get("subtitle_index") or 0) for item in items if int(item.get("subtitle_index") or 0) > 0]
        if decision == "drop_left":
            return indices[:1]
        if decision == "drop_right":
            return indices[-1:]
        if decision == "keep_longest_drop_others":
            longest_index = 0
            longest_len = -1
            for idx, item in enumerate(items):
                text_len = len(normalize_text(str(item.get("text") or "")))
                if text_len > longest_len:
                    longest_index = idx
                    longest_len = text_len
            keep = int(items[longest_index].get("subtitle_index") or 0)
            return [index for index in indices if index != keep]
        return []

    def _repack(self, segments: list[FinalTimelineSegment]) -> list[FinalTimelineSegment]:
        repacked: list[FinalTimelineSegment] = []
        target_cursor = 0
        for index, segment in enumerate(segments, start=1):
            duration = segment.source_end_us - segment.source_start_us
            repacked.append(
                replace(
                    segment,
                    segment_id=f"v21_seg_{index:06d}",
                    target_start_us=target_cursor,
                    target_end_us=target_cursor + duration,
                )
            )
            target_cursor += duration
        return repacked

    def _set_plan_list(self, decision_plan: DecisionPlan, field_name: str, values: list[str]) -> None:
        target = getattr(decision_plan, field_name)
        target.clear()
        target.extend(values)

    def _clear_resolved_semantic_requests(self, decision_plan: DecisionPlan, resolved_ids: set[str]) -> None:
        decision_plan.semantic_request_payloads[:] = [
            payload
            for payload in decision_plan.semantic_request_payloads
            if str(payload.get("issue_id") or payload.get("cluster_id") or "") not in resolved_ids
        ]
        clear_codes = {
            "FINAL_TARGET_REPEAT_SEMANTIC_DECISION_REQUIRED",
            "FINAL_TARGET_REPEAT_REQUIRES_HUMAN_REVIEW",
            "FINAL_TARGET_REPEAT_HIGH_FATAL_KEEP_ALL_REJECTED",
            "V21_AUTO_PROVIDER_ROUTING_SKIPPED_REQUIRED_REQUEST",
            "V21_SEMANTIC_ADJUDICATION_PROVIDER_FAILED",
            "SEMANTIC_DECISION_NOT_PROVIDED",
        }
        decision_plan.blockers[:] = [
            blocker
            for blocker in decision_plan.blockers
            if not (
                blocker.severity == "write_blocker"
                and blocker.code in clear_codes
                and str(blocker.context.get("cluster_id") or "") in resolved_ids
            )
        ]

    def _recompute_semantic_state(self, decision_plan: DecisionPlan) -> None:
        unresolved_payload_ids = {
            str(payload.get("issue_id") or payload.get("cluster_id") or "")
            for payload in decision_plan.semantic_request_payloads
            if isinstance(payload, dict) and str(payload.get("issue_id") or payload.get("cluster_id") or "")
        }
        write_blockers = [blocker for blocker in decision_plan.blockers if blocker.severity == "write_blocker"]
        fatal_blockers = [blocker for blocker in decision_plan.blockers if blocker.severity == "fatal"]
        human_review_decisions = [decision for decision in decision_plan.decisions if decision.requires_human_review]
        object.__setattr__(decision_plan, "semantic_unresolved_count", len(unresolved_payload_ids))
        object.__setattr__(decision_plan, "requires_human_review", bool(unresolved_payload_ids or write_blockers or human_review_decisions))
        object.__setattr__(decision_plan, "write_allowed", not unresolved_payload_ids and not write_blockers and not fatal_blockers and not human_review_decisions)
        object.__setattr__(decision_plan, "dry_run_continued_for_discovery", bool(unresolved_payload_ids))
