from __future__ import annotations

import unittest

from aroll_v21.decision import SemanticDecisionPlanner, SemanticDecisionsJsonPlanner
from aroll_v21.decision.final_target_repeat_resolver import FinalTargetRepeatResolver
from aroll_v21.ir import DecisionPlan
from tests.test_aroll_v21_final_target_repeat_resolver import segment


class HighFatalFinalTargetRepeatResolver(FinalTargetRepeatResolver):
    def _clusters(self, segments):  # type: ignore[override]
        clusters = super()._clusters(segments)
        return [
            dict(
                clusters[0],
                confidence="high",
                severity="high",
                requires_llm=True,
                recommended_drop_index=None,
            )
        ]


def _plan_with_rows(rows: list[dict]) -> DecisionPlan:
    return SemanticDecisionPlanner(deepseek_planner=SemanticDecisionsJsonPlanner(rows)).plan([])


class ArollV21FinalTargetRepeatSemanticDecisionTests(unittest.TestCase):
    def test_keep_longest_drop_others_applies_to_final_timeline(self) -> None:
        plan = _plan_with_rows(
            [
                {
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "keep_longest_drop_others",
                    "reason": "keep longer complete take",
                    "confidence": 0.8,
                    "requires_human_review": False,
                }
            ]
        )

        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "自信的人能拿到结果"), segment(2, "自信的人真的能拿到结果")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["自信的人真的能拿到结果"])
        self.assertEqual(plan.semantic_unresolved_count, 0)
        self.assertTrue(plan.write_allowed)
        self.assertEqual(plan.decision_trace[-1]["decision"], "keep_longest_drop_others")

    def test_keep_longest_uses_source_contiguous_continuation_for_complete_take(self) -> None:
        plan = _plan_with_rows(
            [
                {
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "keep_longest_drop_others",
                    "reason": "keep the semantically complete take including its continuation",
                    "confidence": 0.88,
                    "requires_human_review": False,
                }
            ]
        )
        left_fragment = segment(1, "老子最烦这种深夜准时发作的土味", start_us=0)
        right_restart = segment(2, "老子最烦这种深夜", start_us=1_000_000)
        right_continuation = segment(3, "准时发作的土味纯爱逼", start_us=1_360_000)

        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [left_fragment, right_restart, right_continuation],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["老子最烦这种深夜", "准时发作的土味纯爱逼"])
        self.assertEqual(plan.decision_trace[-1]["decision"], "keep_longest_drop_others")
        self.assertEqual(plan.decision_trace[-1]["dropped_segment_indices"], [1])
        self.assertTrue(plan.write_allowed)

    def test_keep_all_accepts_repeat_without_timeline_delete(self) -> None:
        plan = _plan_with_rows(
            [
                {
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "keep_all",
                    "reason": "intentional repetition",
                    "confidence": 0.9,
                    "requires_human_review": False,
                }
            ]
        )

        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "自信的人能拿到结果"), segment(2, "自信的人真的能拿到结果")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["自信的人能拿到结果", "自信的人真的能拿到结果"])
        self.assertEqual(plan.final_target_repeat_accepted_cluster_ids, ["final_target_repeat_tc_0001"])
        self.assertEqual(plan.decision_trace[-1]["validator_effect"], "accepted_repeat_not_fatal")

    def test_partial_final_target_decisions_keep_write_disallowed(self) -> None:
        plan = _plan_with_rows(
            [
                {
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "keep_all",
                    "reason": "first one accepted",
                    "confidence": 0.9,
                    "requires_human_review": False,
                }
            ]
        )

        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [
                segment(1, "自信的人能拿到结果"),
                segment(2, "自信的人真的能拿到结果"),
                segment(3, "努力的人能得到反馈"),
                segment(4, "努力的人真的能得到反馈"),
            ],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual(len(final_timeline), 4)
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertFalse(plan.write_allowed)
        self.assertEqual(plan.final_target_repeat_accepted_cluster_ids, ["final_target_repeat_tc_0001"])
        self.assertEqual(plan.final_target_repeat_unresolved_cluster_ids, ["final_target_repeat_tc_0002"])

    def test_final_target_semantic_decision_with_physical_field_blocks(self) -> None:
        plan = _plan_with_rows(
            [
                {
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "keep_all",
                    "source_start_us": 123,
                }
            ]
        )

        _final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "自信的人能拿到结果"), segment(2, "自信的人真的能拿到结果")],
            plan,
        )

        self.assertEqual([blocker.code for blocker in blockers], ["SEMANTIC_DECISION_HAS_PHYSICAL_FIELDS"])

    def test_final_target_repeat_rejects_keep_all_for_high_fatal_semantic_containment(self) -> None:
        plan = _plan_with_rows(
            [
                {
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "keep_all",
                    "reason": "intentional repetition",
                    "confidence": 0.9,
                    "requires_human_review": False,
                }
            ]
        )

        final_timeline, blockers = HighFatalFinalTargetRepeatResolver().resolve(
            [segment(1, "自信的人能拿到结果"), segment(2, "自信的人真的能拿到结果")],
            plan,
        )

        self.assertEqual([row.text for row in final_timeline], ["自信的人能拿到结果", "自信的人真的能拿到结果"])
        self.assertIn("FINAL_TARGET_REPEAT_HIGH_FATAL_KEEP_ALL_REJECTED", [blocker.code for blocker in blockers])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertFalse(plan.write_allowed)
        self.assertFalse(any(row.get("decision") == "keep_all" and row.get("applied") for row in plan.decision_trace))


if __name__ == "__main__":
    unittest.main()
