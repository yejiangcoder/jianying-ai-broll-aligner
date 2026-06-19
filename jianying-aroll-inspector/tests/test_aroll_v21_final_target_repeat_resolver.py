from __future__ import annotations

import unittest

from aroll_v21.decision.final_target_repeat_resolver import FinalTargetRepeatResolver
from aroll_v21.ir import DecisionPlan, FinalTimelineSegment


def segment(index: int, text: str, *, start_us: int | None = None) -> FinalTimelineSegment:
    start = (index - 1) * 500_000 if start_us is None else start_us
    return FinalTimelineSegment(
        segment_id=f"v21_seg_{index:06d}",
        source_material_id="main",
        source_segment_id="clip",
        source_start_us=start,
        source_end_us=start + max(200_000, len(text) * 40_000),
        target_start_us=start,
        target_end_us=start + max(200_000, len(text) * 40_000),
        word_ids=[f"w_{index:06d}"],
        text=text,
        decision_ids=[],
    )


class ArollV21FinalTargetRepeatResolverTests(unittest.TestCase):
    def test_high_near_duplicate_take_drops_recommended_segment(self) -> None:
        plan = DecisionPlan(decisions=[])
        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "恨不得给人家当牛做马"), segment(2, "中间句"), segment(3, "恨不得给人家当牛做马")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["中间句", "恨不得给人家当牛做马"])
        self.assertEqual(final_timeline[0].target_start_us, 0)
        self.assertEqual(final_timeline[0].target_end_us, final_timeline[1].target_start_us)
        trace = [row for row in plan.decision_trace if row.get("route") == "final_target_repeat"]
        self.assertEqual(trace[0]["decision"], "auto_drop_high_confidence_exact_repeat")
        self.assertTrue(trace[0]["applied"])

    def test_medium_semantic_containment_emits_request_without_auto_drop(self) -> None:
        plan = DecisionPlan(decisions=[])
        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "自信的人能拿到结果"), segment(2, "自信的人真的能拿到结果")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["自信的人能拿到结果", "自信的人真的能拿到结果"])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertFalse(plan.write_allowed)
        self.assertEqual(plan.semantic_request_payloads[0]["type"], "final_target_repeat")
        self.assertEqual(plan.semantic_request_payloads[0]["cluster_type"], "semantic_containment_take")
        forbidden = {
            "source_start_us",
            "source_end_us",
            "target_start_us",
            "target_end_us",
            "edl",
            "final_edl",
            "draft_content",
            "material_id",
            "segment_id",
        }
        self.assertFalse(forbidden & set(plan.semantic_request_payloads[0]))

    def test_medium_restart_take_emits_semantic_request_before_validator(self) -> None:
        plan = DecisionPlan(decisions=[])
        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "她实打实的去旅行了"), segment(2, "过渡句"), segment(3, "她实打实的去喝了下午茶")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["她实打实的去旅行了", "过渡句", "她实打实的去喝了下午茶"])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertEqual(plan.semantic_request_payloads[0]["type"], "final_target_repeat")
        self.assertEqual(plan.semantic_request_payloads[0]["cluster_type"], "restart_take")
        self.assertEqual(plan.semantic_request_payloads[0]["issue_type"], "ambiguous_repeat")

    def test_provider_decision_matches_by_text_pair_when_cluster_id_drifts(self) -> None:
        plan = DecisionPlan(decisions=[])
        plan.semantic_request_payloads.append(
            {
                "cluster_id": "final_target_repeat_tc_9999",
                "issue_id": "final_target_repeat_tc_9999",
                "type": "final_target_repeat",
                "cluster_type": "semantic_containment_take",
                "severity": "medium",
                "provider_required": True,
                "left_text": "自信的人能拿到结果",
                "right_text": "自信的人真的能拿到结果",
                "candidates": [
                    {"role": "left", "text": "自信的人能拿到结果", "candidate_id": "old_left"},
                    {"role": "right", "text": "自信的人真的能拿到结果", "candidate_id": "old_right"},
                ],
            }
        )
        plan.final_target_repeat_unresolved_cluster_ids.append("final_target_repeat_tc_9999")
        plan.semantic_decision_rows.append(
            {
                "cluster_id": "final_target_repeat_tc_9999",
                "decision": "drop_left",
                "reason": "right side is the complete retained phrase",
                "confidence": 0.9,
                "requires_human_review": False,
            }
        )

        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "自信的人能拿到结果"), segment(2, "自信的人真的能拿到结果")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["自信的人真的能拿到结果"])
        self.assertEqual(plan.semantic_unresolved_count, 0)
        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.final_target_repeat_unresolved_cluster_ids, [])
        self.assertTrue(plan.write_allowed)

    def test_final_target_repeat_consumes_provider_decision_by_text_pair_when_cluster_id_drifts(self) -> None:
        plan = DecisionPlan(decisions=[])
        plan.semantic_request_payloads.append(
            {
                "cluster_id": "final_target_repeat_old_request",
                "issue_id": "final_target_repeat_old_request",
                "type": "final_target_repeat",
                "cluster_type": "semantic_containment_take",
                "severity": "medium",
                "provider_required": True,
                "left_text": "自信的人能拿到结果",
                "right_text": "自信的人真的能拿到结果",
                "candidates": [
                    {"role": "left", "text": "自信的人能拿到结果"},
                    {"role": "right", "text": "自信的人真的能拿到结果"},
                ],
            }
        )
        plan.final_target_repeat_unresolved_cluster_ids.append("final_target_repeat_old_request")
        plan.semantic_decision_rows.append(
            {
                "cluster_id": "final_target_repeat_old_request",
                "decision": "drop_left",
                "reason": "provider selected the complete side",
                "confidence": 0.95,
                "requires_human_review": False,
            }
        )

        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "自信的人能拿到结果"), segment(2, "自信的人真的能拿到结果")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["自信的人真的能拿到结果"])
        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.semantic_unresolved_count, 0)
        self.assertEqual(plan.final_target_repeat_unresolved_cluster_ids, [])

    def test_final_target_repeat_clears_stale_unresolved_after_drop_applied(self) -> None:
        plan = DecisionPlan(decisions=[])
        plan.semantic_request_payloads.append(
            {
                "cluster_id": "final_target_repeat_stale_request",
                "issue_id": "final_target_repeat_stale_request",
                "type": "final_target_repeat",
                "cluster_type": "semantic_containment_take",
                "severity": "medium",
                "provider_required": True,
                "left_text": "自信的人能拿到结果",
                "right_text": "自信的人能拿到结果",
                "candidates": [
                    {"role": "left", "text": "自信的人能拿到结果"},
                    {"role": "right", "text": "自信的人能拿到结果"},
                ],
            }
        )
        plan.final_target_repeat_unresolved_cluster_ids.append("final_target_repeat_stale_request")

        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "自信的人能拿到结果"), segment(2, "自信的人能拿到结果")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["自信的人能拿到结果"])
        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.final_target_repeat_unresolved_cluster_ids, [])
        self.assertEqual(plan.semantic_unresolved_count, 0)
        self.assertTrue(
            any(
                row.get("decision") == "resolved_stale_semantic_request"
                for row in plan.decision_trace
            )
        )

    def test_final_target_restart_take_enters_semantic_request(self) -> None:
        plan = DecisionPlan(decisions=[])

        final_timeline, blockers = FinalTargetRepeatResolver().resolve(
            [segment(1, "她实打实的去旅行了"), segment(2, "过渡句"), segment(3, "她实打实的去喝了下午茶")],
            plan,
        )

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], ["她实打实的去旅行了", "过渡句", "她实打实的去喝了下午茶"])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertEqual(plan.semantic_request_payloads[0]["cluster_type"], "restart_take")
        self.assertEqual(plan.semantic_request_payloads[0]["issue_type"], "ambiguous_repeat")

    def test_final_target_restart_take_provider_keep_all_clears_validator_blocker(self) -> None:
        segments = [segment(1, "她实打实的去旅行了"), segment(2, "过渡句"), segment(3, "她实打实的去喝了下午茶")]
        plan = DecisionPlan(decisions=[])
        resolver = FinalTargetRepeatResolver()
        resolver.resolve(segments, plan)
        request_id = plan.semantic_request_payloads[0]["cluster_id"]
        self.assertEqual(plan.semantic_unresolved_count, 1)

        plan.semantic_decision_rows.append(
            {
                "cluster_id": request_id,
                "decision": "keep_all",
                "reason": "provider judged the two takes as complementary context",
                "confidence": 0.86,
                "requires_human_review": False,
            }
        )
        final_timeline, blockers = resolver.resolve(segments, plan)

        self.assertEqual(blockers, [])
        self.assertEqual([row.text for row in final_timeline], [row.text for row in segments])
        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.semantic_unresolved_count, 0)
        self.assertTrue(plan.write_allowed)
        self.assertNotIn(
            "FINAL_TARGET_REPEAT_SEMANTIC_DECISION_REQUIRED",
            [blocker.code for blocker in plan.blockers],
        )


if __name__ == "__main__":
    unittest.main()
