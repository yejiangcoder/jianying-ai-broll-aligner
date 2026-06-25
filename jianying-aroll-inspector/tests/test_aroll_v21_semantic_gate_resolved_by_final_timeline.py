from __future__ import annotations

import unittest

from aroll_v21 import ArollEngine
from aroll_v21.ir import Blocker, CaptionRenderUnit, DecisionPlan, FinalTimelineSegment
from tests.test_aroll_v21_semantic_unconfigured_dryrun_policy import semantic_run_input


def resolved_modifier_input(*, mode: str = "dry-run"):
    payload = semantic_run_input(mode=mode, text="随意的肆意的踩踏")
    payload.word_timeline[0]["word_text"] = "肆意的踩踏"
    return payload


class ArollV21SemanticGateResolvedByFinalTimelineTests(unittest.TestCase):
    def test_single_variant_cluster_absent_from_final_output_is_resolved(self) -> None:
        report = ArollEngine().run(resolved_modifier_input())

        self.assertEqual(report.decision_plan.semantic_unresolved_count, 0)
        self.assertTrue(report.decision_plan.write_allowed)
        self.assertEqual(report.decision_plan.semantic_request_payloads, [])
        self.assertFalse(any(blocker.code == "DEEPSEEK_SEMANTIC_PLANNER_NOT_CONFIGURED" for blocker in report.blocker_report.blockers))
        self.assertEqual([segment.text for segment in report.final_timeline], ["肆意的踩踏"])
        self.assertEqual([caption.text for caption in report.captions], ["肆意的踩踏"])
        trace = [row for row in report.decision_trace if row.get("decision") == "resolved_by_final_timeline"]
        self.assertEqual(trace[0]["route"], "semantic_gate")
        self.assertEqual(trace[0]["cluster_id"], "repeat_002000")
        self.assertFalse(trace[0]["requires_semantic_decision"])

    def test_cluster_text_still_present_remains_fail_closed(self) -> None:
        report = ArollEngine().run(semantic_run_input(text="随意的肆意的踩踏"))

        self.assertEqual(report.decision_plan.semantic_unresolved_count, 1)
        self.assertFalse(report.decision_plan.write_allowed)
        self.assertTrue(report.decision_plan.semantic_request_payloads)
        self.assertTrue(any(blocker.code == "DEEPSEEK_SEMANTIC_PLANNER_NOT_CONFIGURED" for blocker in report.blocker_report.blockers))
        self.assertEqual([segment.text for segment in report.final_timeline], ["随意的肆意的踩踏"])

    def test_resolved_modifier_payload_clears_modifier_unresolved_state(self) -> None:
        plan = DecisionPlan(
            decisions=[],
            semantic_request_payloads=[
                {
                    "cluster_id": "repeat_002000",
                    "repeat_type": "modifier_redundancy",
                    "type": "single_variant_modifier_redundancy",
                    "text": "随意的肆意的踩踏",
                }
            ],
            modifier_redundancy_unresolved_cluster_ids=["repeat_002000"],
            semantic_unresolved_count=1,
            requires_human_review=True,
            write_allowed=False,
            blockers=[
                Blocker(
                    "FINAL_MODIFIER_REDUNDANCY_SEMANTIC_DECISION_REQUIRED",
                    "needs decision",
                    "decision",
                    severity="write_blocker",
                    context={"cluster_id": "repeat_002000"},
                )
            ],
        )
        final_timeline = [
            FinalTimelineSegment(
                segment_id="seg1",
                source_material_id="main",
                source_segment_id="clip",
                source_start_us=0,
                source_end_us=500_000,
                target_start_us=0,
                target_end_us=500_000,
                word_ids=["w1"],
                text="肆意的踩踏",
                decision_ids=[],
            )
        ]
        captions = [
            CaptionRenderUnit(
                caption_id="cap1",
                timeline_segment_ids=["seg1"],
                word_ids=["w1"],
                text="肆意的踩踏",
                target_start_us=0,
                target_end_us=500_000,
                source_subtitle_uids=["s1"],
                style_template_id="tmpl",
            )
        ]

        ArollEngine()._sync_semantic_gate_with_final_output(plan, final_timeline, captions)

        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.modifier_redundancy_unresolved_cluster_ids, [])
        self.assertEqual(plan.semantic_unresolved_count, 0)
        self.assertTrue(plan.write_allowed)
        self.assertEqual(plan.blockers, [])

    def test_stale_final_target_payload_clears_when_text_pair_is_no_longer_active_cluster(self) -> None:
        plan = DecisionPlan(
            decisions=[],
            semantic_request_payloads=[
                {
                    "cluster_id": "final_target_repeat_tc_0002",
                    "issue_id": "final_target_repeat_tc_0002",
                    "type": "final_target_repeat",
                    "cluster_type": "semantic_containment_take",
                    "left_text": "雪松的冷",
                    "right_text": "那么雪松的冷",
                    "candidates": [
                        {"role": "left", "text": "雪松的冷"},
                        {"role": "right", "text": "那么雪松的冷"},
                    ],
                }
            ],
            final_target_repeat_unresolved_cluster_ids=["final_target_repeat_tc_0002"],
            semantic_unresolved_count=1,
            requires_human_review=True,
            write_allowed=False,
            blockers=[
                Blocker(
                    "FINAL_TARGET_REPEAT_SEMANTIC_DECISION_REQUIRED",
                    "needs decision",
                    "decision",
                    severity="write_blocker",
                    context={"cluster_id": "final_target_repeat_tc_0002"},
                )
            ],
        )
        final_timeline = [
            FinalTimelineSegment(
                segment_id="seg1",
                source_material_id="main",
                source_segment_id="clip",
                source_start_us=0,
                source_end_us=700_000,
                target_start_us=0,
                target_end_us=700_000,
                word_ids=["w1"],
                text="雪松檀木",
                decision_ids=[],
            ),
            FinalTimelineSegment(
                segment_id="seg2",
                source_material_id="main",
                source_segment_id="clip",
                source_start_us=900_000,
                source_end_us=1_800_000,
                target_start_us=700_000,
                target_end_us=1_600_000,
                word_ids=["w2"],
                text="香根草那么雪松的冷",
                decision_ids=[],
            ),
        ]
        captions = [
            CaptionRenderUnit(
                caption_id="cap1",
                timeline_segment_ids=["seg1"],
                word_ids=["w1"],
                text="雪松檀木",
                target_start_us=0,
                target_end_us=700_000,
                source_subtitle_uids=["s1"],
                style_template_id="tmpl",
            ),
            CaptionRenderUnit(
                caption_id="cap2",
                timeline_segment_ids=["seg2"],
                word_ids=["w2"],
                text="香根草那么雪松的冷",
                target_start_us=700_000,
                target_end_us=1_600_000,
                source_subtitle_uids=["s2"],
                style_template_id="tmpl",
            ),
        ]

        ArollEngine()._sync_semantic_gate_with_final_output(plan, final_timeline, captions)

        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.final_target_repeat_unresolved_cluster_ids, [])
        self.assertEqual(plan.semantic_unresolved_count, 0)
        self.assertTrue(plan.write_allowed)
        self.assertEqual(plan.blockers, [])
        self.assertTrue(
            any(row.get("decision") == "resolved_by_final_target_cluster_absence" for row in plan.decision_trace),
            plan.decision_trace,
        )


if __name__ == "__main__":
    unittest.main()
