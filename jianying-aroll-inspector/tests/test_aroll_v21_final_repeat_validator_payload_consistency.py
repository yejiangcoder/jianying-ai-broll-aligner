from __future__ import annotations

import unittest

from aroll_v21 import ArollEngine
from aroll_v21.ir import DecisionPlan
from aroll_v21.validate.validators import ReadOnlyValidators


class ArollV21FinalRepeatValidatorPayloadConsistencyTests(unittest.TestCase):
    def test_final_repeat_validator_fatal_with_empty_payloads_reports_internal_consistency_error(self) -> None:
        validator_report = {
            "final_repeat_validator": {
                "final_repeat_gate_passed": False,
                "blocking_issues": [
                    {
                        "type": "boundary_prefix_containment",
                        "issue_type": "cjk_adjacent_subtitle_boundary_overlap",
                        "severity": "fatal",
                        "left_text": "甲乙重复",
                        "right_text": "重复丙丁",
                        "overlap": "重复",
                        "row_index": 1,
                        "next_row_index": 2,
                        "reason": "normalized suffix of the previous final subtitle is contained in the next subtitle prefix",
                    }
                ],
            },
            "hidden_audio_repeat_validator": {"hidden_audio_repeat_gate_passed": True, "blocking_issues": []},
        }

        blockers = ArollEngine()._semantic_request_consistency_blockers(DecisionPlan(decisions=[]), validator_report)

        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].code, "INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_FINAL_REPEAT_VALIDATOR")
        self.assertEqual(blockers[0].context["candidate_type"], "boundary_prefix_containment")
        self.assertEqual(blockers[0].context["left_text"], "甲乙重复")
        self.assertEqual(blockers[0].context["right_text"], "重复丙丁")
        self.assertEqual(blockers[0].context["overlap"], "重复")

    def test_existing_payload_suppresses_final_repeat_missing_payload_consistency_error(self) -> None:
        plan = DecisionPlan(
            decisions=[],
            semantic_request_payloads=[
                {
                    "cluster_id": "repeat_existing",
                    "type": "final_target_repeat",
                    "repeat_type": "semantic_containment_take",
                }
            ],
        )
        validator_report = {
            "final_repeat_validator": {
                "final_repeat_gate_passed": False,
                "blocking_issues": [
                    {
                        "type": "boundary_prefix_containment",
                        "left_text": "甲乙重复",
                        "right_text": "重复丙丁",
                        "overlap": "重复",
                    }
                ],
            },
            "hidden_audio_repeat_validator": {"hidden_audio_repeat_gate_passed": True, "blocking_issues": []},
        }

        blockers = ArollEngine()._semantic_request_consistency_blockers(plan, validator_report)

        self.assertNotIn(
            "INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_FINAL_REPEAT_VALIDATOR",
            [blocker.code for blocker in blockers],
        )

    def test_protected_final_target_bridge_trace_accepts_validator_candidate_when_cluster_id_drifts(self) -> None:
        plan = DecisionPlan(decisions=[])
        plan.decision_trace.append(
            {
                "route": "final_target_repeat",
                "cluster_id": "final_target_repeat_tc_0003",
                "decision": "keep_all_protected_semantic_bridge",
                "left_text": "你必须具备超越",
                "right_text": "必须具备",
                "applied": True,
            }
        )
        repeat_report = {
            "final_repeat_gate_passed": False,
            "final_target_repeat_candidates": [
                {
                    "cluster_id": "tc_0001",
                    "cluster_type": "semantic_containment_take",
                    "confidence": "medium",
                    "items": [
                        {"subtitle_index": 56, "text": "你必须具备超越"},
                        {"subtitle_index": 60, "text": "必须具备"},
                    ],
                }
            ],
        }

        updated = ReadOnlyValidators()._final_repeat_semantic_status(repeat_report, plan)

        self.assertTrue(updated["final_repeat_gate_passed"])
        self.assertEqual(updated["final_target_repeat_accepted_count"], 1)
        self.assertEqual(updated["final_target_repeat_medium_count"], 0)
        self.assertEqual(
            updated["final_target_repeat_candidates"][0]["v21_resolution"],
            "accepted_by_protected_semantic_bridge",
        )


if __name__ == "__main__":
    unittest.main()
