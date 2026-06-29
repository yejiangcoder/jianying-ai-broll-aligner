from __future__ import annotations

import unittest

from aroll_v21 import ArollEngine
from aroll_v21.decision import SemanticAdjudicationDecision, SemanticAdjudicationDecisionType
from aroll_v21.engine_summary import build_run_summary
from aroll_v21.ir import Blocker, CaptionRenderUnit, DecisionPlan
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from tests.test_aroll_v21_semantic_adjudication_layer import _two_caption_input


class FinalVisibleAdvisoryProvider:
    provider_name = "fake_final_visible_deepseek"

    def __init__(self, decision: SemanticAdjudicationDecisionType) -> None:
        self.decision = decision
        self.requests = []
        self.provider_called_count = 0

    def decide(self, requests):  # type: ignore[no-untyped-def]
        self.requests.extend(requests)
        self.provider_called_count += 1
        return [
            SemanticAdjudicationDecision(
                issue_id=request.issue_id,
                decision=self.decision,
                reason="fake final-visible advisory decision",
                confidence=0.88,
                provider_name=self.provider_name,
            )
            for request in requests
        ]


def _caption(index: int, segment_id: str, start_us: int, end_us: int, text: str) -> CaptionRenderUnit:
    return CaptionRenderUnit(
        caption_id=f"v21_cap_{index:06d}",
        timeline_segment_ids=[segment_id],
        word_ids=[f"w{index:03d}"],
        text=text,
        target_start_us=start_us,
        target_end_us=end_us,
        source_subtitle_uids=[f"s{index:03d}"],
        style_template_id="canonical_caption_template",
    )


def _validator_report_for_visible_gate(gate: dict) -> dict:
    return {
        "validator_report_ok": True,
        "final_caption_visible_repeat_gate": gate,
        "quality_gate_report": {
            "ready_for_user_manual_qc_preconditions_passed": True,
            "effective_speed_gate": {"gate_passed": True, "blocker_codes": [], "prewrite_pending": True},
        },
        "final_repeat_convergence_gate": {"gate_passed": True, "blocker_codes": [], "detector_report_present": True},
        "visual_pacing_gate": {
            "gate_passed": True,
            "visual_pacing_executed": True,
            "visual_merge_safety_gate_passed": True,
            "blocker_codes": [],
        },
        "caption_alignment_gate": {
            "gate_passed": True,
            "caption_gui_track_gate_passed": True,
            "subtitle_readability_gate_passed": True,
            "blocker_codes": [],
        },
        "final_timeline_quality_guard_report": {"gate_passed": True, "blocker_codes": []},
        "prewrite_projected_write_view": {"prewrite_projected_write_view_gate_passed": True},
    }


class ArollV21SemanticRequestConsistencyGateTests(unittest.TestCase):
    def test_missing_request_for_validator_modifier_fatal_blocks_internally(self) -> None:
        validator_report = {
            "final_repeat_validator": {
                "final_repeat_gate_passed": False,
                "blocking_issues": [
                    {
                        "type": "adjacent_modifier_semantic_redundancy",
                        "text": "快乐的开心的孩子",
                        "phrase": "快乐的开心的孩子",
                    }
                ],
            },
            "hidden_audio_repeat_validator": {"hidden_audio_repeat_gate_passed": True, "blocking_issues": []},
        }

        blockers = ArollEngine()._semantic_request_consistency_blockers(DecisionPlan(decisions=[]), validator_report)

        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].code, "INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_FATAL_REPEAT")

    def test_matching_modifier_request_satisfies_consistency_gate(self) -> None:
        plan = DecisionPlan(
            decisions=[],
            semantic_request_payloads=[
                {
                    "cluster_id": "repeat_002000",
                    "repeat_type": "modifier_redundancy",
                    "type": "single_variant_modifier_redundancy",
                    "text": "快乐的开心的孩子",
                }
            ],
        )
        validator_report = {
            "final_repeat_validator": {
                "final_repeat_gate_passed": False,
                "blocking_issues": [
                    {
                        "type": "adjacent_modifier_semantic_redundancy",
                        "text": "快乐的开心的孩子",
                        "phrase": "快乐的开心的孩子",
                    }
                ],
            },
            "hidden_audio_repeat_validator": {"hidden_audio_repeat_gate_passed": True, "blocking_issues": []},
        }

        blockers = ArollEngine()._semantic_request_consistency_blockers(plan, validator_report)

        self.assertEqual(blockers, [])

    def test_missing_request_for_unit_split_human_review_blocks_internally(self) -> None:
        plan = DecisionPlan(
            decisions=[],
            blockers=[
                Blocker(
                    code="UNIT_SPLIT_REQUIRES_HUMAN_REVIEW",
                    message="unit split needs semantic review",
                    layer="decision",
                    context={"cluster_id": "repeat_unit_split"},
                )
            ],
            semantic_request_payloads=[],
        )

        blockers = ArollEngine()._semantic_request_consistency_blockers(plan, {})

        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].code, "INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_UNIT_SPLIT")
        self.assertEqual(blockers[0].context["cluster_id"], "repeat_unit_split")

    def test_matching_unit_split_request_satisfies_consistency_gate(self) -> None:
        plan = DecisionPlan(
            decisions=[],
            blockers=[
                Blocker(
                    code="UNIT_SPLIT_REQUIRES_HUMAN_REVIEW",
                    message="unit split needs semantic review",
                    layer="decision",
                    context={"cluster_id": "repeat_unit_split"},
                )
            ],
            semantic_request_payloads=[
                {
                    "cluster_id": "repeat_unit_split",
                    "type": "unit_split_requires_human_review",
                    "repeat_type": "unit_split",
                    "suggested_for_rough_cut": "apply_suggested_split",
                }
            ],
        )

        blockers = ArollEngine()._semantic_request_consistency_blockers(plan, {})

        self.assertEqual(blockers, [])

    def test_missing_request_for_semantic_decision_not_provided_blocks_internally(self) -> None:
        plan = DecisionPlan(
            decisions=[],
            blockers=[
                Blocker(
                    code="SEMANTIC_DECISION_NOT_PROVIDED",
                    message="semantic decisions json does not cover this unresolved cluster",
                    layer="decision",
                    severity="write_blocker",
                    context={"cluster_id": "repeat_missing"},
                )
            ],
            semantic_request_payloads=[],
        )

        blockers = ArollEngine()._semantic_request_consistency_blockers(plan, {})

        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].code, "INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_DECISION_NOT_PROVIDED")
        self.assertEqual(blockers[0].context["cluster_id"], "repeat_missing")

    def test_matching_request_for_semantic_decision_not_provided_satisfies_gate(self) -> None:
        plan = DecisionPlan(
            decisions=[],
            blockers=[
                Blocker(
                    code="SEMANTIC_DECISION_NOT_PROVIDED",
                    message="semantic decisions json does not cover this unresolved cluster",
                    layer="decision",
                    severity="write_blocker",
                    context={"cluster_id": "repeat_missing"},
                )
            ],
            semantic_request_payloads=[
                {
                    "cluster_id": "repeat_missing",
                    "type": "single_variant_modifier_redundancy",
                    "repeat_type": "modifier_redundancy",
                    "text": "甲的乙的项",
                    "allowed_decisions": ["drop_redundant_modifier", "keep_all", "requires_human_review"],
                }
            ],
        )

        blockers = ArollEngine()._semantic_request_consistency_blockers(plan, {})

        self.assertEqual(blockers, [])

    def test_final_visible_ambiguous_repeat_payload_merges_into_semantic_report_as_warning_only(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_600_000, "他的购物车全是投资和享受"),
                _caption(2, "v21_seg_000002", 1_620_000, 2_300_000, "你的购物车"),
            ]
        )
        plan = DecisionPlan(decisions=[])
        report = _validator_report_for_visible_gate(gate)
        engine = ArollEngine()

        merged = engine._merge_final_visible_repeat_semantic_requests(plan, report)
        engine._refresh_semantic_adjudication_report(plan)
        engine._refresh_validator_semantic_gate_after_request_merge(report, plan)

        self.assertTrue(merged)
        self.assertEqual(len(plan.semantic_request_payloads), 1)
        payload = plan.semantic_request_payloads[0]
        self.assertEqual(payload["issue_type"], "ambiguous_repeat")
        self.assertEqual(payload["cluster_type"], "final_visible_ambiguous_repeat")
        self.assertTrue(payload["warning_only"])
        self.assertFalse(payload["provider_required"])
        self.assertEqual(plan.semantic_adjudication_report["semantic_request_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["semantic_request_unresolved_count"], 0)
        self.assertTrue(plan.semantic_adjudication_report["semantic_adjudication_gate_passed"])
        self.assertEqual(report["semantic_final_review_validator"]["semantic_request_count"], 1)
        self.assertTrue(report["semantic_final_review_validator"]["semantic_final_review_validator_passed"])
        self.assertTrue(report["quality_gate_report"]["gate_passed"])
        self.assertEqual(report["quality_gate_report"]["semantic_request_count"], 1)
        self.assertEqual(report["quality_gate_report"]["semantic_request_unresolved_count"], 0)

    def test_final_visible_ambiguous_repeat_payload_merge_is_idempotent(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_600_000, "他的购物车全是投资和享受"),
                _caption(2, "v21_seg_000002", 1_620_000, 2_300_000, "你的购物车"),
            ]
        )
        plan = DecisionPlan(decisions=[])
        report = _validator_report_for_visible_gate(gate)
        engine = ArollEngine()

        first = engine._merge_final_visible_repeat_semantic_requests(plan, report)
        second = engine._merge_final_visible_repeat_semantic_requests(plan, report)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(plan.semantic_request_payloads), 1)

    def test_final_visible_warning_payload_routes_to_provider_as_advisory_only(self) -> None:
        provider = FinalVisibleAdvisoryProvider(SemanticAdjudicationDecisionType.KEEP_ALL)
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_600_000, "他的购物车全是投资和享受"),
                _caption(2, "v21_seg_000002", 1_620_000, 2_300_000, "你的购物车"),
            ]
        )
        plan = DecisionPlan(decisions=[])
        report = _validator_report_for_visible_gate(gate)
        engine = ArollEngine(semantic_mode="auto", semantic_provider=provider)

        self.assertTrue(engine._merge_final_visible_repeat_semantic_requests(plan, report))
        self.assertTrue(engine._route_final_visible_repeat_semantic_requests(plan))
        engine._refresh_semantic_adjudication_report(plan)
        engine._refresh_validator_semantic_gate_after_request_merge(report, plan)

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0].issue_type.value, "ambiguous_repeat")
        self.assertEqual(provider.requests[0].text_before, "他的购物车全是投资和享受")
        self.assertEqual(provider.requests[0].text_after, "你的购物车")
        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(len(plan.semantic_decision_rows), 1)
        self.assertEqual(plan.semantic_decision_rows[0]["_decision_kind"], "advisory_final_visible_repeat")
        self.assertFalse(plan.semantic_decision_rows[0]["applied"])
        self.assertEqual(plan.semantic_adjudication_report["deepseek_provider_called_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["semantic_request_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["semantic_request_unresolved_count"], 0)
        self.assertTrue(plan.semantic_adjudication_report["semantic_adjudication_gate_passed"])
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_result_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_keep_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_drop_candidate_count"], 0)
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_applied_count"], 0)
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_policy"], "advisory_only_no_timeline_mutation")
        self.assertTrue(report["quality_gate_report"]["gate_passed"])
        self.assertEqual(report["quality_gate_report"]["final_visible_repeat_advisory_keep_count"], 1)
        self.assertEqual(report["quality_gate_report"]["final_visible_repeat_advisory_applied_count"], 0)

    def test_final_visible_provider_drop_decision_does_not_mutate_timeline(self) -> None:
        provider = FinalVisibleAdvisoryProvider(SemanticAdjudicationDecisionType.DROP_LEFT)

        report = ArollEngine(semantic_mode="auto", semantic_provider=provider).run(
            _two_caption_input("他的购物车全是投资和享受", "你的购物车")
        )

        final_text = "".join(segment.text for segment in report.final_timeline)
        caption_text = "".join(caption.text for caption in report.captions)
        self.assertEqual(len(provider.requests), 1)
        self.assertIn("他的购物车全是投资和享受", final_text)
        self.assertIn("你的购物车", final_text)
        self.assertIn("他的购物车全是投资和享受", caption_text)
        self.assertIn("你的购物车", caption_text)
        self.assertEqual(report.decision_plan.semantic_request_payloads, [])
        self.assertTrue(
            any(
                row.get("_decision_kind") == "advisory_final_visible_repeat"
                and row.get("decision") == "drop_left"
                and row.get("applied") is False
                for row in report.decision_plan.semantic_decision_rows
            )
        )
        self.assertTrue(
            any(
                row.get("route") == "final_visible_repeat"
                and row.get("decision") == "drop_left"
                and row.get("applied") is False
                and row.get("warning_only") is True
                for row in report.decision_trace
            )
        )
        semantic_report = report.decision_plan.semantic_adjudication_report
        summary = build_run_summary(report)
        self.assertEqual(semantic_report["final_visible_repeat_advisory_drop_candidate_count"], 1)
        self.assertEqual(semantic_report["final_visible_repeat_advisory_applied_count"], 0)
        self.assertEqual(semantic_report["final_visible_repeat_advisory_decision_counts"], {"drop_left": 1})
        self.assertEqual(summary["final_visible_repeat_advisory_drop_candidate_count"], 1)
        self.assertEqual(summary["final_visible_repeat_advisory_applied_count"], 0)
        self.assertEqual(summary["final_visible_repeat_advisory_policy"], "advisory_only_no_timeline_mutation")

    def test_final_visible_provider_human_review_decision_is_reported_without_blocking_write(self) -> None:
        provider = FinalVisibleAdvisoryProvider(SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW)
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_600_000, "他的购物车全是投资和享受"),
                _caption(2, "v21_seg_000002", 1_620_000, 2_300_000, "你的购物车"),
            ]
        )
        plan = DecisionPlan(decisions=[])
        report = _validator_report_for_visible_gate(gate)
        engine = ArollEngine(semantic_mode="auto", semantic_provider=provider)

        self.assertTrue(engine._merge_final_visible_repeat_semantic_requests(plan, report))
        self.assertTrue(engine._route_final_visible_repeat_semantic_requests(plan))
        engine._refresh_semantic_adjudication_report(plan)
        engine._refresh_validator_semantic_gate_after_request_merge(report, plan)

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.semantic_decision_rows, [])
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_count"], 0)
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_result_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_review_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["final_visible_repeat_advisory_unresolved_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["semantic_request_unresolved_count"], 0)
        self.assertTrue(plan.semantic_adjudication_report["semantic_adjudication_gate_passed"])
        self.assertTrue(report["quality_gate_report"]["gate_passed"])
        self.assertEqual(report["quality_gate_report"]["final_visible_repeat_advisory_review_count"], 1)
        self.assertEqual(report["quality_gate_report"]["final_visible_repeat_advisory_unresolved_count"], 1)


if __name__ == "__main__":
    unittest.main()
