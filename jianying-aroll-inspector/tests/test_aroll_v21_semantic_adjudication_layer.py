from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.decision import (
    SemanticAdjudicationDecision,
    SemanticAdjudicationDecisionType,
    SemanticAdjudicationRequest,
    SemanticDecisionPlanner,
    SemanticDecisionsJsonPlanner,
)
from aroll_v21.decision.semantic_adjudication import request_from_cluster
from aroll_v21.evidence import CandidateEvidenceBuilder
from aroll_v21.ingest import DraftIngest
from aroll_v21.quality.quality_gate import build_quality_gate_report
from aroll_v21.operator import ArollV21OperatorConfig, run_operator
from tests.test_aroll_v21_semantic_planner_contract import _semantic_cluster
from tests.test_aroll_v21_semantic_request_modifier_redundancy import final_modifier_fixture_input
from tests.test_aroll_v21_semantic_unconfigured_dryrun_policy import semantic_run_input
from tests.test_aroll_v21_single_variant_modifier_redundancy import _graph_for_text


ROOT = Path(__file__).resolve().parents[1]


class FakeSemanticProvider:
    provider_name = "fake_deepseek"

    def __init__(self, decision: SemanticAdjudicationDecisionType) -> None:
        self.decision = decision
        self.requests = []

    def decide(self, requests):
        self.requests.extend(requests)
        rows = []
        for request in requests:
            rows.append(
                SemanticAdjudicationDecision(
                    issue_id=request.issue_id,
                    decision=self.decision,
                    reason="fake provider decision",
                    confidence=0.91,
                    provider_name=self.provider_name,
                )
            )
        return rows


class PhysicalFieldSemanticProvider:
    provider_name = "fake_physical_deepseek"

    def decide(self, requests):
        return [
            SemanticAdjudicationDecision(
                issue_id=requests[0].issue_id,
                decision=SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW,
                reason="invalid physical metadata",
                confidence=0.91,
                provider_name=self.provider_name,
                metadata={"source_start_us": 123},
            )
        ]


class FailingSemanticProvider:
    provider_name = "failing_deepseek"

    def decide(self, requests):
        raise AssertionError("DeepSeek provider must not be called")


def _material_rows() -> tuple[list[dict], list[dict]]:
    payload = json.loads((ROOT / "fixtures" / "real_materials" / "normal_caption_template.json").read_text("utf-8"))
    return [payload["material"]], [payload["segment"]]


def _two_caption_input(left: str, right: str) -> ArollRunInput:
    text_materials, text_segments = _material_rows()
    words = []
    subtitles = []
    cursor = 0
    for subtitle_index, text in enumerate([left, right], start=1):
        word_ids = []
        for char in text:
            word_id = f"w_{len(words) + 1:06d}"
            word_ids.append(word_id)
            words.append(
                {
                    "word_id": word_id,
                    "word_text": char,
                    "source_start_us": cursor,
                    "source_end_us": cursor + 90_000,
                    "subtitle_uid": f"s{subtitle_index}",
                    "subtitle_index": subtitle_index,
                }
            )
            cursor += 90_000
        subtitles.append({"subtitle_uid": f"s{subtitle_index}", "subtitle_index": subtitle_index, "text": text, "word_ids": word_ids})
    return ArollRunInput(
        source_segments=[{"id": "clip", "material_id": "main_video", "source_start_us": 0, "source_end_us": cursor + 1_000_000}],
        word_timeline=words,
        subtitles=subtitles,
        text_materials=text_materials,
        text_segments=text_segments,
        postwrite_mode="simulated",
    )


def _write_input(path: Path, run_input: ArollRunInput) -> None:
    path.write_text(
        json.dumps(
            {
                "source_segments": run_input.source_segments,
                "word_timeline": run_input.word_timeline,
                "subtitles": run_input.subtitles,
                "text_materials": run_input.text_materials,
                "text_segments": run_input.text_segments,
            },
            ensure_ascii=False,
        ),
        "utf-8",
    )


def _write_deepseek_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "deepseek:",
                "  api" + "-key: local-test-token",
                "  base-url: https://api.deepseek.com",
                "app:",
                "  ai:",
                "    semantic:",
                "      model: deepseek-v4-pro",
            ]
        ),
        "utf-8",
    )


def _provider_drop_aborted(calls: list):
    def decide(_provider, requests):
        calls.extend(requests)
        return [
            SemanticAdjudicationDecision(
                issue_id=request.issue_id,
                decision=SemanticAdjudicationDecisionType.DROP_ABORTED,
                reason="test provider drops aborted phrase",
                confidence=0.91,
                provider_name="deepseek_semantic_planner",
            )
            for request in requests
        ]

    return decide


def _self_repair_clusters():
    run_input = _two_caption_input("这个问题需要从", "这个问题可以看")
    source_graph = DraftIngest().build_source_graph(
        word_timeline=run_input.word_timeline,
        subtitles=run_input.subtitles,
        source_segments=[{"id": "clip", "material_id": "main_video", "source_start_us": 0, "source_end_us": 2_000_000}],
    )
    return CandidateEvidenceBuilder().build(source_graph)


def _high_fatal_semantic_cluster(issue_type: str, severity: str = "fatal"):
    cluster = _semantic_cluster()
    evidence = replace(
        cluster.evidence[0],
        metadata={"candidate": {"issue_type": issue_type, "severity": severity}},
    )
    return replace(
        cluster,
        repeat_type="semantic_retry",
        evidence=[evidence],
        local_recommendation="semantic_review",
    )


class ArollV21SemanticAdjudicationLayerTests(unittest.TestCase):
    def test_auto_mode_uses_local_decision_for_exact_repeat_without_deepseek(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.DROP_ABORTED)
        report = ArollEngine(semantic_mode="auto", semantic_provider=provider).run(_two_caption_input("相信自己", "相信自己"))

        self.assertEqual(provider.requests, [])
        self.assertEqual([segment.text for segment in report.final_timeline], ["相信自己"])
        self.assertGreaterEqual(report.decision_plan.semantic_adjudication_report["semantic_auto_route_count"], 1)
        self.assertGreaterEqual(report.decision_plan.semantic_adjudication_report["semantic_local_decision_count"], 1)
        self.assertEqual(report.decision_plan.semantic_adjudication_report["semantic_provider_required_count"], 0)

    def test_auto_mode_calls_deepseek_for_fatal_modifier_when_local_uncertain(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW)
        plan = SemanticDecisionPlanner(semantic_mode="auto", semantic_provider=provider).plan([_semantic_cluster()])

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0].issue_type.value, "modifier_redundancy")
        self.assertEqual(plan.semantic_adjudication_report["semantic_provider_required_count"], 1)
        self.assertEqual(plan.semantic_adjudication_report["deepseek_provider_called_count"], 1)

    def test_auto_mode_calls_deepseek_for_self_repair_aborted_phrase_when_ambiguous(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.DROP_ABORTED)
        report = ArollEngine(semantic_mode="auto", semantic_provider=provider).run(_two_caption_input("这个问题需要从", "这个问题可以看"))

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0].issue_type.value, "self_repair_aborted_phrase")
        self.assertEqual([segment.text for segment in report.final_timeline], ["这个问题可以看"])
        self.assertEqual(report.decision_plan.semantic_adjudication_report["semantic_provider_required_count"], 1)

    def test_auto_mode_blocks_high_fatal_when_provider_missing(self) -> None:
        plan = SemanticDecisionPlanner(semantic_mode="auto").plan([_semantic_cluster()])

        self.assertIn("V21_SEMANTIC_ADJUDICATION_PROVIDER_MISSING", [blocker.code for blocker in plan.blockers])
        self.assertFalse(plan.write_allowed)

    def test_auto_mode_never_silent_keep_all_for_provider_required_issue(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.KEEP_ALL)
        plan = SemanticDecisionPlanner(semantic_mode="auto", semantic_provider=provider).plan(_self_repair_clusters())

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(plan.decisions, [])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertIn("V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED", [blocker.code for blocker in plan.blockers])

    def test_deepseek_provider_not_used_for_audio_coverage_or_text_residue(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.DROP_ABORTED)
        report = ArollEngine(semantic_mode="auto", semantic_provider=provider).run(_two_caption_input("正常字幕", "继续表达"))

        self.assertEqual(provider.requests, [])
        self.assertEqual(report.decision_plan.semantic_adjudication_report["semantic_provider_required_count"], 0)

    def test_deepseek_provider_called_for_fatal_modifier_redundancy(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW)
        plan = SemanticDecisionPlanner(semantic_mode="deepseek", semantic_provider=provider).plan([_semantic_cluster()])

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0].issue_type.value, "modifier_redundancy")
        self.assertEqual(plan.semantic_adjudication_report["deepseek_provider_called_count"], 1)

    def test_deepseek_provider_physical_metadata_blocks(self) -> None:
        plan = SemanticDecisionPlanner(
            semantic_mode="deepseek",
            semantic_provider=PhysicalFieldSemanticProvider(),
        ).plan([_semantic_cluster()])

        self.assertEqual(plan.decisions, [])
        self.assertIn("DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS", [blocker.code for blocker in plan.blockers])
        self.assertIn("source_start_us", plan.blockers[0].context.get("forbidden_fields") or [])

    def test_deepseek_provider_called_for_self_repair_aborted_phrase(self) -> None:
        clusters = _self_repair_clusters()
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.DROP_ABORTED)
        plan = SemanticDecisionPlanner(semantic_mode="deepseek", semantic_provider=provider).plan(clusters)

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0].issue_type.value, "self_repair_aborted_phrase")
        self.assertEqual(plan.decisions[0].drop_unit_ids, ["s1"])

    def test_self_repair_provider_keep_all_rejected(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.KEEP_ALL)
        plan = SemanticDecisionPlanner(semantic_mode="deepseek", semantic_provider=provider).plan(_self_repair_clusters())

        self.assertEqual(plan.decisions, [])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertIn("V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED", [blocker.code for blocker in plan.blockers])

    def test_self_repair_request_allows_dropping_bad_right_side(self) -> None:
        request = request_from_cluster(_self_repair_clusters()[0])

        self.assertIn(SemanticAdjudicationDecisionType.DROP_RIGHT.value, request.allowed_decisions)

    def test_self_repair_provider_drop_right_keeps_left_side(self) -> None:
        class DropRightProvider:
            provider_name = "drop_right_provider"

            def decide(self, requests: list[SemanticAdjudicationRequest]) -> list[SemanticAdjudicationDecision]:
                return [
                    SemanticAdjudicationDecision(
                        issue_id=request.issue_id,
                        decision=SemanticAdjudicationDecisionType.DROP_RIGHT,
                        reason="right side is likely ASR garbage, keep the left phrase",
                        confidence=0.82,
                        provider_name=self.provider_name,
                    )
                    for request in requests
                ]

        clusters = _self_repair_clusters()
        plan = SemanticDecisionPlanner(semantic_mode="deepseek", semantic_provider=DropRightProvider()).plan(clusters)

        self.assertFalse(plan.blocked)
        self.assertEqual(plan.decisions[0].keep_unit_id, clusters[0].variants[0].unit_id)
        self.assertEqual(plan.decisions[0].drop_unit_ids, [clusters[0].variants[1].unit_id])

    def test_self_repair_external_keep_all_rejected(self) -> None:
        clusters = _self_repair_clusters()
        plan = SemanticDecisionPlanner(
            deepseek_planner=SemanticDecisionsJsonPlanner(
                [{"cluster_id": clusters[0].cluster_id, "decision": "keep_all", "reason": "reject me", "confidence": 0.8}]
            )
        ).plan(clusters)

        self.assertEqual(plan.decisions, [])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertIn("V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED", [blocker.code for blocker in plan.blockers])

    def test_deepseek_missing_provider_blocks_high_fatal_issue(self) -> None:
        plan = SemanticDecisionPlanner(semantic_mode="deepseek").plan([_semantic_cluster()])

        self.assertIn("V21_SEMANTIC_ADJUDICATION_PROVIDER_MISSING", [blocker.code for blocker in plan.blockers])
        self.assertFalse(plan.write_allowed)

    def test_deterministic_baseline_cannot_handle_high_fatal_semantic_issue_as_keep_all(self) -> None:
        plan = SemanticDecisionPlanner(semantic_mode="deterministic-baseline").plan([_semantic_cluster()])

        self.assertEqual(plan.decisions, [])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertEqual(plan.semantic_adjudication_report["deterministic_baseline_refused_count"], 1)
        self.assertNotIn("keep_all", json.dumps(plan.semantic_decision_rows, ensure_ascii=False))

    def test_deterministic_baseline_cannot_keep_all_high_fatal_semantic_issue(self) -> None:
        self.test_deterministic_baseline_cannot_handle_high_fatal_semantic_issue_as_keep_all()

    def test_semantic_request_payload_emitted_for_unresolved_high_fatal_issue(self) -> None:
        plan = SemanticDecisionPlanner().plan([_semantic_cluster()])
        payload = plan.semantic_request_payloads[0]

        for key in (
            "issue_id",
            "issue_type",
            "severity",
            "candidate_segment_ids",
            "candidate_caption_ids",
            "word_ids",
            "source_start_us",
            "source_end_us",
            "target_start_us",
            "target_end_us",
            "text_before",
            "text_after",
            "local_context",
            "recommended_action",
            "why_local_policy_cannot_decide",
        ):
            self.assertIn(key, payload)

    def test_fatal_modifier_redundancy_cannot_keep_all_any_layer(self) -> None:
        report = ArollEngine(
            deepseek_planner=SemanticDecisionsJsonPlanner(
                [{"cluster_id": "repeat_002000", "decision": "keep_all", "reason": "reject me", "confidence": 0.8}]
            )
        ).run(final_modifier_fixture_input())

        self.assertEqual(report.status, "blocked")
        self.assertIn("V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED", [blocker.code for blocker in report.blocker_report.blockers])
        self.assertFalse(any(row.get("decision") == "keep_all" and row.get("applied") for row in report.decision_trace))

    def test_no_keep_all_backdoor_for_fatal_modifier_in_compiler(self) -> None:
        self.test_fatal_modifier_redundancy_cannot_keep_all_any_layer()

    def test_no_keep_all_backdoor_for_self_repair_in_compiler(self) -> None:
        clusters = _self_repair_clusters()
        plan = SemanticDecisionPlanner(
            deepseek_planner=SemanticDecisionsJsonPlanner(
                [{"cluster_id": clusters[0].cluster_id, "decision": "keep_all", "reason": "reject me", "confidence": 0.8}]
            )
        ).plan(clusters)

        self.assertEqual(plan.decisions, [])
        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertIn("V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED", [blocker.code for blocker in plan.blockers])

    def test_external_semantic_decision_keep_all_rejected_for_fatal_modifier(self) -> None:
        plan = SemanticDecisionPlanner(
            deepseek_planner=SemanticDecisionsJsonPlanner(
                [{"cluster_id": "cluster_1", "decision": "keep_all", "reason": "not redundant", "confidence": 0.8}]
            )
        ).plan([_semantic_cluster()])

        self.assertEqual(plan.semantic_unresolved_count, 1)
        self.assertEqual(len(plan.semantic_request_payloads), 1)
        self.assertIn("V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED", [blocker.code for blocker in plan.blockers])

    def test_external_semantic_keep_all_rejected_for_high_fatal(self) -> None:
        cases = [
            ("modifier_redundancy", _semantic_cluster(), "V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED"),
            ("self_repair_aborted_phrase", _high_fatal_semantic_cluster("self_repair_aborted_phrase"), "V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED"),
            ("semantic_containment", _high_fatal_semantic_cluster("semantic_containment"), "SEMANTIC_DECISION_NOT_PROVIDED"),
            ("near_duplicate_take", _high_fatal_semantic_cluster("near_duplicate_take", "high"), "SEMANTIC_DECISION_NOT_PROVIDED"),
            ("visible_caption_repeat", _high_fatal_semantic_cluster("visible_caption_repeat", "high"), "SEMANTIC_DECISION_NOT_PROVIDED"),
        ]

        for issue_type, cluster, blocker_code in cases:
            with self.subTest(issue_type=issue_type):
                plan = SemanticDecisionPlanner(
                    deepseek_planner=SemanticDecisionsJsonPlanner(
                        [{"cluster_id": cluster.cluster_id, "decision": "keep_all", "reason": "reject me", "confidence": 0.8}]
                    )
                ).plan([cluster])
                self.assertEqual(plan.decisions, [])
                self.assertEqual(plan.semantic_unresolved_count, 1)
                self.assertIn(blocker_code, [blocker.code for blocker in plan.blockers])

    def test_modifier_redundancy_repairs_redundant_modifier_without_hardcoding_sample(self) -> None:
        source_graph = _graph_for_text("快乐的开心的孩子")
        clusters = CandidateEvidenceBuilder().build(source_graph)
        plan = SemanticDecisionPlanner().plan(clusters)

        self.assertEqual(plan.split_decisions[0].drop_word_ids, ["w_000001", "w_000002", "w_000003"])
        self.assertEqual(plan.semantic_unresolved_count, 0)

    def test_modifier_redundancy_unresolved_without_provider_blocks_ready(self) -> None:
        report = ArollEngine().run(semantic_run_input())

        self.assertFalse(report.blocker_report.summary["READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT"])
        self.assertFalse(report.validator_report["quality_gate_report"]["semantic_adjudication_gate_passed"])

    def test_self_repair_aborted_phrase_drops_incomplete_restart(self) -> None:
        report = ArollEngine().run(_two_caption_input("但在金融市场的角", "但在金融的视角下"))

        self.assertEqual([segment.text for segment in report.final_timeline], ["但在金融的视角下"])

    def test_self_repair_keeps_completed_rephrase(self) -> None:
        report = ArollEngine().run(_two_caption_input("但在金融市场的角", "但在金融的视角下"))

        self.assertEqual([caption.text for caption in report.captions], ["但在金融的视角下"])

    def test_self_repair_ambiguous_case_requires_semantic_adjudication(self) -> None:
        report = ArollEngine().run(_two_caption_input("这个问题需要从", "这个问题可以看"))

        self.assertEqual(report.decision_plan.semantic_unresolved_count, 1)
        self.assertEqual(report.decision_plan.semantic_request_payloads[0]["issue_type"], "self_repair_aborted_phrase")
        self.assertIn("V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED", [blocker.code for blocker in report.blocker_report.blockers])

    def test_self_repair_provider_missing_blocks_high_confidence_unresolved(self) -> None:
        report = ArollEngine(semantic_mode="deepseek").run(_two_caption_input("这个问题需要从", "这个问题可以看"))

        self.assertIn("V21_SEMANTIC_ADJUDICATION_PROVIDER_MISSING", [blocker.code for blocker in report.blocker_report.blockers])
        self.assertIn("V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED", [blocker.code for blocker in report.blocker_report.blockers])

    def test_no_hardcoded_finance_angle_phrase(self) -> None:
        source = "\n".join(path.read_text("utf-8") for path in (ROOT / "src" / "aroll_v21").rglob("*.py"))

        self.assertNotIn("金融市场的角", source)
        self.assertNotIn("金融的视角下", source)

    def test_no_asr_completion_hardcoded_du(self) -> None:
        source = "\n".join(path.read_text("utf-8") for path in (ROOT / "src" / "aroll_v21").rglob("*.py"))

        self.assertNotIn("金融市场的角度", source)
        self.assertNotIn("补度", source)

    def test_no_hardcoded_suiyi_siyi_phrase(self) -> None:
        source = "\n".join(path.read_text("utf-8") for path in (ROOT / "src" / "aroll_v21").rglob("*.py"))

        self.assertNotIn("随意的肆意的踩踏", source)

    def test_no_hardcoded_finance_angle_or_suiyi_siyi_phrase(self) -> None:
        source = "\n".join(path.read_text("utf-8") for path in (ROOT / "src" / "aroll_v21").rglob("*.py"))

        self.assertNotIn("金融市场的角", source)
        self.assertNotIn("金融的视角下", source)
        self.assertNotIn("随意的肆意的踩踏", source)

    def test_unresolved_fatal_semantic_issue_emits_request_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            _write_input(input_json, semantic_run_input())

            run_operator(ArollV21OperatorConfig(mode="dry-run", run_dir=root / "run", input_json=input_json))

            payloads = json.loads((root / "run" / "semantic_request_payloads.json").read_text("utf-8"))
            report = json.loads((root / "run" / "semantic_adjudication_report.json").read_text("utf-8"))
            self.assertEqual(len(payloads), 1)
            self.assertEqual(report["semantic_request_unresolved_count"], 1)

    def test_unresolved_fatal_semantic_issue_blocks_ready(self) -> None:
        report = ArollEngine().run(semantic_run_input())

        self.assertFalse(report.blocker_report.summary["READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT"])
        self.assertGreater(report.validator_report["quality_gate_report"]["fatal_semantic_issue_count"], 0)

    def test_quality_gate_blocks_unresolved_fatal_after_compiler(self) -> None:
        self.test_unresolved_fatal_semantic_issue_blocks_ready()

    def test_resolved_semantic_issue_records_decision_trace(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.DROP_ABORTED)
        report = ArollEngine(semantic_mode="deepseek", semantic_provider=provider).run(_two_caption_input("这个问题需要从", "这个问题可以看"))

        self.assertEqual([segment.text for segment in report.final_timeline], ["这个问题可以看"])
        self.assertTrue(report.decision_plan.semantic_adjudication_report["results"])
        self.assertIn("deepseek_required", {row.get("route") for row in report.decision_trace})

    def test_operator_deepseek_mode_passes_env_provider_through_quality_gate(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.DROP_ABORTED)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            _write_input(input_json, _two_caption_input("这个问题需要从", "这个问题可以看"))

            with patch("aroll_v21.operator.deepseek_provider_from_env", return_value=provider):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="dry-run",
                        run_dir=root / "run",
                        input_json=input_json,
                        semantic_mode="deepseek",
                    )
                )

            semantic_report = json.loads((root / "run" / "semantic_adjudication_report.json").read_text("utf-8"))
            quality_gate = json.loads((root / "run" / "quality_gate_report.json").read_text("utf-8"))
            final_timeline = json.loads((root / "run" / "final_timeline.json").read_text("utf-8"))

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0].issue_type.value, "self_repair_aborted_phrase")
        self.assertEqual([segment["text"] for segment in final_timeline], ["这个问题可以看"])
        self.assertEqual(summary["deepseek_provider_called_count"], 1)
        self.assertEqual(summary["deepseek_provider_not_called_reason"], "")
        self.assertEqual(semantic_report["deepseek_provider_called_count"], 1)
        self.assertTrue(semantic_report["semantic_adjudication_gate_passed"])
        self.assertTrue(quality_gate["semantic_adjudication_gate_passed"])

    def test_dry_run_persists_semantic_decision_cache(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.DROP_ABORTED)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            run_dir = root / "run"
            _write_input(input_json, _two_caption_input("这个问题需要从", "这个问题可以看"))

            with patch("aroll_v21.operator.deepseek_provider_from_env", return_value=provider):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="dry-run",
                        run_dir=run_dir,
                        input_json=input_json,
                        semantic_mode="auto",
                    )
                )

            resolved = json.loads((run_dir / "semantic_decisions.resolved.json").read_text("utf-8"))
            cache = json.loads((run_dir / "semantic_decision_cache.json").read_text("utf-8"))

        self.assertEqual(summary["semantic_mode"], "auto")
        self.assertEqual(summary["deepseek_provider_called_count"], 1)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(cache, resolved)
        self.assertEqual(cache[0]["cluster_id"], "repeat_005000")

    def test_commit_reuses_dry_run_semantic_decision_cache(self) -> None:
        provider = FakeSemanticProvider(SemanticAdjudicationDecisionType.DROP_ABORTED)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            run_dir = root / "run"
            _write_input(input_json, _two_caption_input("这个问题需要从", "这个问题可以看"))

            with patch("aroll_v21.operator.deepseek_provider_from_env", return_value=provider):
                run_operator(
                    ArollV21OperatorConfig(
                        mode="dry-run",
                        run_dir=run_dir,
                        input_json=input_json,
                        semantic_mode="auto",
                    )
                )

            with patch("aroll_v21.operator.deepseek_provider_from_env", return_value=FailingSemanticProvider()):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="write",
                        run_dir=run_dir,
                        input_json=input_json,
                        semantic_mode="auto",
                    )
                )

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(summary["deepseek_provider_called_count"], 0)
        self.assertIn(summary["write_status"], {"blocked_actual_decrypt_unavailable", "blocked_by_prewrite_validators"})

    def test_commit_does_not_recall_deepseek_when_cache_exists(self) -> None:
        self.test_commit_reuses_dry_run_semantic_decision_cache()

    def test_auto_mode_calls_provider_for_unresolved_semantic_requests_in_uat_path(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "reference-application.yaml"
            input_json = root / "input.json"
            run_dir = root / "run"
            _write_deepseek_config(config_path)
            _write_input(input_json, _two_caption_input("这个问题需要从", "这个问题可以看"))

            with patch.dict(
                "os.environ",
                {"REFERENCE_VIDEO_DATA_CATCHER_DEEPSEEK_CONFIG_PATH": str(config_path)},
                clear=True,
            ):
                with patch(
                    "aroll_v21.decision.deepseek_semantic_planner.DeepSeekSemanticProvider.decide",
                    new=_provider_drop_aborted(calls),
                ):
                    summary = run_operator(
                        ArollV21OperatorConfig(
                            mode="dry-run",
                            run_dir=run_dir,
                            input_json=input_json,
                            semantic_mode="auto",
                        )
                    )

            semantic_report = json.loads((run_dir / "semantic_adjudication_report.json").read_text("utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].issue_type.value, "self_repair_aborted_phrase")
        self.assertTrue(summary["deepseek_provider_configured"])
        self.assertEqual(summary["deepseek_provider_called_count"], 1)
        self.assertEqual(semantic_report["deepseek_provider_called_count"], 1)

    def test_auto_mode_reports_provider_config_source_without_secret(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "reference-application.yaml"
            input_json = root / "input.json"
            run_dir = root / "run"
            _write_deepseek_config(config_path)
            _write_input(input_json, _two_caption_input("这个问题需要从", "这个问题可以看"))

            with patch.dict(
                "os.environ",
                {"REFERENCE_VIDEO_DATA_CATCHER_DEEPSEEK_CONFIG_PATH": str(config_path)},
                clear=True,
            ):
                with patch(
                    "aroll_v21.decision.deepseek_semantic_planner.DeepSeekSemanticProvider.decide",
                    new=_provider_drop_aborted(calls),
                ):
                    summary = run_operator(
                        ArollV21OperatorConfig(
                            mode="dry-run",
                            run_dir=run_dir,
                            input_json=input_json,
                            semantic_mode="auto",
                        )
                    )

        self.assertEqual(summary["deepseek_provider_config_source"], "reference-application.yaml")
        self.assertNotIn("local-test-token", summary["deepseek_provider_config_source"])
        self.assertNotIn(str(config_path.parent), summary["deepseek_provider_config_source"])

    def test_commit_reuses_cache_after_auto_provider_decision(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "reference-application.yaml"
            input_json = root / "input.json"
            run_dir = root / "run"
            _write_deepseek_config(config_path)
            _write_input(input_json, _two_caption_input("这个问题需要从", "这个问题可以看"))

            with patch.dict(
                "os.environ",
                {"REFERENCE_VIDEO_DATA_CATCHER_DEEPSEEK_CONFIG_PATH": str(config_path)},
                clear=True,
            ):
                with patch(
                    "aroll_v21.decision.deepseek_semantic_planner.DeepSeekSemanticProvider.decide",
                    new=_provider_drop_aborted(calls),
                ):
                    dry_summary = run_operator(
                        ArollV21OperatorConfig(
                            mode="dry-run",
                            run_dir=run_dir,
                            input_json=input_json,
                            semantic_mode="auto",
                        )
                    )
                with patch(
                    "aroll_v21.decision.deepseek_semantic_planner.DeepSeekSemanticProvider.decide",
                    side_effect=AssertionError("DeepSeek provider must not be called when cache exists"),
                ):
                    write_summary = run_operator(
                        ArollV21OperatorConfig(
                            mode="write",
                            run_dir=run_dir,
                            input_json=input_json,
                            semantic_mode="auto",
                        )
                    )

            cache = json.loads((run_dir / "semantic_decision_cache.json").read_text("utf-8"))

        self.assertEqual(dry_summary["deepseek_provider_called_count"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(cache), 1)
        self.assertEqual(write_summary["deepseek_provider_called_count"], 0)
        self.assertTrue(write_summary["semantic_decision_cache_used"])

    def test_ready_requires_semantic_adjudication_gate(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": []},
            semantic_adjudication_gate={
                "semantic_adjudication_gate_passed": False,
                "semantic_request_unresolved_count": 1,
                "fatal_semantic_issue_count": 0,
                "blocker_codes": [],
            },
            visual_pacing_gate={"gate_passed": True, "visual_pacing_executed": True, "visual_merge_safety_gate_passed": True, "blocker_codes": []},
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertIn("V21_SEMANTIC_ADJUDICATION_GATE_FAILED", quality["blocker_codes"])

    def test_quality_gate_blocks_unresolved_fatal_semantic_issue(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": []},
            semantic_adjudication_gate={
                "semantic_adjudication_gate_passed": False,
                "semantic_request_unresolved_count": 1,
                "fatal_semantic_issue_count": 1,
                "blocker_codes": ["V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED"],
            },
            visual_pacing_gate={"gate_passed": True, "visual_pacing_executed": True, "visual_merge_safety_gate_passed": True, "blocker_codes": []},
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertIn("V21_FATAL_SEMANTIC_ISSUE_UNRESOLVED", quality["blocker_codes"])


if __name__ == "__main__":
    unittest.main()
