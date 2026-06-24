from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aroll_v21.decision.deepseek_semantic_planner import FORBIDDEN_PROVIDER_FIELDS
from aroll_v21.pre_review.full_text_pre_review import (
    REPORT_KIND,
    build_review_payload,
    run_full_text_pre_review,
)


class FakePreReviewProvider:
    provider_name = "fake_full_text_pre_review"

    def __init__(self, response: dict) -> None:
        self.response = response

    def review(self, payload: dict) -> dict:
        self.last_payload = payload
        return self.response


class ArollV21FullTextPreReviewTests(unittest.TestCase):
    def test_build_payload_is_sidecar_and_excludes_physical_edit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run_dir(Path(tmp))
            payload = build_review_payload(run_dir)

            self.assertEqual(payload["report_kind"], REPORT_KIND)
            self.assertTrue(payload["sidecar_only"])
            self.assertTrue(payload["non_blocking"])
            self.assertEqual(payload["review_item_count"], 2)
            self.assertEqual(payload["review_items"][0]["review_item_id"], "review_0000")
            self.assertIn("visible_caption_texts", payload["review_items"][0])
            serialized = json.dumps(payload, ensure_ascii=False)
            for field in FORBIDDEN_PROVIDER_FIELDS:
                self.assertNotIn(f'"{field}":', serialized)

    def test_payload_keeps_full_visible_caption_sequence_for_long_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run_dir(Path(tmp))
            self._write_json(
                run_dir / "captions.json",
                [
                    {"caption_id": f"cap{index}", "timeline_segment_ids": ["seg1"], "text": f"第{index}条字幕"}
                    for index in range(1, 7)
                ],
            )

            payload = build_review_payload(run_dir)

            item = payload["review_items"][0]
            self.assertEqual(item["visible_caption_count"], 6)
            self.assertIn("第6条字幕", item["visible_caption_text_sequence"])
            self.assertEqual(item["visible_caption_texts"][-1], "第6条字幕")
            self.assertFalse(item["visible_caption_texts_truncated"])

    def test_run_writes_report_markdown_and_hotspots_without_changing_ready_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run_dir(Path(tmp))
            provider = FakePreReviewProvider(
                {
                    "qc_recommendation": "warning",
                    "summary": "发现一处疑似断句，需要人工复听。",
                    "issues": [
                        {
                            "issue_type": "semantic_jump",
                            "severity": "warning",
                            "confidence": 0.88,
                            "review_item_ids": ["review_0001"],
                            "evidence_text": "突然接到下一句",
                            "reason": "上下句连接像中间缺了一块。",
                            "suggested_human_qc": "复听该处前后 3 秒。",
                        }
                    ],
                }
            )

            result = run_full_text_pre_review(run_dir, provider=provider)

            self.assertEqual(result.report["status"], "ok")
            self.assertTrue(result.report["does_not_change_ready_gate"])
            self.assertTrue(result.report["does_not_write_draft"])
            self.assertEqual(result.report["issue_count"], 1)
            self.assertEqual(result.hotspots["hotspot_count"], 1)
            self.assertEqual(result.triage["summary"]["human_audio_review_count"], 1)
            self.assertTrue((run_dir / "quality" / "full_text_pre_review.json").exists())
            self.assertTrue((run_dir / "quality" / "full_text_pre_review.md").exists())
            self.assertTrue((run_dir / "quality" / "full_text_pre_review_hotspots.json").exists())
            self.assertTrue((run_dir / "quality" / "full_text_pre_review_triage.json").exists())
            self.assertTrue((run_dir / "quality" / "full_text_pre_review_triage.md").exists())

            report = json.loads((run_dir / "quality" / "full_text_pre_review.json").read_text("utf-8"))
            self.assertEqual(report["qc_recommendation"], "warning")
            self.assertEqual(report["issues"][0]["issue_id"], "FTPR-001")
            self.assertEqual(report["triage_summary"]["human_audio_review_count"], 1)

            triage = json.loads((run_dir / "quality" / "full_text_pre_review_triage.json").read_text("utf-8"))
            self.assertTrue(triage["sidecar_only"])
            self.assertTrue(triage["does_not_change_ready_gate"])
            self.assertFalse(triage["automated_timeline_mutation_allowed"])
            self.assertEqual(triage["triage_items"][0]["triage_bucket"], "human_audio_review")

    def test_triage_classifies_repeat_as_deterministic_backlog_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run_dir(Path(tmp))
            provider = FakePreReviewProvider(
                {
                    "qc_recommendation": "block_candidate",
                    "summary": "发现重复。",
                    "issues": [
                        {
                            "issue_type": "bad_repeat",
                            "severity": "fatal_candidate",
                            "confidence": 0.95,
                            "review_item_ids": ["review_0000"],
                            "evidence_text": "重复两遍",
                            "reason": "机械重复。",
                            "suggested_human_qc": "确认是否要做本地规则。",
                        }
                    ],
                }
            )

            result = run_full_text_pre_review(run_dir, provider=provider)

            triage_item = result.triage["triage_items"][0]
            self.assertEqual(triage_item["triage_bucket"], "deterministic_rule_backlog")
            self.assertEqual(triage_item["automation_policy"], "collect_as_rule_backlog_only_no_direct_edit")
            self.assertIn("regression_test_required", triage_item["tags"])
            self.assertTrue(result.triage["block_candidate_is_qc_only"])

    def test_triage_classifies_visible_caption_text_match_as_asr_review_not_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run_dir(Path(tmp))
            provider = FakePreReviewProvider(
                {
                    "qc_recommendation": "warning",
                    "summary": "可见字幕文本和最终文本一致，但词义疑似 ASR 错。",
                    "issues": [
                        {
                            "issue_type": "visible_caption_mismatch",
                            "severity": "warning",
                            "confidence": 0.9,
                            "review_item_ids": ["review_0000"],
                            "evidence_text": "文本疑似误识别",
                            "reason": "字幕和最终文本一致，问题更像文案复核。",
                            "suggested_human_qc": "听原音频确认。",
                        }
                    ],
                }
            )

            result = run_full_text_pre_review(run_dir, provider=provider)

            triage_item = result.triage["triage_items"][0]
            self.assertEqual(triage_item["triage_bucket"], "asr_or_text_mismatch_review")
            self.assertIn("visible_caption_sequence_matches_final_text", triage_item["tags"])
            self.assertEqual(triage_item["automation_policy"], "manual_wording_review_no_auto_rewrite")

    def test_triage_invalid_review_item_id_lowers_priority_as_false_positive_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run_dir(Path(tmp))
            provider = FakePreReviewProvider(
                {
                    "qc_recommendation": "warning",
                    "summary": "bad ids",
                    "issues": [
                        {
                            "issue_type": "caption_fragment",
                            "severity": "warning",
                            "confidence": 0.6,
                            "review_item_ids": ["not_real"],
                            "evidence_text": "missing id",
                            "reason": "provider referenced invalid item id.",
                            "suggested_human_qc": "ignore unless reproducible.",
                        }
                    ],
                }
            )

            result = run_full_text_pre_review(run_dir, provider=provider)

            triage_item = result.triage["triage_items"][0]
            self.assertEqual(triage_item["review_item_ids"], [])
            self.assertEqual(triage_item["triage_bucket"], "pre_review_false_positive_candidate")
            self.assertIn("no_valid_review_item_id", triage_item["tags"])

    def test_provider_forbidden_physical_fields_are_dropped_from_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run_dir(Path(tmp))
            provider = FakePreReviewProvider(
                {
                    "qc_recommendation": "block_candidate",
                    "summary": "bad provider payload",
                    "source_start_us": 1,
                    "issues": [
                        {
                            "issue_type": "bad_repeat",
                            "severity": "fatal_candidate",
                            "confidence": 0.9,
                            "review_item_ids": ["review_0000", "not_real"],
                            "evidence_text": "重复两遍",
                            "reason": "机械重复",
                            "suggested_human_qc": "复听重复",
                            "target_start_us": 2,
                            "segment_id": "danger",
                        }
                    ],
                }
            )

            result = run_full_text_pre_review(run_dir, provider=provider)
            serialized = json.dumps(result.report, ensure_ascii=False)

            self.assertEqual(result.report["status"], "ok")
            self.assertEqual(result.report["issues"][0]["review_item_ids"], ["review_0000"])
            self.assertIn("source_start_us", result.report["provider_forbidden_fields_dropped"])
            self.assertIn("target_start_us", result.report["provider_forbidden_fields_dropped"])
            self.assertIn("segment_id", result.report["provider_forbidden_fields_dropped"])
            for field in ("source_start_us", "target_start_us", "segment_id"):
                self.assertNotIn(f'"{field}":', serialized)

    def test_missing_provider_writes_non_blocking_warning_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run_dir(Path(tmp))

            result = run_full_text_pre_review(run_dir, provider=None)

            self.assertEqual(result.report["status"], "provider_missing")
            self.assertEqual(result.report["qc_recommendation"], "warning")
            self.assertTrue(result.report["non_blocking"])
            self.assertTrue((run_dir / "quality" / "full_text_pre_review.json").exists())

    def _write_run_dir(self, root: Path) -> Path:
        run_dir = root / "run"
        run_dir.mkdir(parents=True)
        self._write_json(
            run_dir / "source_graph.json",
            {
                "words": [
                    {
                        "word_id": "w1",
                        "text": "你",
                        "source_start_us": 0,
                        "source_end_us": 100,
                        "source_material_id": "m1",
                    },
                    {
                        "word_id": "w2",
                        "text": "好",
                        "source_start_us": 100,
                        "source_end_us": 200,
                        "source_material_id": "m1",
                    },
                ]
            },
        )
        self._write_json(
            run_dir / "final_timeline.json",
            [
                {
                    "segment_id": "seg1",
                    "source_material_id": "m1",
                    "source_start_us": 0,
                    "source_end_us": 1000,
                    "target_start_us": 0,
                    "target_end_us": 1000,
                    "word_ids": ["w1"],
                    "text": "你根本不知道什么叫爽",
                },
                {
                    "segment_id": "seg2",
                    "source_material_id": "m1",
                    "source_start_us": 1000,
                    "source_end_us": 2000,
                    "target_start_us": 1000,
                    "target_end_us": 2000,
                    "word_ids": ["w2"],
                    "text": "突然接一句阶层跃迁",
                },
            ],
        )
        self._write_json(
            run_dir / "captions.json",
            [
                {"caption_id": "cap1", "timeline_segment_ids": ["seg1"], "text": "你根本不知道什么叫爽"},
                {"caption_id": "cap2", "timeline_segment_ids": ["seg2"], "text": "突然接一句阶层跃迁"},
            ],
        )
        self._write_json(
            run_dir / "run_summary.json",
            {
                "status": "ok",
                "READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT": True,
                "caption_count": 2,
                "final_video_segment_count": 2,
            },
        )
        self._write_json(run_dir / "quality_gate_report.json", {"gate_passed": True})
        self._write_json(run_dir / "final_caption_visible_repeat_gate.json", {"gate_passed": True})
        self._write_json(run_dir / "final_visible_caption_repair_report.json", {"status": "ok"})
        self._write_json(run_dir / "semantic_adjudication_report.json", {"semantic_unresolved_count": 0})
        self._write_json(run_dir / "decision_trace.json", [])
        return run_dir

    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    unittest.main()
