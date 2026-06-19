from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.aroll_quality_defect_ledger import QCIssueInput, build_ledger, main


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def write_json_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def make_run_dir(root: Path) -> Path:
    run_dir = root / "run"
    write_json(
        run_dir / "run_summary.json",
        {
            "status": "ok",
            "write_status": "committed_after_postwrite_verification",
            "READY_FOR_USER_MANUAL_QC": True,
            "blocker_codes": [],
            "report_profile": "standard",
        },
    )
    write_json(
        run_dir / "captions.json",
        [
            {
                "caption_id": "cap_001",
                "timeline_segment_ids": ["seg_001"],
                "word_ids": ["w001", "w002"],
                "text": "你真的以为你在评论",
                "target_start_us": 0,
                "target_end_us": 900000,
            },
            {
                "caption_id": "cap_002",
                "timeline_segment_ids": ["seg_002"],
                "word_ids": ["w003", "w004"],
                "text": "区里面敲了几个字",
                "target_start_us": 900000,
                "target_end_us": 1500000,
            },
            {
                "caption_id": "cap_003",
                "timeline_segment_ids": ["seg_003"],
                "word_ids": ["w005"],
                "text": "后面一句",
                "target_start_us": 1500000,
                "target_end_us": 1900000,
            },
        ],
    )
    write_json(
        run_dir / "final_timeline.json",
        [
            {
                "segment_id": "seg_001",
                "source_material_id": "main_video",
                "source_segment_id": "clip_a",
                "source_start_us": 100000,
                "source_end_us": 1000000,
                "target_start_us": 0,
                "target_end_us": 900000,
                "word_ids": ["w001", "w002"],
                "text": "你真的以为你在评论",
                "decision_ids": [],
            },
            {
                "segment_id": "seg_002",
                "source_material_id": "main_video",
                "source_segment_id": "clip_a",
                "source_start_us": 1000000,
                "source_end_us": 1600000,
                "target_start_us": 900000,
                "target_end_us": 1500000,
                "word_ids": ["w003", "w004"],
                "text": "区里面敲了几个字",
                "decision_ids": [],
            },
        ],
    )
    write_json_gz(
        run_dir / "source_graph.json.gz",
        {
            "words": [
                {"word_id": "w001", "text": "你真的", "source_start_us": 100000, "source_end_us": 400000, "confidence": 0.99},
                {"word_id": "w002", "text": "以为你在评论", "source_start_us": 400000, "source_end_us": 1000000, "confidence": 0.99},
                {"word_id": "w003", "text": "区里面", "source_start_us": 1000000, "source_end_us": 1300000, "confidence": 0.99},
                {"word_id": "w004", "text": "敲了几个字", "source_start_us": 1300000, "source_end_us": 1600000, "confidence": 0.99},
            ]
        },
    )
    write_json(
        run_dir / "semantic_request_payloads.json",
        [
            {
                "cluster_id": "cluster_visible_001",
                "issue_type": "cross_caption_semantic_containment",
                "word_ids": ["w001", "w002", "w003", "w004"],
                "candidate_text": "你真的以为你在评论区里面敲了几个字",
            }
        ],
    )
    write_json(
        run_dir / "deepseek_decisions.json",
        [
            {
                "cluster_id": "cluster_visible_001",
                "decision": "requires_human_review",
                "reason": "cross-caption issue",
                "confidence": 0.8,
            }
        ],
    )
    write_json(run_dir / "semantic_decisions.resolved.json", [])
    write_json(
        run_dir / "quality_gate_report.json",
        {
            "quality_gate_passed": True,
            "dangling_prefix_suffix_count": 0,
            "semantic_garbage_or_asr_suspect_count": 0,
            "cross_caption_semantic_containment_count": 0,
            "restart_repeat_visible_count": 0,
        },
    )
    write_json(run_dir / "final_caption_visible_repeat_gate.json", {"gate_passed": True, "restart_repeat_visible_count": 0})
    write_json(run_dir / "final_visible_caption_repair_report.json", {"final_visible_repair_success": True, "final_visible_repair_unresolved": 0})
    return run_dir


class ArollQualityDefectLedgerTests(unittest.TestCase):
    def test_build_ledger_extracts_caption_timeline_gate_and_semantic_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = make_run_dir(root)
            out_root = root / "ledger"

            ledger = build_ledger(
                run_dir=run_dir,
                out_root=out_root,
                case_id="case_manual_qc",
                issues=[
                    QCIssueInput(
                        issue_no="1",
                        bad_visible_text="你真的以为你在评论/区里面敲了几个字",
                        root_cause="cross_caption_semantic_containment",
                        note="manual QC found split semantic unit",
                    )
                ],
            )

            issue = ledger["issues"][0]
            self.assertEqual(issue["caption_ids"], ["cap_001", "cap_002"])
            self.assertEqual(issue["final_timeline_segment_ids"], ["seg_001", "seg_002"])
            self.assertEqual(issue["source_media"], "main_video")
            self.assertEqual(issue["source_start_us"], 100000)
            self.assertEqual(issue["source_end_us"], 1600000)
            self.assertEqual(issue["native_words_text"], "你真的以为你在评论区里面敲了几个字")
            self.assertTrue(issue["entered_semantic_request_payloads"])
            self.assertTrue(issue["entered_deepseek"])
            self.assertIn("quality_gate_passed_without_issue_candidate_match", issue["why_gate_passed"])
            self.assertEqual(
                issue["suggested_regression_test"]["test_file"],
                "tests/test_aroll_v21_final_visible_generic_qc_regressions.py",
            )
            self.assertTrue((out_root / "case_manual_qc" / "defect_ledger.json").exists())
            markdown = (out_root / "case_manual_qc" / "defect_ledger.md").read_text("utf-8")
            self.assertIn("A-Roll Quality Defect Ledger", markdown)
            self.assertIn("cross-caption semantic containment", markdown)

    def test_cli_uses_configured_external_ledger_dir_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = make_run_dir(root)
            ledger_root = root / "runtime" / "aroll_v21_audits" / "quality_defect_ledger"
            issues_json = root / "issues.json"
            write_json(
                issues_json,
                {
                    "issues": [
                        {
                            "issue_no": "qc2",
                            "bad_visible_text": "你真的以为你在评论区里面敲了几个字",
                            "root_cause": "cross_caption_semantic_containment",
                        }
                    ]
                },
            )

            with patch.dict("os.environ", {"AUTO_CLIP_AROLL_QUALITY_DEFECT_LEDGER_DIR": str(ledger_root)}, clear=False):
                exit_code = main(["--run-dir", str(run_dir), "--issues-json", str(issues_json), "--case-id", "case_cli"])

            self.assertEqual(exit_code, 0)
            ledger_path = ledger_root / "case_cli" / "defect_ledger.json"
            self.assertTrue(ledger_path.exists())
            payload = json.loads(ledger_path.read_text("utf-8"))
            self.assertEqual(payload["runtime_config"]["effective_quality_defect_ledger_dir"], str(ledger_root.resolve()))
            self.assertFalse(payload["runtime_config"]["override_used"])

    def test_cli_rejects_sparse_parallel_issue_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = make_run_dir(root)

            with self.assertRaisesRegex(ValueError, "--expected-visible-text count .* must match --issue-text count"):
                main(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--case-id",
                        "case_sparse_cli",
                        "--issue-text",
                        "第一条 QC 问题",
                        "--issue-text",
                        "第二条 QC 问题",
                        "--expected-visible-text",
                        "只给第二条的期望文本",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
