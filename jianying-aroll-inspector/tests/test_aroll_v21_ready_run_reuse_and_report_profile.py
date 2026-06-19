from __future__ import annotations

import json
import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aroll_v21.operator import ArollV21OperatorConfig, run_operator
from tests.test_aroll_v21_sacrificial_write_override import (
    FakeAdapter,
    create_disposable_draft,
    fake_real_draft_result,
    fake_writeback_factory,
)


def read_json(path: Path):
    return json.loads(path.read_text("utf-8"))


class ArollV21ReadyRunReuseAndReportProfileTests(unittest.TestCase):
    def _ready_run(self, root: Path) -> tuple[Path, Path, Path]:
        draft_dir, draft_content, template = create_disposable_draft(root)
        run_dir = root / "ready_run"
        with patch(
            "aroll_v21.operator.RealDraftIngestAdapter",
            lambda *a, **k: FakeAdapter(result=fake_real_draft_result(root=root)),
        ):
            summary = run_operator(
                ArollV21OperatorConfig(
                    mode="dry-run",
                    run_dir=run_dir,
                    draft_dir=draft_dir,
                    jy_draftc=root / "jy-draftc.exe",
                    report_profile="standard",
                )
            )
        self.assertEqual(summary["status"], "ok")
        self.assertTrue(summary["READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT"])
        self.assertTrue((run_dir / "artifact_manifest.json").exists())
        self.assertFalse((run_dir / "source_graph.json").exists())
        self.assertTrue((run_dir / "source_graph.json.gz").exists())
        self.assertFalse((run_dir / "validator_report.json").exists())
        self.assertTrue((run_dir / "validator_report.json.gz").exists())
        return run_dir, draft_content, template

    def test_commit_from_ready_run_dir_skips_replanning_and_deepseek(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_run, draft_content, template = self._ready_run(root)
            before_draft = draft_content.read_text("utf-8")
            before_template = template.read_text("utf-8")

            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: FakeAdapter(result=fake_real_draft_result(root=root)),
            ), patch("aroll_v21.operator.RealDraftWriteback", fake_writeback_factory), patch(
                "aroll_v21.operator.deepseek_provider_from_env",
                side_effect=AssertionError("DeepSeek must not be called when reusing READY run"),
            ), patch("aroll_v21.operator.ArollEngine.run", side_effect=AssertionError("planning must not rerun")):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="write",
                        run_dir=root / "write_run",
                        draft_dir=root / "draft",
                        jy_draftc=root / "jy-draftc.exe",
                        ready_run_dir=ready_run,
                        commit=True,
                        allow_sacrificial_write_without_postwrite_decrypt=True,
                        report_profile="standard",
                    )
                )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["write_status"], "committed_from_ready_run_without_replanning")
            self.assertTrue(summary["commit_performed"])
            self.assertTrue(summary["writeback_success"])
            self.assertNotEqual(draft_content.read_text("utf-8"), before_draft)
            self.assertNotEqual(template.read_text("utf-8"), before_template)
            self.assertFalse((root / "write_run" / "run_report.json").exists())
            self.assertFalse((root / "write_run" / "source_graph.json").exists())
            self.assertTrue((root / "write_run" / "source_graph.json.gz").exists())
            writeback = read_json(root / "write_run" / "writeback_report.json")
            self.assertTrue(writeback["writeback_success"])
            self.assertNotIn("post_write_actual_draft_audit", writeback)
            self.assertIn("compact_report_omitted_debug_payloads", writeback)

    def test_commit_from_ready_run_dir_rejects_changed_draft_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_run, draft_content, _template = self._ready_run(root)
            draft_content.write_text("user changed draft after dry-run", "utf-8")

            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: FakeAdapter(result=fake_real_draft_result(root=root)),
            ), patch("aroll_v21.operator.RealDraftWriteback", fake_writeback_factory):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="write",
                        run_dir=root / "write_run",
                        draft_dir=root / "draft",
                        jy_draftc=root / "jy-draftc.exe",
                        ready_run_dir=ready_run,
                        commit=True,
                        allow_sacrificial_write_without_postwrite_decrypt=True,
                    )
                )

            self.assertEqual(summary["status"], "blocked")
            self.assertIn("READY_RUN_REUSE_REJECTED", summary["blocker_codes"])
            blocker_report = read_json(root / "write_run" / "blocker_report.json")
            self.assertIn("mismatch", blocker_report["blockers"][0]["context"]["reason"])

    def test_commit_from_ready_run_dir_rejects_unready_or_blocked_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_run, _draft_content, _template = self._ready_run(root)
            summary_path = ready_run / "run_summary.json"
            summary_payload = read_json(summary_path)
            summary_payload["READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT"] = False
            summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), "utf-8")

            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: FakeAdapter(result=fake_real_draft_result(root=root)),
            ):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="write",
                        run_dir=root / "write_run",
                        draft_dir=root / "draft",
                        ready_run_dir=ready_run,
                        commit=True,
                        allow_sacrificial_write_without_postwrite_decrypt=True,
                    )
                )

            self.assertEqual(summary["status"], "blocked")
            self.assertIn("READY_RUN_REUSE_REJECTED", summary["blocker_codes"])

    def test_report_profile_minimal_writes_only_manifest_summary_blocker_and_writeback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fake_real_draft_result()
            input_json = root / "input.json"
            input_json.write_text(
                json.dumps(
                    {
                        "source_segments": result.source_segments,
                        "source_materials": result.source_materials,
                        "word_timeline": result.word_timeline,
                        "subtitles": result.subtitles,
                        "text_materials": result.text_materials,
                        "text_segments": result.text_segments,
                    },
                    ensure_ascii=False,
                ),
                "utf-8",
            )
            run_dir = root / "run"
            summary = run_operator(
                ArollV21OperatorConfig(
                    mode="dry-run",
                    run_dir=run_dir,
                    input_json=input_json,
                    report_profile="minimal",
                )
            )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(
                sorted(path.name for path in run_dir.glob("*.json")),
                ["artifact_manifest.json", "blocker_report.json", "run_summary.json", "writeback_report.json"],
            )

    def test_report_profile_debug_writes_full_debug_artifacts_to_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fake_real_draft_result()
            input_json = root / "input.json"
            input_json.write_text(
                json.dumps(
                    {
                        "source_segments": result.source_segments,
                        "source_materials": result.source_materials,
                        "word_timeline": result.word_timeline,
                        "subtitles": result.subtitles,
                        "text_materials": result.text_materials,
                        "text_segments": result.text_segments,
                    },
                    ensure_ascii=False,
                ),
                "utf-8",
            )
            run_dir = root / "run"
            summary = run_operator(
                ArollV21OperatorConfig(
                    mode="dry-run",
                    run_dir=run_dir,
                    input_json=input_json,
                    report_profile="debug",
                )
            )

            self.assertEqual(summary["status"], "ok")
            self.assertTrue((run_dir / "run_report.json").exists())
            self.assertTrue((run_dir / "source_graph.json").exists())
            self.assertTrue((run_dir / "deepseek_batch_request.json").exists())
            self.assertTrue((run_dir / "artifact_manifest.json").exists())

    def test_report_profile_standard_writes_large_debug_artifacts_compressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fake_real_draft_result()
            input_json = root / "input.json"
            input_json.write_text(
                json.dumps(
                    {
                        "source_segments": result.source_segments,
                        "source_materials": result.source_materials,
                        "word_timeline": result.word_timeline,
                        "subtitles": result.subtitles,
                        "text_materials": result.text_materials,
                        "text_segments": result.text_segments,
                    },
                    ensure_ascii=False,
                ),
                "utf-8",
            )
            run_dir = root / "run"
            summary = run_operator(
                ArollV21OperatorConfig(
                    mode="dry-run",
                    run_dir=run_dir,
                    input_json=input_json,
                    report_profile="standard",
                )
            )

            self.assertEqual(summary["status"], "ok")
            self.assertFalse((run_dir / "source_graph.json").exists())
            self.assertTrue((run_dir / "source_graph.json.gz").exists())
            with gzip.open(run_dir / "source_graph.json.gz", "rt", encoding="utf-8") as f:
                source_graph = json.load(f)
            self.assertIn("words", source_graph)
            manifest = read_json(run_dir / "artifact_manifest.json")
            self.assertIn("source_graph.json.gz", manifest["artifact_files"])

    def test_run_summary_contains_stage_timing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fake_real_draft_result()
            input_json = root / "input.json"
            input_json.write_text(
                json.dumps(
                    {
                        "source_segments": result.source_segments,
                        "source_materials": result.source_materials,
                        "word_timeline": result.word_timeline,
                        "subtitles": result.subtitles,
                        "text_materials": result.text_materials,
                        "text_segments": result.text_segments,
                    },
                    ensure_ascii=False,
                ),
                "utf-8",
            )
            summary = run_operator(ArollV21OperatorConfig(mode="dry-run", run_dir=root / "run", input_json=input_json))

            for key in (
                "total_seconds",
                "dry_run_seconds",
                "planning_seconds",
                "semantic_adjudication_seconds",
                "quality_gate_seconds",
                "report_write_seconds",
                "writeback_seconds",
                "postwrite_core_audit_seconds",
                "postwrite_debug_audit_seconds",
            ):
                self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
