from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import jy_bridge
from aroll_v21.writeback import RealDraftWriteback
from tests.test_aroll_v21_real_writeback_backend import run_report_from_result
from tests.test_aroll_v21_sacrificial_write_override import (
    create_disposable_draft,
    fake_encrypt,
    fake_integrity_ok,
    fake_real_draft_result,
    fake_real_writeback,
)


def root_mirror_required(_draft_dir: Path, _jy_draftc: Path, _run_dir: Path, _timeline_id: str) -> bool:
    return True


def root_mirror_raises(_draft_dir: Path, _jy_draftc: Path, _run_dir: Path, _timeline_id: str) -> bool:
    raise RuntimeError("mirror check unavailable")


def writeback_with_default_root_mirror() -> RealDraftWriteback:
    return RealDraftWriteback(
        encrypt_func=fake_encrypt,
        timeline_content_check_func=fake_integrity_ok,
        layout_check_func=fake_integrity_ok,
        project_folder_check_func=fake_integrity_ok,
    )


class ArollV21WritebackRootMirrorTargetTests(unittest.TestCase):
    def test_root_mirror_bridge_symbol_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            (draft_dir / "draft_content.json").write_bytes(draft_content.read_bytes())
            (draft_dir / "template-2.tmp").write_bytes(template.read_bytes())

            self.assertTrue(callable(jy_bridge.root_mirrors_timeline_id))
            self.assertTrue(jy_bridge.root_mirrors_timeline_id(draft_dir, Path("draftc"), root / "run", "timeline_001"))

    def test_writeback_does_not_raise_jy_bridge_symbol_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = writeback_with_default_root_mirror().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            serialized = json.dumps(writeback_result.report, ensure_ascii=False)
            self.assertNotIn("JY_BRIDGE_SYMBOL_NOT_FOUND", serialized)

    def test_root_mirror_required_writes_root_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            root_draft = draft_dir / "draft_content.json"
            root_template = draft_dir / "template-2.tmp"
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = fake_real_writeback(root_mirror_func=root_mirror_required).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            self.assertTrue(writeback_result.report["root_mirror_required"])
            self.assertTrue(writeback_result.report["root_mirror_written"])
            for target in (draft_content, template, root_draft, root_template):
                self.assertTrue(writeback_result.report["target_writes"][str(target)])
                self.assertTrue(target.exists())

    def test_root_mirror_check_failure_blocks_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = fake_real_writeback(root_mirror_func=root_mirror_raises).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertFalse(writeback_result.success)
            self.assertEqual(writeback_result.blockers[0].code, "V21_WRITEBACK_ROOT_MIRROR_DETECTION_FAILED")
            self.assertTrue(writeback_result.report["root_mirror_check_failed"])
            self.assertIsNone(writeback_result.report["root_mirror_required"])
            self.assertEqual(writeback_result.report["target_writes"], {})
            self.assertNotIn(str(draft_dir / "draft_content.json"), writeback_result.report["target_writes"])

    def test_root_mirror_detection_report_written_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)
            run_dir = root / "run"

            writeback_result = fake_real_writeback(root_mirror_func=root_mirror_raises).commit(
                draft_dir=draft_dir,
                run_dir=run_dir,
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            report_path = run_dir / "root_mirror_detection_report.json"
            payload = json.loads(report_path.read_text("utf-8"))
            self.assertFalse(writeback_result.success)
            self.assertTrue(report_path.exists())
            self.assertEqual(payload["root_mirror_error"], "mirror check unavailable")
            self.assertEqual(payload["expected_root_path"], str(draft_dir / "draft_content.json"))
            self.assertEqual(payload["timeline_path"], str(draft_dir / "Timelines" / "timeline_001" / "draft_content.json"))
            self.assertEqual(payload["mirror_path"], str(draft_dir / "template-2.tmp"))
            self.assertFalse(payload["root_mirror_match"])
            self.assertEqual(payload["reason"], "root mirror detection function failed")

    def test_writeback_blocks_when_root_mirror_cannot_be_determined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            (draft_dir / "draft_content.json").write_bytes(draft_content.read_bytes())
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)
            run_dir = root / "run"

            writeback_result = writeback_with_default_root_mirror().commit(
                draft_dir=draft_dir,
                run_dir=run_dir,
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            payload = json.loads((run_dir / "root_mirror_detection_report.json").read_text("utf-8"))
            self.assertFalse(writeback_result.success)
            self.assertEqual(writeback_result.blockers[0].code, "V21_WRITEBACK_ROOT_MIRROR_DETECTION_FAILED")
            self.assertEqual(writeback_result.report["root_mirror_error"], "ROOT_MIRROR_PARTIAL_FILES_PRESENT")
            self.assertEqual(payload["root_mirror_error"], "ROOT_MIRROR_PARTIAL_FILES_PRESENT")
            self.assertTrue(payload["root_exists"])
            self.assertTrue(payload["timeline_exists"])
            self.assertFalse(payload["mirror_exists"])
            self.assertFalse(payload["root_mirror_match"])
            self.assertNotIn("JY_BRIDGE_SYMBOL_NOT_FOUND", json.dumps(writeback_result.report, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
