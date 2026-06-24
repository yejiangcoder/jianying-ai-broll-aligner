from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "tools" / "jianying_draft_rollback_analyzer.py"


spec = importlib.util.spec_from_file_location("jianying_draft_rollback_analyzer", ANALYZER_PATH)
assert spec is not None and spec.loader is not None
rollback_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollback_analyzer)


def _draft_payload(media_path: str) -> dict:
    return {
        "tracks": [{"type": "video", "segments": [{"id": "seg_1"}]}],
        "materials": {
            "videos": [
                {
                    "id": "video_1",
                    "material_name": Path(media_path).name,
                    "path": media_path,
                    "duration": 1_000_000,
                }
            ]
        },
    }


class JianyingDraftRollbackAnalyzerTests(unittest.TestCase):
    def test_expected_source_media_root_keeps_matching_candidate_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.bak"
            candidate.write_text("encrypted", "utf-8")

            def fake_decrypt(_jy_draftc: Path, _path: Path, output_path: Path) -> None:
                output_path.write_text(
                    json.dumps(_draft_payload("D:/media/show_a/Output/1-1.mp4"), ensure_ascii=False),
                    "utf-8",
                )

            with patch.object(rollback_analyzer, "decrypt", fake_decrypt):
                record = rollback_analyzer.decrypt_candidate(
                    root / "jy-draftc.exe",
                    candidate,
                    root / "out",
                    keep_decrypted=False,
                    expected_source_media_root="D:\\media\\show_a\\Output",
                )

        self.assertTrue(record["source_media_root_ok"])
        self.assertTrue(record["clean_like"])
        self.assertEqual(record["source_media_root_mismatches"], [])

    def test_expected_source_media_root_rejects_wrong_candidate_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.bak"
            candidate.write_text("encrypted", "utf-8")

            def fake_decrypt(_jy_draftc: Path, _path: Path, output_path: Path) -> None:
                output_path.write_text(
                    json.dumps(_draft_payload("D:/media/wrong_show/Output/1-1.mp4"), ensure_ascii=False),
                    "utf-8",
                )

            with patch.object(rollback_analyzer, "decrypt", fake_decrypt):
                record = rollback_analyzer.decrypt_candidate(
                    root / "jy-draftc.exe",
                    candidate,
                    root / "out",
                    keep_decrypted=False,
                    expected_source_media_root="D:\\media\\show_a\\Output",
                )

        self.assertFalse(record["source_media_root_ok"])
        self.assertFalse(record["clean_like"])
        self.assertEqual(record["source_media_root_mismatches"], ["D:/media/wrong_show/Output/1-1.mp4"])

    def test_rollback_apply_blocks_medium_confidence_without_manual_override(self) -> None:
        script = (ROOT / "scripts" / "rollback_jianying_draft.ps1").read_text("utf-8")
        self.assertIn('$report.selection.confidence -eq "medium"', script)
        self.assertIn("-not $userProvidedBaselinePath", script)
        self.assertIn("ExpectedSourceMediaRoot", script)
        self.assertIn("-not $expectedSourceMediaRootResolved -and -not $Force", script)
        self.assertIn("Rollback Apply requires -ExpectedSourceMediaRoot", script)

    def test_rollback_run_dir_uses_atomic_unique_directory_allocation(self) -> None:
        script = (ROOT / "scripts" / "rollback_jianying_draft.ps1").read_text("utf-8")
        self.assertIn("function New-UniqueRollbackRunDir", script)
        self.assertIn('ToString("yyyyMMdd_HHmmss_fff")', script)
        self.assertIn("$PID", script)
        self.assertIn("[Guid]::NewGuid()", script)
        self.assertIn("New-Item -ItemType Directory -Path $candidate -ErrorAction Stop", script)
        self.assertNotIn('Get-Date -Format "yyyyMMdd_HHmmss"', script)
        self.assertNotIn('New-Item -ItemType Directory -Force -Path $runDir', script)


if __name__ == "__main__":
    unittest.main()
