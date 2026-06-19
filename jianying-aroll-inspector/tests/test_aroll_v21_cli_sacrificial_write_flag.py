from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from aroll_v21.cli import parse_args


class ArollV21CliSacrificialWriteFlagTests(unittest.TestCase):
    def test_python_cli_accepts_sacrificial_write_flag(self) -> None:
        with patch(
            "sys.argv",
            [
                "aroll_v21.cli",
                "--draft-dir",
                "sample_draft",
                "--output-dir",
                "sample_run",
                "--mode",
                "write",
                "--semantic-mode",
                "fail-closed",
                "--commit",
                "--allow-sacrificial-write-without-postwrite-decrypt",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.draft_dir, Path("sample_draft"))
        self.assertEqual(args.run_dir, Path("sample_run"))
        self.assertEqual(args.mode, "write")
        self.assertEqual(args.semantic_mode, "fail-closed")
        self.assertTrue(args.commit)
        self.assertTrue(args.allow_sacrificial_write_without_postwrite_decrypt)

    def test_powershell_entry_forwards_sacrificial_write_and_semantic_mode(self) -> None:
        script = Path("run_aroll_v21_operator.ps1").read_text("utf-8")
        self.assertIn("SemanticMode", script)
        self.assertIn("--semantic-mode", script)
        self.assertIn("AllowSacrificialWriteWithoutPostwriteDecrypt", script)
        self.assertIn("--allow-sacrificial-write-without-postwrite-decrypt", script)

    def test_uat_fresh_draft_exposes_semantic_mode_without_hardcoded_baseline(self) -> None:
        script = Path("scripts/uat_fresh_draft.ps1").read_text("utf-8")
        self.assertIn("SemanticMode", script)
        self.assertIn("--semantic-mode", script)
        self.assertIn('"auto"', script)
        self.assertIn('"semantic-requests-only"', script)
        self.assertIn('[string]$SemanticMode = "auto"', script)
        self.assertNotIn('"--semantic-mode", "deterministic-baseline"', script)

    def test_semantic_mode_auto_is_default_for_cli(self) -> None:
        with patch(
            "sys.argv",
            [
                "aroll_v21.cli",
                "--input-json",
                "input.json",
                "--output-dir",
                "sample_run",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.semantic_mode, "auto")

    def test_semantic_mode_auto_is_default_for_uat(self) -> None:
        script = Path("scripts/uat_fresh_draft.ps1").read_text("utf-8")

        self.assertIn('[string]$SemanticMode = "auto"', script)


if __name__ == "__main__":
    unittest.main()
