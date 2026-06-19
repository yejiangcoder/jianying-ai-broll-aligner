from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aroll_runtime_paths


ROOT = Path(__file__).resolve().parents[1]


class RuntimeConfigPathTests(unittest.TestCase):
    def test_runtime_paths_config_controls_aligner_and_deepseek_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligner_root = root / "aligner"
            deepseek_config = root / "config" / "deepseek.yaml"
            local_config = root / "runtime_paths.local.yaml"
            local_config.write_text(
                "\n".join(
                    [
                        "runtime_root: " + repr(str(root / "runtime")),
                        "jianying:",
                        "  aligner_root: " + repr(str(aligner_root)),
                        "deepseek:",
                        "  config_path: " + repr(str(deepseek_config)),
                    ]
                ),
                "utf-8",
            )

            with patch.dict("os.environ", {}, clear=True), patch.object(
                aroll_runtime_paths,
                "LOCAL_CONFIG",
                local_config,
            ), patch.object(aroll_runtime_paths, "EXAMPLE_CONFIG", root / "missing.yaml"):
                self.assertEqual(aroll_runtime_paths.get_aligner_root(), aligner_root)
                self.assertEqual(aroll_runtime_paths.get_deepseek_config_path(), deepseek_config)

    def test_source_and_tools_do_not_hardcode_windows_drive_paths(self) -> None:
        drive_path = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
        hits = []
        for scan_root in (ROOT / "src", ROOT / "tools"):
            for path in scan_root.rglob("*.py"):
                text = path.read_text("utf-8")
                if drive_path.search(text):
                    hits.append(str(path.relative_to(ROOT)))

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
