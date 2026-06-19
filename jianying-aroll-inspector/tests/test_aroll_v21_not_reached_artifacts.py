from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aroll_v21.operator import ArollV21OperatorConfig, run_operator


ROOT = Path(__file__).resolve().parents[1]


def _write_ingest_blocking_input(path: Path) -> None:
    material = json.loads((ROOT / "fixtures" / "real_materials" / "normal_caption_template.json").read_text("utf-8"))
    payload = {
        "source_segments": [{"id": "clip_1", "material_id": "main_video", "source_start_us": 0, "source_end_us": 1000000}],
        "word_timeline": [],
        "subtitles": [{"subtitle_uid": "s1", "subtitle_index": 1, "text": "整句字幕", "start_us": 0, "end_us": 1000000}],
        "text_materials": [material["material"]],
        "text_segments": [material["segment"]],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")


class ArollV21NotReachedArtifactsTests(unittest.TestCase):
    def test_ingest_block_writes_not_reached_downstream_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            run_dir = root / "run"
            _write_ingest_blocking_input(input_json)

            summary = run_operator(ArollV21OperatorConfig(mode="dry-run", run_dir=run_dir, input_json=input_json))

            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(summary["requested_report_profile"], "standard")
            self.assertEqual(summary["effective_report_profile"], "debug")
            final_timeline = json.loads((run_dir / "final_timeline.json").read_text("utf-8"))
            captions = json.loads((run_dir / "captions.json").read_text("utf-8"))
            validator = json.loads((run_dir / "validator_report.json").read_text("utf-8"))
            manifest = json.loads((run_dir / "artifact_manifest.json").read_text("utf-8"))

            self.assertEqual(final_timeline["status"], "not_reached")
            self.assertEqual(final_timeline["blocked_by_stage"], "ingest")
            self.assertEqual(captions["status"], "not_reached")
            self.assertEqual(validator["status"], "not_reached")
            self.assertEqual(manifest["requested_report_profile"], "standard")
            self.assertEqual(manifest["effective_report_profile"], "debug")


if __name__ == "__main__":
    unittest.main()
