from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aroll_v21.operator import ArollV21OperatorConfig, REQUIRED_ARTIFACTS, run_operator


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str):
    return json.loads((ROOT / path).read_text("utf-8"))


def write_input(path: Path) -> None:
    material = load_json("fixtures/real_materials/normal_caption_template.json")
    payload = load_json("fixtures/real_timelines/multi_clip_gap.json")
    payload["text_materials"] = [material["material"]]
    payload["text_segments"] = [material["segment"]]
    path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")


class ArollV21RunArtifactTests(unittest.TestCase):
    def test_run_dir_contains_complete_v21_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            run_dir = root / "run"
            write_input(input_json)
            summary = run_operator(ArollV21OperatorConfig(mode="dry-run", run_dir=run_dir, input_json=input_json, report_profile="debug"))
            self.assertEqual(summary["status"], "ok")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((run_dir / artifact).exists(), artifact)
            run_summary = json.loads((run_dir / "run_summary.json").read_text("utf-8"))
            for key in (
                "single_source_graph_ok",
                "all_final_segments_have_word_ids",
                "all_captions_derived_from_final_timeline",
                "all_materials_from_canonical_template",
                "no_writer_fallback",
                "validators_readonly",
                "final_repeat_count",
                "hidden_audio_repeat_count",
                "cut_inside_word_count",
                "partial_multichar_cut_count",
                "giant_subtitle_count",
                "template_fingerprint_mismatch_count",
                "content_schema_error_count",
                "caption_coverage_gap_count",
                "prewrite_style_gate_ok",
                "postwrite_style_gate_ok",
                "postwrite_decrypt_ok",
                "commit_only_after_all_validators",
            ):
                self.assertIn(key, run_summary)


if __name__ == "__main__":
    unittest.main()
