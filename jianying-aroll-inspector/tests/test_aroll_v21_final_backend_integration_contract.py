from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.operator import ArollV21OperatorConfig, run_operator
from tests.test_aroll_v21_full_chain_internal_self_check import ExternalWordTimelineAdapter
from tests.test_aroll_v21_sacrificial_write_override import create_disposable_draft, fake_real_draft_result, fake_real_writeback


def semantic_modifier_result(root: Path):
    result = fake_real_draft_result(root=root)
    text = "甲的乙的项"
    words = []
    word_ids = []
    cursor = 100_000
    for index, char in enumerate(text, start=1):
        word_id = f"w{index:03d}"
        word_ids.append(word_id)
        words.append(
            {
                "word_id": word_id,
                "word_text": char,
                "start_us": cursor,
                "end_us": cursor + 120_000,
                "source_start_us": cursor,
                "source_end_us": cursor + 120_000,
                "source_material_id": "main_video_a",
                "source_segment_id": "clip",
                "subtitle_uid": "s001",
                "subtitle_index": 1,
            }
        )
        cursor += 120_000
    subtitles = [{"subtitle_uid": "s001", "subtitle_index": 1, "text": text, "word_ids": word_ids, "text_material_id": "caption_template_001"}]
    return replace(result, word_timeline=words, subtitles=subtitles)


def add_selected_track_callout(result) -> None:
    callout_material = {"id": "callout_material", "role": "callout", "type": "text", "text": "callout"}
    callout_segment = {
        "id": "callout_segment",
        "type": "callout",
        "material_id": "callout_material",
        "target_timerange": {"start": 500_000, "duration": 500_000},
    }
    result.draft_data["materials"]["texts"].append(callout_material)
    text_track = next(track for track in result.draft_data["tracks"] if track["id"] == "text_track")
    text_track["segments"].append(callout_segment)
    result.text_segments.append(callout_segment | {"track_id": "text_track", "track_type": "text"})
    result.text_materials.append(callout_material)


def read_json(path: Path):
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    gzip_path = path.with_suffix(path.suffix + ".gz")
    if gzip_path.exists():
        with gzip.open(gzip_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text("utf-8"))


class ArollV21FinalBackendIntegrationContractTests(unittest.TestCase):
    def test_external_words_to_semantic_decision_to_fake_writeback_reports_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = semantic_modifier_result(root)
            add_selected_track_callout(result)

            discovery = ArollEngine().run(
                ArollRunInput(
                    mode="dry-run",
                    draft_data=result.draft_data,
                    word_timeline=result.word_timeline,
                    subtitles=result.subtitles,
                    source_segments=result.source_segments,
                    source_materials=result.source_materials,
                    text_materials=result.text_materials,
                    text_segments=result.text_segments,
                    postwrite_mode="simulated",
                )
            )
            self.assertEqual(discovery.decision_plan.semantic_request_payloads, [])
            self.assertTrue(discovery.decision_plan.split_decisions)
            self.assertEqual([caption.text for caption in discovery.captions], ["乙的项"])

            word_timeline_json = root / "external_words.json"
            word_timeline_json.write_text(json.dumps(result.word_timeline, ensure_ascii=False), "utf-8")
            semantic_decisions_json = root / "semantic_decisions.json"
            semantic_decisions_json.write_text(
                json.dumps(
                    [
                        {
                            "cluster_id": "repeat_002000",
                            "decision": "drop_redundant_modifier",
                            "reason": "drop redundant modifier in rough cut",
                            "confidence": 0.9,
                            "requires_human_review": False,
                        }
                    ],
                    ensure_ascii=False,
                ),
                "utf-8",
            )

            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: ExternalWordTimelineAdapter(result=result),
            ), patch("aroll_v21.operator.RealDraftWriteback", lambda *a, **k: fake_real_writeback(jy_draftc=k.get("jy_draftc"))):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="write",
                        run_dir=root / "run",
                        draft_dir=draft_dir,
                        jy_draftc=root / "jy-draftc.exe",
                        word_timeline_json=word_timeline_json,
                        semantic_decisions_json=semantic_decisions_json,
                        commit=True,
                        allow_sacrificial_write_without_postwrite_decrypt=True,
                    )
                )

            run_dir = root / "run"
            validator_report = read_json(run_dir / "validator_report.json")
            blocker_report = read_json(run_dir / "blocker_report.json")
            final_timeline = read_json(run_dir / "final_timeline.json")
            captions = read_json(run_dir / "captions.json")
            material_write_plan = read_json(run_dir / "material_write_plan.json")
            writeback_report = read_json(run_dir / "writeback_report.json")
            semantic_decisions = read_json(run_dir / "semantic_decisions.json")
            rough = validator_report["rough_cut_quality_validator"]
            writeback_rough = writeback_report["rough_cut_quality"]

            self.assertTrue(summary["READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT"])
            self.assertTrue(summary["write_allowed"])
            self.assertTrue(validator_report["validator_report_ok"])
            self.assertEqual(summary["semantic_unresolved_count"], 0)
            self.assertEqual(summary["writer_fallback_count"], 0)
            self.assertEqual(summary["fatal_blocker"], None)
            self.assertEqual(blocker_report["blockers"], [])
            for code in (
                "FINAL_REPEAT_VALIDATOR_FAILED",
                "HIDDEN_AUDIO_REPEAT_VALIDATOR_FAILED",
                "ROUGH_CUT_QUALITY_VALIDATOR_FAILED",
                "INTERNAL_SEMANTIC_REQUEST_MISSING_FOR_FATAL_REPEAT",
                "ROUGH_CUT_RESIDUAL_MICRO_SEGMENT_UNMERGEABLE",
            ):
                self.assertNotIn(code, summary["blocker_codes"])

            self.assertEqual(rough["segments_lt_300ms"], 0)
            self.assertEqual(rough["one_char_captions"], 0)
            self.assertEqual(rough["adjacent_duplicate_text_count"], 0)
            self.assertEqual(len(final_timeline), len(captions))
            self.assertEqual(len(captions), len(material_write_plan["materials"]))
            self.assertEqual(len(captions), len(material_write_plan["segments"]))
            self.assertEqual(semantic_decisions[0]["decision"], "drop_redundant_modifier")

            self.assertEqual(writeback_report["visible_caption_track_count"], 1)
            self.assertEqual(writeback_report["old_subtitle_residue_track_count"], 0)
            self.assertEqual(writeback_report["overlapping_caption_segments_count"], 0)
            self.assertEqual(writeback_report["canonical_caption_segment_count"], len(captions))
            self.assertEqual(writeback_rough["canonical_caption_segment_count"], len(captions))
            self.assertTrue(writeback_report["non_subtitle_text_segments_preserved"])
            self.assertTrue(writeback_report["non_subtitle_text_materials_preserved"])


if __name__ == "__main__":
    unittest.main()
