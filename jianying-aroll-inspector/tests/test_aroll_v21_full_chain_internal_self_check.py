from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.test_aroll_v21_real_writeback_backend import run_report_from_result
from aroll_v21.operator import ArollV21OperatorConfig, run_operator

from tests.test_aroll_v21_sacrificial_write_override import (
    create_disposable_draft,
    fake_real_draft_result,
    fake_real_writeback,
)


def root_mirror_required(_draft_dir: Path, _jy_draftc: Path, _run_dir: Path, _timeline_id: str) -> bool:
    return True


class ExternalWordTimelineAdapter:
    def __init__(self, *args, result, **kwargs) -> None:
        self.result = result

    def load(self, _draft_dir: Path, _run_dir: Path, *, word_timeline_json: Path | None = None):
        if word_timeline_json is None:
            return self.result
        words = json.loads(Path(word_timeline_json).read_text("utf-8"))
        return replace(self.result, word_timeline=words)


def writeback_factory(*args, **kwargs):
    return fake_real_writeback(jy_draftc=kwargs.get("jy_draftc"), root_mirror_func=root_mirror_required)


def add_non_aroll_tracks(result) -> None:
    title_material = {"id": "title_material", "type": "text", "text": "标题"}
    title_segment = {
        "id": "title_segment",
        "type": "text",
        "material_id": "title_material",
        "track_id": "title_track",
        "track_type": "text",
        "target_timerange": {"start": 0, "duration": 1000000},
    }
    callout_material = {"id": "callout_material", "type": "text", "text": "贴纸文字"}
    callout_segment = {
        "id": "callout_segment",
        "type": "text",
        "material_id": "callout_material",
        "track_id": "callout_track",
        "track_type": "text",
        "target_timerange": {"start": 500000, "duration": 500000},
    }
    broll_segment = {
        "id": "broll_segment",
        "type": "video",
        "material_id": "broll_material",
        "source_timerange": {"start": 0, "duration": 1000000},
        "target_timerange": {"start": 0, "duration": 1000000},
    }
    result.draft_data["materials"]["texts"].insert(0, title_material)
    result.draft_data["materials"]["texts"].append(callout_material)
    result.draft_data["tracks"].insert(0, {"id": "broll_track", "type": "video", "segments": [broll_segment]})
    result.draft_data["tracks"].insert(1, {"id": "title_track", "type": "text", "segments": [title_segment]})
    result.draft_data["tracks"].append({"id": "callout_track", "type": "text", "segments": [callout_segment]})
    result.draft_data["tracks"].append({"id": "audio_track", "type": "audio", "segments": [{"id": "audio_segment"}]})
    result.draft_data["tracks"].append(
        {
            "id": "filter_track",
            "type": "filter",
            "segments": [{"id": "filter_segment", "target_timerange": {"start": 0, "duration": 1000000}}],
        }
    )
    result.text_segments.extend([title_segment, callout_segment])


class ArollV21FullChainInternalSelfCheckTests(unittest.TestCase):
    def test_full_chain_operator_to_real_writeback_temp_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            add_non_aroll_tracks(result)
            word_timeline_json = root / "external_word_timeline.json"
            word_timeline_json.write_text(json.dumps(result.word_timeline, ensure_ascii=False), "utf-8")
            semantic_decisions_json = root / "semantic_decisions.json"
            semantic_decisions_json.write_text("[]", "utf-8")
            old_draft_content = draft_content.read_text("utf-8")
            old_template = template.read_text("utf-8")

            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: ExternalWordTimelineAdapter(result=result),
            ), patch("aroll_v21.operator.RealDraftWriteback", writeback_factory):
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
                        report_profile="debug",
                    )
                )

            run_dir = root / "run"
            for artifact in (
                "source_graph.json",
                "edit_units.json",
                "repeat_clusters.json",
                "decision_plan.json",
                "semantic_request_payloads.json",
                "final_timeline.json",
                "captions.json",
                "material_write_plan.json",
                "validator_report.json",
                "writeback_report.json",
                "run_summary.json",
            ):
                self.assertTrue((run_dir / artifact).exists(), artifact)

            self.assertEqual(summary["status"], "ok")
            self.assertTrue(summary["commit_performed"])
            self.assertTrue(summary["writeback_success"])
            self.assertTrue(summary["WRITE_SUCCESS"])
            self.assertTrue(summary["ENCRYPT_SUCCESS"])
            self.assertTrue(summary["validator_write_allowed"])
            self.assertTrue(summary["write_allowed"])
            self.assertEqual(summary["semantic_unresolved_count"], 0)

            final_timeline = json.loads((run_dir / "final_timeline.json").read_text("utf-8"))
            captions = json.loads((run_dir / "captions.json").read_text("utf-8"))
            material_write_plan = json.loads((run_dir / "material_write_plan.json").read_text("utf-8"))
            validator_report = json.loads((run_dir / "validator_report.json").read_text("utf-8"))
            writeback_report = json.loads((run_dir / "writeback_report.json").read_text("utf-8"))
            self.assertGreater(len(final_timeline), 0)
            self.assertEqual(len(final_timeline), len(captions))
            self.assertEqual(len(captions), len(material_write_plan["segments"]))
            self.assertEqual(len(captions), len(material_write_plan["materials"]))
            self.assertTrue(validator_report["validator_report_ok"])
            self.assertEqual(writeback_report["selected_text_track_id"], "text_track")
            self.assertEqual(writeback_report["selected_video_track_id"], "video_track")
            self.assertTrue(writeback_report["root_mirror_required"])
            self.assertTrue(writeback_report["root_mirror_written"])
            self.assertTrue(writeback_report["non_subtitle_text_tracks_preserved"])
            self.assertTrue(writeback_report["timeline_integrity_checks"]["timeline_content_id_ok"])
            self.assertEqual(writeback_report["source_mapping_mode"], "dynamic_source_binding")
            self.assertTrue(writeback_report["video_preflight"]["main_video_speed_safe"])
            self.assertTrue(writeback_report["audio_preflight"]["independent_audio_track_detected"])
            self.assertTrue(writeback_report["filter_preflight"]["filter_track_detected"])

            self.assertNotEqual(draft_content.read_text("utf-8"), old_draft_content)
            self.assertNotEqual(template.read_text("utf-8"), old_template)
            self.assertTrue((draft_dir / "draft_content.json").exists())
            self.assertTrue((draft_dir / "template-2.tmp").exists())
            written = json.loads(draft_content.read_text("utf-8"))
            text_track = next(track for track in written["tracks"] if track["id"] == "text_track")
            title_track = next(track for track in written["tracks"] if track["id"] == "title_track")
            callout_track = next(track for track in written["tracks"] if track["id"] == "callout_track")
            video_track = next(track for track in written["tracks"] if track["id"] == "video_track")
            broll_track = next(track for track in written["tracks"] if track["id"] == "broll_track")
            self.assertTrue(all(segment["id"].startswith("v21_caption_segment_") for segment in text_track["segments"]))
            self.assertEqual(title_track["segments"][0]["id"], "title_segment")
            self.assertEqual(callout_track["segments"][0]["id"], "callout_segment")
            self.assertTrue(all(segment["id"].startswith("v21_video_segment_") for segment in video_track["segments"]))
            self.assertEqual(broll_track["segments"][0]["id"], "broll_segment")
            material_ids = {row["id"] for row in written["materials"]["texts"]}
            self.assertIn("title_material", material_ids)
            self.assertIn("callout_material", material_ids)

    def test_actual_postwrite_commit_path_calls_writeback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            add_non_aroll_tracks(result)
            precomputed = run_report_from_result(result)
            postwrite_materials_json = root / "postwrite_materials.json"
            postwrite_materials_json.write_text(
                json.dumps(precomputed.material_write_plan["materials"], ensure_ascii=False),
                "utf-8",
            )
            old_draft_content = draft_content.read_text("utf-8")
            old_template = template.read_text("utf-8")

            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: ExternalWordTimelineAdapter(result=result),
            ), patch("aroll_v21.operator.RealDraftWriteback", writeback_factory):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="write",
                        run_dir=root / "run",
                        draft_dir=draft_dir,
                        jy_draftc=root / "jy-draftc.exe",
                        postwrite_materials_json=postwrite_materials_json,
                        commit=True,
                    )
                )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["write_status"], "committed_after_postwrite_verification")
            self.assertTrue(summary["postwrite_decrypt_ok"])
            self.assertTrue(summary["commit_performed"])
            self.assertTrue(summary["writeback_success"])
            self.assertTrue(summary["commit_only_after_all_validators"])
            self.assertNotEqual(draft_content.read_text("utf-8"), old_draft_content)
            self.assertNotEqual(template.read_text("utf-8"), old_template)
            writeback_report = json.loads((root / "run" / "writeback_report.json").read_text("utf-8"))
            self.assertTrue(writeback_report["writeback_success"])
            self.assertEqual(writeback_report["selected_text_track_id"], "text_track")
            self.assertEqual(writeback_report["selected_video_track_id"], "video_track")

    def test_verify_only_real_draft_path_uses_postwrite_materials_json_without_writeback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            precomputed = run_report_from_result(result)
            postwrite_materials_json = root / "postwrite_materials.json"
            postwrite_materials_json.write_text(
                json.dumps(precomputed.material_write_plan["materials"], ensure_ascii=False),
                "utf-8",
            )
            old_draft_content = draft_content.read_text("utf-8")
            old_template = template.read_text("utf-8")

            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: ExternalWordTimelineAdapter(result=result),
            ):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="verify-only",
                        run_dir=root / "run",
                        draft_dir=draft_dir,
                        jy_draftc=root / "jy-draftc.exe",
                        postwrite_materials_json=postwrite_materials_json,
                    )
                )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["write_status"], "verify_only_passed")
            self.assertTrue(summary["postwrite_decrypt_ok"])
            self.assertFalse(summary["commit_performed"])
            self.assertFalse(summary["writeback_success"])
            self.assertEqual(draft_content.read_text("utf-8"), old_draft_content)
            self.assertEqual(template.read_text("utf-8"), old_template)
            writeback_report = json.loads((root / "run" / "writeback_report.json").read_text("utf-8"))
            self.assertFalse(writeback_report["writeback_attempted"])

    def test_invalid_postwrite_materials_json_blocks_at_operator_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            invalid_postwrite = root / "invalid_postwrite_materials.json"
            invalid_postwrite.write_text(json.dumps({"not": "a list"}, ensure_ascii=False), "utf-8")
            old_draft_content = draft_content.read_text("utf-8")
            old_template = template.read_text("utf-8")

            summary = run_operator(
                ArollV21OperatorConfig(
                    mode="verify-only",
                    run_dir=root / "run",
                    draft_dir=draft_dir,
                    jy_draftc=root / "jy-draftc.exe",
                    postwrite_materials_json=invalid_postwrite,
                )
            )

            self.assertEqual(summary["status"], "blocked")
            self.assertIn("POSTWRITE_MATERIALS_JSON_INVALID", summary["blocker_codes"])
            self.assertFalse(summary["commit_performed"])
            self.assertEqual(draft_content.read_text("utf-8"), old_draft_content)
            self.assertEqual(template.read_text("utf-8"), old_template)


if __name__ == "__main__":
    unittest.main()
