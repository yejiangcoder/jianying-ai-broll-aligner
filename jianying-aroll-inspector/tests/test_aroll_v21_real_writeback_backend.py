from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.ir.models import CaptionRenderUnit
from aroll_v21.writeback.dynamic_source_binding_preflight import DynamicSourceBindingPreflight
from tests.test_aroll_v21_sacrificial_write_override import (
    create_disposable_draft,
    fake_encrypt,
    fake_real_draft_result,
    fake_real_writeback,
    fake_root_mirror_not_required,
)


def bind_report_to_result(report, result):
    if report.status != "ok":
        return report
    preflight = DynamicSourceBindingPreflight(root_mirror_func=fake_root_mirror_not_required).preflight(
        draft_dir=Path(str((result.metadata or {}).get("draft_dir") or "")),
        real_draft_result=result,
        run_report=report,
        run_dir=Path(str((result.metadata or {}).get("draft_dir") or "")) / "run",
    )
    if not preflight.success:
        return report
    return replace(
        report,
        resolved_template_map=dict(preflight.report.get("resolved_template_map") or {}),
        source_binding_report=dict(preflight.report),
    )


def preflight_source_templates(*, draft_dir, real_draft_result, run_report, run_dir=None):
    return DynamicSourceBindingPreflight(root_mirror_func=fake_root_mirror_not_required).preflight(
        draft_dir=draft_dir,
        real_draft_result=real_draft_result,
        run_report=run_report,
        run_dir=Path(run_dir) if run_dir is not None else Path(draft_dir) / "run",
    )


def run_report_from_result(result) :
    report = ArollEngine().run(
        ArollRunInput(
            mode="write",
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
    return bind_report_to_result(report, result)


class ArollV21RealWritebackBackendTests(unittest.TestCase):
    def test_real_writeback_writes_materials_segments_video_track_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)
            self.assertEqual(report.status, "ok")

            writeback = fake_real_writeback(jy_draftc=root / "jy-draftc.exe", encrypt_func=fake_encrypt)
            writeback_result = writeback.commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            self.assertTrue(writeback_result.report["writeback_success"])
            self.assertEqual(writeback_result.report["source_mapping_mode"], "dynamic_source_binding")
            self.assertEqual(writeback_result.report["selected_text_track_id"], "text_track")
            self.assertEqual(writeback_result.report["selected_video_track_id"], "video_track")
            self.assertIn("timeline_integrity_checks", writeback_result.report)
            self.assertIn("video_preflight", writeback_result.report)
            self.assertIn("audio_preflight", writeback_result.report)
            self.assertIn("filter_preflight", writeback_result.report)
            self.assertTrue(writeback_result.report["target_writes"][str(draft_content)])
            self.assertTrue(writeback_result.report["target_writes"][str(template)])
            written = json.loads(draft_content.read_text("utf-8"))
            self.assertEqual(written["duration"], writeback_result.report["gapless_final_video_end_us"])
            self.assertEqual(len(written["materials"]["texts"]), len(report.material_write_plan["materials"]))
            text_track = next(track for track in written["tracks"] if track["type"] == "text")
            video_track = next(track for track in written["tracks"] if track["type"] == "video")
            self.assertEqual(len(text_track["segments"]), len(report.material_write_plan["segments"]))
            self.assertEqual(len(video_track["segments"]), len(report.final_timeline))
            self.assertTrue(writeback_result.report["safe_handle_policy_enabled"])
            self.assertGreater(writeback_result.report["lead_handle_applied_count"], 0)
            self.assertGreater(writeback_result.report["tail_handle_applied_count"], 0)
            self.assertLessEqual(
                video_track["segments"][0]["source_timerange"]["start"],
                report.final_timeline[0].spoken_source_start_us if report.final_timeline[0].spoken_source_start_us is not None else report.final_timeline[0].source_start_us,
            )
            self.assertIn("rough_cut_quality", writeback_result.report)
            self.assertTrue((root / "run" / "draft_content.v21.modified.dec.json").exists())
            self.assertTrue((root / "run" / "draft_content.v21.modified.enc.json").exists())

    def test_writeback_video_rows_follow_final_timeline_when_captions_exceed_video_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)
            self.assertEqual(report.status, "ok")
            base_segment = report.final_timeline[0]
            final_segment = replace(
                base_segment,
                source_start_us=0,
                source_end_us=2_400_000,
                target_start_us=0,
                target_end_us=2_000_000,
                spoken_source_start_us=0,
                spoken_source_end_us=2_400_000,
                clip_source_start_us=0,
                clip_source_end_us=2_400_000,
                word_ids=["w001", "w002"],
                text="测试第一句测试第二句",
                debug_hints={},
            )
            captions = [
                CaptionRenderUnit(
                    caption_id="v21_cap_000001",
                    timeline_segment_ids=[final_segment.segment_id],
                    word_ids=["w001"],
                    text="测试第一句",
                    target_start_us=0,
                    target_end_us=900_000,
                    source_subtitle_uids=["s001"],
                    style_template_id="canonical_caption_template",
                    spoken_source_start_us=0,
                    spoken_source_end_us=1_080_000,
                    containing_video_segment_id=final_segment.segment_id,
                ),
                CaptionRenderUnit(
                    caption_id="v21_cap_000002",
                    timeline_segment_ids=[final_segment.segment_id],
                    word_ids=["w002"],
                    text="测试第二句",
                    target_start_us=900_000,
                    target_end_us=2_000_000,
                    source_subtitle_uids=["s002"],
                    style_template_id="canonical_caption_template",
                    spoken_source_start_us=1_080_000,
                    spoken_source_end_us=2_400_000,
                    containing_video_segment_id=final_segment.segment_id,
                ),
            ]
            text_segments = []
            for index, caption in enumerate(captions, start=1):
                segment = deepcopy(report.material_write_plan["segments"][0])
                segment["id"] = f"v21_caption_segment_{index:06d}"
                segment["material_id"] = f"v21_caption_material_{index:06d}"
                segment["target_timerange"] = {
                    "start": caption.target_start_us,
                    "duration": caption.target_end_us - caption.target_start_us,
                }
                text_segments.append(segment)
            text_materials = []
            for index, caption in enumerate(captions, start=1):
                material = deepcopy(report.material_write_plan["materials"][0])
                material["id"] = f"v21_caption_material_{index:06d}"
                material["content"] = json.dumps({"text": caption.text}, ensure_ascii=False)
                material["base_content"] = json.dumps({"text": caption.text}, ensure_ascii=False)
                text_materials.append(material)
            modified_report = replace(
                report,
                final_timeline=[final_segment],
                captions=captions,
                material_write_plan={
                    **report.material_write_plan,
                    "segments": text_segments,
                    "materials": text_materials,
                },
            )

            writeback = fake_real_writeback()
            data, mutation_report = writeback._modified_draft_data(result, modified_report)

            video_track = next(track for track in data["tracks"] if track["type"] == "video")
            text_track = next(track for track in data["tracks"] if track["type"] == "text")
            self.assertEqual(len(video_track["segments"]), len(modified_report.final_timeline))
            self.assertEqual(len(text_track["segments"]), len(modified_report.captions))
            self.assertEqual(mutation_report["gapless_video_row_count"], len(modified_report.final_timeline))
            self.assertEqual(text_track["segments"][0]["target_timerange"]["start"], captions[0].target_start_us)
            self.assertEqual(
                text_track["segments"][1]["target_timerange"]["duration"],
                captions[1].target_end_us - captions[1].target_start_us,
            )

    def test_canonical_sync_accepts_multiple_captions_inside_one_video_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run_report_from_result(fake_real_draft_result(root=root))
            base_segment = report.final_timeline[0]
            final_segment = replace(
                base_segment,
                source_start_us=0,
                source_end_us=2_400_000,
                target_start_us=0,
                target_end_us=2_000_000,
                spoken_source_start_us=0,
                spoken_source_end_us=2_400_000,
                clip_source_start_us=0,
                clip_source_end_us=2_400_000,
                word_ids=["w001", "w002"],
                text="测试第一句测试第二句",
                debug_hints={},
            )
            captions = [
                CaptionRenderUnit(
                    caption_id="v21_cap_000001",
                    timeline_segment_ids=[final_segment.segment_id],
                    word_ids=["w001"],
                    text="测试第一句",
                    target_start_us=0,
                    target_end_us=900_000,
                    source_subtitle_uids=["s001"],
                    style_template_id="canonical_caption_template",
                    containing_video_segment_id=final_segment.segment_id,
                ),
                CaptionRenderUnit(
                    caption_id="v21_cap_000002",
                    timeline_segment_ids=[final_segment.segment_id],
                    word_ids=["w002"],
                    text="测试第二句",
                    target_start_us=900_000,
                    target_end_us=2_000_000,
                    source_subtitle_uids=["s002"],
                    style_template_id="canonical_caption_template",
                    containing_video_segment_id=final_segment.segment_id,
                ),
            ]
            modified_report = replace(report, final_timeline=[final_segment], captions=captions)
            actual_video_segments = [
                {
                    "id": "v21_video_segment_000001",
                    "target_timerange": {"start": 0, "duration": 2_000_000},
                    "source_timerange": {"start": 0, "duration": 2_400_000},
                    "_v21_audio_coverage": {
                        "timeline_segment_id": final_segment.segment_id,
                        "source_start_us": 0,
                        "source_end_us": 2_400_000,
                        "spoken_source_start_us": 0,
                        "spoken_source_end_us": 2_400_000,
                        "word_ids": ["w001", "w002"],
                    },
                }
            ]
            caption_rows = [
                {
                    "caption_id": caption.caption_id,
                    "text": caption.text,
                    "target_start_us": caption.target_start_us,
                    "target_end_us": caption.target_end_us,
                    "segment_id": f"v21_caption_segment_{index:06d}",
                    "material_id": f"v21_caption_material_{index:06d}",
                }
                for index, caption in enumerate(captions, start=1)
            ]

            sync_report = fake_real_writeback()._jianying_canonical_timeline_sync_report(
                actual_video_segments=actual_video_segments,
                generated_caption_rows=caption_rows,
                run_report=modified_report,
            )

            self.assertTrue(sync_report["gate_passed"], sync_report)
            self.assertEqual(sync_report["caption_video_drift_count"], 0)
            self.assertEqual(sync_report["split_caption_container_mismatch_count"], 0)
            self.assertEqual(sync_report["caption_words_not_covered_by_actual_video_count"], 0)

    def test_source_projection_preserves_relative_caption_ranges_inside_one_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run_report_from_result(fake_real_draft_result(root=root))
            base_word = report.source_graph.words[0]
            dropped_word = replace(
                base_word,
                word_id="w_drop",
                text="删掉",
                normalized_text="删掉",
                source_start_us=0,
                source_end_us=80_000,
            )
            first_word = replace(
                base_word,
                word_id="w_keep_1",
                text="第一句",
                normalized_text="第一句",
                source_start_us=100_000,
                source_end_us=300_000,
            )
            second_word = replace(
                base_word,
                word_id="w_keep_2",
                text="第二句",
                normalized_text="第二句",
                source_start_us=300_000,
                source_end_us=600_000,
            )
            final_segment = replace(
                report.final_timeline[0],
                source_start_us=0,
                source_end_us=600_000,
                target_start_us=0,
                target_end_us=600_000,
                spoken_source_start_us=0,
                spoken_source_end_us=600_000,
                clip_source_start_us=0,
                clip_source_end_us=600_000,
                word_ids=["w_drop", "w_keep_1", "w_keep_2"],
                text="删掉第一句第二句",
                debug_hints={},
            )
            captions = [
                CaptionRenderUnit(
                    caption_id="v21_cap_000001",
                    timeline_segment_ids=[final_segment.segment_id],
                    word_ids=["w_keep_1"],
                    text="第一句",
                    target_start_us=100_000,
                    target_end_us=300_000,
                    source_subtitle_uids=["s001"],
                    style_template_id="canonical_caption_template",
                    containing_video_segment_id=final_segment.segment_id,
                ),
                CaptionRenderUnit(
                    caption_id="v21_cap_000002",
                    timeline_segment_ids=[final_segment.segment_id],
                    word_ids=["w_keep_2"],
                    text="第二句",
                    target_start_us=300_000,
                    target_end_us=600_000,
                    source_subtitle_uids=["s002"],
                    style_template_id="canonical_caption_template",
                    containing_video_segment_id=final_segment.segment_id,
                ),
            ]
            modified_report = replace(
                report,
                source_graph=replace(report.source_graph, words=[dropped_word, first_word, second_word]),
                final_timeline=[final_segment],
                captions=captions,
            )

            plan = fake_real_writeback()._gapless_caption_video_projection_plan(modified_report)

            self.assertEqual(len(plan["video_units"]), 1)
            self.assertEqual(plan["video_units"][0].lead_handle_us, 0)
            self.assertEqual(plan["video_units"][0].clip_source_start_us, 100_000)
            self.assertEqual(plan["caption_target_ranges"]["v21_cap_000001"], {"target_start_us": 0, "target_end_us": 200_000})
            self.assertEqual(plan["caption_target_ranges"]["v21_cap_000002"], {"target_start_us": 200_000, "target_end_us": 500_000})

    def test_writeback_projection_omits_spoken_final_segments_without_captions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run_report_from_result(fake_real_draft_result(root=root))
            captioned_segment = report.final_timeline[0]
            uncaptioned_segment = replace(
                captioned_segment,
                segment_id="v21_seg_uncaptioned",
                source_start_us=400_000,
                source_end_us=700_000,
                target_start_us=300_000,
                target_end_us=600_000,
                word_ids=["w_uncaptioned"],
                text="无字幕语音",
                debug_hints={},
            )
            caption = replace(
                report.captions[0],
                containing_video_segment_id=captioned_segment.segment_id,
                timeline_segment_ids=[captioned_segment.segment_id],
            )
            modified_report = replace(
                report,
                final_timeline=[captioned_segment, uncaptioned_segment],
                captions=[caption],
            )

            plan = fake_real_writeback()._gapless_caption_video_projection_plan(modified_report)

            self.assertEqual([segment.segment_id for segment in plan["video_units"]], [captioned_segment.segment_id])
            self.assertNotIn("v21_seg_uncaptioned", [segment.segment_id for segment in plan["video_units"]])

    def test_missing_timeline_metadata_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = replace(fake_real_draft_result(root=root), metadata={})
            report = run_report_from_result(fake_real_draft_result(root=root))

            writeback_result = fake_real_writeback(encrypt_func=fake_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertFalse(writeback_result.success)
            self.assertEqual(writeback_result.blockers[0].code, "V21_WRITEBACK_TIMELINE_METADATA_MISSING")


if __name__ == "__main__":
    unittest.main()
