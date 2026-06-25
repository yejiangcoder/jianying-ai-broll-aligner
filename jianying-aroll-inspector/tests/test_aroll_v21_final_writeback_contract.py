from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aroll_v21.ir.models import (
    BlockerReport,
    CanonicalSourceGraph,
    CanonicalWord,
    CaptionRenderUnit,
    FinalTimelineSegment,
    RunReport,
    SourceGraphInvariantReport,
)
from aroll_v21.operator import ArollV21OperatorConfig, run_operator
from aroll_v21.writeback.real_draft_writeback import RealDraftWriteback
from tests.test_aroll_v21_final_backend_integration_contract import add_selected_track_callout, read_json
from tests.test_aroll_v21_full_chain_internal_self_check import ExternalWordTimelineAdapter
from tests.test_aroll_v21_real_writeback_backend import run_report_from_result
from tests.test_aroll_v21_sacrificial_write_override import create_disposable_draft, fake_real_draft_result, fake_real_writeback


def _run_write_operator(root: Path, result, *, writeback_factory=None):
    draft_dir = root / "draft"
    if writeback_factory is None:
        writeback_factory = lambda *a, **k: fake_real_writeback(jy_draftc=k.get("jy_draftc"))
    with patch(
        "aroll_v21.operator.RealDraftIngestAdapter",
        lambda *a, **k: ExternalWordTimelineAdapter(result=result),
    ), patch("aroll_v21.operator.RealDraftWriteback", writeback_factory):
        return run_operator(
            ArollV21OperatorConfig(
                mode="write",
                run_dir=root / "run",
                draft_dir=draft_dir,
                jy_draftc=root / "jy-draftc.exe",
                commit=True,
                allow_sacrificial_write_without_postwrite_decrypt=True,
            )
        )


class ArollV21FinalWritebackContractTests(unittest.TestCase):
    def test_gapless_projection_keeps_video_unit_for_cross_segment_caption(self) -> None:
        words = [
            CanonicalWord(
                word_id=f"w{index}",
                text=text,
                normalized_text=text,
                source_start_us=start,
                source_end_us=end,
                source_material_id="mat",
                source_segment_id=f"src{index}",
                subtitle_uid=None,
                subtitle_index=None,
                char_start=None,
                char_end=None,
                confidence=None,
                is_cuttable_left=True,
                is_cuttable_right=True,
            )
            for index, (text, start, end) in enumerate(
                [
                    ("你嘲笑嘉豪是对自己人", 0, 1_000_000),
                    ("的规训", 1_000_000, 1_600_000),
                    ("你其实是在嘲笑自己", 2_000_000, 2_600_000),
                ],
                start=1,
            )
        ]
        report = RunReport(
            status="ok",
            source_graph=CanonicalSourceGraph(
                words=words,
                edit_units=[],
                subtitle_rows=[],
                source_materials=[],
                source_segments=[],
                text_materials=[],
                text_segments=[],
                invariant_report=SourceGraphInvariantReport(
                    single_source_graph_ok=True,
                    all_words_have_source_time=True,
                    all_edit_units_have_word_ids=True,
                    unbound_word_count=0,
                    unbound_subtitle_count=0,
                    blocker_count=0,
                ),
            ),
            repeat_clusters=[],
            decision_plan=None,
            final_timeline=[
                FinalTimelineSegment(
                    segment_id="seg1",
                    source_material_id="mat",
                    source_segment_id="src1",
                    source_start_us=0,
                    source_end_us=1_000_000,
                    target_start_us=0,
                    target_end_us=1_000_000,
                    word_ids=["w1"],
                    text="你嘲笑嘉豪是对自己人",
                    decision_ids=[],
                ),
                FinalTimelineSegment(
                    segment_id="seg2",
                    source_material_id="mat",
                    source_segment_id="src2",
                    source_start_us=1_000_000,
                    source_end_us=1_600_000,
                    target_start_us=1_000_000,
                    target_end_us=1_600_000,
                    word_ids=["w2"],
                    text="的规训",
                    decision_ids=[],
                ),
                FinalTimelineSegment(
                    segment_id="seg3",
                    source_material_id="mat",
                    source_segment_id="src3",
                    source_start_us=2_000_000,
                    source_end_us=2_600_000,
                    target_start_us=1_600_000,
                    target_end_us=2_200_000,
                    word_ids=["w3"],
                    text="你其实是在嘲笑自己",
                    decision_ids=[],
                ),
            ],
            captions=[
                CaptionRenderUnit(
                    caption_id="cap1",
                    timeline_segment_ids=["seg1", "seg2"],
                    word_ids=["w1", "w2"],
                    text="你嘲笑嘉豪是对自己人的规训",
                    target_start_us=0,
                    target_end_us=1_600_000,
                    source_subtitle_uids=[],
                    style_template_id="caption_template",
                    containing_video_segment_id=None,
                ),
                CaptionRenderUnit(
                    caption_id="cap2",
                    timeline_segment_ids=["seg3"],
                    word_ids=["w3"],
                    text="你其实是在嘲笑自己",
                    target_start_us=1_600_000,
                    target_end_us=2_200_000,
                    source_subtitle_uids=[],
                    style_template_id="caption_template",
                    containing_video_segment_id="seg3",
                ),
            ],
            material_write_plan={
                "segments": [
                    {"id": "caption_seg_1", "material_id": "caption_mat_1", "target_timerange": {"start": 0, "duration": 1_600_000}},
                    {"id": "caption_seg_2", "material_id": "caption_mat_2", "target_timerange": {"start": 1_600_000, "duration": 600_000}},
                ],
            },
            validator_report={},
            postwrite_report={},
            blocker_report=BlockerReport(blocked=False, blockers=[]),
        )
        writeback = RealDraftWriteback()

        projection = writeback._gapless_caption_video_projection_plan(report)
        segments = [dict(row) for row in report.material_write_plan["segments"]]
        writeback._apply_gapless_caption_ranges(segments, report.captions, projection["caption_target_ranges"])

        self.assertEqual([unit.segment_id for unit in projection["video_units"]], ["seg1", "seg2", "seg3"])
        self.assertEqual(projection["caption_target_ranges"]["cap1"], {"target_start_us": 0, "target_end_us": 1_600_000})
        self.assertEqual(projection["caption_target_ranges"]["cap2"], {"target_start_us": 1_600_000, "target_end_us": 2_200_000})
        first_end = segments[0]["target_timerange"]["start"] + segments[0]["target_timerange"]["duration"]
        second_start = segments[1]["target_timerange"]["start"]
        self.assertEqual(first_end, second_start)

    def test_gapless_projection_does_not_expand_cross_group_caption_to_whole_group_range(self) -> None:
        words = [
            CanonicalWord(
                word_id=word_id,
                text=text,
                normalized_text=text,
                source_start_us=start,
                source_end_us=end,
                source_material_id="mat",
                source_segment_id="src",
                subtitle_uid=None,
                subtitle_index=None,
                char_start=None,
                char_end=None,
                confidence=None,
                is_cuttable_left=True,
                is_cuttable_right=True,
            )
            for word_id, text, start, end in [
                ("w1", "第一", 0, 100_000),
                ("w2", "第二", 100_000, 200_000),
                ("w3", "第三", 200_000, 300_000),
                ("w4", "第四", 300_000, 350_000),
                ("drop1", "删掉", 360_000, 450_000),
                ("drop2", "废话", 460_000, 550_000),
                ("w5", "续接", 560_000, 600_000),
            ]
        ]
        report = RunReport(
            status="ok",
            source_graph=CanonicalSourceGraph(
                words=words,
                edit_units=[],
                subtitle_rows=[],
                source_materials=[],
                source_segments=[],
                text_materials=[],
                text_segments=[],
                invariant_report=SourceGraphInvariantReport(
                    single_source_graph_ok=True,
                    all_words_have_source_time=True,
                    all_edit_units_have_word_ids=True,
                    unbound_word_count=0,
                    unbound_subtitle_count=0,
                    blocker_count=0,
                ),
            ),
            repeat_clusters=[],
            decision_plan=None,
            final_timeline=[
                FinalTimelineSegment(
                    segment_id="seg1",
                    source_material_id="mat",
                    source_segment_id="src",
                    source_start_us=0,
                    source_end_us=600_000,
                    target_start_us=0,
                    target_end_us=600_000,
                    word_ids=["w1", "w2", "w3", "w4", "w5"],
                    text="第一第二第三第四续接",
                    decision_ids=[],
                )
            ],
            captions=[
                CaptionRenderUnit("cap1", ["seg1"], ["w1"], "第一", 0, 100_000, [], "caption_template", containing_video_segment_id="seg1"),
                CaptionRenderUnit("cap2", ["seg1"], ["w2"], "第二", 100_000, 200_000, [], "caption_template", containing_video_segment_id="seg1"),
                CaptionRenderUnit("cap3", ["seg1"], ["w3"], "第三", 200_000, 300_000, [], "caption_template", containing_video_segment_id="seg1"),
                CaptionRenderUnit("cap4", ["seg1"], ["w4", "w5"], "第四续接", 300_000, 400_000, [], "caption_template", containing_video_segment_id="seg1"),
            ],
            material_write_plan={
                "segments": [
                    {"id": f"caption_seg_{index}", "material_id": f"caption_mat_{index}", "target_timerange": {"start": (index - 1) * 100_000, "duration": 100_000}}
                    for index in range(1, 5)
                ],
            },
            validator_report={},
            postwrite_report={},
            blocker_report=BlockerReport(blocked=False, blockers=[]),
        )
        writeback = RealDraftWriteback()

        projection = writeback._gapless_caption_video_projection_plan(report)
        segments = [dict(row) for row in report.material_write_plan["segments"]]
        writeback._apply_gapless_caption_ranges(segments, report.captions, projection["caption_target_ranges"])

        cap3_end = segments[2]["target_timerange"]["start"] + segments[2]["target_timerange"]["duration"]
        cap4_start = segments[3]["target_timerange"]["start"]
        self.assertGreaterEqual(cap4_start, cap3_end)
        self.assertGreater(cap4_start, 0)
        self.assertEqual(
            projection["caption_target_ranges"]["cap4"],
            {"target_start_us": 300_000, "target_end_us": 390_000},
        )
        self.assertEqual(segments[3]["target_timerange"], {"start": 300_000, "duration": 90_000})

    def test_mixed_selected_text_track_cleans_old_subtitles_preserves_callout_and_writes_captions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            add_selected_track_callout(result)
            report = run_report_from_result(fake_real_draft_result(root=root))

            writeback_result = fake_real_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            written = json.loads(draft_content.read_text("utf-8"))
            text_track = next(track for track in written["tracks"] if track["id"] == "text_track")
            segment_ids = {segment["id"] for segment in text_track["segments"]}
            material_ids = {material["id"] for material in written["materials"]["texts"]}

            self.assertNotIn("caption_segment_001", segment_ids)
            self.assertNotIn("caption_template_001", material_ids)
            self.assertIn("callout_segment", segment_ids)
            self.assertIn("callout_material", material_ids)
            self.assertEqual(sum(1 for segment in text_track["segments"] if segment["id"].startswith("v21_caption_segment_")), len(report.captions))
            self.assertGreater(len(text_track["segments"]), len(report.captions))

            rough = writeback_result.report["rough_cut_quality"]
            self.assertEqual(rough["canonical_caption_segment_count"], len(report.captions))
            self.assertEqual(rough["visible_caption_track_count"], 1)
            self.assertEqual(rough["old_subtitle_residue_track_count"], 0)
            self.assertEqual(rough["overlapping_caption_segments_count"], 0)

    def test_unknown_caption_like_text_segment_is_cleaned_without_residue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            unknown_material = {"id": "unknown_text_material", "type": "text", "text": "unknown"}
            unknown_segment = {
                "id": "unknown_text_segment",
                "type": "text",
                "material_id": "unknown_text_material",
                "target_timerange": {"start": 500_000, "duration": 500_000},
            }
            result.draft_data["materials"]["texts"].append(unknown_material)
            text_track = next(track for track in result.draft_data["tracks"] if track["id"] == "text_track")
            text_track["segments"].append(unknown_segment)
            result.text_segments.append(unknown_segment | {"track_id": "text_track", "track_type": "text"})

            summary = _run_write_operator(root, result)
            writeback_report = read_json(root / "run" / "writeback_report.json")

            self.assertEqual(summary["status"], "ok")
            self.assertTrue(summary["commit_performed"])
            self.assertTrue(summary["writeback_success"])
            self.assertTrue(summary["WRITE_SUCCESS"])
            self.assertEqual(writeback_report["post_write_old_subtitle_residue_count"], 0)
            self.assertTrue(writeback_report["post_write_actual_text_residue_gate_passed"])
            written = json.loads(draft_content.read_text("utf-8"))
            text_track = next(track for track in written["tracks"] if track["id"] == "text_track")
            self.assertNotIn("unknown_text_segment", {segment["id"] for segment in text_track["segments"]})

    def test_encrypt_failure_and_target_write_failure_do_not_report_fake_commit_success(self) -> None:
        def failing_encrypt(_jy_draftc: Path, _plain: Path, _encrypted_out: Path) -> None:
            raise RuntimeError("encrypt failed")

        cases = [
            (
                "encrypt",
                lambda *a, **k: fake_real_writeback(jy_draftc=k.get("jy_draftc"), encrypt_func=failing_encrypt),
                "V21_WRITEBACK_ENCRYPT_FAILED",
            ),
            (
                "target_write",
                lambda *a, **k: fake_real_writeback(jy_draftc=k.get("jy_draftc")),
                "V21_WRITEBACK_TARGET_WRITE_FAILED",
            ),
        ]
        for name, writeback_factory, expected_blocker in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                create_disposable_draft(root)
                result = fake_real_draft_result(root=root)
                if name == "target_write":
                    with patch("aroll_v21.writeback.real_draft_writeback.shutil.copyfile", side_effect=OSError("target write failed")):
                        summary = _run_write_operator(root, result, writeback_factory=writeback_factory)
                else:
                    summary = _run_write_operator(root, result, writeback_factory=writeback_factory)

                self.assertEqual(summary["status"], "blocked")
                self.assertFalse(summary["commit_performed"])
                self.assertFalse(summary["writeback_success"])
                self.assertFalse(summary["WRITE_SUCCESS"])
                self.assertEqual(summary["fatal_blocker"], expected_blocker)
                self.assertIn(expected_blocker, summary["blocker_codes"])


if __name__ == "__main__":
    unittest.main()
