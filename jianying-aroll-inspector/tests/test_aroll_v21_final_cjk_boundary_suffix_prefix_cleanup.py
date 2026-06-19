from __future__ import annotations

import unittest

from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.compiler import FinalTimelineCompiler
from aroll_v21.ingest import DraftIngest
from aroll_v21.ir import DecisionPlan
from tests.test_aroll_v21_captions_after_prefix_drop import _template_rows


def _input_from_word_groups(groups: list[list[str]]) -> ArollRunInput:
    materials, text_segments = _template_rows()
    words = []
    subtitles = []
    cursor = 0
    for subtitle_index, group in enumerate(groups, start=1):
        word_ids = []
        for token in group:
            word_id = f"w_{len(words) + 1:06d}"
            duration = max(120_000, len(token) * 100_000)
            word_ids.append(word_id)
            words.append(
                {
                    "word_id": word_id,
                    "word_text": token,
                    "source_start_us": cursor,
                    "source_end_us": cursor + duration,
                    "source_material_id": "main",
                    "source_segment_id": "clip",
                    "subtitle_uid": f"s_{subtitle_index:03d}",
                    "subtitle_index": subtitle_index,
                }
            )
            cursor += duration
        subtitles.append(
            {
                "subtitle_uid": f"s_{subtitle_index:03d}",
                "subtitle_index": subtitle_index,
                "text": "".join(group),
                "word_ids": word_ids,
                "text_material_id": "template_text",
            }
        )
    return ArollRunInput(
        source_segments=[{"id": "clip", "material_id": "main", "source_start_us": 0, "source_end_us": cursor + 1_000_000}],
        word_timeline=words,
        subtitles=subtitles,
        text_materials=materials,
        text_segments=text_segments,
        postwrite_mode="simulated",
    )


class ArollV21FinalCjkBoundarySuffixPrefixCleanupTests(unittest.TestCase):
    def test_suffix_prefix_overlap_drops_left_trailing_whole_words(self) -> None:
        cases = [
            (
                [["在", "舞台", "中央", "大声", "说"], ["中央", "大声", "说", "话的", "自己"]],
                ["在舞台", "中央大声说话的自己"],
            ),
            (
                [["都", "觉得", "羞耻", "嘉豪"], ["嘉豪", "不是", "中二啊"]],
                ["都觉得羞耻", "嘉豪不是中二啊"],
            ),
            (
                [["但你", "不能", "没有", "主角感", "停止"], ["停止", "规训", "你自己的", "同类"]],
                ["但你不能没有主角感", "停止规训你自己的同类"],
            ),
        ]
        for groups, expected in cases:
            with self.subTest(expected=expected):
                report = ArollEngine().run(_input_from_word_groups(groups))

                self.assertEqual(report.status, "ok", [blocker.code for blocker in report.blocker_report.blockers])
                self.assertEqual([caption.text for caption in report.captions], expected)
                self.assertEqual("".join(segment.text for segment in report.final_timeline), "".join(expected))
                self.assertTrue(report.validator_report["final_repeat_validator"]["final_repeat_gate_passed"])
                self.assertTrue(report.validator_report["hidden_audio_repeat_validator"]["hidden_audio_repeat_gate_passed"])
                rough = report.validator_report["rough_cut_quality_validator"]
                self.assertEqual(rough["segments_lt_300ms"], 0)
                self.assertEqual(rough["one_char_captions"], 0)
                self.assertLessEqual(len(report.final_timeline), len(report.captions))
                self.assertTrue(rough["caption_count_covers_video_segments"])
                self.assertEqual(len(report.captions), len(report.material_write_plan["materials"]))
                self.assertEqual(len(report.captions), len(report.material_write_plan["segments"]))

    def test_suffix_prefix_overlap_requires_whole_word_binding(self) -> None:
        materials, text_segments = _template_rows()
        graph = DraftIngest().build_source_graph(
            word_timeline=[
                {
                    "word_id": "w_left",
                    "word_text": "在舞台中央大声说",
                    "source_start_us": 0,
                    "source_end_us": 700_000,
                    "source_material_id": "main",
                    "source_segment_id": "clip",
                    "subtitle_uid": "s_left",
                    "subtitle_index": 1,
                },
                {
                    "word_id": "w_right_1",
                    "word_text": "中央",
                    "source_start_us": 800_000,
                    "source_end_us": 920_000,
                    "source_material_id": "main",
                    "source_segment_id": "clip",
                    "subtitle_uid": "s_right",
                    "subtitle_index": 2,
                },
                {
                    "word_id": "w_right_2",
                    "word_text": "大声说话",
                    "source_start_us": 920_000,
                    "source_end_us": 1_120_000,
                    "source_material_id": "main",
                    "source_segment_id": "clip",
                    "subtitle_uid": "s_right",
                    "subtitle_index": 2,
                },
            ],
            subtitles=[
                {"subtitle_uid": "s_left", "subtitle_index": 1, "text": "在舞台中央大声说", "word_ids": ["w_left"], "text_material_id": "template_text"},
                {"subtitle_uid": "s_right", "subtitle_index": 2, "text": "中央大声说话", "word_ids": ["w_right_1", "w_right_2"], "text_material_id": "template_text"},
            ],
            source_segments=[{"id": "clip", "material_id": "main", "source_start_us": 0, "source_end_us": 2_000_000}],
            text_materials=materials,
            text_segments=text_segments,
        )

        _segments, blockers = FinalTimelineCompiler().compile(graph, DecisionPlan(decisions=[]))

        self.assertIn("BOUNDARY_SUFFIX_PREFIX_OVERLAP_WORD_BINDING_MISSING", [blocker.code for blocker in blockers])


if __name__ == "__main__":
    unittest.main()
