from __future__ import annotations

import unittest

from aroll_v21 import ArollEngine
from aroll_v21.ir import CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
from aroll_v21.render.subtitle_renderer import SubtitleRenderer


def _graph(rows: list[tuple[str, str, int, int, int]]):
    return ArollEngine().ingest.build_source_graph(
        word_timeline=[
            {
                "word_id": word_id,
                "word_text": text,
                "start_us": start,
                "end_us": end,
                "subtitle_index": subtitle_index,
                "subtitle_uid": f"s{subtitle_index:03d}",
            }
            for word_id, text, start, end, subtitle_index in rows
        ],
        subtitles=[
            {
                "subtitle_uid": f"s{subtitle_index:03d}",
                "subtitle_index": subtitle_index,
                "text": "".join(text for _word_id, text, _start, _end, row_subtitle_index in rows if row_subtitle_index == subtitle_index),
                "word_ids": [word_id for word_id, _text, _start, _end, row_subtitle_index in rows if row_subtitle_index == subtitle_index],
            }
            for subtitle_index in sorted({row[4] for row in rows})
        ],
        source_segments=[
            {
                "id": "primary_window",
                "material_id": "main",
                "type": "video",
                "source_start_us": 0,
                "source_end_us": max(end for _word_id, _text, _start, end, _subtitle_index in rows) + 500_000,
            }
        ],
        source_materials=[{"source_material_id": "main", "type": "video", "duration_us": max(end for _word_id, _text, _start, end, _subtitle_index in rows) + 500_000}],
        text_materials=[],
        text_segments=[],
    )


def _segment(segment_id: str, word_ids: list[str], text: str, start: int, end: int) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=segment_id,
        source_material_id="main",
        source_segment_id="primary_window",
        source_start_us=start,
        source_end_us=end,
        target_start_us=start,
        target_end_us=end,
        word_ids=word_ids,
        text=text,
        decision_ids=[],
    )


def _caption(
    caption_id: str,
    segment_id: str,
    word_ids: list[str],
    text: str,
    start: int,
    end: int,
) -> CaptionRenderUnit:
    return CaptionRenderUnit(
        caption_id=caption_id,
        timeline_segment_ids=[segment_id],
        word_ids=word_ids,
        text=text,
        target_start_us=start,
        target_end_us=end,
        source_subtitle_uids=[],
        style_template_id="test",
        containing_video_segment_id=segment_id,
    )


class FinalVisibleGenericQcRegressionTests(unittest.TestCase):
    def test_intraword_cjk_restart_before_result_complement_is_normalized(self) -> None:
        graph = _graph(
            [
                ("w001", "拍拍", 0, 500_000, 1),
                ("w002", "出", 500_000, 650_000, 1),
                ("w003", "展示面", 650_000, 1_200_000, 1),
            ]
        )

        self.assertEqual(graph.words[0].text, "拍")
        self.assertEqual(graph.edit_units[0].text, "拍出展示面")
        self.assertTrue(graph.words[0].debug_hints["intraword_cjk_restart_normalized"])

    def test_intraword_cjk_reduplication_before_noun_is_preserved(self) -> None:
        graph = _graph(
            [
                ("w001", "看看", 0, 500_000, 1),
                ("w002", "照片", 500_000, 900_000, 1),
            ]
        )

        self.assertEqual(graph.words[0].text, "看看")
        self.assertFalse(graph.words[0].debug_hints.get("intraword_cjk_restart_normalized", False))

    def test_negative_predicate_restart_across_visible_boundary_is_removed(self) -> None:
        rows = [
            ("w001", "你的", 0, 200_000, 1),
            ("w002", "脊椎", 200_000, 400_000, 1),
            ("w003", "会", 400_000, 550_000, 1),
            ("w004", "下意识", 550_000, 850_000, 1),
            ("w005", "的", 850_000, 930_000, 1),
            ("w006", "不", 930_000, 1_050_000, 1),
            ("w007", "可控", 1_050_000, 1_350_000, 2),
            ("w008", "不可", 1_650_000, 1_900_000, 2),
            ("w009", "控制", 1_900_000, 2_150_000, 2),
            ("w010", "的", 2_150_000, 2_220_000, 2),
            ("w011", "驼起来", 2_500_000, 3_200_000, 3),
        ]
        graph = _graph(rows)
        timeline = [
            FinalTimelineSegment(
                segment_id="v21_seg_000001",
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_220_000,
                target_start_us=0,
                target_end_us=4_000_000,
                word_ids=[row[0] for row in rows[:-1]],
                text="你的脊椎会下意识的不可控不可控制的",
                decision_ids=[],
            ),
            FinalTimelineSegment(
                segment_id="v21_seg_000002",
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=2_500_000,
                source_end_us=3_200_000,
                target_start_us=4_000_000,
                target_end_us=4_700_000,
                word_ids=["w011"],
                text="驼起来",
                decision_ids=[],
            ),
        ]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=graph,
            render_captions=lambda current: renderer.render(current, graph),
        )

        visible_text = "".join(caption.text for caption in result.captions)
        self.assertNotIn("不可控不可控制", visible_text)
        self.assertIn("不可控制的驼起来", visible_text)
        merged = [segment for segment in result.final_timeline if "不可控制的驼起来" in segment.text][0]
        self.assertEqual(merged.target_end_us - merged.target_start_us, merged.source_end_us - merged.source_start_us)
        self.assertIn("drop_negative_predicate_restart_span", [row["decision"] for row in result.report["final_visible_repair_actions"]])
        self.assertTrue(result.report["final_visible_repair_success"], result.report)

    def test_negative_predicate_restart_drops_unmergeable_micro_residual(self) -> None:
        rows = [
            ("w001", "你", 0, 320_000, 1),
            ("w002", "不是", 420_000, 650_000, 1),
            ("w003", "节俭", 650_000, 950_000, 1),
            ("w004", "你", 950_000, 1_060_000, 1),
            ("w005", "不是", 1_300_000, 1_560_000, 2),
            ("w006", "务实", 1_560_000, 1_860_000, 2),
            ("w007", "啊", 1_860_000, 1_980_000, 2),
        ]
        graph = _graph(rows)
        timeline = [
            _segment("v21_seg_000001", ["w001", "w002", "w003", "w004"], "你不是节俭你", 0, 1_060_000),
            _segment("v21_seg_000002", ["w005", "w006", "w007"], "不是务实啊", 1_300_000, 1_980_000),
        ]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=graph,
            render_captions=lambda current: renderer.render(current, graph),
        )

        visible_text = "".join(caption.text for caption in result.captions)
        self.assertNotIn("不是节俭你不是", visible_text)
        self.assertIn("不是务实啊", visible_text)
        self.assertNotIn("你不是务实啊", visible_text)
        self.assertFalse(
            [
                segment
                for segment in result.final_timeline
                if segment.text == "你" and segment.target_end_us - segment.target_start_us < 500_000
            ],
            result.final_timeline,
        )
        decisions = [row["decision"] for row in result.report["final_visible_repair_actions"]]
        self.assertIn("drop_negative_predicate_restart_span", decisions)
        self.assertIn("cleanup_short_repair_residual_segments", decisions)
        self.assertTrue(result.report["final_visible_repair_success"], result.report)

    def test_partial_phrase_restart_across_caption_boundary_is_removed(self) -> None:
        rows = [
            ("w001", "她", 0, 80_000, 1),
            ("w002", "把", 80_000, 200_000, 1),
            ("w003", "身材", 200_000, 520_000, 1),
            ("w004", "当", 520_000, 720_000, 1),
            ("w005", "成资", 720_000, 1_000_000, 2),
            ("w006", "当成", 1_300_000, 1_620_000, 2),
            ("w007", "资产", 1_620_000, 1_940_000, 2),
            ("w008", "去", 1_940_000, 2_120_000, 2),
            ("w009", "投资", 2_120_000, 2_560_000, 2),
        ]
        graph = _graph(rows)
        timeline = [
            _segment("v21_seg_000001", ["w001", "w002", "w003", "w004"], "她把身材当", 0, 720_000),
            _segment("v21_seg_000002", ["w005", "w006", "w007", "w008", "w009"], "成资当成资产去投资", 720_000, 2_560_000),
        ]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=graph,
            render_captions=lambda current: renderer.render(current, graph),
        )

        visible_text = "".join(caption.text for caption in result.captions)
        self.assertNotIn("当成资当成资产", visible_text)
        self.assertIn("身材当成资产去投资", visible_text)
        for segment in result.final_timeline:
            self.assertEqual(
                segment.target_end_us - segment.target_start_us,
                segment.source_end_us - segment.source_start_us,
            )
        self.assertIn("drop_partial_phrase_restart_span", [row["decision"] for row in result.report["final_visible_repair_actions"]])
        self.assertTrue(result.report["final_visible_repair_success"], result.report)

    def test_compound_suffix_split_by_subtitle_boundary_is_merged(self) -> None:
        rows = [
            ("w001", "你", 0, 150_000, 1),
            ("w002", "在", 150_000, 300_000, 1),
            ("w003", "评论", 300_000, 650_000, 1),
            ("w004", "区", 650_000, 760_000, 2),
            ("w005", "里面", 760_000, 1_000_000, 2),
            ("w006", "敲字", 1_000_000, 1_400_000, 2),
        ]
        graph = _graph(rows)
        timeline = [
            _segment("v21_seg_000001", ["w001", "w002", "w003"], "你在评论", 0, 650_000),
            _segment("v21_seg_000002", ["w004", "w005", "w006"], "区里面敲字", 650_000, 1_400_000),
        ]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=graph,
            render_captions=lambda current: renderer.render(current, graph),
        )

        self.assertEqual([caption.text for caption in result.captions], ["你在评论区里面敲字"])
        self.assertEqual(len(result.final_timeline), 1)
        self.assertIn("merge_source_boundary_compound_suffix", [row["decision"] for row in result.report["final_visible_repair_actions"]])
        self.assertTrue(result.report["final_visible_repair_success"], result.report)

    def test_final_caption_only_dangling_merges_survive_pass_limit(self) -> None:
        rows = [
            ("w001", "你", 0, 180_000, 1),
            ("w002", "找", 180_000, 320_000, 1),
            ("w003", "女生", 320_000, 620_000, 1),
            ("w004", "的", 620_000, 720_000, 1),
            ("w005", "后台", 720_000, 980_000, 1),
            ("w006", "看看", 980_000, 1_240_000, 1),
            ("w007", "我", 1_500_000, 1_650_000, 2),
            ("w008", "看", 1_650_000, 1_800_000, 2),
            ("w009", "内容", 1_800_000, 2_080_000, 2),
            ("w010", "的", 2_080_000, 2_180_000, 2),
            ("w011", "详情", 2_180_000, 2_520_000, 2),
        ]
        graph = _graph(rows)
        timeline = [
            _segment("v21_seg_000001", [row[0] for row in rows[:6]], "你找女生的后台看看", 0, 1_240_000),
            _segment("v21_seg_000002", [row[0] for row in rows[6:]], "我看内容的详情", 1_500_000, 2_520_000),
        ]
        captions = [
            _caption("v21_cap_000001", "v21_seg_000001", ["w001", "w002", "w003"], "你找女生", 0, 620_000),
            _caption("v21_cap_000002", "v21_seg_000001", ["w004", "w005", "w006"], "的后台看看", 620_000, 1_240_000),
            _caption("v21_cap_000003", "v21_seg_000002", ["w007", "w008", "w009"], "我看内容", 1_500_000, 2_080_000),
            _caption("v21_cap_000004", "v21_seg_000002", ["w010", "w011"], "的详情", 2_080_000, 2_520_000),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=graph,
            render_captions=lambda _current: captions,
            max_passes=1,
        )

        texts = [caption.text for caption in result.captions]
        self.assertIn("你找女生的后台看看", texts)
        self.assertIn("我看内容的详情", texts)
        self.assertNotIn("的后台看看", texts)
        self.assertNotIn("的详情", texts)
        self.assertIn(
            "finalize_caption_only_dangling_merge",
            [row["decision"] for row in result.report["final_visible_repair_actions"]],
        )
        self.assertTrue(result.report["final_visible_repair_success"], result.report)


if __name__ == "__main__":
    unittest.main()
