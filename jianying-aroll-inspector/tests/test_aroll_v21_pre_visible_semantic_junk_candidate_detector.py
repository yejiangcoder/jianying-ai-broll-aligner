from __future__ import annotations

import unittest

from aroll_v21 import ArollEngine
from aroll_v21.ir import CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.pre_visible_semantic_junk_candidate_detector import build_pre_visible_semantic_junk_candidate_report
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
from aroll_v21.render.subtitle_renderer import SubtitleRenderer
from tests.test_aroll_v21_captions_after_prefix_drop import _template_rows


def _graph_for_rows(rows: list[tuple[str, str, int, int]]):
    materials, text_segments = _template_rows()
    words = [
        {
            "word_id": word_id,
            "word_text": text,
            "start_us": start,
            "end_us": end,
            "subtitle_index": index,
            "subtitle_uid": f"s{index:03d}",
        }
        for index, (word_id, text, start, end) in enumerate(rows, start=1)
    ]
    return ArollEngine().ingest.build_source_graph(
        word_timeline=words,
        subtitles=[
            {
                "subtitle_uid": row["subtitle_uid"],
                "subtitle_index": row["subtitle_index"],
                "text": row["word_text"],
                "word_ids": [row["word_id"]],
            }
            for row in words
        ],
        source_segments=[
            {
                "id": "primary_window",
                "material_id": "main",
                "type": "video",
                "source_start_us": 0,
                "source_end_us": max(end for _word_id, _text, _start, end in rows) + 500_000,
            }
        ],
        text_materials=materials,
        text_segments=text_segments,
    )


def _segment(index: int, word_id: str, text: str, start: int, end: int) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=f"v21_seg_{index:06d}",
        source_material_id="main",
        source_segment_id="primary_window",
        source_start_us=start,
        source_end_us=end,
        target_start_us=start,
        target_end_us=end,
        word_ids=[word_id],
        text=text,
        decision_ids=[],
    )


def _multi_segment(index: int, word_ids: list[str], text: str, start: int, end: int) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=f"v21_seg_{index:06d}",
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


class PreVisibleSemanticJunkTest(unittest.TestCase):
    def test_detects_and_repairs_aborted_restart_fragment_before_longer_restatement(self) -> None:
        rows = [
            ("w001", "她对你有金融性的喜欢", 0, 1_000_000),
            ("w002", "扫描你第一", 1_080_000, 1_680_000),
            ("w003", "那么扫描你的经济", 1_760_000, 2_900_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [_segment(index, word_id, text, start, end) for index, (word_id, text, start, end) in enumerate(rows, start=1)]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        self.assertTrue(report["pre_visible_semantic_junk_audit_only"], report)
        self.assertFalse(report["pre_visible_semantic_junk_timeline_mutation_allowed"], report)
        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 1, report)
        candidate = report["pre_visible_semantic_junk_candidates"][0]
        self.assertEqual(candidate["type"], "aborted_restart")
        self.assertEqual(candidate["proposed_action"], "drop_fragment")
        self.assertFalse(candidate["provider_required"])

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertEqual([segment.text for segment in result.final_timeline], ["她对你有金融性的喜欢", "那么扫描你的经济"])
        self.assertEqual([caption.text for caption in result.captions], ["她对你有金融性的喜欢", "那么扫描你的经济"])
        self.assertEqual(result.report["pre_visible_semantic_junk_initial_candidate_count"], 1)
        self.assertEqual(result.report["pre_visible_semantic_junk_final_candidate_count"], 0)
        self.assertEqual(result.report["pre_visible_semantic_junk_repair_action_count"], 1)
        self.assertFalse(result.report["pre_visible_semantic_junk_audit_only"])
        self.assertTrue(result.report["pre_visible_semantic_junk_deterministic_apply_enabled"])
        action = result.report["pre_visible_semantic_junk_repair_actions"][0]
        self.assertEqual(action["decision"], "drop_high_confidence_semantic_junk_segment")
        self.assertEqual(action["dropped_word_ids"], ["w002"])

    def test_definition_boundary_is_not_semantic_junk(self) -> None:
        rows = [
            ("w001", "先看这个概念", 0, 700_000),
            ("w002", "这都叫释放信号", 780_000, 1_500_000),
            ("w003", "释放信号等于越界意图", 1_580_000, 2_700_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [_segment(index, word_id, text, start, end) for index, (word_id, text, start, end) in enumerate(rows, start=1)]
        captions = SubtitleRenderer().render(timeline, source_graph)

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 0, report)

    def test_nominal_fragment_completing_previous_identity_clause_is_not_dropped(self) -> None:
        rows = [
            ("w001", "他", 0, 40_000),
            ("w002", "这", 80_000, 200_000),
            ("w003", "具", 280_000, 320_000),
            ("w004", "肉体", 440_000, 680_000),
            ("w005", "就是", 840_000, 1_200_000),
            ("w006", "他", 1_200_000, 1_320_000),
            ("w007", "手里", 1_333_333, 1_653_333),
            ("w008", "唯一", 1_733_333, 1_973_333),
            ("w009", "能", 1_973_333, 2_173_333),
            ("w010", "上桌", 2_173_333, 2_453_333),
            ("w011", "的", 2_533_333, 2_693_333),
            ("w012", "筹码", 2_693_333, 3_166_667),
            ("w013", "他", 3_466_667, 3_546_667),
            ("w014", "就是", 3_626_667, 3_866_667),
            ("w015", "要", 3_866_667, 3_986_667),
            ("w016", "用", 3_986_667, 4_146_667),
            ("w017", "这个", 4_266_667, 4_500_000),
            ("w018", "唯一", 4_500_000, 4_700_000),
            ("w019", "的", 4_700_000, 4_820_000),
            ("w020", "筹码", 4_860_000, 5_020_000),
            ("w021", "去", 5_060_000, 5_220_000),
            ("w022", "唯一", 5_500_000, 5_700_000),
            ("w023", "的", 5_700_000, 5_820_000),
            ("w024", "筹码", 5_860_000, 6_020_000),
            ("w025", "去", 6_060_000, 6_220_000),
            ("w026", "加", 6_220_000, 6_380_000),
            ("w027", "杠杆", 6_460_000, 6_860_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [
            _multi_segment(1, [f"w{index:03d}" for index in range(1, 7)], "他这具肉体就是他", 0, 1_320_000),
            _multi_segment(
                2,
                [f"w{index:03d}" for index in range(7, 13)],
                "手里唯一能上桌的筹码",
                1_333_333,
                3_166_667,
            ),
            _multi_segment(
                3,
                [f"w{index:03d}" for index in range(22, 28)],
                "唯一的筹码去加杠杆",
                5_500_000,
                6_860_000,
            ),
        ]
        captions = [
            CaptionRenderUnit(
                caption_id="v21_cap_000001",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=[f"w{index:03d}" for index in range(1, 7)],
                text="他这具肉体就是他",
                target_start_us=0,
                target_end_us=1_320_000,
                source_subtitle_uids=[f"s{index:03d}" for index in range(1, 7)],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000002",
                timeline_segment_ids=["v21_seg_000002"],
                word_ids=[f"w{index:03d}" for index in range(7, 13)],
                text="手里唯一能上桌的筹码",
                target_start_us=1_333_333,
                target_end_us=3_166_667,
                source_subtitle_uids=[f"s{index:03d}" for index in range(7, 13)],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000002",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000003",
                timeline_segment_ids=["v21_seg_000003"],
                word_ids=[f"w{index:03d}" for index in range(22, 28)],
                text="唯一的筹码去加杠杆",
                target_start_us=5_500_000,
                target_end_us=6_860_000,
                source_subtitle_uids=[f"s{index:03d}" for index in range(22, 28)],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000003",
            ),
        ]

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 0, report)
        self.assertTrue(
            any(
                row["type"] == "lookahead_nominal_restart_fragment"
                and row["proposed_action"] == "keep"
                and row["visible_text"] == "手里唯一能上桌的筹码"
                for row in report["pre_visible_semantic_junk_candidates"]
            ),
            report,
        )

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda _repaired: captions,
        )

        visible = "".join(caption.text for caption in result.captions)
        self.assertIn("他这具肉体就是他手里唯一能上桌的筹码", visible)
        self.assertIn("唯一的筹码去加杠杆", visible)

    def test_repairs_standalone_topic_prefix_restart_before_longer_caption(self) -> None:
        rows = [
            ("w001", "享受到的快乐给抹平吗", 0, 1_200_000),
            ("w002", "集美", 1_260_000, 1_820_000),
            ("w003", "我说集美比你们更懂得爱自己", 1_900_000, 3_400_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [_segment(index, word_id, text, start, end) for index, (word_id, text, start, end) in enumerate(rows, start=1)]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 1, report)
        candidate = report["pre_visible_semantic_junk_candidates"][0]
        self.assertEqual(candidate["type"], "standalone_topic_prefix_restart")
        self.assertEqual(candidate["proposed_action"], "drop_fragment")

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertEqual(
            [segment.text for segment in result.final_timeline],
            ["享受到的快乐给抹平吗", "我说集美比你们更懂得爱自己"],
        )
        self.assertEqual(
            [caption.text for caption in result.captions],
            ["享受到的快乐给抹平吗", "我说集美比你们更懂得爱自己"],
        )
        action = result.report["pre_visible_semantic_junk_repair_actions"][0]
        self.assertEqual(action["candidate_type"], "standalone_topic_prefix_restart")
        self.assertEqual(action["dropped_word_ids"], ["w002"])

    def test_repairs_adjacent_suffix_semantic_recurrence_by_dropping_shorter_tail(self) -> None:
        rows = [
            ("w001", "你真的以为几个字就能把人家实打实的快乐给抹平吗", 0, 2_400_000),
            ("w002", "享受到的快乐给抹平吗", 2_480_000, 3_600_000),
            ("w003", "下一句继续推进观点", 3_680_000, 4_600_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [_segment(index, word_id, text, start, end) for index, (word_id, text, start, end) in enumerate(rows, start=1)]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 1, report)
        candidate = report["pre_visible_semantic_junk_candidates"][0]
        self.assertEqual(candidate["type"], "adjacent_suffix_semantic_recurrence")
        self.assertEqual(candidate["proposed_action"], "drop_fragment")

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertEqual(
            [segment.text for segment in result.final_timeline],
            ["你真的以为几个字就能把人家实打实的快乐给抹平吗", "下一句继续推进观点"],
        )
        self.assertEqual(
            [caption.text for caption in result.captions],
            ["你真的以为几个字就能把人家实打实的快乐给抹平吗", "下一句继续推进观点"],
        )
        action = result.report["pre_visible_semantic_junk_repair_actions"][0]
        self.assertEqual(action["candidate_type"], "adjacent_suffix_semantic_recurrence")
        self.assertEqual(action["dropped_word_ids"], ["w002"])

    def test_repairs_adjacent_reordered_semantic_restart_by_dropping_less_complete_left_caption(self) -> None:
        rows = [
            ("w001", "前面观点继续推进", 0, 800_000),
            ("w002", "而如果国男被扔进去", 880_000, 1_900_000),
            ("w003", "而国南如果被扔进这个场景", 1_900_000, 3_100_000),
            ("w004", "后面解释具体后果", 3_180_000, 4_100_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [_segment(index, word_id, text, start, end) for index, (word_id, text, start, end) in enumerate(rows, start=1)]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        candidates = report["pre_visible_semantic_junk_candidates"]
        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 1, report)
        self.assertEqual(candidates[0]["type"], "adjacent_reordered_semantic_restart")
        self.assertEqual(candidates[0]["proposed_action"], "drop_fragment")

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertEqual(
            [segment.text for segment in result.final_timeline],
            ["前面观点继续推进", "而国南如果被扔进这个场景", "后面解释具体后果"],
        )
        self.assertEqual(
            [caption.text for caption in result.captions],
            ["前面观点继续推进", "而国南如果被扔进这个场景", "后面解释具体后果"],
        )
        action = result.report["pre_visible_semantic_junk_repair_actions"][0]
        self.assertEqual(action["candidate_type"], "adjacent_reordered_semantic_restart")
        self.assertEqual(action["dropped_word_ids"], ["w002"])

    def test_reordered_semantic_restart_keeps_clear_gender_contrast(self) -> None:
        rows = [
            ("w001", "如果女生被扔进去", 0, 900_000),
            ("w002", "如果男生被扔进这个场景", 980_000, 2_100_000),
            ("w003", "后面继续比较差异", 2_180_000, 3_000_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [_segment(index, word_id, text, start, end) for index, (word_id, text, start, end) in enumerate(rows, start=1)]
        captions = SubtitleRenderer().render(timeline, source_graph)

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 0, report)
        self.assertFalse(
            any(row["proposed_action"] == "drop_fragment" for row in report["pre_visible_semantic_junk_candidates"]),
            report,
        )

    def test_enumeration_structure_is_not_semantic_junk(self) -> None:
        rows = [
            ("w001", "她有几个判断标准", 0, 800_000),
            ("w002", "第一观察你", 880_000, 1_500_000),
            ("w003", "第二观察你的消费", 1_580_000, 2_600_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [_segment(index, word_id, text, start, end) for index, (word_id, text, start, end) in enumerate(rows, start=1)]
        captions = SubtitleRenderer().render(timeline, source_graph)

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 0, report)

    def test_parallel_object_scan_structure_is_not_auto_dropped(self) -> None:
        rows = [
            ("w001", "她会持续判断你", 0, 800_000),
            ("w002", "她扫描你的脸", 880_000, 1_600_000),
            ("w003", "扫描你的阶层", 1_680_000, 2_500_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [_segment(index, word_id, text, start, end) for index, (word_id, text, start, end) in enumerate(rows, start=1)]
        captions = SubtitleRenderer().render(timeline, source_graph)

        report = build_pre_visible_semantic_junk_candidate_report(captions, source_graph)

        self.assertEqual(report["pre_visible_semantic_junk_high_confidence_candidate_count"], 0, report)
        self.assertFalse(
            any(row["proposed_action"] == "drop_fragment" for row in report["pre_visible_semantic_junk_candidates"]),
            report,
        )

    def test_repairs_connector_filler_restart_by_trimming_first_connector_and_filler(self) -> None:
        rows = [
            ("w001", "但", 0, 120_000),
            ("w002", "哪", 240_000, 300_000),
            ("w003", "但", 460_000, 660_000),
            ("w004", "人家", 660_000, 860_000),
            ("w005", "集美", 980_000, 1_220_000),
            ("w006", "哪怕", 1_300_000, 1_520_000),
            ("w007", "是", 1_520_000, 1_640_000),
            ("w008", "去", 1_640_000, 1_800_000),
            ("w009", "拼", 1_800_000, 1_940_000),
        ]
        source_graph = _graph_for_rows(rows)
        timeline = [
            _multi_segment(
                1,
                [word_id for word_id, _text, _start, _end in rows],
                "但哪但人家集美哪怕是去拼",
                0,
                1_940_000,
            )
        ]
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertEqual(result.final_timeline[0].word_ids[:2], ["w003", "w004"])
        self.assertEqual(result.final_timeline[0].text, "但人家集美哪怕是去拼")
        self.assertEqual([caption.text for caption in result.captions], ["但人家集美哪怕是去拼"])
        action = next(
            action
            for action in result.report["final_visible_repair_actions"]
            if action["issue_type"] == "connector_filler_restart"
        )
        self.assertEqual(action["decision"], "trim_connector_filler_before_restart")
        self.assertEqual(action["dropped_word_ids"], ["w001", "w002"])


if __name__ == "__main__":
    unittest.main()
