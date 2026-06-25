from __future__ import annotations

import unittest

from aroll_v21.compiler.final_timeline_compiler import FinalTimelineCompiler
from aroll_v21.compiler.suffix_cleanup import _repeated_suffix_island_start as compiler_repeated_suffix_start
from aroll_v21.decision.final_target_repeat_resolver import FinalTargetRepeatResolver
from aroll_v21.ir import (
    CanonicalSourceGraph,
    CanonicalWord,
    DecisionPlan,
    FinalTimelineSegment,
    SourceGraphInvariantReport,
)
from aroll_v21.quality.visual_pacing.suffix_cleanup import _repeated_suffix_island_start as visual_repeated_suffix_start


def _word(index: int, text: str, start_us: int, end_us: int, subtitle_index: int) -> CanonicalWord:
    return CanonicalWord(
        word_id=f"w_{index:06d}",
        text=text,
        normalized_text=text,
        source_start_us=start_us,
        source_end_us=end_us,
        source_material_id="main",
        source_segment_id="clip",
        subtitle_uid=f"s_{subtitle_index:03d}",
        subtitle_index=subtitle_index,
        char_start=None,
        char_end=None,
        confidence=None,
        is_cuttable_left=True,
        is_cuttable_right=True,
    )


def _graph(words: list[CanonicalWord]) -> CanonicalSourceGraph:
    return CanonicalSourceGraph(
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
    )


def _segment(index: int, text: str, start_us: int, end_us: int, word_ids: list[str]) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=f"v21_seg_{index:06d}",
        source_material_id="main",
        source_segment_id="clip",
        source_start_us=start_us,
        source_end_us=end_us,
        target_start_us=start_us,
        target_end_us=end_us,
        word_ids=word_ids,
        text=text,
        decision_ids=[],
    )


class ArollV21QcRound0626Regressions(unittest.TestCase):
    def test_boundary_overlap_keeps_attributive_term_reused_as_definition_subject(self) -> None:
        words = [
            _word(1, "最后", 0, 200_000, 1),
            _word(2, "就是", 200_000, 400_000, 1),
            _word(3, "刀削般", 400_000, 700_000, 1),
            _word(4, "的", 700_000, 800_000, 1),
            _word(5, "下颌", 800_000, 1_000_000, 1),
            _word(6, "线", 1_000_000, 1_100_000, 1),
            _word(7, "下颌", 1_120_000, 1_320_000, 2),
            _word(8, "线", 1_320_000, 1_420_000, 2),
            _word(9, "就是", 1_420_000, 1_620_000, 2),
            _word(10, "男人", 1_620_000, 1_820_000, 2),
            _word(11, "脸上", 1_820_000, 2_020_000, 2),
            _word(12, "的", 2_020_000, 2_120_000, 2),
            _word(13, "冷兵器", 2_120_000, 2_520_000, 2),
        ]
        source_graph = _graph(words)
        segments = [
            _segment(1, "最后就是刀削般的下颌线", 0, 1_100_000, [word.word_id for word in words[:6]]),
            _segment(2, "下颌线就是男人脸上的冷兵器", 1_120_000, 2_520_000, [word.word_id for word in words[6:]]),
        ]

        cleaned, blockers = FinalTimelineCompiler()._final_cjk_boundary_suffix_prefix_overlap_cleanup(
            segments,
            source_graph,
            DecisionPlan(decisions=[]),
        )

        self.assertEqual(blockers, [])
        self.assertEqual([segment.text for segment in cleaned], ["最后就是刀削般的下颌线", "下颌线就是男人脸上的冷兵器"])

    def test_repeated_suffix_cleanup_keeps_coordinated_parallel_noun_phrase(self) -> None:
        tokens = ["骨骼", "优势", "和", "基因", "优势", "放大", "而已"]

        self.assertIsNone(visual_repeated_suffix_start(tokens))
        self.assertIsNone(compiler_repeated_suffix_start(object(), tokens))
        self.assertEqual(visual_repeated_suffix_start(["重复", "短语", "中间", "重复", "短语"]), 3)

    def test_final_target_repeat_keeps_segment_required_by_left_context(self) -> None:
        segments = [
            _segment(1, "男人只要进入", 0, 1_000_000, ["w_000001"]),
            _segment(2, "Pro的阶段", 1_120_000, 1_900_000, ["w_000002"]),
            _segment(3, "终点呢一定是剪裁", 2_000_000, 3_300_000, ["w_000003"]),
            _segment(4, "在Pro的阶段", 3_500_000, 4_700_000, ["w_000004"]),
        ]
        resolver = FinalTargetRepeatResolver()
        clusters = resolver._clusters(segments)

        self.assertTrue(
            any(cluster.get("recommended_drop_index") == 2 for cluster in clusters),
            clusters,
        )
        resolved, blockers = resolver.resolve(segments, DecisionPlan(decisions=[]))

        self.assertEqual(blockers, [])
        self.assertEqual(
            [segment.text for segment in resolved],
            ["男人只要进入", "Pro的阶段", "终点呢一定是剪裁", "在Pro的阶段"],
        )


if __name__ == "__main__":
    unittest.main()
