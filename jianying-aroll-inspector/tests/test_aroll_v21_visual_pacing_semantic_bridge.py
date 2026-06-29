from __future__ import annotations

import unittest

from aroll_v21.ir.models import (
    CanonicalSourceGraph,
    CanonicalWord,
    FinalTimelineSegment,
    SourceGraphInvariantReport,
)
from aroll_v21.quality.visual_pacing.normalizer import VisualPacingNormalizer


def _word(word_id: str, text: str, start_us: int, end_us: int) -> CanonicalWord:
    return CanonicalWord(
        word_id=word_id,
        text=text,
        normalized_text=text,
        source_start_us=start_us,
        source_end_us=end_us,
        source_material_id="main",
        source_segment_id="primary",
        subtitle_uid="s001",
        subtitle_index=1,
        char_start=None,
        char_end=None,
        confidence=0.99,
        is_cuttable_left=True,
        is_cuttable_right=True,
    )


def _graph(words: list[CanonicalWord]) -> CanonicalSourceGraph:
    return CanonicalSourceGraph(
        words=words,
        edit_units=[],
        subtitle_rows=[],
        source_materials=[],
        source_segments=[
            {
                "id": "primary",
                "material_id": "main",
                "type": "video",
                "source_start_us": 0,
                "source_end_us": 10_000_000,
            }
        ],
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


def _segment(segment_id: str, word_ids: list[str], text: str, start_us: int, end_us: int) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=segment_id,
        source_material_id="main",
        source_segment_id="primary",
        source_start_us=start_us,
        source_end_us=end_us,
        target_start_us=start_us,
        target_end_us=end_us,
        word_ids=word_ids,
        text=text,
        decision_ids=[],
    )


class VisualPacingSemanticBridgeTest(unittest.TestCase):
    def test_safe_semantic_bridge_exception_is_merged_before_gate(self) -> None:
        graph = _graph(
            [
                _word("w001", "语义桥", 0, 700_000),
                _word("w002", "继续内容", 700_000, 2_000_000),
            ]
        )
        timeline = [
            _segment("v21_seg_000001", ["w001"], "语义桥", 0, 700_000),
            _segment("v21_seg_000002", ["w002"], "继续内容", 700_000, 2_000_000),
        ]
        normalizer = VisualPacingNormalizer()

        merged, merged_count, unsafe_count = normalizer._merge_safe_semantic_bridge_exceptions(
            timeline,
            graph,
            [("primary", 0, 2_000_000)],
            {word.word_id: word for word in graph.words},
        )

        self.assertEqual(unsafe_count, 0)
        self.assertEqual(merged_count, 1)
        self.assertEqual([segment.text for segment in merged], ["语义桥继续内容"])


if __name__ == "__main__":
    unittest.main()
