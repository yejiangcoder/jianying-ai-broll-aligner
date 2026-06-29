from __future__ import annotations

import unittest

from aroll_v21.ir.models import (
    CanonicalSourceGraph,
    CanonicalWord,
    FinalTimelineSegment,
    SourceGraphInvariantReport,
)
from aroll_v21.quality.final_visible_repair.rules.word_span_edit import _trim_word_ids_from_timeline


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


def _segment(word_ids: list[str], text: str, start_us: int, end_us: int) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id="v21_seg_000001",
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


class FinalVisibleWordSpanEditTest(unittest.TestCase):
    def test_trim_refuses_to_drop_open_predicate_bridge_with_source_context(self) -> None:
        graph = _graph(
            [
                _word("w001", "必须", 0, 260_000),
                _word("w002", "具备", 260_000, 620_000),
                _word("w003", "明显", 700_000, 1_200_000),
            ]
        )
        timeline = [_segment(["w001", "w002", "w003"], "必须具备明显", 0, 1_200_000)]

        repaired = _trim_word_ids_from_timeline(timeline, graph, ["w001", "w002"])

        self.assertIsNone(repaired)

    def test_trim_still_drops_unprotected_prefix(self) -> None:
        graph = _graph(
            [
                _word("w001", "重复", 0, 300_000),
                _word("w002", "正句", 300_000, 1_300_000),
            ]
        )
        timeline = [_segment(["w001", "w002"], "重复正句", 0, 1_300_000)]

        repaired = _trim_word_ids_from_timeline(timeline, graph, ["w001"])

        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual([segment.text for segment in repaired], ["正句"])
        self.assertEqual(repaired[0].word_ids, ["w002"])


if __name__ == "__main__":
    unittest.main()
