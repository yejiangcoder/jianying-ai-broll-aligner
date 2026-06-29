from __future__ import annotations

from dataclasses import replace

from aroll_v21.ir.models import (
    CanonicalSourceGraph,
    CanonicalWord,
    CaptionRenderUnit,
    FinalTimelineSegment,
    SourceGraphInvariantReport,
)
from aroll_v21.quality.caption_alignment import build_caption_alignment_report
from aroll_v21.render.subtitle_renderer import SubtitleRenderer, _cleanup_caption_units


def _word(index: int, text: str, start_us: int, end_us: int) -> CanonicalWord:
    return CanonicalWord(
        word_id=f"w{index}",
        text=text,
        normalized_text=text,
        source_start_us=start_us,
        source_end_us=end_us,
        source_material_id="video",
        source_segment_id="src",
        subtitle_uid=f"s{index}",
        subtitle_index=index,
        char_start=None,
        char_end=None,
        confidence=1.0,
        is_cuttable_left=True,
        is_cuttable_right=True,
    )


def _source_graph() -> CanonicalSourceGraph:
    return CanonicalSourceGraph(
        words=[
            _word(1, "甲", 0, 400_000),
            _word(2, "乙", 400_000, 800_000),
            _word(3, "丙", 800_000, 1_200_000),
        ],
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


def _segment() -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id="v21_seg_000001",
        source_material_id="video",
        source_segment_id="src",
        source_start_us=0,
        source_end_us=1_200_000,
        target_start_us=0,
        target_end_us=1_200_000,
        word_ids=["w1", "w2", "w3"],
        text="甲乙丙",
        decision_ids=[],
    )


def _caption(index: int, word_ids: list[str], start_us: int, end_us: int) -> CaptionRenderUnit:
    return CaptionRenderUnit(
        caption_id=f"v21_cap_{index:06d}",
        timeline_segment_ids=["v21_seg_000001"],
        word_ids=list(word_ids),
        text="".join(word_ids),
        target_start_us=start_us,
        target_end_us=end_us,
        source_subtitle_uids=[f"s{index}"],
        style_template_id="canonical_caption_template",
        containing_video_segment_id="v21_seg_000001",
    )


def test_caption_cleanup_does_not_reduce_word_coverage() -> None:
    segment = replace(_segment(), word_ids=["w1", "w2"], text="甲乙", source_end_us=800_000, target_end_us=800_000)
    captions = [
        _caption(1, ["w1"], 0, 200_000),
        _caption(2, ["w2"], 200_000, 800_000),
    ]

    cleaned = _cleanup_caption_units(captions, {segment.segment_id: segment})

    before_word_ids = {word_id for caption in captions for word_id in caption.word_ids}
    after_word_ids = {word_id for caption in cleaned for word_id in caption.word_ids}
    assert after_word_ids == before_word_ids


def test_final_timeline_spoken_word_missing_from_caption_word_ids_blocks() -> None:
    segment = _segment()
    captions = [_caption(1, ["w1", "w2"], 0, 800_000)]

    report = build_caption_alignment_report(final_timeline=[segment], captions=captions)

    assert report["gate_passed"] is False
    assert report["prewrite_uncaptioned_spoken_word_count"] == 1
    assert report["missing_final_timeline_caption_word_count"] == 1
    assert report["missing_final_timeline_caption_word_ids"] == ["w3"]
    assert "V21_FINAL_TIMELINE_CAPTION_WORD_COVERAGE_FAILED" in report["blocker_codes"]


def test_renderer_cleanup_preserves_final_timeline_caption_word_coverage() -> None:
    source_graph = _source_graph()
    segment = _segment()

    captions = SubtitleRenderer().render([segment], source_graph)
    report = build_caption_alignment_report(final_timeline=[segment], captions=captions)

    assert report["prewrite_uncaptioned_spoken_word_count"] == 0
    assert report["missing_final_timeline_caption_word_count"] == 0
    assert "V21_FINAL_TIMELINE_CAPTION_WORD_COVERAGE_FAILED" not in report["blocker_codes"]


def test_renderer_rebalances_ascii_label_before_open_predicate_caption() -> None:
    words = [
        CanonicalWord(f"w{index:02d}", text, text, start, end, "video", "src", uid, subtitle_index, None, None, 1.0, True, True)
        for index, (text, start, end, uid, subtitle_index) in enumerate(
            [
                ("从来", 0, 280_000, "s1", 1),
                ("不是", 280_000, 560_000, "s1", 1),
                ("为", 560_000, 680_000, "s1", 1),
                ("了", 680_000, 760_000, "s1", 1),
                ("乞求", 800_000, 1_080_000, "s1", 1),
                ("被", 1_080_000, 1_260_000, "s1", 1),
                ("爱", 1_260_000, 1_600_000, "s1", 1),
                ("Pro", 1_660_000, 1_820_000, "s1", 1),
                ("是", 2_180_000, 2_460_000, "s2", 2),
                ("为", 2_460_000, 2_580_000, "s2", 2),
                ("了", 2_580_000, 2_900_000, "s2", 2),
                ("Pro", 3_200_000, 3_360_000, "s3", 3),
                ("阶段", 3_480_000, 3_720_000, "s3", 3),
                ("已经", 3_860_000, 4_140_000, "s3", 3),
                ("开", 4_140_000, 4_580_000, "s3", 3),
            ],
            start=1,
        )
    ]
    graph = CanonicalSourceGraph(
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
    timeline = [
        FinalTimelineSegment("v21_seg_000001", "video", "src", 0, 1_820_000, 0, 1_820_000, [f"w{i:02d}" for i in range(1, 9)], "从来不是为了乞求被爱Pro", []),
        FinalTimelineSegment("v21_seg_000002", "video", "src", 2_180_000, 2_900_000, 1_820_000, 2_540_000, [f"w{i:02d}" for i in range(9, 12)], "是为了", []),
        FinalTimelineSegment("v21_seg_000003", "video", "src", 3_200_000, 4_580_000, 2_540_000, 3_920_000, [f"w{i:02d}" for i in range(12, 16)], "Pro阶段已经开", []),
    ]

    captions = SubtitleRenderer().render(timeline, graph)
    report = build_caption_alignment_report(final_timeline=timeline, captions=captions)

    assert [caption.text for caption in captions[:3]] == ["从来不是为了乞求被爱", "Pro是为了", "Pro阶段已经开"]
    assert captions[1].word_ids == ["w08", "w09", "w10", "w11"]
    assert captions[1].timeline_segment_ids == ["v21_seg_000001", "v21_seg_000002"]
    assert report["gate_passed"] is True
    assert report["prewrite_uncaptioned_spoken_word_count"] == 0
    assert report["missing_final_timeline_caption_word_count"] == 0
