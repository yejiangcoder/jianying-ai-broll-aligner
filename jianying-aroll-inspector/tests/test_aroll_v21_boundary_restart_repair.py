from __future__ import annotations

from dataclasses import replace

from aroll_v21.ir.models import (
    CanonicalSourceGraph,
    CanonicalWord,
    FinalTimelineSegment,
    SourceGraphInvariantReport,
)
from aroll_v21.quality.caption_alignment import build_caption_alignment_report
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
from aroll_v21.quality.final_visible_repair.rules.boundary_restart import build_boundary_restart_proposals
from aroll_v21.render.subtitle_renderer import SubtitleRenderer


def _word(index: int, text: str, start_us: int, end_us: int) -> CanonicalWord:
    return CanonicalWord(
        word_id=f"w{index:03d}",
        text=text,
        normalized_text=text,
        source_start_us=start_us,
        source_end_us=end_us,
        source_material_id="video",
        source_segment_id="src",
        subtitle_uid=f"s{index:03d}",
        subtitle_index=index,
        char_start=None,
        char_end=None,
        confidence=1.0,
        is_cuttable_left=True,
        is_cuttable_right=True,
    )


def _graph_for_words(words: list[str], *, step_us: int = 200_000) -> CanonicalSourceGraph:
    rows = [
        _word(index, text, (index - 1) * step_us, index * step_us)
        for index, text in enumerate(words, start=1)
    ]
    return CanonicalSourceGraph(
        words=rows,
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


def _segment(segment_id: str, word_ids: list[str], start_us: int, end_us: int, text: str) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=segment_id,
        source_material_id="video",
        source_segment_id="src",
        source_start_us=start_us,
        source_end_us=end_us,
        target_start_us=start_us,
        target_end_us=end_us,
        word_ids=list(word_ids),
        text=text,
        decision_ids=[],
    )


def _boundary_restart_fixture() -> tuple[CanonicalSourceGraph, list[FinalTimelineSegment]]:
    graph = _graph_for_words(["咳", "咳", "就", "你", "骂", "就", "你", "骂", "集", "美", "虚", "容", "啊"])
    first = _segment(
        "v21_seg_000001",
        ["w001", "w002", "w003", "w004", "w005"],
        0,
        1_000_000,
        "咳咳就你骂",
    )
    second = _segment(
        "v21_seg_000002",
        ["w006", "w007", "w008", "w009", "w010", "w011", "w012", "w013"],
        1_000_000,
        2_600_000,
        "就你骂集美虚容啊",
    )
    return graph, [first, second]


def test_boundary_restart_generates_suffix_trim_proposal() -> None:
    graph, timeline = _boundary_restart_fixture()

    proposals = build_boundary_restart_proposals(timeline, graph)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.issue_type == "boundary_restart"
    assert proposal.repair_action == "suffix_trim"
    assert proposal.target_segment_id == "v21_seg_000001"
    assert proposal.target_word_ids == ["w003", "w004", "w005"]
    assert proposal.target_text == "就你骂"
    assert proposal.confidence >= 0.9
    assert proposal.evidence["prev_segment_id"] == "v21_seg_000001"
    assert proposal.evidence["next_segment_id"] == "v21_seg_000002"
    assert proposal.evidence["next_prefix_word_ids"] == ["w006", "w007", "w008"]
    assert proposal.evidence["gap_us"] == 0


def test_boundary_restart_repair_trims_previous_suffix_and_keeps_next_complete() -> None:
    graph, timeline = _boundary_restart_fixture()
    renderer = SubtitleRenderer()
    captions = renderer.render(timeline, graph)

    result = repair_final_visible_caption_issues(
        final_timeline=timeline,
        captions=captions,
        source_graph=graph,
        render_captions=lambda rows: renderer.render(rows, graph),
    )

    assert len(result.final_timeline) == 1
    assert result.final_timeline[0].word_ids == ["w006", "w007", "w008", "w009", "w010", "w011", "w012", "w013"]
    assert result.final_timeline[0].text == "就你骂集美虚容啊"
    actions = [action for action in result.report["final_visible_repair_actions"] if action["issue_type"] == "boundary_restart"]
    assert actions
    assert actions[0]["decision"] == "suffix_trim"
    assert actions[0]["target_segment_id"] == "v21_seg_000001"
    assert actions[0]["target_word_ids"] == ["w003", "w004", "w005"]
    tiny_actions = [action for action in result.report["final_visible_repair_actions"] if action["issue_type"] == "tiny_caption_residual"]
    assert tiny_actions
    assert tiny_actions[0]["target_word_ids"] == ["w001", "w002"]
    coverage = build_caption_alignment_report(final_timeline=result.final_timeline, captions=result.captions)
    assert coverage["missing_final_timeline_caption_word_count"] == 0
    assert coverage["prewrite_uncaptioned_spoken_word_count"] == 0


def test_boundary_restart_does_not_repair_non_adjacent_far_repeat() -> None:
    graph, timeline = _boundary_restart_fixture()
    far_second = replace(timeline[1], target_start_us=1_600_001, target_end_us=3_200_001)

    proposals = build_boundary_restart_proposals([timeline[0], far_second], graph)

    assert proposals == []


def test_boundary_restart_does_not_repair_far_topic_reuse() -> None:
    graph = _graph_for_words(["主", "题", "词", "很", "多", "主", "题", "词", "延", "伸"])
    first = _segment("v21_seg_000001", ["w001", "w002", "w003"], 0, 600_000, "主题词")
    second = _segment("v21_seg_000002", ["w006", "w007", "w008", "w009", "w010"], 1_200_001, 2_200_001, "主题词延伸")

    proposals = build_boundary_restart_proposals([first, second], graph)

    assert proposals == []


def test_boundary_restart_does_not_trim_command_tail_when_it_leaves_open_clause() -> None:
    graph = _graph_for_words(["立刻", "给", "老子", "关", "了", "老子", "最烦", "这种"])
    first = _segment("v21_seg_000001", ["w001", "w002", "w003", "w004", "w005"], 0, 1_000_000, "立刻给老子关了")
    second = _segment("v21_seg_000002", ["w006", "w007", "w008"], 1_000_000, 1_600_000, "老子最烦这种")

    proposals = build_boundary_restart_proposals([first, second], graph)

    assert proposals == []


def test_boundary_restart_does_not_trim_parallel_enumeration_tail() -> None:
    graph = _graph_for_words(["那么", "第二", "他", "要", "观察", "你", "的", "结构", "观察", "你", "的", "状态"])
    first = _segment(
        "v21_seg_000001",
        ["w001", "w002", "w003", "w004", "w005", "w006", "w007", "w008"],
        0,
        1_600_000,
        "那么第二他要观察你的结构",
    )
    second = _segment(
        "v21_seg_000002",
        ["w009", "w010", "w011", "w012"],
        1_600_000,
        2_400_000,
        "观察你的状态",
    )

    proposals = build_boundary_restart_proposals([first, second], graph)

    assert proposals == []


def test_boundary_restart_does_not_trim_attributive_term_reused_as_definition_subject() -> None:
    graph = _graph_for_words(["最后", "就是", "刀削般", "的", "下颌", "线", "下颌", "线", "就是", "男人", "脸上", "的", "冷兵器"])
    first = _segment(
        "v21_seg_000001",
        ["w001", "w002", "w003", "w004", "w005", "w006"],
        0,
        1_200_000,
        "最后就是刀削般的下颌线",
    )
    second = _segment(
        "v21_seg_000002",
        ["w007", "w008", "w009", "w010", "w011", "w012", "w013"],
        1_200_000,
        2_600_000,
        "下颌线就是男人脸上的冷兵器",
    )

    proposals = build_boundary_restart_proposals([first, second], graph)

    assert proposals == []


def test_boundary_restart_without_clear_word_ids_does_not_apply() -> None:
    graph, timeline = _boundary_restart_fixture()
    without_words = replace(timeline[0], word_ids=[])

    proposals = build_boundary_restart_proposals([without_words, timeline[1]], graph)

    assert proposals == []
