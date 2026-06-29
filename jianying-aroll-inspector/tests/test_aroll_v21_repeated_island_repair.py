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
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
from aroll_v21.quality.final_visible_repair import apply_timeline_repair_proposal
from aroll_v21.quality.final_visible_repair.rules.repeated_island import (
    build_repeated_island_proposals,
    detect_repeated_island_candidates,
)
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


def _graph_for_tokens(tokens: list[str], *, step_us: int = 200_000) -> CanonicalSourceGraph:
    words = [
        _word(index, text, (index - 1) * step_us, index * step_us)
        for index, text in enumerate(tokens, start=1)
    ]
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


def _single_segment(graph: CanonicalSourceGraph, *, segment_id: str = "v21_seg_000001") -> FinalTimelineSegment:
    word_ids = [word.word_id for word in graph.words]
    return FinalTimelineSegment(
        segment_id=segment_id,
        source_material_id="video",
        source_segment_id="src",
        source_start_us=int(graph.words[0].source_start_us),
        source_end_us=int(graph.words[-1].source_end_us),
        target_start_us=int(graph.words[0].source_start_us),
        target_end_us=int(graph.words[-1].source_end_us),
        word_ids=word_ids,
        text="".join(word.text for word in graph.words),
        decision_ids=[],
    )


def _high_confidence_fixture() -> tuple[CanonicalSourceGraph, list[FinalTimelineSegment]]:
    graph = _graph_for_tokens(["她", "说", "她", "是", "其", "实", "女", "人", "她", "是", "嘴", "上"])
    return graph, [_single_segment(graph)]


class _BadCoverageRenderer:
    def render(self, final_timeline, source_graph):  # type: ignore[no-untyped-def]
        segment = final_timeline[0]
        return [
            CaptionRenderUnit(
                caption_id="v21_cap_000001",
                timeline_segment_ids=[segment.segment_id],
                word_ids=list(segment.word_ids[:1]),
                text=str(segment.text or "")[:1],
                target_start_us=int(segment.target_start_us),
                target_end_us=int(segment.target_end_us),
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                containing_video_segment_id=segment.segment_id,
            )
        ]


def test_repeated_island_high_confidence_generates_proposal() -> None:
    graph, timeline = _high_confidence_fixture()

    proposals = build_repeated_island_proposals(timeline, graph)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.issue_type == "repeated_island"
    assert proposal.repair_action == "internal_drop"
    assert proposal.target_segment_id == "v21_seg_000001"
    assert proposal.target_word_ids == ["w003", "w004"]
    assert proposal.target_text == "她是"
    assert proposal.evidence["confidence"] == "high"
    assert proposal.evidence["first_word_ids"] == ["w003", "w004"]
    assert proposal.evidence["second_word_ids"] == ["w009", "w010"]
    assert proposal.evidence["after_second_text"] == "嘴上"


def test_repeated_island_high_confidence_repair_drops_first_island_and_keeps_second() -> None:
    graph, timeline = _high_confidence_fixture()
    renderer = SubtitleRenderer()
    captions = renderer.render(timeline, graph)

    result = repair_final_visible_caption_issues(
        final_timeline=timeline,
        captions=captions,
        source_graph=graph,
        render_captions=lambda rows: renderer.render(rows, graph),
    )

    repaired_text = "".join(segment.text for segment in result.final_timeline)
    assert repaired_text == "她说其实女人她是嘴上"
    assert repaired_text.count("她是") == 1
    assert len(result.final_timeline) == 2
    assert result.final_timeline[0].word_ids == ["w001", "w002"]
    assert result.final_timeline[1].word_ids == ["w005", "w006", "w007", "w008", "w009", "w010", "w011", "w012"]
    actions = [action for action in result.report["final_visible_repair_actions"] if action["issue_type"] == "repeated_island"]
    assert len(actions) == 1
    assert actions[0]["decision"] == "internal_drop"
    assert actions[0]["target_word_ids"] == ["w003", "w004"]
    assert result.report["repeated_island_repair_action_count"] == 1


def test_repeated_island_source_boundary_restart_drops_abandoned_middle_fragment() -> None:
    graph = _graph_for_tokens(["他", "说", "需要", "旧方案", "需要", "新方案"])
    words = [
        replace(word, subtitle_uid="s_intro", subtitle_index=1)
        if index < 2
        else replace(word, subtitle_uid="s_aborted", subtitle_index=2)
        if index < 4
        else replace(word, subtitle_uid="s_restart", subtitle_index=3)
        for index, word in enumerate(graph.words)
    ]
    graph = replace(graph, words=words)
    timeline = [_single_segment(graph)]
    renderer = SubtitleRenderer()

    proposals = build_repeated_island_proposals(timeline, graph)

    assert len(proposals) == 1
    assert proposals[0].target_word_ids == ["w003", "w004"]
    assert proposals[0].target_text == "需要旧方案"
    assert proposals[0].evidence["proposal_policy"] == "drop_first_island_and_middle_fragment_keep_source_boundary_restart"

    result = repair_final_visible_caption_issues(
        final_timeline=timeline,
        captions=renderer.render(timeline, graph),
        source_graph=graph,
        render_captions=lambda rows: renderer.render(rows, graph),
    )

    repaired_text = "".join(segment.text for segment in result.final_timeline)
    assert repaired_text == "他说需要新方案"
    actions = [action for action in result.report["final_visible_repair_actions"] if action["issue_type"] == "repeated_island"]
    assert actions[0]["target_word_ids"] == ["w003", "w004"]


def test_repeated_island_replay_repairs_middle_fragment_after_prior_island_drop() -> None:
    graph = _graph_for_tokens(["他", "说", "需要", "旧方案", "需要", "新方案"])
    words = [
        replace(word, subtitle_uid="s_intro", subtitle_index=1)
        if index < 2
        else replace(word, subtitle_uid="s_aborted", subtitle_index=2)
        if index < 4
        else replace(word, subtitle_uid="s_restart", subtitle_index=3)
        for index, word in enumerate(graph.words)
    ]
    graph = replace(graph, words=words)
    base = _single_segment(graph)
    timeline = [
        replace(
            base,
            segment_id="v21_seg_000001",
            source_start_us=0,
            source_end_us=400_000,
            target_start_us=0,
            target_end_us=400_000,
            word_ids=["w001", "w002"],
            text="他说",
        ),
        replace(
            base,
            segment_id="v21_seg_000002",
            source_start_us=600_000,
            source_end_us=1_200_000,
            target_start_us=400_000,
            target_end_us=1_000_000,
            word_ids=["w004", "w005", "w006"],
            text="旧方案需要新方案",
        ),
    ]
    renderer = SubtitleRenderer()

    proposals = build_repeated_island_proposals(timeline, graph)

    assert len(proposals) == 1
    assert proposals[0].target_word_ids == ["w004"]
    assert proposals[0].target_text == "旧方案"
    assert proposals[0].evidence["proposal_policy"] == "trim_abandoned_middle_fragment_left_after_prior_restart_island_drop"

    result = repair_final_visible_caption_issues(
        final_timeline=timeline,
        captions=renderer.render(timeline, graph),
        source_graph=graph,
        render_captions=lambda rows: renderer.render(rows, graph),
    )

    repaired_text = "".join(segment.text for segment in result.final_timeline)
    assert repaired_text == "他说需要新方案"


def test_repeated_island_repair_rerenders_and_preserves_caption_word_coverage() -> None:
    graph, timeline = _high_confidence_fixture()
    renderer = SubtitleRenderer()

    result = repair_final_visible_caption_issues(
        final_timeline=timeline,
        captions=renderer.render(timeline, graph),
        source_graph=graph,
        render_captions=lambda rows: renderer.render(rows, graph),
    )

    coverage = build_caption_alignment_report(final_timeline=result.final_timeline, captions=result.captions)
    assert result.captions
    assert coverage["missing_final_timeline_caption_word_count"] == 0
    assert coverage["prewrite_uncaptioned_spoken_word_count"] == 0


def test_repeated_island_without_word_ids_does_not_auto_apply() -> None:
    graph, timeline = _high_confidence_fixture()
    without_word_ids = [replace(timeline[0], word_ids=[])]

    proposals = build_repeated_island_proposals(without_word_ids, graph)

    assert proposals == []


def test_repeated_island_medium_confidence_reports_warning_without_apply() -> None:
    graph = _graph_for_tokens(["我", "们", "讨", "论", "他", "们", "我", "们", "继", "续"])
    timeline = [_single_segment(graph)]
    renderer = SubtitleRenderer()

    result = repair_final_visible_caption_issues(
        final_timeline=timeline,
        captions=renderer.render(timeline, graph),
        source_graph=graph,
        render_captions=lambda rows: renderer.render(rows, graph),
    )

    assert build_repeated_island_proposals(timeline, graph) == []
    assert result.final_timeline == timeline
    assert result.report["repeated_island_medium_confidence_count"] >= 1
    assert result.report["repeated_island_warning_count"] >= 1
    assert result.report["repeated_island_repair_action_count"] == 0


def test_repeated_island_low_confidence_allow() -> None:
    graph = _graph_for_tokens(["我", "说", "我", "去"])
    timeline = [_single_segment(graph)]
    candidates = detect_repeated_island_candidates(timeline, graph)

    assert any(candidate.confidence == "low" for candidate in candidates)
    assert build_repeated_island_proposals(timeline, graph) == []


def test_repeated_island_repairs_leading_discourse_particle_restart() -> None:
    graph = _graph_for_tokens(["还", "那", "还", "说", "个", "der"])
    timeline = [_single_segment(graph)]
    renderer = SubtitleRenderer()

    proposals = build_repeated_island_proposals(timeline, graph)

    assert len(proposals) == 1
    assert proposals[0].target_word_ids == ["w001"]
    assert proposals[0].target_text == "还"
    assert proposals[0].evidence["confidence"] == "high"

    result = repair_final_visible_caption_issues(
        final_timeline=timeline,
        captions=renderer.render(timeline, graph),
        source_graph=graph,
        render_captions=lambda rows: renderer.render(rows, graph),
    )

    assert "".join(segment.text for segment in result.final_timeline) == "那还说个der"
    actions = [action for action in result.report["final_visible_repair_actions"] if action["issue_type"] == "repeated_island"]
    assert actions[0]["target_word_ids"] == ["w001"]


def test_repeated_island_single_char_topic_reuse_stays_low_confidence() -> None:
    graph = _graph_for_tokens(["人", "和", "人", "对话"])
    timeline = [_single_segment(graph)]

    candidates = detect_repeated_island_candidates(timeline, graph)

    assert any(candidate.confidence == "low" for candidate in candidates)
    assert build_repeated_island_proposals(timeline, graph) == []


def test_repeated_island_a_not_a_structure_not_repaired() -> None:
    graph = _graph_for_tokens(["去", "不", "去", "看", "看"])
    timeline = [_single_segment(graph)]
    candidates = detect_repeated_island_candidates(timeline, graph)

    assert any("a_not_a_structure" in candidate.risk_tags for candidate in candidates)
    assert build_repeated_island_proposals(timeline, graph) == []


def test_repeated_island_adjacent_reduplication_not_repaired() -> None:
    graph = _graph_for_tokens(["人", "人", "都", "来"])
    timeline = [_single_segment(graph)]

    assert detect_repeated_island_candidates(timeline, graph) == []
    assert build_repeated_island_proposals(timeline, graph) == []


def test_repeated_island_definition_or_emphasis_not_repaired() -> None:
    graph = _graph_for_tokens(["自由", "就是", "自由", "不是", "任性"])
    timeline = [_single_segment(graph)]
    candidates = detect_repeated_island_candidates(timeline, graph)

    assert any("definition_or_emphasis_structure" in candidate.risk_tags for candidate in candidates)
    assert build_repeated_island_proposals(timeline, graph) == []


def test_repeated_island_topic_reuse_not_repaired() -> None:
    graph = _graph_for_tokens(["算法", "影响", "很多", "算法", "需要", "审计"])
    timeline = [_single_segment(graph)]
    candidates = detect_repeated_island_candidates(timeline, graph)

    assert any(candidate.confidence == "medium" for candidate in candidates)
    assert build_repeated_island_proposals(timeline, graph) == []


def test_internal_drop_does_not_fake_continuous_source_span() -> None:
    graph, timeline = _high_confidence_fixture()
    proposal = build_repeated_island_proposals(timeline, graph)[0]

    result = apply_timeline_repair_proposal(proposal, timeline, graph)

    assert result.applied is True
    assert len(result.final_timeline) == 2
    left, right = result.final_timeline
    assert left.source_start_us == 0
    assert left.source_end_us == 400_000
    assert right.source_start_us == 800_000
    assert right.source_end_us == 2_400_000
    assert right.source_start_us > left.source_end_us


def test_repeated_island_coverage_mismatch_fail_closed() -> None:
    graph, timeline = _high_confidence_fixture()
    proposal = build_repeated_island_proposals(timeline, graph)[0]

    result = apply_timeline_repair_proposal(
        proposal,
        timeline,
        graph,
        renderer=_BadCoverageRenderer(),  # type: ignore[arg-type]
    )

    assert result.applied is False
    assert result.blocker_code == "V21_TIMELINE_REPAIR_CAPTION_WORD_COVERAGE_FAILED"
    assert result.coverage_report["missing_final_timeline_caption_word_count"] > 0
