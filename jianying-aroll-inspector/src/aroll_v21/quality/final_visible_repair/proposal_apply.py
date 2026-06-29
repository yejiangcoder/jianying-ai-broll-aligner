from __future__ import annotations

from typing import Any, Callable

from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.final_visible_repair.proposal import TimelineRepairProposal
from aroll_v21.quality.final_visible_repair.report import _action
from aroll_v21.quality.final_visible_repair.result import _RepairStep
from aroll_v21.quality.final_visible_repair.rules import (
    boundary_restart as _boundary_restart_rules,
    repeated_island as _repeated_island_rules,
)
from aroll_v21.quality.final_visible_repair.timeline_materializer import (
    apply_timeline_repair_proposal as _apply_timeline_repair_proposal,
)


class RenderCallbackAdapter:
    def __init__(self, render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]]) -> None:
        self._render_captions = render_captions

    def render(
        self,
        final_timeline: list[FinalTimelineSegment],
        _source_graph: CanonicalSourceGraph,
    ) -> list[CaptionRenderUnit]:
        return self._render_captions(final_timeline)


def repair_boundary_restart_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    proposals = _boundary_restart_rules.build_boundary_restart_proposals(final_timeline, source_graph)
    return apply_first_timeline_repair_proposal(
        proposals=proposals,
        final_timeline=final_timeline,
        source_graph=source_graph,
        render_captions=render_captions,
        pass_index=pass_index,
        decision="suffix_trim",
    )


def repair_repeated_island_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    proposals = _repeated_island_rules.build_repeated_island_proposals(final_timeline, source_graph)
    return apply_first_timeline_repair_proposal(
        proposals=proposals,
        final_timeline=final_timeline,
        source_graph=source_graph,
        render_captions=render_captions,
        pass_index=pass_index,
        decision="internal_drop",
    )


def build_caption_span_drop_proposal(
    *,
    proposal_id: str,
    issue_type: str,
    confidence: float,
    repair_action: str,
    caption: CaptionRenderUnit,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    risk_tags: list[str],
    evidence: dict[str, Any],
) -> TimelineRepairProposal | None:
    target_segment_id = target_segment_id_for_caption(caption, final_timeline)
    if not target_segment_id:
        no_proposal: TimelineRepairProposal | None = None
        return no_proposal
    target_word_ids = [str(word_id) for word_id in caption.word_ids if str(word_id)]
    if not target_word_ids:
        no_proposal: TimelineRepairProposal | None = None
        return no_proposal
    segments_by_id = {segment.segment_id: segment for segment in final_timeline}
    target_segment = segments_by_id.get(target_segment_id)
    if target_segment is None:
        no_proposal: TimelineRepairProposal | None = None
        return no_proposal
    return TimelineRepairProposal(
        proposal_id=proposal_id,
        issue_type=issue_type,
        confidence=confidence,
        target_segment_id=target_segment_id,
        target_word_ids=target_word_ids,
        target_source_start_us=int(
            caption.spoken_source_start_us
            or word_source_start_us(target_word_ids, source_graph)
            or target_segment.source_start_us
        ),
        target_source_end_us=int(
            caption.spoken_source_end_us
            or word_source_end_us(target_word_ids, source_graph)
            or target_segment.source_end_us
        ),
        target_text=str(caption.text or ""),
        repair_action=repair_action,
        risk_tags=risk_tags,
        evidence={
            **evidence,
            "target_segment_id": target_segment_id,
            "target_word_ids": target_word_ids,
        },
    )


def target_segment_id_for_caption(
    caption: CaptionRenderUnit,
    final_timeline: list[FinalTimelineSegment],
) -> str:
    if caption.containing_video_segment_id:
        return str(caption.containing_video_segment_id)
    if len(caption.timeline_segment_ids) == 1:
        return str(caption.timeline_segment_ids[0])
    caption_word_ids = {str(word_id) for word_id in caption.word_ids if str(word_id)}
    if not caption_word_ids:
        return ""
    for segment in final_timeline:
        segment_word_ids = {str(word_id) for word_id in segment.word_ids if str(word_id)}
        if caption_word_ids <= segment_word_ids:
            return str(segment.segment_id)
    return ""


def word_source_start_us(word_ids: list[str], source_graph: CanonicalSourceGraph) -> int:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for word_id in word_ids:
        word = words_by_id.get(word_id)
        if word is not None:
            return int(word.source_start_us)
    return 0


def word_source_end_us(word_ids: list[str], source_graph: CanonicalSourceGraph) -> int:
    words_by_id = {word.word_id: word for word in source_graph.words}
    for word_id in reversed(word_ids):
        word = words_by_id.get(word_id)
        if word is not None:
            return int(word.source_end_us)
    return 0


def apply_first_timeline_repair_proposal(
    *,
    proposals: list[TimelineRepairProposal],
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
    decision: str,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    if not proposals:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    return apply_timeline_repair_proposal_as_step(
        proposal=proposals[0],
        final_timeline=final_timeline,
        source_graph=source_graph,
        render_captions=render_captions,
        pass_index=pass_index,
        decision=decision,
    )


def apply_timeline_repair_proposal_as_step(
    *,
    proposal: TimelineRepairProposal,
    final_timeline: list[FinalTimelineSegment],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
    decision: str,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    materialized = _apply_timeline_repair_proposal(
        proposal,
        final_timeline,
        source_graph,
        renderer=RenderCallbackAdapter(render_captions),
    )
    if not materialized.applied:
        no_step: _RepairStep | None = None
        return no_step, proposal_unresolved(proposal, pass_index=pass_index, materialized=materialized)
    return (
        _RepairStep(
            final_timeline=materialized.final_timeline,
            captions=materialized.captions,
            timeline_changed=True,
            action=proposal_action(
                proposal,
                decision=decision,
                pass_index=pass_index,
                coverage_report=materialized.coverage_report,
            ),
        ),
        None,
    )


def proposal_unresolved(
    proposal: TimelineRepairProposal,
    *,
    pass_index: int,
    materialized: Any,
) -> dict[str, Any]:
    return {
        "pass_index": pass_index,
        "issue_type": proposal.issue_type,
        "proposal_id": proposal.proposal_id,
        "reason": materialized.reason,
        "blocker_code": materialized.blocker_code,
        "target_segment_id": proposal.target_segment_id,
        "target_word_ids": list(proposal.target_word_ids),
        "evidence": dict(proposal.evidence),
    }


def proposal_action(
    proposal: TimelineRepairProposal,
    *,
    decision: str,
    pass_index: int,
    coverage_report: dict[str, Any],
) -> dict[str, Any]:
    return _action(
        proposal.issue_type,
        decision,
        pass_index,
        dict(proposal.evidence),
        proposal_id=proposal.proposal_id,
        repair_action=proposal.repair_action,
        confidence=float(proposal.confidence),
        target_segment_id=proposal.target_segment_id,
        target_word_ids=list(proposal.target_word_ids),
        target_text=proposal.target_text,
        risk_tags=list(proposal.risk_tags),
        evidence=dict(proposal.evidence),
        coverage_report={
            "missing_final_timeline_caption_word_count": int(
                coverage_report.get("missing_final_timeline_caption_word_count") or 0
            ),
            "prewrite_uncaptioned_spoken_word_count": int(
                coverage_report.get("prewrite_uncaptioned_spoken_word_count") or 0
            ),
        },
    )
