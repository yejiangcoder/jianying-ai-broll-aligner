from __future__ import annotations

from typing import Any, Callable

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.final_visible_repair.proposal_apply import (
    apply_timeline_repair_proposal_as_step,
    build_caption_span_drop_proposal,
)
from aroll_v21.quality.final_visible_repair.proposal import TimelineRepairProposal
from aroll_v21.quality.final_visible_repair.report import _action
from aroll_v21.quality.final_visible_repair.result import _RepairStep
from aroll_v21.quality.final_visible_repair.rules.caption_only_merge import (
    _merge_adjacent_caption_segments,
    _merge_adjacent_captions,
)
from aroll_v21.quality.final_visible_repair.timeline_utils import (
    caption_segment_ids,
    ordered_captions,
    renumber_captions,
)
from aroll_v21.quality.subtitle_readability import HARD_MAX_CHARS
from aroll_v21.quality.tiny_caption_classification import build_tiny_caption_classification_report


CONTAINED_SHORT_FRAGMENT_OPEN_TAIL_CHARS = set("\u7684\u5f97\u5730\u4e4b\u5728\u4ece\u5bf9\u628a\u88ab\u5c06\u8ba9\u4f7f\u8ddf\u548c\u4e0e\u6216\u53ca\u4ee5\u4e3a\u4e8e\u5230")
OPEN_TAIL_SHORT_CAPTION_MAX_CHARS = 5
OPEN_TAIL_SHORT_CAPTION_MAX_GAP_US = 120_000
OPEN_TAIL_SHORT_CAPTION_MERGE_TAILS = set("\u7684\u5f97\u5730\u4e4b")
OPEN_TAIL_OBJECT_CAPTION_MERGE_TAILS = set("\u628a\u88ab\u7ed9\u8ba9\u4f7f\u5bf9\u5411\u4e3a\u5c06")
SHORT_ABORTED_PREFIX_MAX_CHARS = 5
SHORT_ABORTED_PREFIX_MAX_GAP_US = 300_000
COMMON_CLOSED_DE_PHRASES = {
    "可以的",
    "不会的",
    "不是的",
    "对的",
    "好的",
    "真的",
    "假的",
    "是的",
}


def repair_contained_short_fragment_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    gate = build_final_caption_visible_repeat_gate(captions)
    candidates = [
        row
        for row in list(gate.get("containment_repeat_candidates") or [])
        if str(row.get("severity") or "") in {"fatal", "high"}
        and str(row.get("classification") or "") == "local_containment_restart"
        and str(row.get("distance_kind") or "") in {"adjacent", "near"}
    ]
    if not candidates:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    captions_by_id = {caption.caption_id: caption for caption in captions}
    for candidate in candidates:
        left = captions_by_id.get(str(candidate.get("caption_id") or ""))
        right = captions_by_id.get(str(candidate.get("related_caption_id") or ""))
        drop_caption, kept_caption = contained_short_fragment_drop_caption(left, right)
        if drop_caption is None or kept_caption is None:
            continue
        proposal = build_caption_span_drop_proposal(
            proposal_id=f"contained_short_fragment_{pass_index:06d}_{drop_caption.caption_id}",
            issue_type="contained_short_caption_fragment",
            confidence=0.94,
            repair_action="span_drop",
            caption=drop_caption,
            final_timeline=final_timeline,
            source_graph=source_graph,
            risk_tags=["local_containment_restart", "contained_short_fragment"],
            evidence={
                "candidate": dict(candidate),
                "dropped_caption_id": drop_caption.caption_id,
                "kept_caption_id": kept_caption.caption_id,
                "dropped_text": drop_caption.text,
                "kept_text": kept_caption.text,
                "policy": "drop_open_tail_short_fragment_that_is_prefix_of_adjacent_complete_caption",
            },
        )
        if proposal is None:
            continue
        step, unresolved = apply_timeline_repair_proposal_as_step(
            proposal=proposal,
            final_timeline=final_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index,
            decision="span_drop",
        )
        if step is not None or unresolved is not None:
            return step, unresolved
    no_step: _RepairStep | None = None
    no_unresolved: dict[str, Any] | None = None
    return no_step, no_unresolved


def repair_self_repair_aborted_phrase_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    gate = build_final_caption_visible_repeat_gate(captions)
    candidates = [
        row
        for row in list(gate.get("self_repair_aborted_phrase_candidates") or [])
        if bool(row.get("deterministic_drop_left"))
    ]
    if not candidates:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    captions_by_id = {caption.caption_id: caption for caption in captions}
    for candidate in candidates:
        drop_caption = captions_by_id.get(str(candidate.get("caption_id") or ""))
        kept_caption = captions_by_id.get(str(candidate.get("related_caption_id") or ""))
        if drop_caption is None or kept_caption is None:
            continue
        proposal = build_caption_span_drop_proposal(
            proposal_id=f"self_repair_aborted_phrase_{pass_index:06d}_{drop_caption.caption_id}",
            issue_type="self_repair_aborted_phrase",
            confidence=float(candidate.get("similarity") or 0.9),
            repair_action="span_drop",
            caption=drop_caption,
            final_timeline=final_timeline,
            source_graph=source_graph,
            risk_tags=["self_repair_aborted_phrase", "drop_left_keep_right"],
            evidence={
                "candidate": dict(candidate),
                "dropped_caption_id": drop_caption.caption_id,
                "kept_caption_id": kept_caption.caption_id,
                "dropped_text": drop_caption.text,
                "kept_text": kept_caption.text,
                "policy": "drop_left_aborted_phrase_keep_completed_restart",
            },
        )
        if proposal is None:
            continue
        step, unresolved = apply_timeline_repair_proposal_as_step(
            proposal=proposal,
            final_timeline=final_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index,
            decision="drop_left_keep_right",
        )
        if step is not None or unresolved is not None:
            return step, unresolved
    no_step: _RepairStep | None = None
    no_unresolved: dict[str, Any] | None = None
    return no_step, no_unresolved


def repair_short_aborted_prefix_caption_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    ordered = ordered_captions(captions)
    for left, right in zip(ordered, ordered[1:]):
        row = short_aborted_prefix_candidate(left, right)
        if not row:
            continue
        proposal = build_caption_span_drop_proposal(
            proposal_id=f"short_aborted_prefix_caption_{pass_index:06d}_{left.caption_id}",
            issue_type="short_aborted_prefix_caption",
            confidence=float(row.get("confidence") or 0.92),
            repair_action="span_drop",
            caption=left,
            final_timeline=final_timeline,
            source_graph=source_graph,
            risk_tags=["short_aborted_prefix_caption", "single_char_asr_tail"],
            evidence={
                **row,
                "dropped_caption_id": left.caption_id,
                "kept_caption_id": right.caption_id,
                "dropped_text": left.text,
                "kept_text": right.text,
                "policy": "drop_short_caption_restarted_by_adjacent_longer_caption_with_single_char_tail_mismatch",
            },
        )
        if proposal is None:
            continue
        step, unresolved = apply_timeline_repair_proposal_as_step(
            proposal=proposal,
            final_timeline=final_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index,
            decision="span_drop",
        )
        if step is not None or unresolved is not None:
            return step, unresolved
    no_step: _RepairStep | None = None
    no_unresolved: dict[str, Any] | None = None
    return no_step, no_unresolved


def repair_open_tail_short_caption_with_next(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
    render_captions_preserving_caption_only_materializations: Callable[
        [list[FinalTimelineSegment], list[CaptionRenderUnit], Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]]],
        list[CaptionRenderUnit],
    ],
) -> _RepairStep | None:
    ordered = ordered_captions(captions)
    for index in range(len(ordered) - 1):
        current = ordered[index]
        next_caption = ordered[index + 1]
        if not open_tail_short_caption_should_merge(current, next_caption):
            continue
        merged_timeline = _merge_adjacent_caption_segments(final_timeline, current, next_caption, source_graph)
        if merged_timeline is not None:
            rendered_captions = render_captions_preserving_caption_only_materializations(
                merged_timeline,
                captions,
                render_captions,
            )
            return _RepairStep(
                final_timeline=merged_timeline,
                captions=rendered_captions,
                timeline_changed=True,
                action=_action(
                    "open_tail_short_caption",
                    "merge_with_next_segment",
                    pass_index,
                    {
                        "caption_id": current.caption_id,
                        "related_caption_id": next_caption.caption_id,
                        "text": current.text,
                        "related_text": next_caption.text,
                    },
                    affected_caption_ids=[current.caption_id, next_caption.caption_id],
                    target_gap_us=int(next_caption.target_start_us) - int(current.target_end_us),
                ),
            )
        merged_caption_result = _merge_adjacent_captions(current, next_caption)
        if merged_caption_result is None:
            continue
        merged_caption, merge_decision = merged_caption_result
        if not bool(build_final_caption_visible_repeat_gate([merged_caption]).get("gate_passed")):
            continue
        repaired = [*ordered[:index], merged_caption, *ordered[index + 2 :]]
        return _RepairStep(
            final_timeline=final_timeline,
            captions=renumber_captions(repaired),
            timeline_changed=False,
            action=_action(
                "open_tail_short_caption",
                "caption_only_merge_open_tail_with_next",
                pass_index,
                {
                    "caption_id": current.caption_id,
                    "related_caption_id": next_caption.caption_id,
                    "text": current.text,
                    "related_text": next_caption.text,
                },
                affected_caption_ids=[current.caption_id, next_caption.caption_id],
                target_gap_us=int(next_caption.target_start_us) - int(current.target_end_us),
                video_segment_merged=False,
                caption_only_merge_materialized=True,
                caption_only_merge_decision=merge_decision,
                merged_into_caption_id=current.caption_id,
                consumed_caption_id=next_caption.caption_id,
                consumed_caption_state="consumed_by_open_tail_caption_merge",
                merged_caption_text=merged_caption.text,
                merged_caption_timeline_segment_ids=list(merged_caption.timeline_segment_ids),
                merged_caption_target_start_us=int(merged_caption.target_start_us),
                merged_caption_target_end_us=int(merged_caption.target_end_us),
            ),
        )
    no_step: _RepairStep | None = None
    return no_step


def repair_fatal_tiny_caption_with_proposal(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    render_captions: Callable[[list[FinalTimelineSegment]], list[CaptionRenderUnit]],
    pass_index: int,
) -> tuple[_RepairStep | None, dict[str, Any] | None]:
    tiny_report = build_tiny_caption_classification_report(captions)
    fatal_rows = [
        row
        for row in list(tiny_report.get("tiny_caption_classifications") or [])
        if str(row.get("severity") or "") == "fatal"
    ]
    if not fatal_rows:
        no_step: _RepairStep | None = None
        no_unresolved: dict[str, Any] | None = None
        return no_step, no_unresolved
    captions_by_id = {caption.caption_id: caption for caption in captions}
    segments_by_id = {segment.segment_id: segment for segment in final_timeline}
    for row in fatal_rows:
        caption = captions_by_id.get(str(row.get("caption_id") or ""))
        if caption is None:
            continue
        target_segment_id = str(caption.containing_video_segment_id or "")
        if not target_segment_id and len(caption.timeline_segment_ids) == 1:
            target_segment_id = str(caption.timeline_segment_ids[0])
        if target_segment_id not in segments_by_id:
            continue
        target_word_ids = [str(word_id) for word_id in caption.word_ids if str(word_id)]
        if not target_word_ids:
            continue
        proposal = TimelineRepairProposal(
            proposal_id=f"tiny_caption_residual_{pass_index:06d}_{target_segment_id}",
            issue_type="tiny_caption_residual",
            confidence=0.95,
            target_segment_id=target_segment_id,
            target_word_ids=target_word_ids,
            target_source_start_us=int(caption.spoken_source_start_us or segments_by_id[target_segment_id].source_start_us),
            target_source_end_us=int(caption.spoken_source_end_us or segments_by_id[target_segment_id].source_end_us),
            target_text=str(caption.text or row.get("caption_text") or ""),
            repair_action="span_drop",
            risk_tags=[*list(row.get("risk_tags") or []), "tiny_caption_residual"],
            evidence={
                "caption_id": caption.caption_id,
                "classification": str(row.get("classification") or ""),
                "classification_reason": str(row.get("classification_reason") or ""),
                "caption_text": str(row.get("caption_text") or caption.text or ""),
                "word_ids": target_word_ids,
            },
        )
        step, unresolved = apply_timeline_repair_proposal_as_step(
            proposal=proposal,
            final_timeline=final_timeline,
            source_graph=source_graph,
            render_captions=render_captions,
            pass_index=pass_index,
            decision="span_drop",
        )
        if step is not None or unresolved is not None:
            return step, unresolved
    no_step: _RepairStep | None = None
    no_unresolved: dict[str, Any] | None = None
    return no_step, no_unresolved


def short_aborted_prefix_candidate(
    left: CaptionRenderUnit,
    right: CaptionRenderUnit,
) -> dict[str, Any] | None:
    no_candidate: dict[str, Any] | None = None
    left_text = normalize_text(left.text)
    right_text = normalize_text(right.text)
    if not left_text or not right_text or left_text == right_text:
        return no_candidate
    gap_us = int(right.target_start_us) - int(left.target_end_us)
    if gap_us < -80_000 or gap_us > SHORT_ABORTED_PREFIX_MAX_GAP_US:
        return no_candidate
    if len(left_text) > SHORT_ABORTED_PREFIX_MAX_CHARS or len(right_text) < len(left_text) + 2:
        return no_candidate
    prefix_len = common_prefix_len(left_text, right_text)
    left_tail = left_text[prefix_len:]
    right_tail = right_text[prefix_len:]
    if prefix_len < 2 or len(left_tail) != 1 or len(right_tail) < 2:
        return no_candidate
    if right_text.startswith(left_text):
        return no_candidate
    return {
        "reason": "short caption is reopened by the next caption with a longer continuation and a single-character tail mismatch",
        "shared_prefix": left_text[:prefix_len],
        "left_tail": left_tail,
        "right_tail": right_tail,
        "gap_us": gap_us,
        "confidence": 0.92,
    }


def open_tail_short_caption_should_merge(
    current: CaptionRenderUnit,
    next_caption: CaptionRenderUnit,
) -> bool:
    text = normalize_text(current.text)
    next_text = normalize_text(next_caption.text)
    if not text or not next_text:
        return False
    current_segments = set(caption_segment_ids(current))
    next_segments = set(caption_segment_ids(next_caption))
    same_visible_segment = bool(current_segments & next_segments) or (
        bool(current.containing_video_segment_id)
        and str(current.containing_video_segment_id) == str(next_caption.containing_video_segment_id or "")
    )
    tail = text[-1]
    short_tail_merge = len(text) <= OPEN_TAIL_SHORT_CAPTION_MAX_CHARS and tail in OPEN_TAIL_SHORT_CAPTION_MERGE_TAILS
    object_tail_merge = same_visible_segment and tail in OPEN_TAIL_OBJECT_CAPTION_MERGE_TAILS
    if not short_tail_merge and not object_tail_merge:
        return False
    if text in COMMON_CLOSED_DE_PHRASES:
        return False
    if text.startswith("是") and text.endswith("的"):
        return False
    if next_text.startswith(("的", "的是")):
        return False
    gap_us = int(next_caption.target_start_us) - int(current.target_end_us)
    if gap_us < -80_000 or gap_us > OPEN_TAIL_SHORT_CAPTION_MAX_GAP_US:
        return False
    combined = normalize_text(f"{current.text}{next_caption.text}")
    return bool(combined) and len(combined) <= HARD_MAX_CHARS


def common_prefix_len(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def contained_short_fragment_drop_caption(
    left: CaptionRenderUnit | None,
    right: CaptionRenderUnit | None,
) -> tuple[CaptionRenderUnit | None, CaptionRenderUnit | None]:
    if left is None or right is None:
        return None, None
    left_text = normalize_text(left.text)
    right_text = normalize_text(right.text)
    if not left_text or not right_text or left_text == right_text:
        return None, None
    if right_text.startswith(left_text) and len(right_text) > len(left_text) and safe_contained_short_fragment(left_text):
        return left, right
    if left_text.startswith(right_text) and len(left_text) > len(right_text) and safe_contained_short_fragment(right_text):
        return right, left
    return None, None


def safe_contained_short_fragment(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized or len(normalized) > 8:
        return False
    return normalized[-1] in CONTAINED_SHORT_FRAGMENT_OPEN_TAIL_CHARS
