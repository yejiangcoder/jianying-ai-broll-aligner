from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.caption_alignment import build_caption_alignment_report
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.final_timeline_quality_guard import build_final_timeline_quality_guard_report
from aroll_v21.quality.visual_pacing import build_visual_pacing_report


FINAL_VISIBLE_FATAL_COUNT_FIELDS = (
    "visible_repeat_fatal_candidate_count",
    "modifier_redundancy_residual_count",
    "self_repair_aborted_phrase_count",
    "dangling_prefix_suffix_count",
    "semantic_garbage_or_asr_suspect_count",
    "semantic_integrity_count",
    "cross_caption_semantic_containment_count",
    "restart_repeat_visible_count",
)


@dataclass(frozen=True)
class QualitySnapshot:
    timeline_signature: tuple[Any, ...]
    caption_signature: tuple[Any, ...]
    final_segment_count: int
    caption_count: int
    final_visible_gate_passed: bool
    final_visible_blocker_codes: tuple[str, ...]
    final_visible_fatal_count: int
    visual_pacing_gate_passed: bool
    visual_blocker_codes: tuple[str, ...]
    blocking_short_segment_count: int
    caption_alignment_gate_passed: bool
    caption_alignment_blocker_codes: tuple[str, ...]
    final_timeline_quality_guard_gate_present: bool
    final_timeline_quality_guard_gate_passed: bool
    final_timeline_quality_guard_blocker_codes: tuple[str, ...]
    final_timeline_quality_guard_blocking_candidate_count: int

    def blocker_codes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(
                    [
                        *self.final_visible_blocker_codes,
                        *self.visual_blocker_codes,
                        *self.caption_alignment_blocker_codes,
                        *self.final_timeline_quality_guard_blocker_codes,
                    ]
                )
            )
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "timeline_signature_hash": _signature_hash(self.timeline_signature),
            "caption_signature_hash": _signature_hash(self.caption_signature),
            "final_segment_count": int(self.final_segment_count),
            "caption_count": int(self.caption_count),
            "final_visible_gate_passed": bool(self.final_visible_gate_passed),
            "final_visible_blocker_codes": list(self.final_visible_blocker_codes),
            "final_visible_fatal_count": int(self.final_visible_fatal_count),
            "visual_pacing_gate_passed": bool(self.visual_pacing_gate_passed),
            "visual_blocker_codes": list(self.visual_blocker_codes),
            "blocking_short_segment_count": int(self.blocking_short_segment_count),
            "caption_alignment_gate_passed": bool(self.caption_alignment_gate_passed),
            "caption_alignment_blocker_codes": list(self.caption_alignment_blocker_codes),
            "final_timeline_quality_guard_gate_present": bool(self.final_timeline_quality_guard_gate_present),
            "final_timeline_quality_guard_gate_passed": bool(self.final_timeline_quality_guard_gate_passed),
            "final_timeline_quality_guard_blocker_codes": list(self.final_timeline_quality_guard_blocker_codes),
            "final_timeline_quality_guard_blocking_candidate_count": int(
                self.final_timeline_quality_guard_blocking_candidate_count
            ),
            "blocker_codes": list(self.blocker_codes()),
        }


@dataclass(frozen=True)
class TimelineMutation:
    phase: str
    rule_name: str
    before: QualitySnapshot
    after: QualitySnapshot
    action: dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    rejection_reason: str = ""
    introduced_blocker_codes: tuple[str, ...] = field(default_factory=tuple)
    cleared_blocker_codes: tuple[str, ...] = field(default_factory=tuple)

    def to_report(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "rule_name": self.rule_name,
            "accepted": bool(self.accepted),
            "rejection_reason": self.rejection_reason,
            "action": dict(self.action),
            "introduced_blocker_codes": list(self.introduced_blocker_codes),
            "cleared_blocker_codes": list(self.cleared_blocker_codes),
            "before": self.before.to_report(),
            "after": self.after.to_report(),
        }


def build_quality_snapshot(
    *,
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph | None = None,
    visual_pacing_report: dict[str, Any] | None = None,
) -> QualitySnapshot:
    visible_gate = build_final_caption_visible_repeat_gate(list(captions))
    visual_gate = build_visual_pacing_report(
        final_timeline=list(final_timeline),
        captions=list(captions),
        executed=True,
        merge_report=visual_pacing_report,
        source_graph=source_graph,
    )
    alignment_gate = build_caption_alignment_report(final_timeline=list(final_timeline), captions=list(captions))
    final_timeline_quality_guard = (
        build_final_timeline_quality_guard_report(
            source_graph=source_graph,
            final_timeline=list(final_timeline),
            captions=list(captions),
        )
        if source_graph is not None
        else {"gate_passed": True, "blocker_codes": [], "blocking_candidate_count": 0}
    )
    return QualitySnapshot(
        timeline_signature=_timeline_signature(final_timeline),
        caption_signature=_caption_signature(captions),
        final_segment_count=len(final_timeline),
        caption_count=len(captions),
        final_visible_gate_passed=bool(visible_gate.get("gate_passed")),
        final_visible_blocker_codes=_blocker_codes(visible_gate),
        final_visible_fatal_count=sum(int(visible_gate.get(field_name) or 0) for field_name in FINAL_VISIBLE_FATAL_COUNT_FIELDS),
        visual_pacing_gate_passed=bool(visual_gate.get("gate_passed")),
        visual_blocker_codes=_blocker_codes(visual_gate),
        blocking_short_segment_count=int(visual_gate.get("visual_short_segment_count_lt_1200ms_after_blocking") or 0),
        caption_alignment_gate_passed=bool(alignment_gate.get("gate_passed")),
        caption_alignment_blocker_codes=_blocker_codes(alignment_gate),
        final_timeline_quality_guard_gate_present=source_graph is not None,
        final_timeline_quality_guard_gate_passed=bool(final_timeline_quality_guard.get("gate_passed")),
        final_timeline_quality_guard_blocker_codes=_blocker_codes(final_timeline_quality_guard),
        final_timeline_quality_guard_blocking_candidate_count=int(
            final_timeline_quality_guard.get("blocking_candidate_count") or 0
        ),
    )


def build_timeline_mutation(
    *,
    phase: str,
    rule_name: str,
    before: QualitySnapshot,
    after: QualitySnapshot,
    action: dict[str, Any] | None = None,
) -> TimelineMutation:
    introduced = tuple(sorted(set(after.blocker_codes()) - set(before.blocker_codes())))
    cleared = tuple(sorted(set(before.blocker_codes()) - set(after.blocker_codes())))
    rejection_reason = _quality_regression_reason(before, after)
    return TimelineMutation(
        phase=phase,
        rule_name=rule_name,
        before=before,
        after=after,
        action=dict(action or {}),
        accepted=not rejection_reason,
        rejection_reason=rejection_reason,
        introduced_blocker_codes=introduced,
        cleared_blocker_codes=cleared,
    )


def _quality_regression_reason(before: QualitySnapshot, after: QualitySnapshot) -> str:
    if after.final_visible_fatal_count > before.final_visible_fatal_count:
        return "final_visible_fatal_count_increased"
    if set(after.final_timeline_quality_guard_blocker_codes) - set(before.final_timeline_quality_guard_blocker_codes):
        return "final_timeline_quality_guard_blocker_introduced"
    if after.blocking_short_segment_count > before.blocking_short_segment_count:
        return "blocking_short_segment_count_increased"
    if set(after.caption_alignment_blocker_codes) - set(before.caption_alignment_blocker_codes):
        return "caption_alignment_blocker_introduced"
    if set(after.visual_blocker_codes) - set(before.visual_blocker_codes):
        return "visual_blocker_introduced"
    return ""


def _timeline_signature(final_timeline: list[FinalTimelineSegment]) -> tuple[Any, ...]:
    return tuple(
        (
            str(segment.segment_id),
            tuple(str(word_id) for word_id in list(segment.word_ids or [])),
            normalize_text(str(segment.text or "")),
            int(segment.source_start_us),
            int(segment.source_end_us),
            int(segment.target_start_us),
            int(segment.target_end_us),
        )
        for segment in list(final_timeline or [])
    )


def _caption_signature(captions: list[CaptionRenderUnit]) -> tuple[Any, ...]:
    return tuple(
        (
            str(caption.caption_id),
            tuple(str(segment_id) for segment_id in list(caption.timeline_segment_ids or [])),
            tuple(str(word_id) for word_id in list(caption.word_ids or [])),
            normalize_text(str(caption.text or "")),
            int(caption.target_start_us),
            int(caption.target_end_us),
        )
        for caption in list(captions or [])
    )


def _blocker_codes(report: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({str(code) for code in list(report.get("blocker_codes") or []) if str(code)}))


def _signature_hash(signature: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:16]
