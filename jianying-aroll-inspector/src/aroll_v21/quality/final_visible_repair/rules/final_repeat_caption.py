from __future__ import annotations

from typing import Any

from aroll_final_repeat_gate import build_final_repeat_gate_report
from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.final_visible_repair.result import _RepairStep
from aroll_v21.quality.final_visible_repair.rules.de_shi_bridge import _drop_repeated_caption_span


def repair_caption_level_final_repeat_aborted_containment(
    final_timeline: list[FinalTimelineSegment],
    captions: list[CaptionRenderUnit],
    source_graph: CanonicalSourceGraph,
    pass_index: int,
) -> _RepairStep | None:
    repeat_report = build_final_repeat_gate_report({"issues": []}, final_repeat_caption_rows(captions))
    for candidate in list(repeat_report.get("final_target_repeat_candidates") or []):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("cluster_type") or "") != "semantic_containment_take":
            continue
        drop_caption_id = caption_level_final_repeat_aborted_drop_caption_id(candidate)
        if not drop_caption_id:
            continue
        repair_candidate = dict(candidate)
        repair_candidate.update(
            {
                "caption_id": drop_caption_id,
                "reason": "caption_level_aborted_semantic_containment",
                "drop_caption_id": drop_caption_id,
            }
        )
        return _drop_repeated_caption_span(
            final_timeline,
            captions,
            source_graph,
            repair_candidate,
            "final_target_repeat_caption_containment",
            pass_index,
        )
    no_step: _RepairStep | None = None
    return no_step


def final_repeat_caption_rows(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    return [
        {
            "fragment_id": caption.caption_id,
            "fragment_text": caption.text,
            "text": caption.text,
            "word_ids": list(caption.word_ids),
            "target_start_us": int(caption.target_start_us),
            "target_duration_us": max(0, int(caption.target_end_us) - int(caption.target_start_us)),
            "source_subtitle_uids": list(caption.source_subtitle_uids),
        }
        for caption in list(captions or [])
    ]


def caption_level_final_repeat_aborted_drop_caption_id(candidate: dict[str, Any]) -> str:
    rows = [row for row in list(candidate.get("candidates") or []) if isinstance(row, dict)]
    if len(rows) < 2:
        return ""
    completed_texts = [
        normalize_text(str(row.get("text") or row.get("norm_text") or ""))
        for row in rows
        if not bool(row.get("is_aborted_start"))
    ]
    completed_texts = [text for text in completed_texts if text]
    if not completed_texts:
        return ""
    for row in rows:
        if not bool(row.get("is_aborted_start")):
            continue
        text = normalize_text(str(row.get("text") or row.get("norm_text") or ""))
        if len(text) < 2:
            continue
        if not any(caption_level_containment_match(text, completed, candidate) for completed in completed_texts):
            continue
        caption_ids = [str(value) for value in list(row.get("subtitle_uids") or []) if str(value)]
        if caption_ids:
            return caption_ids[0]
    return ""


def caption_level_containment_match(short_text: str, completed_text: str, candidate: dict[str, Any]) -> bool:
    if not short_text or not completed_text or short_text == completed_text:
        return False
    if short_text in completed_text:
        return True
    relaxed_short = relaxed_containment_text(short_text)
    relaxed_completed = relaxed_containment_text(completed_text)
    if relaxed_short and relaxed_short != relaxed_completed and relaxed_short in relaxed_completed:
        return True
    return any(
        isinstance(row, dict)
        and str(row.get("cluster_type") or "") == "semantic_containment_take"
        and float(row.get("containment") or 0.0) >= 1.0
        and float(row.get("similarity") or 0.0) >= 0.5
        for row in list(candidate.get("pairwise_evidence") or [])
    )


def relaxed_containment_text(text: str) -> str:
    return "".join(char for char in normalize_text(text) if char not in {"的", "地", "得"})
