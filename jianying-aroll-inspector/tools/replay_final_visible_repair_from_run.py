from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aroll_v21.operator_io import _captions, _final_timeline, _safe_read_json, _source_graph, write_json
from aroll_v21.quality.caption_alignment import build_caption_alignment_report
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
from aroll_v21.quality.quality_gate import build_quality_gate_report
from aroll_v21.quality.visual_pacing import build_visual_pacing_report
from aroll_v21.render.subtitle_renderer import SubtitleRenderer


REPORT_KIND = "aroll_v21_final_visible_repair_replay"


def replay_final_visible_repair_from_run(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    max_cycles: int = 8,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    output_dir = output_dir.resolve() if output_dir else _default_output_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    source_graph = _source_graph(_safe_read_json(run_dir / "source_graph.json"))
    if source_graph is None:
        raise ValueError(f"source_graph artifact missing or invalid under {run_dir}")
    final_timeline = _final_timeline(_safe_read_json(run_dir / "final_timeline.json"))
    captions = _captions(_safe_read_json(run_dir / "captions.json"))
    if not final_timeline:
        raise ValueError(f"final_timeline artifact is empty under {run_dir}")

    original_signature = _state_signature(final_timeline, captions)
    renderer = SubtitleRenderer()
    cycle_reports: list[dict[str, Any]] = []
    for cycle_index in range(max(1, int(max_cycles))):
        before = _state_signature(final_timeline, captions)
        result = repair_final_visible_caption_issues(
            final_timeline=final_timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda rows: renderer.render(rows, source_graph),
        )
        cycle_report = dict(result.report)
        cycle_report["replay_cycle_index"] = cycle_index
        cycle_reports.append(cycle_report)
        final_timeline = result.final_timeline
        captions = result.captions
        action_count = int(cycle_report.get("final_visible_repair_action_count") or 0)
        after = _state_signature(final_timeline, captions)
        if action_count <= 0 or after == before:
            break

    combined_repair_report = _combine_repair_reports(cycle_reports)
    caption_alignment_gate = build_caption_alignment_report(
        final_timeline=final_timeline,
        captions=captions,
        visible_caption_track_count=_old_caption_track_count(run_dir),
        caption_lane_count=_old_caption_lane_count(run_dir),
    )
    final_caption_visible_repeat_gate = build_final_caption_visible_repeat_gate(captions)
    old_quality = _safe_read_json(run_dir / "quality_gate_report.json") or {}
    visual_pacing_gate = build_visual_pacing_report(
        final_timeline=final_timeline,
        captions=captions,
        executed=bool(_old_visual_pacing_gate(old_quality).get("visual_pacing_executed", True)),
        merge_report={
            **_old_visual_pacing_gate(old_quality),
            "final_visible_repair_action_count": int(combined_repair_report.get("final_visible_repair_action_count") or 0),
        },
        source_graph=source_graph,
    )
    quality_gate_report = build_quality_gate_report(
        effective_speed_gate=_old_effective_speed_gate(old_quality),
        final_repeat_convergence_gate=_old_final_repeat_gate(old_quality),
        final_caption_visible_repeat_gate=final_caption_visible_repeat_gate,
        semantic_adjudication_gate=_old_semantic_gate(run_dir, old_quality),
        visual_pacing_gate=visual_pacing_gate,
        caption_alignment_gate=caption_alignment_gate,
        ready_for_user_manual_qc_preconditions_passed=bool((_safe_read_json(run_dir / "run_summary.json") or {}).get("ready_for_disposable_write_pre_audit")),
    )

    changed = _state_signature(final_timeline, captions) != original_signature
    elapsed = round(time.perf_counter() - started, 6)
    report = {
        "report_kind": REPORT_KIND,
        "report_version": 1,
        "status": "ok",
        "input_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "read_only_input": True,
        "does_not_write_draft": True,
        "does_not_call_semantic_provider": True,
        "does_not_replan_decisions": True,
        "visual_pacing_normalize_executed": False,
        "reused_effective_speed_gate": True,
        "reused_final_repeat_convergence_gate": True,
        "reused_semantic_adjudication_gate": True,
        "max_cycles": int(max_cycles),
        "cycle_count": len(cycle_reports),
        "changed": changed,
        "elapsed_seconds": elapsed,
        "input_final_timeline_segment_count": len(_final_timeline(_safe_read_json(run_dir / "final_timeline.json"))),
        "output_final_timeline_segment_count": len(final_timeline),
        "input_caption_count": len(_captions(_safe_read_json(run_dir / "captions.json"))),
        "output_caption_count": len(captions),
        "final_visible_repair_action_count": int(combined_repair_report.get("final_visible_repair_action_count") or 0),
        "quality_gate_passed": bool(quality_gate_report.get("gate_passed")),
        "quality_gate_blocker_codes": list(quality_gate_report.get("blocker_codes") or []),
        "caption_alignment_gate_passed": bool(caption_alignment_gate.get("gate_passed")),
        "final_caption_visible_repeat_gate_passed": bool(final_caption_visible_repeat_gate.get("gate_passed")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    write_json(output_dir / "final_timeline.json", final_timeline)
    write_json(output_dir / "captions.json", captions)
    write_json(output_dir / "final_visible_caption_repair_report.json", combined_repair_report)
    write_json(output_dir / "caption_alignment_gate.json", caption_alignment_gate)
    write_json(output_dir / "final_caption_visible_repeat_gate.json", final_caption_visible_repeat_gate)
    write_json(output_dir / "visual_pacing_gate.json", visual_pacing_gate)
    write_json(output_dir / "quality_gate_report.json", quality_gate_report)
    write_json(output_dir / "replay_final_visible_repair_report.json", report)
    write_json(output_dir / "replay_final_visible_repair_cycles.json", cycle_reports)
    return report


def _default_output_dir(run_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return run_dir / "quality" / f"replay_final_visible_repair_{stamp}_{uuid4().hex[:8]}"


def _state_signature(final_timeline: list[Any], captions: list[Any]) -> tuple[Any, Any]:
    return (
        tuple(
            (
                str(segment.segment_id),
                tuple(str(word_id) for word_id in segment.word_ids),
                int(segment.source_start_us),
                int(segment.source_end_us),
                int(segment.target_start_us),
                int(segment.target_end_us),
                str(segment.text),
            )
            for segment in final_timeline
        ),
        tuple(
            (
                str(caption.caption_id),
                tuple(str(word_id) for word_id in caption.word_ids),
                int(caption.target_start_us),
                int(caption.target_end_us),
                str(caption.text),
                str(caption.containing_video_segment_id or ""),
            )
            for caption in captions
        ),
    )


def _combine_repair_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"final_visible_repair_success": True, "final_visible_repair_action_count": 0, "cycle_reports": []}
    latest = dict(reports[-1])
    actions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for report in reports:
        report_actions = list(report.get("final_visible_repair_actions") or [])
        if not report_actions:
            report_actions = list(report.get("timeline_repair_proposal_actions") or [])
        actions.extend(report_actions)
        unresolved.extend(list(report.get("timeline_repair_proposal_unresolved") or []))
    latest["final_visible_repair_action_count"] = len(actions)
    latest["final_visible_repair_actions"] = actions
    latest["timeline_repair_proposal_unresolved"] = unresolved
    latest["replay_cycle_count"] = len(reports)
    latest["cycle_reports"] = reports
    latest["report_kind"] = "aroll_v21_final_visible_repair_replay_combined"
    return latest


def _old_caption_track_count(run_dir: Path) -> int | None:
    gate = (_safe_read_json(run_dir / "quality_gate_report.json") or {}).get("caption_alignment_gate") or {}
    return int(gate["visible_caption_track_count"]) if gate.get("visible_caption_track_count") is not None else None


def _old_caption_lane_count(run_dir: Path) -> int | None:
    gate = (_safe_read_json(run_dir / "quality_gate_report.json") or {}).get("caption_alignment_gate") or {}
    return int(gate["caption_lane_count"]) if gate.get("caption_lane_count") is not None else None


def _old_effective_speed_gate(old_quality: dict[str, Any]) -> dict[str, Any]:
    return dict(old_quality.get("effective_speed_gate") or {"gate_passed": True, "blocker_codes": []})


def _old_final_repeat_gate(old_quality: dict[str, Any]) -> dict[str, Any]:
    return dict(old_quality.get("final_repeat_convergence_gate") or {"gate_passed": True, "blocker_codes": []})


def _old_visual_pacing_gate(old_quality: dict[str, Any]) -> dict[str, Any]:
    return dict(old_quality.get("visual_pacing_gate") or {"gate_passed": True, "visual_pacing_executed": True, "blocker_codes": []})


def _old_semantic_gate(run_dir: Path, old_quality: dict[str, Any]) -> dict[str, Any]:
    if isinstance(old_quality.get("semantic_adjudication_gate"), dict):
        return dict(old_quality["semantic_adjudication_gate"])
    semantic = _safe_read_json(run_dir / "semantic_adjudication_report.json") or {}
    return {
        "semantic_adjudication_gate_passed": not bool(semantic.get("blocker_codes")),
        "semantic_request_count": int(semantic.get("semantic_request_count") or 0),
        "semantic_request_unresolved_count": int(semantic.get("semantic_request_unresolved_count") or 0),
        "fatal_semantic_issue_count": int(semantic.get("fatal_semantic_issue_count") or 0),
        "blocker_codes": list(semantic.get("blocker_codes") or []),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay only final-visible repair from an existing A-Roll V21 run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cycles", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = replay_final_visible_repair_from_run(
        args.run_dir,
        output_dir=args.output_dir,
        max_cycles=args.max_cycles,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
