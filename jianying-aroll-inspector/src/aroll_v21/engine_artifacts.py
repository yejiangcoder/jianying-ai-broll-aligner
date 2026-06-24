from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from aroll_v21.engine_report_compaction import _compact_runtime_report_payload, _resolved_semantic_decision_rows
from aroll_v21.engine_summary import build_run_summary
from aroll_v21.ir.models import RunReport, dataclass_to_dict

def write_run_artifacts(run_report: RunReport, output_dir: Path, *, report_profile: str = "standard") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = str(report_profile or "standard").strip().lower()
    if profile not in {"minimal", "standard", "debug"}:
        profile = "standard"
    effective_profile = "debug" if profile == "standard" and run_report.status != "ok" else profile
    blocked_by_stage = str((run_report.blocker_report.summary or {}).get("stage") or "")
    blocked_by_codes = [blocker.code for blocker in (run_report.blocker_report.blockers if run_report.blocker_report else [])]

    def not_reached(stage: str, blocked_by: str | None = None) -> dict[str, Any]:
        return {
            "stage": stage,
            "status": "not_reached",
            "blocked_by_stage": blocked_by or blocked_by_stage,
            "blocked_by_codes": blocked_by_codes,
            "items": [],
        }

    def reached_after(stage: str) -> bool:
        if run_report.status != "blocked":
            return True
        order = {
            "ingest": 0,
            "decision": 1,
            "compiler": 2,
            "renderer": 3,
            "writer": 4,
            "validate": 5,
        }
        blocked_order = order.get(blocked_by_stage, 99)
        return order[stage] <= blocked_order

    final_timeline_payload: Any = run_report.final_timeline
    final_edl_payload: Any = [
        {
            "clip_id": segment.segment_id,
            "source_material_id": segment.source_material_id,
            "source_segment_id": segment.source_segment_id,
            "source_start_us": segment.source_start_us,
            "source_end_us": segment.source_end_us,
            "target_start_us": segment.target_start_us,
            "target_duration_us": segment.target_end_us - segment.target_start_us,
            "word_ids": segment.word_ids,
            "text": segment.text,
            "decision_ids": segment.decision_ids,
        }
        for segment in run_report.final_timeline
    ]
    if not reached_after("compiler") or (run_report.status == "blocked" and blocked_by_stage == "decision"):
        blocked_by = "SemanticDecisionPlanner" if blocked_by_stage == "decision" else blocked_by_stage
        final_timeline_payload = not_reached("FinalTimelineCompiler", blocked_by)
        final_edl_payload = not_reached("FinalTimelineCompiler", blocked_by)

    captions_payload: Any = run_report.captions
    if not reached_after("renderer") or (run_report.status == "blocked" and blocked_by_stage in {"decision", "compiler"}):
        captions_payload = not_reached("SubtitleRenderer")

    canonical_template_payload: Any = (run_report.material_write_plan or {}).get("canonical_caption_template") or {}
    material_write_plan_payload: Any = run_report.material_write_plan
    if not reached_after("writer") or (run_report.status == "blocked" and blocked_by_stage in {"decision", "compiler", "writer"} and not run_report.material_write_plan):
        canonical_template_payload = not_reached("CaptionMaterialWriter")
        material_write_plan_payload = not_reached("CaptionMaterialWriter")

    validator_payload: Any = run_report.validator_report
    postwrite_payload: Any = run_report.postwrite_report
    if not run_report.validator_report:
        validator_payload = not_reached("ReadOnlyValidators")
        postwrite_payload = not_reached("PostwriteVerification")
    if effective_profile != "debug":
        postwrite_payload = _compact_runtime_report_payload(postwrite_payload)

    decision_plan = run_report.decision_plan
    local_policy_decisions = []
    deepseek_decisions = []
    if decision_plan is not None:
        local_policy_decisions = [
            dataclass_to_dict(item)
            for item in [*decision_plan.decisions, *decision_plan.split_decisions]
            if str(getattr(item, "source", "")) == "local_policy"
        ]
        deepseek_decisions = [
            dataclass_to_dict(item)
            for item in [*decision_plan.decisions, *decision_plan.split_decisions]
            if str(getattr(item, "source", "")) == "deepseek_semantic_planner"
        ]
    resolved_semantic_rows = _resolved_semantic_decision_rows(decision_plan)
    semantic_report_payload = run_report.decision_plan.semantic_adjudication_report if run_report.decision_plan else {}
    artifacts = {
        "source_graph.json": run_report.source_graph,
        "edit_units.json": run_report.source_graph.edit_units if run_report.source_graph else [],
        "repeat_clusters.json": run_report.repeat_clusters,
        "decision_plan.json": run_report.decision_plan,
        "semantic_request_payloads.json": (run_report.decision_plan.semantic_request_payloads if run_report.decision_plan else []),
        "semantic_decisions.json": (run_report.decision_plan.semantic_decision_rows if run_report.decision_plan else []),
        "semantic_decisions.resolved.json": resolved_semantic_rows,
        "semantic_decision_cache.json": resolved_semantic_rows,
        "semantic_adjudication_report.json": semantic_report_payload,
        "deepseek_batch_request.json": semantic_report_payload.get("deepseek_batch_request") or {},
        "deepseek_batch_response.json": semantic_report_payload.get("deepseek_batch_response") or {},
        "deepseek_batch_error.json": semantic_report_payload.get("deepseek_batch_error_payload") or (
            {"error": semantic_report_payload.get("deepseek_batch_error")}
            if semantic_report_payload.get("deepseek_batch_error")
            else {}
        ),
        "final_timeline.json": final_timeline_payload,
        "final_edl.json": final_edl_payload,
        "captions.json": captions_payload,
        "canonical_caption_template.json": canonical_template_payload,
        "material_write_plan.json": material_write_plan_payload,
        "validator_report.json": validator_payload,
        "postwrite_report.json": postwrite_payload,
        "final_caption_visible_repeat_gate.json": (validator_payload or {}).get("final_caption_visible_repeat_gate") if isinstance(validator_payload, dict) else not_reached("FinalCaptionVisibleRepeatGate"),
        "final_visible_caption_repair_report.json": (validator_payload or {}).get("final_visible_caption_repair_report") if isinstance(validator_payload, dict) else not_reached("FinalVisibleCaptionRepair"),
        "quality_gate_report.json": (validator_payload or {}).get("quality_gate_report") if isinstance(validator_payload, dict) else not_reached("QualityGate"),
        "blocker_report.json": run_report.blocker_report,
        "decision_trace.json": run_report.decision_trace,
        "local_policy_decisions.json": local_policy_decisions,
        "deepseek_decisions.json": deepseek_decisions,
        "run_summary.json": build_run_summary(run_report),
        "run_report.json": run_report,
    }
    repair_payload = (validator_payload or {}).get("final_visible_caption_repair_report") if isinstance(validator_payload, dict) else {}
    semantic_junk_report = repair_payload.get("pre_visible_semantic_junk_report") if isinstance(repair_payload, dict) else {}
    if isinstance(semantic_junk_report, dict):
        artifacts["quality/pre_visible_semantic_junk_report.json"] = semantic_junk_report
        artifacts["quality/semantic_junk_candidates.json"] = list(semantic_junk_report.get("pre_visible_semantic_junk_candidates") or [])
        artifacts["quality/quality_defect_ledger.json"] = {
            "ledger_kind": "quality_defect_ledger_runtime_seed",
            "source": "pre_visible_semantic_junk_candidate_detector",
            "candidate_count": int(semantic_junk_report.get("pre_visible_semantic_junk_candidate_count") or 0),
            "candidates": list(semantic_junk_report.get("pre_visible_semantic_junk_candidates") or []),
        }
    compressed_artifacts: dict[str, Any] = {}
    if effective_profile == "minimal":
        artifacts = {
            "blocker_report.json": run_report.blocker_report,
            "writeback_report.json": postwrite_payload,
        }
    elif effective_profile == "standard":
        compressed_artifacts = {
            "source_graph.json.gz": run_report.source_graph,
            "validator_report.json.gz": validator_payload,
        }
        for debug_name in (
            "run_report.json",
            "source_graph.json",
            "validator_report.json",
            "deepseek_batch_request.json",
            "deepseek_batch_response.json",
        ):
            artifacts.pop(debug_name, None)
        if not semantic_report_payload.get("deepseek_batch_error"):
            artifacts.pop("deepseek_batch_error.json", None)
    for name, payload in artifacts.items():
        artifact_path = output_dir / name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(dataclass_to_dict(payload), ensure_ascii=False, indent=2), "utf-8")
    for name, payload in compressed_artifacts.items():
        artifact_path = output_dir / name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(artifact_path, "wt", encoding="utf-8") as f:
            json.dump(dataclass_to_dict(payload), f, ensure_ascii=False, separators=(",", ":"))

