from __future__ import annotations

from typing import Any

from aroll_v21.contracts import (
    CaptionAlignmentReport,
    EffectiveSpeedGateReport,
    FinalRepeatConvergenceReport,
    QualityGateReport,
    VisualPacingReport,
    contract_to_dict,
)


def build_quality_gate_report(
    *,
    effective_speed_gate: dict[str, Any] | None = None,
    final_repeat_convergence_gate: dict[str, Any] | None = None,
    final_caption_visible_repeat_gate: dict[str, Any] | None = None,
    semantic_adjudication_gate: dict[str, Any] | None = None,
    visual_pacing_gate: dict[str, Any] | None = None,
    caption_alignment_gate: dict[str, Any] | None = None,
    final_timeline_quality_guard_gate: dict[str, Any] | None = None,
    prewrite_projection_gate: dict[str, Any] | None = None,
    ready_for_user_manual_qc_preconditions_passed: bool = False,
) -> dict[str, Any]:
    missing = []
    if effective_speed_gate is None:
        missing.append("effective_speed_gate")
    if final_repeat_convergence_gate is None:
        missing.append("final_repeat_convergence_gate")
    if final_caption_visible_repeat_gate is None:
        missing.append("final_caption_visible_repeat_gate")
    if semantic_adjudication_gate is None:
        missing.append("semantic_adjudication_gate")
    if visual_pacing_gate is None:
        missing.append("visual_pacing_gate")
    if caption_alignment_gate is None:
        missing.append("caption_alignment_gate")
    if final_timeline_quality_guard_gate is None:
        missing.append("final_timeline_quality_guard_gate")
    speed = _normalize_effective_speed_gate(
        effective_speed_gate
        or (contract_to_dict(EffectiveSpeedGateReport(gate_passed=False, blocker_codes=["V21_QUALITY_GATE_MISSING_REQUIRED_GATE"])))
    )
    repeat = final_repeat_convergence_gate or contract_to_dict(FinalRepeatConvergenceReport(gate_passed=False, blocker_codes=["V21_QUALITY_GATE_MISSING_REQUIRED_GATE"]))
    visual = visual_pacing_gate or contract_to_dict(VisualPacingReport(gate_passed=False, blocker_codes=["V21_QUALITY_GATE_MISSING_REQUIRED_GATE"]))
    caption = caption_alignment_gate or contract_to_dict(CaptionAlignmentReport(gate_passed=False, blocker_codes=["V21_QUALITY_GATE_MISSING_REQUIRED_GATE"]))
    semantic = semantic_adjudication_gate or {
        "semantic_adjudication_gate_passed": False,
        "semantic_request_count": 0,
        "semantic_request_unresolved_count": 0,
        "fatal_semantic_issue_count": 0,
        "deepseek_provider_configured": False,
        "deepseek_provider_called_count": 0,
        "deepseek_provider_skipped_count": 0,
        "deepseek_provider_skipped_reasons": {},
        "semantic_decision_cache_used": False,
        "semantic_auto_route_count": 0,
        "semantic_local_decision_count": 0,
        "semantic_provider_required_count": 0,
        "deterministic_baseline_refused_count": 0,
        "blocker_codes": [],
    }
    blocker_codes = []
    for report in (speed, repeat, visual, caption):
        blocker_codes.extend(str(code) for code in report.get("blocker_codes") or [])
    if final_repeat_convergence_gate is not None and repeat.get("detector_report_present") is False:
        blocker_codes.append("V21_FINAL_REPEAT_DETECTOR_REPORT_MISSING")
    if effective_speed_gate is not None and not _effective_speed_satisfied(speed) and not list(speed.get("blocker_codes") or []):
        blocker_codes.append("V21_EFFECTIVE_SPEED_GATE_FAILED")
    if isinstance(final_caption_visible_repeat_gate, dict):
        blocker_codes.extend(str(code) for code in final_caption_visible_repeat_gate.get("blocker_codes") or [])
    else:
        blocker_codes.append("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_MISSING")
    final_timeline_quality_passed = False
    if isinstance(final_timeline_quality_guard_gate, dict):
        final_timeline_quality_passed = bool(final_timeline_quality_guard_gate.get("gate_passed"))
        blocker_codes.extend(str(code) for code in final_timeline_quality_guard_gate.get("blocker_codes") or [])
        if not final_timeline_quality_passed:
            blocker_codes.append("V21_FINAL_TIMELINE_QUALITY_GUARD_FAILED")
    else:
        blocker_codes.append("V21_FINAL_TIMELINE_QUALITY_GUARD_MISSING")
    prewrite_projection_passed = True
    if isinstance(prewrite_projection_gate, dict):
        prewrite_projection_passed = bool(
            prewrite_projection_gate.get(
                "prewrite_projected_write_view_gate_passed",
                prewrite_projection_gate.get("gate_passed"),
            )
        )
        projection_blocker_codes = prewrite_projection_gate.get("prewrite_projected_write_view_blocker_codes") or []
        blocker_codes.extend(str(code) for code in projection_blocker_codes)
        if not prewrite_projection_passed:
            blocker_codes.append("V21_PREWRITE_PROJECTED_WRITE_VIEW_FAILED")
    if semantic_adjudication_gate is not None:
        blocker_codes.extend(str(code) for code in semantic.get("blocker_codes") or [])
        if not bool(semantic.get("semantic_adjudication_gate_passed")):
            blocker_codes.append("V21_SEMANTIC_ADJUDICATION_GATE_FAILED")
        if int(semantic.get("fatal_semantic_issue_count") or 0) > 0:
            blocker_codes.append("V21_FATAL_SEMANTIC_ISSUE_UNRESOLVED")
        if int(semantic.get("semantic_request_unresolved_count") or 0) > 0:
            blocker_codes.append("V21_SEMANTIC_REQUEST_UNRESOLVED")
    if missing:
        blocker_codes.append("V21_QUALITY_GATE_MISSING_REQUIRED_GATE")
    if visual_pacing_gate is not None and not bool(visual.get("visual_pacing_executed")):
        blocker_codes.append("V21_VISUAL_PACING_NOT_EXECUTED")
    visual_blocking_short = int(
        visual.get("visual_short_segment_count_lt_1200ms_after_blocking")
        if visual.get("visual_short_segment_count_lt_1200ms_after_blocking") is not None
        else max(
            0,
            int(visual.get("visual_short_segment_count_lt_1200ms_after") or visual.get("visual_short_segment_count_lt_1200ms") or 0)
            - int(visual.get("semantic_bridge_short_segment_count") or 0),
        )
    )
    visual_allowed_short = int(visual.get("visual_pacing_allowed_short_segment_threshold") or 0)
    if visual_pacing_gate is not None and visual_blocking_short > visual_allowed_short:
        blocker_codes.append("V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN")
    if visual_pacing_gate is not None and not bool(visual.get("visual_merge_safety_gate_passed")):
        blocker_codes.append("V21_VISUAL_PACING_UNSAFE_MERGE")
    if visual_pacing_gate is not None and int(visual.get("unsafe_merge_group_count") or 0) > 0:
        blocker_codes.append("V21_VISUAL_PACING_UNSAFE_MERGE")
    if visual_pacing_gate is not None and int(visual.get("dropped_content_reintroduced_count") or 0) > 0:
        blocker_codes.append("V21_VISUAL_PACING_UNSAFE_MERGE")
    semantic_bridge_cap = int(visual.get("semantic_bridge_cap") or 8)
    if visual_pacing_gate is not None and (
        int(visual.get("semantic_bridge_short_segment_count") or 0) > semantic_bridge_cap
        or int(visual.get("semantic_bridge_safe_merge_candidate_count") or 0) > 0
    ):
        blocker_codes.append("V21_VISUAL_SEMANTIC_BRIDGE_ABUSE")
    if visual_pacing_gate is not None and bool(visual.get("cut_density_gate_enabled")) and (
        float(visual.get("cuts_per_minute") or 0.0) > float(((visual.get("cut_density_thresholds") or {}).get("max_cuts_per_minute") or 30.0))
        or int(visual.get("max_cuts_in_5s") or 0) > int(((visual.get("cut_density_thresholds") or {}).get("max_cuts_in_5s") or 5))
        or int(visual.get("burst_cut_count") or 0) > int(((visual.get("cut_density_thresholds") or {}).get("max_burst_cut_count") or 0))
    ):
        blocker_codes.append("V21_VISUAL_CUT_DENSITY_FAILED")
    caption_gui_failed = (
        not bool(caption.get("caption_gui_track_gate_passed"))
        if "caption_gui_track_gate_passed" in caption
        else (
            ("visible_caption_track_count" in caption and int(caption.get("visible_caption_track_count") or 0) != 1)
            or ("caption_lane_count" in caption and int(caption.get("caption_lane_count") or 0) != 1)
            or int(caption.get("orphan_caption_count") or 0) > 0
            or int(caption.get("floating_caption_count") or 0) > 0
            or int(caption.get("caption_without_video_container_count") or 0) > 0
        )
    )
    if caption_alignment_gate is not None and caption_gui_failed:
        blocker_codes.append("V21_CAPTION_GUI_TRACK_GATE_FAILED")
    readability_failed = (
        not bool(caption.get("subtitle_readability_gate_passed"))
        if "subtitle_readability_gate_passed" in caption
        else (
            int(caption.get("tiny_caption_fatal_count") or 0) > 0
            or int(caption.get("tiny_caption_residual_density_window_count") or 0) > 0
            or int(caption.get("subtitle_hard_max_char_count") or 0) > 0
            or int(caption.get("subtitle_interval_too_short_count") or 0) > 0
            or int(caption.get("subtitle_interval_too_long_count") or 0) > 0
            or int(caption.get("caption_burst_density_count") or 0) > 0
        )
    )
    if caption_alignment_gate is not None and readability_failed:
        blocker_codes.append("V21_SUBTITLE_READABILITY_GATE_FAILED")
    core_gate_passed = _effective_speed_satisfied(speed) and all(
        bool(report.get("gate_passed"))
        for report in (repeat, visual, caption)
    )
    caption_repeat_passed = (
        bool(final_caption_visible_repeat_gate.get("gate_passed"))
        if isinstance(final_caption_visible_repeat_gate, dict)
        else False
    )
    semantic_passed = bool(semantic.get("semantic_adjudication_gate_passed"))
    gate_passed = (
        core_gate_passed
        and caption_repeat_passed
        and semantic_passed
        and final_timeline_quality_passed
        and prewrite_projection_passed
        and not blocker_codes
    )
    payload = contract_to_dict(
        QualityGateReport(
            gate_passed=gate_passed,
            effective_speed_gate=_effective_speed_contract(speed),
            final_repeat_convergence_gate=_final_repeat_contract(repeat),
            visual_pacing_gate=_visual_contract(visual),
            caption_alignment_gate=_caption_contract(caption),
            ready_for_user_manual_qc_preconditions_passed=ready_for_user_manual_qc_preconditions_passed and gate_passed,
            blocker_codes=sorted(set(blocker_codes)),
        )
    )
    payload.update(
        {
            "semantic_adjudication_gate_passed": semantic_passed,
            "semantic_request_count": int(semantic.get("semantic_request_count") or 0),
            "semantic_request_unresolved_count": int(semantic.get("semantic_request_unresolved_count") or 0),
            "fatal_semantic_issue_count": int(semantic.get("fatal_semantic_issue_count") or 0),
            "deepseek_provider_configured": bool(semantic.get("deepseek_provider_configured")),
            "deepseek_provider_called_count": int(semantic.get("deepseek_provider_called_count") or 0),
            "deepseek_provider_error": str(semantic.get("deepseek_provider_error") or ""),
            "deepseek_batch_enabled": bool(semantic.get("deepseek_batch_enabled")),
            "deepseek_batch_request_count": int(semantic.get("deepseek_batch_request_count") or 0),
            "deepseek_batch_attempt_count": int(semantic.get("deepseek_batch_attempt_count") or 0),
            "deepseek_batch_retry_count": int(semantic.get("deepseek_batch_retry_count") or 0),
            "deepseek_batch_issue_count": int(semantic.get("deepseek_batch_issue_count") or 0),
            "deepseek_batch_resolved_count": int(semantic.get("deepseek_batch_resolved_count") or 0),
            "deepseek_batch_unresolved_count": int(semantic.get("deepseek_batch_unresolved_count") or 0),
            "deepseek_batch_missing_issue_ids": list(semantic.get("deepseek_batch_missing_issue_ids") or []),
            "deepseek_batch_error": str(semantic.get("deepseek_batch_error") or ""),
            "deepseek_provider_skipped_count": int(semantic.get("deepseek_provider_skipped_count") or 0),
            "deepseek_provider_skipped_reasons": dict(semantic.get("deepseek_provider_skipped_reasons") or {}),
            "semantic_decision_cache_used": bool(semantic.get("semantic_decision_cache_used")),
            "commit_reused_semantic_cache": bool(semantic.get("commit_reused_semantic_cache")),
            "semantic_cache_input_hash": str(semantic.get("semantic_cache_input_hash") or ""),
            "semantic_cache_issue_count": int(semantic.get("semantic_cache_issue_count") or 0),
            "semantic_cache_resolved_count": int(semantic.get("semantic_cache_resolved_count") or 0),
            "semantic_cache_unresolved_count": int(semantic.get("semantic_cache_unresolved_count") or 0),
            "semantic_auto_route_count": int(semantic.get("semantic_auto_route_count") or 0),
            "semantic_local_decision_count": int(semantic.get("semantic_local_decision_count") or 0),
            "semantic_provider_required_count": int(semantic.get("semantic_provider_required_count") or 0),
            "deterministic_baseline_refused_count": int(semantic.get("deterministic_baseline_refused_count") or 0),
            "final_visible_repeat_advisory_count": int(semantic.get("final_visible_repeat_advisory_count") or 0),
            "final_visible_repeat_advisory_result_count": int(semantic.get("final_visible_repeat_advisory_result_count") or 0),
            "final_visible_repeat_advisory_decision_counts": dict(semantic.get("final_visible_repeat_advisory_decision_counts") or {}),
            "final_visible_repeat_advisory_keep_count": int(semantic.get("final_visible_repeat_advisory_keep_count") or 0),
            "final_visible_repeat_advisory_drop_candidate_count": int(semantic.get("final_visible_repeat_advisory_drop_candidate_count") or 0),
            "final_visible_repeat_advisory_review_count": int(semantic.get("final_visible_repeat_advisory_review_count") or 0),
            "final_visible_repeat_advisory_unresolved_count": int(semantic.get("final_visible_repeat_advisory_unresolved_count") or 0),
            "final_visible_repeat_advisory_applied_count": int(semantic.get("final_visible_repeat_advisory_applied_count") or 0),
            "final_visible_repeat_advisory_provider_called_count": int(semantic.get("final_visible_repeat_advisory_provider_called_count") or 0),
            "final_visible_repeat_advisory_policy": str(semantic.get("final_visible_repeat_advisory_policy") or ""),
            "effective_speed_gate_present": effective_speed_gate is not None,
            "final_repeat_convergence_gate_present": final_repeat_convergence_gate is not None,
            "semantic_adjudication_gate_present": semantic_adjudication_gate is not None,
            "visual_pacing_gate_present": visual_pacing_gate is not None,
            "caption_alignment_gate_present": caption_alignment_gate is not None,
            "final_caption_visible_repeat_gate_present": final_caption_visible_repeat_gate is not None,
            "final_timeline_quality_guard_gate_present": final_timeline_quality_guard_gate is not None,
            "final_timeline_quality_guard_gate_passed": final_timeline_quality_passed,
            "prewrite_projected_write_view_gate_present": prewrite_projection_gate is not None,
            "prewrite_projected_write_view_gate_passed": prewrite_projection_passed,
            "missing_required_gates": missing,
            "post_write_actual_draft_audit_required_on_commit": True,
        }
    )
    if isinstance(prewrite_projection_gate, dict):
        payload["prewrite_projected_write_view_gate"] = {
            key: prewrite_projection_gate[key]
            for key in (
                "prewrite_projected_write_view_applied",
                "prewrite_projected_write_view_gate_passed",
                "prewrite_projection_video_write_plan_gapless",
                "prewrite_projection_error_code",
                "prewrite_projection_error_message",
                "prewrite_projected_write_view_blocker_codes",
            )
            if key in prewrite_projection_gate
        }
    if isinstance(final_timeline_quality_guard_gate, dict):
        payload["final_timeline_quality_guard_gate"] = {
            key: final_timeline_quality_guard_gate[key]
            for key in (
                "gate_passed",
                "write_gate_passed",
                "quality_guard_passed",
                "candidate_count",
                "high_risk_candidate_count",
                "blocking_candidate_count",
                "physical_blocking_candidate_count",
                "caption_masking_candidate_count",
                "candidate_type_counts",
                "blocking_candidate_type_counts",
                "blocker_codes",
                "blocking_candidates",
                "repair_intent_count",
                "repair_intent_type_counts",
            )
            if key in final_timeline_quality_guard_gate
        }
    if isinstance(final_caption_visible_repeat_gate, dict):
        payload["final_caption_visible_repeat_gate"] = {
            key: final_caption_visible_repeat_gate[key]
            for key in (
                "gate_passed",
                "blocker_codes",
                "visible_repeat_candidate_count",
                "visible_repeat_fatal_candidate_count",
                "visible_repeat_warning_candidate_count",
                "visible_repeat_allow_candidate_count",
                "repeat_classification_candidate_count",
                "repeat_classification_candidates",
                "repeat_semantic_arbitration_candidate_count",
                "repeat_semantic_arbitration_request_count",
                "repeat_semantic_arbitration_candidate_ids",
                "repeat_semantic_arbitration_candidates",
                "repeat_semantic_arbitration_request_payloads",
                "repeat_semantic_arbitration_mode",
                "repeat_semantic_arbitration_provider",
                "visible_repeat_warning_candidates",
                "visible_repeat_allow_candidates",
                "containment_repeat_count",
                "containment_repeat_raw_count",
                "prefix_suffix_overlap_count",
                "ngram_repeat_count",
                "ngram_repeat_raw_count",
                "near_duplicate_visible_caption_count",
                "modifier_redundancy_residual_count",
                "self_repair_aborted_phrase_count",
                "dangling_prefix_suffix_count",
                "semantic_garbage_or_asr_suspect_count",
                "semantic_integrity_count",
                "semantic_integrity_reason_counts",
                "cross_caption_semantic_containment_count",
                "cross_caption_semantic_containment_raw_count",
                "restart_repeat_visible_count",
                "visible_repeat_candidates",
                "containment_repeat_candidates",
                "prefix_suffix_overlap_candidates",
                "ngram_repeat_candidates",
                "near_duplicate_visible_caption_candidates",
                "modifier_redundancy_residual_candidates",
                "self_repair_aborted_phrase_candidates",
                "dangling_prefix_suffix_candidates",
                "semantic_garbage_or_asr_suspect_candidates",
                "semantic_integrity_candidates",
                "cross_caption_semantic_containment_candidates",
                "restart_repeat_visible_candidates",
                "final_caption_visible_repeat_gate_enabled",
                "final_visible_recheck_allowed_decisions",
                "final_visible_repair_success",
                "final_visible_repair_unresolved_count",
                "final_visible_repair_final_timeline_counts",
                "final_visible_effective_caption_count",
                "caption_only_materialized_merge_count",
                "caption_only_consumed_caption_ids",
                "caption_only_materialized_merges",
                "ngram_size",
                "prefix_suffix_min_overlap",
                "near_duplicate_ratio",
            )
            if key in final_caption_visible_repeat_gate
        }
    if isinstance(speed, dict):
        payload["effective_speed_gate"].update(
            {
                key: speed[key]
                for key in (
                    "effective_speed_projected_row_missing_count",
                    "effective_speed_projected_row_count",
                    "prewrite_pending",
                    "not_applicable",
                    "not_applicable_reason",
                    "safe_handle_policy_enabled",
                    "lead_handle_requested_count",
                    "tail_handle_requested_count",
                    "lead_handle_applied_count",
                    "tail_handle_applied_count",
                    "segments_with_no_lead_handle",
                    "segments_with_no_tail_handle",
                    "handle_blocked_count",
                    "handle_blocked_reasons",
                )
                if key in speed
            }
        )
    if isinstance(final_repeat_convergence_gate, dict):
        payload["final_repeat_convergence_gate"].update(
            {
                key: final_repeat_convergence_gate[key]
                for key in (
                    "detector_report_present",
                    "dropped_cluster_count",
                    "dropped_segment_count",
                    "final_repeat_dropped_segment_count",
                    "clusters_per_dropped_segment",
                )
                if key in final_repeat_convergence_gate
            }
        )
    if isinstance(visual_pacing_gate, dict):
        payload["visual_pacing_gate"].update(
            {
                key: visual_pacing_gate[key]
                for key in (
                    "visual_pacing_executed",
                    "visual_pacing_merge_attempted_count",
                    "visual_pacing_merged_count",
                    "visual_short_segment_count_lt_1200ms_before",
                    "visual_short_segment_count_lt_1200ms_after",
                    "visual_short_segment_count_lt_1200ms_after_blocking",
                    "semantic_bridge_short_segment_count",
                    "visual_pacing_allowed_short_segment_threshold",
                    "visual_pacing_allowed_short_segment_policy",
                    "visual_pacing_blocker_codes",
                    "residual_visual_short_segments",
                    "semantic_bridge_short_segment_details",
                    "semantic_bridge_reason_counts",
                    "semantic_bridge_cap",
                    "semantic_bridge_safe_merge_candidate_count",
                    "semantic_bridge_safe_merge_candidates",
                    "cuts_per_minute",
                    "max_cuts_in_5s",
                    "burst_cut_count",
                    "cut_density_gate_enabled",
                    "cut_density_gate_passed",
                    "cut_density_thresholds",
                    "cut_density_window_us",
                    "visual_merge_safety_gate_passed",
                    "unsafe_merge_group_count",
                    "dropped_content_reintroduced_count",
                    "max_bridged_gap_us",
                    "total_bridged_gap_us",
                    "unspoken_bridge_ratio",
                    "visual_merge_safety_report",
                    "visual_merge_groups",
                )
                if key in visual_pacing_gate
            }
        )
    if isinstance(caption_alignment_gate, dict):
        payload["caption_alignment_gate"].update(
            {
                key: caption_alignment_gate[key]
                for key in (
                    "caption_too_short_count",
                    "caption_cross_primary_window_count",
                    "prewrite_uncaptioned_spoken_word_count",
                    "prewrite_uncaptioned_spoken_segment_count",
                    "prewrite_uncaptioned_spoken_word_rows",
                    "missing_final_timeline_caption_word_count",
                    "missing_final_timeline_caption_word_ids",
                    "caption_alignment_ok",
                    "caption_gui_track_gate_passed",
                    "visible_caption_track_count",
                    "caption_lane_count",
                    "orphan_caption_count",
                    "floating_caption_count",
                    "caption_render_order_stable",
                    "subtitle_readability_gate_passed",
                    "subtitle_interval_overlap_count",
                    "subtitle_interval_gap_violation_count",
                    "subtitle_interval_too_short_count",
                    "subtitle_interval_too_long_count",
                    "subtitle_hard_max_char_count",
                    "captions_le_3_chars",
                    "captions_le_3_chars_cap",
                    "tiny_caption_classification_enabled",
                    "tiny_caption_classification_count",
                    "tiny_caption_fatal_count",
                    "tiny_caption_warning_count",
                    "tiny_caption_allow_count",
                    "tiny_caption_classifications",
                    "tiny_caption_residual_density_window_count",
                    "tiny_caption_residual_density_windows",
                    "tiny_caption_residual_density_window_us",
                    "tiny_caption_residual_density_threshold",
                    "caption_density_per_minute",
                    "max_captions_in_5s",
                    "caption_burst_density_count",
                    "caption_density_window_us",
                    "max_captions_in_5s_threshold",
                    "tiny_caption_details",
                    "subtitle_hard_max_char_details",
                    "subtitle_too_short_details",
                    "subtitle_too_long_details",
                    "residual_too_short_captions",
                    "residual_one_char_captions",
                )
                if key in caption_alignment_gate
            }
        )
    return payload


def _effective_speed_contract(report: dict[str, Any]) -> EffectiveSpeedGateReport:
    return EffectiveSpeedGateReport(
        gate_passed=bool(report.get("gate_passed", False)),
        expected_speeds=list(report.get("expected_speeds") or []),
        effective_speed_min=report.get("effective_speed_min"),
        effective_speed_max=report.get("effective_speed_max"),
        effective_speed_drift_count=int(report.get("effective_speed_drift_count") or 0),
        segment_reports=list(report.get("segment_reports") or []),  # type: ignore[arg-type]
        blocker_codes=list(report.get("blocker_codes") or []),
    )


def _normalize_effective_speed_gate(report: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(report)
    if (
        bool(normalized.get("prewrite_pending"))
        and normalized.get("effective_speed_min") is None
        and normalized.get("effective_speed_max") is None
    ):
        normalized["gate_passed"] = False
        normalized["not_applicable"] = True
        normalized.setdefault("not_applicable_reason", "prewrite_source_binding_pending")
    return normalized


def _effective_speed_satisfied(report: dict[str, Any]) -> bool:
    if bool(report.get("not_applicable")):
        return bool(report.get("prewrite_pending")) and bool(str(report.get("not_applicable_reason") or ""))
    return bool(report.get("gate_passed"))


def _final_repeat_contract(report: dict[str, Any]) -> FinalRepeatConvergenceReport:
    return FinalRepeatConvergenceReport(
        enabled=bool(report.get("enabled")),
        iterations=int(report.get("iterations") or 0),
        dropped_cluster_ids=list(report.get("dropped_cluster_ids") or []),
        dropped_segment_indices=[int(item) for item in report.get("dropped_segment_indices") or []],
        final_repeat_high_count_before=int(report.get("final_repeat_high_count_before") or 0),
        final_repeat_high_count_after=int(report.get("final_repeat_high_count_after") or 0),
        unresolved_high_cluster_ids=list(report.get("unresolved_high_cluster_ids") or []),
        gate_passed=bool(report.get("gate_passed", False)),
        blocker_codes=list(report.get("blocker_codes") or []),
    )


def _visual_contract(report: dict[str, Any]) -> VisualPacingReport:
    return VisualPacingReport(
        gate_passed=bool(report.get("gate_passed", False)),
        final_video_segment_count=int(report.get("final_video_segment_count") or 0),
        caption_count=int(report.get("caption_count") or 0),
        visual_short_segment_count_lt_1200ms=int(report.get("visual_short_segment_count_lt_1200ms") or 0),
        median_segment_duration_us=int(report.get("median_segment_duration_us") or 0),
        p10_segment_duration_us=int(report.get("p10_segment_duration_us") or 0),
        caption_per_video_segment_ratio=float(report.get("caption_per_video_segment_ratio") or 0.0),
        blocker_codes=list(report.get("blocker_codes") or []),
    )


def _caption_contract(report: dict[str, Any]) -> CaptionAlignmentReport:
    return CaptionAlignmentReport(
        gate_passed=bool(report.get("gate_passed", False)),
        caption_count=int(report.get("caption_count") or 0),
        caption_outside_video_count=int(report.get("caption_outside_video_count") or 0),
        caption_overlap_count=int(report.get("caption_overlap_count") or 0),
        caption_too_short_count=int(report.get("caption_too_short_count") or 0),
        one_char_caption_count=int(report.get("one_char_caption_count") or 0),
        caption_without_video_container_count=int(report.get("caption_without_video_container_count") or 0),
        caption_cross_primary_window_count=int(report.get("caption_cross_primary_window_count") or 0),
        blocker_codes=list(report.get("blocker_codes") or []),
    )
