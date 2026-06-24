from __future__ import annotations

from typing import Any

from aroll_v21.ir.models import RunReport

def build_run_summary(run_report: RunReport, *, commit_performed: bool = False, write_status: str = "") -> dict[str, Any]:
    validator = run_report.validator_report or {}
    style = validator.get("subtitle_style_validator") or {}
    coverage = validator.get("subtitle_coverage_validator") or {}
    final_repeat = validator.get("final_repeat_validator") or {}
    final_repeat_convergence = validator.get("final_repeat_convergence_gate") or {}
    final_caption_visible_repeat = validator.get("final_caption_visible_repeat_gate") or {}
    final_visible_caption_repair = validator.get("final_visible_caption_repair_report") or {}
    hidden = validator.get("hidden_audio_repeat_validator") or {}
    safe_cut = validator.get("safe_cut_validator") or {}
    rough_cut = validator.get("rough_cut_quality_validator") or {}
    visual_pacing = validator.get("visual_pacing_gate") or {}
    caption_alignment = validator.get("caption_alignment_gate") or {}
    quality_gate = validator.get("quality_gate_report") or {}
    semantic_gate = validator.get("semantic_final_review_validator") or {}
    effective_speed_gate = quality_gate.get("effective_speed_gate") if isinstance(quality_gate.get("effective_speed_gate"), dict) else {}
    postwrite = run_report.postwrite_report or {}
    blocker_summary = run_report.blocker_report.summary if run_report.blocker_report else {}
    ingest_metadata = blocker_summary.get("ingest_metadata") if isinstance(blocker_summary.get("ingest_metadata"), dict) else {}
    blockers = run_report.blocker_report.blockers if run_report.blocker_report else []
    mode = str((run_report.blocker_report.summary or {}).get("mode") or "")
    semantic_unresolved_count = int(run_report.decision_plan.semantic_unresolved_count if run_report.decision_plan else 0)
    semantic_report = run_report.decision_plan.semantic_adjudication_report if run_report.decision_plan else {}
    semantic_mode = str(
        postwrite.get("semantic_mode")
        or semantic_report.get("semantic_mode")
        or quality_gate.get("semantic_mode")
        or semantic_gate.get("semantic_mode")
        or "default"
    )
    semantic_request_count = int(
        quality_gate.get(
            "semantic_request_count",
            semantic_gate.get("semantic_request_count", semantic_report.get("semantic_request_count") or 0),
        )
        or 0
    )
    semantic_request_unresolved_count = int(
        quality_gate.get(
            "semantic_request_unresolved_count",
            semantic_gate.get("semantic_request_unresolved_count", semantic_report.get("semantic_request_unresolved_count") or 0),
        )
        or 0
    )
    fatal_semantic_issue_count = int(
        quality_gate.get(
            "fatal_semantic_issue_count",
            semantic_gate.get("fatal_semantic_issue_count", semantic_report.get("fatal_semantic_issue_count") or 0),
        )
        or 0
    )
    deepseek_provider_configured = bool(
        quality_gate.get(
            "deepseek_provider_configured",
            semantic_gate.get("deepseek_provider_configured", semantic_report.get("deepseek_provider_configured")),
        )
    )
    deepseek_provider_called_count = int(
        quality_gate.get(
            "deepseek_provider_called_count",
            semantic_gate.get("deepseek_provider_called_count", semantic_report.get("deepseek_provider_called_count") or 0),
        )
        or 0
    )
    deepseek_provider_error = str(
        quality_gate.get(
            "deepseek_provider_error",
            semantic_gate.get("deepseek_provider_error", semantic_report.get("deepseek_provider_error") or ""),
        )
        or ""
    )

    def semantic_int(key: str) -> int:
        return int(
            semantic_report.get(key)
            or semantic_gate.get(key)
            or quality_gate.get(key)
            or 0
        )

    def semantic_bool(key: str) -> bool:
        return bool(
            semantic_report.get(key)
            if key in semantic_report
            else semantic_gate.get(key)
            if key in semantic_gate
            else quality_gate.get(key)
        )

    def semantic_list(key: str) -> list[Any]:
        value = semantic_report.get(key) or semantic_gate.get(key) or quality_gate.get(key) or []
        return list(value) if isinstance(value, list) else []

    def semantic_str(key: str) -> str:
        return str(semantic_report.get(key) or semantic_gate.get(key) or quality_gate.get(key) or "")

    deepseek_provider_skipped_count = int(
        semantic_report.get("deepseek_provider_skipped_count")
        or semantic_gate.get("deepseek_provider_skipped_count")
        or quality_gate.get("deepseek_provider_skipped_count")
        or 0
    )
    deepseek_provider_skipped_reasons = dict(
        semantic_report.get("deepseek_provider_skipped_reasons")
        or semantic_gate.get("deepseek_provider_skipped_reasons")
        or quality_gate.get("deepseek_provider_skipped_reasons")
        or {}
    )
    semantic_decision_cache_used = bool(
        semantic_report.get("semantic_decision_cache_used")
        or semantic_gate.get("semantic_decision_cache_used")
        or quality_gate.get("semantic_decision_cache_used")
    )
    semantic_auto_route_count = int(
        max(
            int(quality_gate.get("semantic_auto_route_count") or 0),
            int(semantic_gate.get("semantic_auto_route_count") or 0),
            int(semantic_report.get("semantic_auto_route_count") or 0),
        )
    )
    semantic_local_decision_count = int(
        max(
            int(quality_gate.get("semantic_local_decision_count") or 0),
            int(semantic_gate.get("semantic_local_decision_count") or 0),
            int(semantic_report.get("semantic_local_decision_count") or 0),
        )
    )
    semantic_provider_required_count = int(
        max(
            int(quality_gate.get("semantic_provider_required_count") or 0),
            int(semantic_gate.get("semantic_provider_required_count") or 0),
            int(semantic_report.get("semantic_provider_required_count") or 0),
        )
    )
    deterministic_baseline_refused_count = int(
        max(
            int(quality_gate.get("deterministic_baseline_refused_count") or 0),
            int(semantic_gate.get("deterministic_baseline_refused_count") or 0),
            int(semantic_report.get("deterministic_baseline_refused_count") or 0),
        )
    )
    write_blocker_count = sum(1 for blocker in blockers if blocker.severity == "write_blocker")
    semantic_write_allowed = bool(
        run_report.decision_plan
        and run_report.decision_plan.semantic_unresolved_count == 0
        and run_report.decision_plan.write_allowed
    )
    validator_write_allowed = bool(validator.get("validator_report_ok"))
    writer_fallback_count = int((run_report.material_write_plan or {}).get("writer_fallback_count") or 0)
    ready_for_write = bool(run_report.status == "ok" and semantic_write_allowed and validator_write_allowed and writer_fallback_count == 0)
    sacrificial_postwrite_skip = bool(postwrite.get("postwrite_decrypt_skipped_for_sacrificial_draft"))
    source_stage_blocked = any(blocker.layer in {"operator", "ingest"} for blocker in blockers)
    audit = postwrite.get("post_write_actual_draft_audit") if isinstance(postwrite.get("post_write_actual_draft_audit"), dict) else {}
    post_write_audit_required_on_commit = True
    post_write_audit_commit_required = bool(commit_performed or postwrite.get("commit_performed") or postwrite.get("WRITE_SUCCESS") or postwrite.get("writeback_success"))
    post_write_audit_executed = bool(postwrite.get("post_write_actual_draft_audit_executed") or audit.get("executed"))
    post_write_audit_gate_passed = bool(postwrite.get("post_write_actual_draft_audit_gate_passed") or audit.get("gate_passed"))
    post_write_audit_ready = (not post_write_audit_commit_required) or (post_write_audit_executed and post_write_audit_gate_passed)
    ready_for_user_manual_qc = bool(postwrite.get("ready_for_user_manual_qc")) and post_write_audit_ready
    writeback_success = bool(postwrite.get("writeback_success"))
    effective_speed_min = postwrite.get("effective_speed_min", effective_speed_gate.get("effective_speed_min"))
    effective_speed_max = postwrite.get("effective_speed_max", effective_speed_gate.get("effective_speed_max"))
    effective_speed_not_applicable = bool(effective_speed_gate.get("not_applicable")) or (
        bool(effective_speed_gate.get("prewrite_pending"))
        and effective_speed_gate.get("effective_speed_min") is None
        and effective_speed_gate.get("effective_speed_max") is None
    )
    effective_speed_passed_raw = bool(effective_speed_gate.get("gate_passed") or postwrite.get("effective_speed_gate_passed"))
    effective_speed_has_bounds = effective_speed_min is not None and effective_speed_max is not None

    def postwrite_bool(flat_key: str, audit_key: str) -> bool:
        if flat_key in postwrite:
            return bool(postwrite.get(flat_key))
        return bool(audit.get(audit_key))

    def postwrite_int(flat_key: str, audit_key: str) -> int:
        if flat_key in postwrite:
            return int(postwrite.get(flat_key) or 0)
        return int(audit.get(audit_key) or 0)

    summary = {
        "status": run_report.status,
        "write_status": write_status or ("blocked" if run_report.status == "blocked" else "not_requested"),
        "commit_performed": bool(commit_performed),
        "single_source_graph_ok": bool(
            run_report.source_graph and run_report.source_graph.invariant_report.single_source_graph_ok and not source_stage_blocked
        ),
        "all_final_segments_have_word_ids": bool(run_report.final_timeline)
        and all(bool(segment.word_ids) for segment in run_report.final_timeline),
        "all_captions_derived_from_final_timeline": bool(coverage.get("all_captions_derived_from_final_timeline")),
        "all_materials_from_canonical_template": bool((run_report.material_write_plan or {}).get("canonical_caption_template_id")),
        "no_writer_fallback": bool((run_report.material_write_plan or {}).get("no_writer_fallback")),
        "writer_fallback_count": writer_fallback_count,
        "validators_readonly": bool(validator.get("validators_read_only")),
        "final_repeat_count": int(final_repeat.get("final_text_repeat_high_count") or 0)
        + int(final_repeat.get("final_text_repeat_medium_count") or 0)
        + int(final_repeat.get("final_cjk_short_repeat_fatal_count") or 0)
        + int(final_repeat.get("adjacent_modifier_semantic_redundancy_fatal_count") or 0)
        + int(final_repeat.get("final_target_repeat_high_count") or 0)
        + int(final_repeat.get("final_target_repeat_medium_count") or 0),
        "hidden_audio_repeat_count": int(hidden.get("word_timeline_hidden_repeat_count") or 0)
        + int(hidden.get("word_timeline_repeated_island_count") or 0)
        + int(hidden.get("final_spoken_text_short_repeat_fatal_count") or 0)
        + int(hidden.get("adjacent_modifier_semantic_redundancy_fatal_count") or 0),
        "cut_inside_word_count": int(safe_cut.get("cut_inside_word_count") or 0),
        "partial_multichar_cut_count": int(safe_cut.get("partial_multichar_cut_count") or 0),
        "giant_subtitle_count": int(style.get("giant_subtitle_count") or 0),
        "template_fingerprint_mismatch_count": int(style.get("template_fingerprint_mismatch_count") or 0),
        "content_schema_error_count": int(postwrite.get("content_schema_error_count") or 0),
        "caption_coverage_gap_count": len(coverage.get("missing_caption_segment_ids") or []),
        "caption_word_coverage_gap_count": int(caption_alignment.get("prewrite_uncaptioned_spoken_word_count") or 0),
        "missing_final_timeline_caption_word_count": int(coverage.get("missing_final_timeline_caption_word_count") or 0),
        "missing_final_timeline_caption_word_ids": list(coverage.get("missing_final_timeline_caption_word_ids") or []),
        "rough_cut_quality_gate_passed": bool(rough_cut.get("rough_cut_quality_gate_passed")),
        "quality_gate_passed": bool(quality_gate.get("gate_passed")),
        "quality_gate_blocker_codes": list(quality_gate.get("blocker_codes") or []),
        "semantic_adjudication_gate_passed": bool(
            quality_gate.get("semantic_adjudication_gate_passed", semantic_gate.get("semantic_adjudication_gate_passed"))
        ),
        "semantic_request_count": semantic_request_count,
        "semantic_request_unresolved_count": semantic_request_unresolved_count,
        "fatal_semantic_issue_count": fatal_semantic_issue_count,
        "deepseek_provider_configured": deepseek_provider_configured,
        "deepseek_provider_called_count": deepseek_provider_called_count,
        "deepseek_provider_error": deepseek_provider_error,
        "deepseek_batch_enabled": semantic_bool("deepseek_batch_enabled"),
        "deepseek_batch_request_count": semantic_int("deepseek_batch_request_count"),
        "deepseek_batch_attempt_count": semantic_int("deepseek_batch_attempt_count"),
        "deepseek_batch_retry_count": semantic_int("deepseek_batch_retry_count"),
        "deepseek_batch_issue_count": semantic_int("deepseek_batch_issue_count"),
        "deepseek_batch_resolved_count": semantic_int("deepseek_batch_resolved_count"),
        "deepseek_batch_unresolved_count": semantic_int("deepseek_batch_unresolved_count"),
        "deepseek_batch_missing_issue_ids": semantic_list("deepseek_batch_missing_issue_ids"),
        "deepseek_batch_error": semantic_str("deepseek_batch_error"),
        "commit_reused_semantic_cache": semantic_bool("commit_reused_semantic_cache"),
        "semantic_cache_input_hash": semantic_str("semantic_cache_input_hash"),
        "semantic_cache_issue_count": semantic_int("semantic_cache_issue_count"),
        "semantic_cache_resolved_count": semantic_int("semantic_cache_resolved_count"),
        "semantic_cache_unresolved_count": semantic_int("semantic_cache_unresolved_count"),
        "deepseek_provider_not_called_reason": _deepseek_provider_not_called_reason(
            semantic_mode=semantic_mode,
            provider_configured=deepseek_provider_configured,
            provider_called_count=deepseek_provider_called_count,
            semantic_request_count=semantic_request_count,
            semantic_decision_cache_used=semantic_decision_cache_used,
        ),
        "deepseek_provider_skipped_count": deepseek_provider_skipped_count,
        "deepseek_provider_skipped_reasons": deepseek_provider_skipped_reasons,
        "semantic_decision_cache_used": semantic_decision_cache_used,
        "semantic_auto_route_count": semantic_auto_route_count,
        "semantic_local_decision_count": semantic_local_decision_count,
        "semantic_provider_required_count": semantic_provider_required_count,
        "deterministic_baseline_refused_count": deterministic_baseline_refused_count,
        "effective_speed_gate_passed": bool(effective_speed_passed_raw and effective_speed_has_bounds and not effective_speed_not_applicable),
        "effective_speed_not_applicable": effective_speed_not_applicable,
        "effective_speed_not_applicable_reason": str(
            effective_speed_gate.get("not_applicable_reason")
            or ("prewrite_source_binding_pending" if effective_speed_not_applicable else "")
        ),
        "effective_speed_min": effective_speed_min,
        "effective_speed_max": effective_speed_max,
        "effective_speed_drift_count": int(postwrite.get("effective_speed_drift_count") or (quality_gate.get("effective_speed_gate") or {}).get("effective_speed_drift_count") or 0),
        "safe_handle_policy_enabled": bool(postwrite.get("safe_handle_policy_enabled", effective_speed_gate.get("safe_handle_policy_enabled"))),
        "lead_handle_requested_count": int(postwrite.get("lead_handle_requested_count", effective_speed_gate.get("lead_handle_requested_count") or 0) or 0),
        "tail_handle_requested_count": int(postwrite.get("tail_handle_requested_count", effective_speed_gate.get("tail_handle_requested_count") or 0) or 0),
        "lead_handle_applied_count": int(postwrite.get("lead_handle_applied_count", effective_speed_gate.get("lead_handle_applied_count") or 0) or 0),
        "tail_handle_applied_count": int(postwrite.get("tail_handle_applied_count", effective_speed_gate.get("tail_handle_applied_count") or 0) or 0),
        "segments_with_no_lead_handle": int(postwrite.get("segments_with_no_lead_handle", effective_speed_gate.get("segments_with_no_lead_handle") or 0) or 0),
        "segments_with_no_tail_handle": int(postwrite.get("segments_with_no_tail_handle", effective_speed_gate.get("segments_with_no_tail_handle") or 0) or 0),
        "handle_blocked_count": int(postwrite.get("handle_blocked_count", effective_speed_gate.get("handle_blocked_count") or 0) or 0),
        "handle_blocked_reasons": dict(postwrite.get("handle_blocked_reasons", effective_speed_gate.get("handle_blocked_reasons") or {}) or {}),
        "final_repeat_convergence_gate_passed": bool(final_repeat_convergence.get("gate_passed")),
        "final_repeat_high_count_after_convergence": int(final_repeat_convergence.get("final_repeat_high_count_after") or 0),
        "final_repeat_dropped_segment_count": int(
            final_repeat_convergence.get("dropped_segment_count")
            or final_repeat_convergence.get("final_repeat_dropped_segment_count")
            or len(final_repeat_convergence.get("dropped_segment_indices") or [])
        ),
        "dropped_cluster_count": int(final_repeat_convergence.get("dropped_cluster_count") or len(final_repeat_convergence.get("dropped_cluster_ids") or [])),
        "dropped_segment_count": int(final_repeat_convergence.get("dropped_segment_count") or len(final_repeat_convergence.get("dropped_segment_indices") or [])),
        "dropped_cluster_ids": list(final_repeat_convergence.get("dropped_cluster_ids") or []),
        "dropped_segment_indices": list(final_repeat_convergence.get("dropped_segment_indices") or []),
        "final_caption_visible_repeat_gate_passed": bool(final_caption_visible_repeat.get("gate_passed")),
        "visible_repeat_candidate_count": int(final_caption_visible_repeat.get("visible_repeat_candidate_count") or 0),
        "containment_repeat_count": int(final_caption_visible_repeat.get("containment_repeat_count") or 0),
        "prefix_suffix_overlap_count": int(final_caption_visible_repeat.get("prefix_suffix_overlap_count") or 0),
        "ngram_repeat_count": int(final_caption_visible_repeat.get("ngram_repeat_count") or 0),
        "near_duplicate_visible_caption_count": int(final_caption_visible_repeat.get("near_duplicate_visible_caption_count") or 0),
        "modifier_redundancy_residual_count": int(final_caption_visible_repeat.get("modifier_redundancy_residual_count") or 0),
        "self_repair_aborted_phrase_count": int(final_caption_visible_repeat.get("self_repair_aborted_phrase_count") or 0),
        "dangling_prefix_suffix_count": int(final_caption_visible_repeat.get("dangling_prefix_suffix_count") or 0),
        "semantic_garbage_or_asr_suspect_count": int(final_caption_visible_repeat.get("semantic_garbage_or_asr_suspect_count") or 0),
        "cross_caption_semantic_containment_count": int(final_caption_visible_repeat.get("cross_caption_semantic_containment_count") or 0),
        "restart_repeat_visible_count": int(final_caption_visible_repeat.get("restart_repeat_visible_count") or 0),
        "final_visible_repair_attempted": bool(final_visible_caption_repair.get("final_visible_repair_attempted")),
        "final_visible_repair_success": bool(final_visible_caption_repair.get("final_visible_repair_success")),
        "final_visible_repair_action_count": int(final_visible_caption_repair.get("final_visible_repair_action_count") or 0),
        "final_visible_repair_initial_counts": dict(final_visible_caption_repair.get("final_visible_repair_initial_counts") or {}),
        "final_visible_repair_final_counts": dict(final_visible_caption_repair.get("final_visible_repair_final_counts") or {}),
        "final_visible_repair_final_timeline_counts": dict(final_visible_caption_repair.get("final_visible_repair_final_timeline_counts") or {}),
        "final_visible_effective_caption_count": int(final_visible_caption_repair.get("final_visible_effective_caption_count") or 0),
        "caption_only_materialized_merge_count": int(final_visible_caption_repair.get("caption_only_materialized_merge_count") or 0),
        "caption_only_consumed_caption_ids": list(final_visible_caption_repair.get("caption_only_consumed_caption_ids") or []),
        "final_visible_repair_unresolved": list(final_visible_caption_repair.get("final_visible_repair_unresolved") or []),
        "final_caption_visible_repeat_blocker_codes": list(final_caption_visible_repeat.get("blocker_codes") or []),
        "final_caption_visible_repeat_candidates": list(final_caption_visible_repeat.get("visible_repeat_candidates") or []),
        "modifier_redundancy_residual_candidates": list(final_caption_visible_repeat.get("modifier_redundancy_residual_candidates") or []),
        "self_repair_aborted_phrase_candidates": list(final_caption_visible_repeat.get("self_repair_aborted_phrase_candidates") or []),
        "dangling_prefix_suffix_candidates": list(final_caption_visible_repeat.get("dangling_prefix_suffix_candidates") or []),
        "semantic_garbage_or_asr_suspect_candidates": list(final_caption_visible_repeat.get("semantic_garbage_or_asr_suspect_candidates") or []),
        "cross_caption_semantic_containment_candidates": list(final_caption_visible_repeat.get("cross_caption_semantic_containment_candidates") or []),
        "restart_repeat_visible_candidates": list(final_caption_visible_repeat.get("restart_repeat_visible_candidates") or []),
        "visual_pacing_gate_passed": bool(visual_pacing.get("gate_passed")),
        "visual_pacing_executed": bool(visual_pacing.get("visual_pacing_executed")),
        "visual_pacing_merge_attempted_count": int(visual_pacing.get("visual_pacing_merge_attempted_count") or 0),
        "visual_pacing_merged_count": int(visual_pacing.get("visual_pacing_merged_count") or 0),
        "visual_merge_safety_gate_passed": bool(visual_pacing.get("visual_merge_safety_gate_passed")),
        "unsafe_merge_group_count": int(visual_pacing.get("unsafe_merge_group_count") or 0),
        "dropped_content_reintroduced_count": int(visual_pacing.get("dropped_content_reintroduced_count") or 0),
        "max_bridged_gap_us": int(visual_pacing.get("max_bridged_gap_us") or 0),
        "total_bridged_gap_us": int(visual_pacing.get("total_bridged_gap_us") or 0),
        "unspoken_bridge_ratio": float(visual_pacing.get("unspoken_bridge_ratio") or 0.0),
        "final_video_segment_count": int(visual_pacing.get("final_video_segment_count") or len(run_report.final_timeline)),
        "caption_count": int(visual_pacing.get("caption_count") or len(run_report.captions)),
        "visual_short_segment_count_lt_1200ms": int(visual_pacing.get("visual_short_segment_count_lt_1200ms") or 0),
        "visual_short_segment_count_lt_1200ms_before": int(visual_pacing.get("visual_short_segment_count_lt_1200ms_before") or 0),
        "visual_short_segment_count_lt_1200ms_after": int(visual_pacing.get("visual_short_segment_count_lt_1200ms_after") or 0),
        "visual_short_segment_count_lt_1200ms_after_blocking": int(
            visual_pacing.get("visual_short_segment_count_lt_1200ms_after_blocking") or 0
        ),
        "semantic_bridge_short_segment_count": int(visual_pacing.get("semantic_bridge_short_segment_count") or 0),
        "semantic_bridge_short_segment_details": list(visual_pacing.get("semantic_bridge_short_segment_details") or []),
        "semantic_bridge_reason_counts": dict(visual_pacing.get("semantic_bridge_reason_counts") or {}),
        "semantic_bridge_cap": int(visual_pacing.get("semantic_bridge_cap") or 0),
        "semantic_bridge_safe_merge_candidate_count": int(visual_pacing.get("semantic_bridge_safe_merge_candidate_count") or 0),
        "semantic_bridge_safe_merge_candidates": list(visual_pacing.get("semantic_bridge_safe_merge_candidates") or []),
        "cuts_per_minute": float(visual_pacing.get("cuts_per_minute") or 0.0),
        "max_cuts_in_5s": int(visual_pacing.get("max_cuts_in_5s") or 0),
        "burst_cut_count": int(visual_pacing.get("burst_cut_count") or 0),
        "cut_density_gate_enabled": bool(visual_pacing.get("cut_density_gate_enabled")),
        "cut_density_gate_passed": bool(visual_pacing.get("cut_density_gate_passed")),
        "cut_density_thresholds": dict(visual_pacing.get("cut_density_thresholds") or {}),
        "large_intra_segment_gap_candidate_count": int(visual_pacing.get("large_intra_segment_gap_candidate_count") or 0),
        "large_intra_segment_gap_split_count": int(visual_pacing.get("large_intra_segment_gap_split_count") or 0),
        "large_intra_segment_gap_unsafe_count": int(visual_pacing.get("large_intra_segment_gap_unsafe_count") or 0),
        "large_intra_segment_gap_max_us": int(visual_pacing.get("large_intra_segment_gap_max_us") or 0),
        "large_intra_segment_gap_threshold_us": int(visual_pacing.get("large_intra_segment_gap_threshold_us") or 0),
        "large_intra_segment_gap_candidates": list(visual_pacing.get("large_intra_segment_gap_candidates") or []),
        "visual_pacing_allowed_short_segment_threshold": int(visual_pacing.get("visual_pacing_allowed_short_segment_threshold") or 0),
        "visual_pacing_blocker_codes": list(visual_pacing.get("visual_pacing_blocker_codes") or []),
        "residual_visual_short_segments": list(visual_pacing.get("residual_visual_short_segments") or []),
        "hidden_repeat_cleanup_dropped_word_count": int(visual_pacing.get("visual_pacing_hidden_repeat_dropped_word_count") or 0),
        "boundary_overlap_cleanup_dropped_word_count": int(visual_pacing.get("visual_pacing_boundary_overlap_dropped_word_count") or 0),
        "median_segment_duration_us": int(visual_pacing.get("median_segment_duration_us") or 0),
        "p10_segment_duration_us": int(visual_pacing.get("p10_segment_duration_us") or 0),
        "caption_per_video_segment_ratio": float(visual_pacing.get("caption_per_video_segment_ratio") or 0.0),
        "caption_alignment_gate_passed": bool(caption_alignment.get("gate_passed")),
        "caption_gui_track_gate_passed": bool(caption_alignment.get("caption_gui_track_gate_passed")),
        "subtitle_readability_gate_passed": bool(caption_alignment.get("subtitle_readability_gate_passed")),
        "visible_caption_track_count": int(caption_alignment.get("visible_caption_track_count") or 0),
        "caption_lane_count": int(caption_alignment.get("caption_lane_count") or 0),
        "orphan_caption_count": int(caption_alignment.get("orphan_caption_count") or 0),
        "floating_caption_count": int(caption_alignment.get("floating_caption_count") or 0),
        "caption_render_order_stable": bool(caption_alignment.get("caption_render_order_stable")),
        "caption_outside_video_count": int(caption_alignment.get("caption_outside_video_count") or 0),
        "caption_overlap_count": int(caption_alignment.get("caption_overlap_count") or 0),
        "caption_too_short_count": int(caption_alignment.get("caption_too_short_count") or 0),
        "one_char_caption_count": int(caption_alignment.get("one_char_caption_count") or 0),
        "prewrite_uncaptioned_spoken_word_count": int(caption_alignment.get("prewrite_uncaptioned_spoken_word_count") or 0),
        "prewrite_uncaptioned_spoken_segment_count": int(caption_alignment.get("prewrite_uncaptioned_spoken_segment_count") or 0),
        "prewrite_uncaptioned_spoken_word_rows": list(caption_alignment.get("prewrite_uncaptioned_spoken_word_rows") or []),
        "residual_too_short_captions": list(caption_alignment.get("residual_too_short_captions") or []),
        "residual_one_char_captions": list(caption_alignment.get("residual_one_char_captions") or []),
        "caption_without_video_container_count": int(caption_alignment.get("caption_without_video_container_count") or 0),
        "caption_without_container_count": int(caption_alignment.get("caption_without_video_container_count") or 0),
        "caption_cross_primary_window_count": int(caption_alignment.get("caption_cross_primary_window_count") or 0),
        "captions_le_3_chars": int(caption_alignment.get("captions_le_3_chars") or 0),
        "captions_le_3_chars_cap": int(caption_alignment.get("captions_le_3_chars_cap") or 0),
        "subtitle_interval_too_short_count": int(caption_alignment.get("subtitle_interval_too_short_count") or 0),
        "subtitle_interval_too_long_count": int(caption_alignment.get("subtitle_interval_too_long_count") or 0),
        "subtitle_hard_max_char_count": int(caption_alignment.get("subtitle_hard_max_char_count") or 0),
        "caption_density_per_minute": float(caption_alignment.get("caption_density_per_minute") or 0.0),
        "max_captions_in_5s": int(caption_alignment.get("max_captions_in_5s") or 0),
        "caption_burst_density_count": int(caption_alignment.get("caption_burst_density_count") or 0),
        "prewrite_style_gate_ok": bool(style.get("prewrite_style_gate_ok")),
        "postwrite_style_gate_ok": bool(postwrite.get("postwrite_material_gate_ok")),
        "postwrite_decrypt_ok": bool(postwrite.get("postwrite_decrypt_ok")),
        "postwrite_mode": str(postwrite.get("postwrite_mode") or ""),
        "sacrificial_write_override_used": bool(postwrite.get("sacrificial_write_override_used")),
        "postwrite_decrypt_skipped_for_sacrificial_draft": sacrificial_postwrite_skip,
        "postwrite_decrypt_skip_reason": str(postwrite.get("postwrite_decrypt_skip_reason") or ""),
        "ready_for_user_manual_qc": ready_for_user_manual_qc,
        "writeback_success": writeback_success,
        "WRITE_SUCCESS": bool(postwrite.get("WRITE_SUCCESS")),
        "ENCRYPT_SUCCESS": bool(postwrite.get("ENCRYPT_SUCCESS")),
        "post_write_actual_draft_audit_required_on_commit": post_write_audit_required_on_commit,
        "post_write_actual_draft_audit_executed": post_write_audit_executed,
        "post_write_actual_draft_audit_gate_passed": post_write_audit_gate_passed,
        "post_write_actual_draft_audit_blocker_codes": list(
            postwrite.get("post_write_actual_draft_audit_blocker_codes")
            or audit.get("blocker_codes")
            or []
        ),
        "post_write_actual_draft_audit_failure_reasons": list(
            postwrite.get("post_write_actual_draft_audit_failure_reasons")
            or audit.get("failure_reasons")
            or []
        ),
        "post_write_actual_draft_loaded": bool(postwrite.get("post_write_actual_draft_loaded") or audit.get("actual_draft_loaded")),
        "post_write_actual_video_rows_match_plan": bool(postwrite.get("post_write_actual_video_rows_match_plan") or audit.get("actual_video_rows_match_plan")),
        "post_write_actual_caption_rows_match_plan": bool(postwrite.get("post_write_actual_caption_rows_match_plan") or audit.get("actual_caption_rows_match_plan")),
        "post_write_expected_caption_rows_present": postwrite_bool("post_write_expected_caption_rows_present", "expected_caption_rows_present"),
        "post_write_actual_has_no_extra_caption_like_text_segments": postwrite_bool(
            "post_write_actual_has_no_extra_caption_like_text_segments",
            "actual_has_no_extra_caption_like_text_segments",
        ),
        "post_write_actual_caption_rows_exact_match_plan": postwrite_bool(
            "post_write_actual_caption_rows_exact_match_plan",
            "actual_caption_rows_exact_match_plan",
        ),
        "post_write_actual_text_residue_gate_passed": postwrite_bool("post_write_actual_text_residue_gate_passed", "actual_text_residue_gate_passed"),
        "post_write_actual_audio_coverage_gate_passed": postwrite_bool("post_write_actual_audio_coverage_gate_passed", "actual_audio_coverage_gate_passed"),
        "post_write_actual_visible_text_repeat_gate_passed": postwrite_bool(
            "post_write_actual_visible_text_repeat_gate_passed",
            "actual_visible_text_repeat_gate_passed",
        ),
        "post_write_actual_text_segment_count": postwrite_int("post_write_actual_text_segment_count", "actual_text_segment_count"),
        "post_write_generated_caption_segment_count": postwrite_int("post_write_generated_caption_segment_count", "generated_caption_segment_count"),
        "post_write_preserved_non_subtitle_count": postwrite_int("post_write_preserved_non_subtitle_count", "preserved_non_subtitle_count"),
        "post_write_old_subtitle_residue_count": postwrite_int("post_write_old_subtitle_residue_count", "old_subtitle_residue_count"),
        "post_write_orphan_text_segment_count": postwrite_int("post_write_orphan_text_segment_count", "orphan_text_segment_count"),
        "post_write_text_after_final_video_end_count": postwrite_int("post_write_text_after_final_video_end_count", "text_after_final_video_end_count"),
        "post_write_floating_caption_count": postwrite_int("post_write_floating_caption_count", "floating_caption_count"),
        "post_write_audio_coverage_failure_count": postwrite_int("post_write_audio_coverage_failure_count", "audio_coverage_failure_count"),
        "post_write_heard_but_uncaptioned_word_count": postwrite_int("post_write_heard_but_uncaptioned_word_count", "heard_but_uncaptioned_word_count"),
        "post_write_dropped_but_reintroduced_word_count": postwrite_int(
            "post_write_dropped_but_reintroduced_word_count",
            "dropped_but_reintroduced_word_count",
        ),
        "post_write_actual_visible_repeat_candidate_count": postwrite_int(
            "post_write_actual_visible_repeat_candidate_count",
            "actual_visible_repeat_candidate_count",
        ),
        "jianying_canonical_timeline_sync_gate_passed": postwrite_bool(
            "jianying_canonical_timeline_sync_gate_passed",
            "jianying_canonical_timeline_sync_gate_passed",
        ),
        "final_video_end_us": postwrite_int("final_video_end_us", "final_video_end_us"),
        "max_caption_end_us": postwrite_int("max_caption_end_us", "max_caption_end_us"),
        "captions_after_final_video_end_count": postwrite_int(
            "captions_after_final_video_end_count",
            "captions_after_final_video_end_count",
        ),
        "post_write_video_target_gap_count_gt_300ms": postwrite_int(
            "post_write_video_target_gap_count_gt_300ms",
            "post_write_video_target_gap_count_gt_300ms",
        ),
        "post_write_total_video_target_gap_us": postwrite_int(
            "post_write_total_video_target_gap_us",
            "post_write_total_video_target_gap_us",
        ),
        "caption_video_drift_count": postwrite_int("caption_video_drift_count", "caption_video_drift_count"),
        "max_caption_video_drift_us": postwrite_int("max_caption_video_drift_us", "max_caption_video_drift_us"),
        "split_caption_container_mismatch_count": postwrite_int(
            "split_caption_container_mismatch_count",
            "split_caption_container_mismatch_count",
        ),
        "caption_crosses_video_split_gap_count": postwrite_int(
            "caption_crosses_video_split_gap_count",
            "caption_crosses_video_split_gap_count",
        ),
        "caption_words_not_covered_by_actual_video_count": postwrite_int(
            "caption_words_not_covered_by_actual_video_count",
            "caption_words_not_covered_by_actual_video_count",
        ),
        "actual_has_no_extra_caption_like_text_segments": postwrite_bool(
            "post_write_actual_has_no_extra_caption_like_text_segments",
            "actual_has_no_extra_caption_like_text_segments",
        ),
        "actual_caption_rows_exact_match_plan": postwrite_bool("post_write_actual_caption_rows_exact_match_plan", "actual_caption_rows_exact_match_plan"),
        "actual_text_residue_gate_passed": postwrite_bool("post_write_actual_text_residue_gate_passed", "actual_text_residue_gate_passed"),
        "actual_audio_coverage_gate_passed": postwrite_bool("post_write_actual_audio_coverage_gate_passed", "actual_audio_coverage_gate_passed"),
        "actual_visible_text_repeat_gate_passed": postwrite_bool(
            "post_write_actual_visible_text_repeat_gate_passed",
            "actual_visible_text_repeat_gate_passed",
        ),
        "actual_audio_coverage_failure_count": postwrite_int("post_write_audio_coverage_failure_count", "audio_coverage_failure_count"),
        "heard_but_uncaptioned_word_count": postwrite_int("post_write_heard_but_uncaptioned_word_count", "heard_but_uncaptioned_word_count"),
        "dropped_but_reintroduced_word_count": postwrite_int(
            "post_write_dropped_but_reintroduced_word_count",
            "dropped_but_reintroduced_word_count",
        ),
        "old_subtitle_residue_count": postwrite_int("post_write_old_subtitle_residue_count", "old_subtitle_residue_count"),
        "orphan_text_segment_count": postwrite_int("post_write_orphan_text_segment_count", "orphan_text_segment_count"),
        "text_after_final_video_end_count": postwrite_int("post_write_text_after_final_video_end_count", "text_after_final_video_end_count"),
        "post_write_actual_effective_speed_gate_passed": bool(postwrite.get("post_write_actual_effective_speed_gate_passed") or audit.get("actual_effective_speed_gate_passed")),
        "post_write_actual_visual_pacing_gate_passed": bool(postwrite.get("post_write_actual_visual_pacing_gate_passed") or audit.get("actual_visual_pacing_gate_passed")),
        "post_write_actual_caption_gui_readability_gate_passed": bool(
            postwrite.get("post_write_actual_caption_gui_readability_gate_passed")
            or audit.get("actual_caption_gui_readability_gate_passed")
        ),
        "post_write_actual_final_caption_visible_repeat_gate_passed": bool(
            postwrite.get("post_write_actual_final_caption_visible_repeat_gate_passed")
            or audit.get("actual_final_caption_visible_repeat_gate_passed")
        ),
        "post_write_actual_caption_alignment_gate_passed": bool(postwrite.get("post_write_actual_caption_alignment_gate_passed") or audit.get("actual_caption_alignment_gate_passed")),
        "draft_dir": str(postwrite.get("draft_dir") or ""),
        "jy_draftc_path": str(postwrite.get("jy_draftc_path") or ""),
        "jy_install_dir": str(postwrite.get("jy_install_dir") or ""),
        "postwrite_decrypt_cwd": str(postwrite.get("postwrite_decrypt_cwd") or ""),
        "draft_content_path": str(postwrite.get("draft_content_path") or ""),
        "only_specified_draft_written": bool(postwrite.get("only_specified_draft_written")),
        "source_segment_template_exact_match_count": int(postwrite.get("source_segment_template_exact_match_count") or 0),
        "source_segment_template_rebind_count": int(postwrite.get("source_segment_template_rebind_count") or 0),
        "source_segment_template_missing_count": int(postwrite.get("source_segment_template_missing_count") or 0),
        "source_segment_template_ambiguous_count": int(postwrite.get("source_segment_template_ambiguous_count") or 0),
        "resolved_template_map_count": int(postwrite.get("resolved_template_map_count") or len(run_report.resolved_template_map or {})),
        "current_draft_video_track_count": int(postwrite.get("current_draft_video_track_count") or 0),
        "current_draft_video_segment_count": int(postwrite.get("current_draft_video_segment_count") or 0),
        "current_draft_video_material_count": int(postwrite.get("current_draft_video_material_count") or 0),
        "current_source_template_candidate_count": int(postwrite.get("current_source_template_candidate_count") or 0),
        "speech_timeline_provider": str(blocker_summary.get("speech_timeline_provider") or ingest_metadata.get("speech_timeline_provider") or ""),
        "speech_timeline_granularity": str(blocker_summary.get("speech_timeline_granularity") or ingest_metadata.get("speech_timeline_granularity") or ""),
        "speech_timeline_precision": str(blocker_summary.get("speech_timeline_precision") or ingest_metadata.get("speech_timeline_precision") or ""),
        "speech_timeline_can_cut_inside_caption": bool(
            blocker_summary.get("speech_timeline_can_cut_inside_caption")
            or ingest_metadata.get("speech_timeline_can_cut_inside_caption")
        ),
        "word_timeline_count": int(ingest_metadata.get("word_timeline_count") or (len(run_report.source_graph.words) if run_report.source_graph else 0)),
        "word_timeline_count_source": "ingest_metadata" if ingest_metadata.get("word_timeline_count") is not None else "source_graph",
        "semantic_unresolved_count": semantic_unresolved_count,
        "semantic_mode": semantic_mode,
        "semantic_decisions_generated_from_current_draft": bool(postwrite.get("semantic_decisions_generated_from_current_draft")),
        "semantic_decisions_reused_from_old_draft": bool(postwrite.get("semantic_decisions_reused_from_old_draft")),
        "semantic_review_blocker_count": semantic_unresolved_count,
        "write_blocker_count": write_blocker_count,
        "requires_human_review": bool(semantic_unresolved_count),
        "semantic_write_allowed": semantic_write_allowed,
        "validator_write_allowed": validator_write_allowed,
        "validator_report_ok": validator_write_allowed,
        "write_allowed": ready_for_write,
        "ready_for_write": ready_for_write,
        "READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT": ready_for_write,
        "dry_run_continued_for_discovery": bool(mode == "dry-run" and semantic_unresolved_count > 0 and run_report.final_timeline),
        "commit_only_after_all_validators": bool(
            (not commit_performed)
            or (
                validator.get("validator_report_ok")
                and postwrite.get("postwrite_material_gate_ok")
                and postwrite.get("writeback_success")
                and (postwrite.get("postwrite_decrypt_ok") or sacrificial_postwrite_skip)
                and post_write_audit_ready
            )
        ),
        "blocker_count": len(blockers),
        "blocker_codes": [blocker.code for blocker in blockers],
        "fatal_blocker": blockers[0].code if blockers else None,
        "rough_cut_quality": postwrite.get("rough_cut_quality") or rough_cut,
    }
    return summary


def _deepseek_provider_not_called_reason(
    *,
    semantic_mode: str,
    provider_configured: bool,
    provider_called_count: int,
    semantic_request_count: int,
    semantic_decision_cache_used: bool = False,
) -> str:
    if provider_called_count > 0:
        return ""
    if semantic_decision_cache_used:
        return "semantic_decision_cache_used"
    if semantic_mode not in {"auto", "deepseek"}:
        return f"semantic_mode={semantic_mode}"
    if not provider_configured:
        if semantic_mode == "auto" and semantic_request_count == 0:
            return "no_provider_required"
        return "deepseek_provider_not_configured"
    if semantic_request_count == 0:
        return "no_provider_required" if semantic_mode == "auto" else "no_semantic_requests"
    return "deepseek_provider_not_called_with_pending_semantic_requests"


def _normalize_effective_speed_prewrite_placeholder(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "gate_passed": False,
            "blocker_codes": [],
            "prewrite_pending": True,
            "not_applicable": True,
            "not_applicable_reason": "prewrite_source_binding_pending",
        }
    normalized = dict(payload)
    if (
        bool(normalized.get("prewrite_pending"))
        and normalized.get("effective_speed_min") is None
        and normalized.get("effective_speed_max") is None
    ):
        normalized["gate_passed"] = False
        normalized["not_applicable"] = True
        normalized.setdefault("not_applicable_reason", "prewrite_source_binding_pending")
    return normalized

