from __future__ import annotations

import time
from dataclasses import replace

from aroll_v21.decision.deepseek_semantic_planner import (
    deepseek_provider_from_runtime_config as deepseek_provider_from_env,
)
from aroll_v21.decision.semantic_adjudication import normalize_semantic_mode
from aroll_v21.artifact_manifest import (
    DEBUG_ONLY_ARTIFACTS,
    MINIMAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    _code_version_hash,
    _draft_hashes_from_real_result,
    _hash_file,
    _hash_json,
    _pipeline_config_hash,
    _run_metadata,
    _semantic_artifact_input_hash,
    _stage_timing_defaults,
    _write_artifact_manifest,
    write_boundary_block,
    write_operator_artifacts,
)
from aroll_v21.engine import ArollEngine
from aroll_v21.ingest.real_draft_adapter import RealDraftIngestAdapter, RealDraftIngestResult
from aroll_v21.ir.models import Blocker, RunReport
from aroll_v21.operator_config import (
    ArollV21OperatorConfig,
    Mode,
    ReportProfile,
    _effective_report_profile,
    _normalize_report_profile,
)
from aroll_v21.operator_io import (
    _blocker_report,
    _captions,
    _compact_runtime_report,
    _dataclass_from_dict,
    _decision_plan,
    _edit_units,
    _final_timeline,
    _load_ready_run_report,
    _repeat_clusters,
    _run_input_from_real_draft_result,
    _safe_read_json,
    _source_graph,
    _words,
    load_run_input,
    read_json,
    read_postwrite_materials_json,
    write_json,
    write_profiled_report_json,
)
from aroll_v21.operator_summary import (
    _annotate_postwrite_environment,
    _merge_prewrite_quality_gate,
    _postwrite_environment,
    _summary_blocker_codes,
    _writeback_blocked_report,
)
from aroll_v21.ready_run_reuse import (
    _commit_from_ready_run_dir as _commit_from_ready_run_dir_impl,
    _ready_reuse_rejected,
    _validate_ready_run_dir,
)
from aroll_v21.semantic_provider_factory import (
    DeterministicBaselineSemanticPlanner,
    _semantic_cache_input_hash,
    _semantic_decisions_planner as _semantic_decisions_planner_impl,
)
from aroll_v21.writeback import RealDraftWriteback
from aroll_v21.writeback.dynamic_source_binding_preflight import DynamicSourceBindingPreflight

SOURCE_TEMPLATE_AVAILABILITY_BLOCKERS = {
    "V21_WRITEBACK_CURRENT_SOURCE_TEMPLATE_INDEX_EMPTY",
    "V21_WRITEBACK_SOURCE_SEGMENT_TEMPLATE_MISSING",
    "V21_WRITEBACK_SOURCE_SEGMENT_TEMPLATE_AMBIGUOUS",
    "V21_DYNAMIC_BINDING_CANDIDATE_INDEX_EMPTY",
    "V21_DYNAMIC_BINDING_MISSING",
    "V21_DYNAMIC_BINDING_AMBIGUOUS",
    "V21_DYNAMIC_BINDING_REQUIRED",
    "V21_DYNAMIC_BINDING_FORBIDS_LOGICAL_SOURCE_SEGMENT_ID",
    "V21_DYNAMIC_BINDING_PRIMARY_VIDEO_MISSING",
    "V21_DYNAMIC_BINDING_PRIMARY_VIDEO_AMBIGUOUS",
    "V21_DYNAMIC_BINDING_DURATION_UNPARSEABLE",
    "V21_WRITEBACK_UNSUPPORTED_SPEED_MAPPING",
    "V21_WRITEBACK_UNSUPPORTED_COMPLEX_EFFECT_TRACK",
    "V21_WRITEBACK_ROOT_MIRROR_DETECTION_FAILED",
}


def _preflight_source_segment_templates(
    report: RunReport,
    config: ArollV21OperatorConfig,
    real_draft_result: RealDraftIngestResult | None,
) -> RunReport:
    if report.status != "ok" or config.input_json is not None or config.draft_dir is None or real_draft_result is None:
        return report
    root_mirror_func = None
    if config.jy_draftc is not None and (
        not isinstance(RealDraftWriteback, type) or str(getattr(RealDraftWriteback, "__module__", "")).startswith("aroll_v21")
    ):
        writeback_backend = RealDraftWriteback(jy_draftc=config.jy_draftc)
        root_mirror_func = getattr(writeback_backend, "root_mirror_func", None)
    root_mirror_func = root_mirror_func or (lambda *_args: False)
    writeback_result = DynamicSourceBindingPreflight(
        jy_draftc=config.jy_draftc,
        root_mirror_func=root_mirror_func,
    ).preflight(
        draft_dir=config.draft_dir,
        real_draft_result=real_draft_result,
        run_report=report,
        run_dir=config.run_dir,
    )
    write_profiled_report_json(config.run_dir / "writeback_report.json", writeback_result.report, config.report_profile)
    if not writeback_result.success:
        blocked = _writeback_blocked_report(report, writeback_result)
        _annotate_postwrite_environment(blocked, config, writeback_report=writeback_result.report)
        return blocked
    bound_report = replace(
        report,
        resolved_template_map=dict(writeback_result.report.get("resolved_template_map") or {}),
        source_binding_report=dict(writeback_result.report),
    )
    _annotate_postwrite_environment(bound_report, config, writeback_report=writeback_result.report)
    return bound_report


def _blocked_by_source_template_availability(report: RunReport) -> bool:
    blockers = report.blocker_report.blockers if report.blocker_report else []
    return any(blocker.code in SOURCE_TEMPLATE_AVAILABILITY_BLOCKERS for blocker in blockers)


def _postwrite_materials_config_blocker(config: ArollV21OperatorConfig) -> Blocker | None:
    if config.postwrite_materials_json is None:
        return None
    try:
        read_postwrite_materials_json(config.postwrite_materials_json)
    except Exception as exc:
        return Blocker(
            code="POSTWRITE_MATERIALS_JSON_INVALID",
            message="postwrite materials json could not be parsed as a list of material objects",
            layer="operator",
            context={"path": str(config.postwrite_materials_json), "error": str(exc)},
        )
    return None


def _commit_from_ready_run_dir(
    config: ArollV21OperatorConfig,
    *,
    real_draft_result: RealDraftIngestResult,
    runtime_metadata: dict[str, Any],
    stage_timings: dict[str, float],
) -> dict[str, Any]:
    return _commit_from_ready_run_dir_impl(
        config,
        real_draft_result=real_draft_result,
        runtime_metadata=runtime_metadata,
        stage_timings=stage_timings,
        writeback_cls=RealDraftWriteback,
    )


def _semantic_decisions_planner(config: ArollV21OperatorConfig):
    return _semantic_decisions_planner_impl(config, provider_factory=deepseek_provider_from_env)


def run_operator(config: ArollV21OperatorConfig) -> dict[str, Any]:
    total_started = time.monotonic()
    stage_timings: dict[str, float] = {}
    profile = _normalize_report_profile(config.report_profile)
    if profile != config.report_profile:
        config = replace(config, report_profile=profile)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if config.input_json is not None and config.draft_dir is not None:
        return write_boundary_block(
            config,
            Blocker(
                code="REAL_DRAFT_INPUT_JSON_NOT_ALLOWED_WITH_DRAFT_DIR",
                message="sanitized input_json cannot be used to masquerade as a real draft ingest",
                layer="operator",
                context={"draft_dir": str(config.draft_dir), "input_json": str(config.input_json)},
            ),
        )
    if config.input_json is None and config.draft_dir is None:
        return write_boundary_block(
            config,
            Blocker(
                code="REAL_DRAFT_DIR_REQUIRED",
                message="pass a disposable DraftDir for V21 real ingest or omit DraftDir and pass input_json for offline fixture mode",
                layer="operator",
            ),
        )

    postwrite_materials_blocker = _postwrite_materials_config_blocker(config)
    if postwrite_materials_blocker is not None:
        return write_boundary_block(config, postwrite_materials_blocker)

    real_draft_result: RealDraftIngestResult | None = None
    if config.input_json is None and config.draft_dir is not None:
        ingest_started = time.monotonic()
        real_draft_result = RealDraftIngestAdapter(jy_draftc=config.jy_draftc).load(
            config.draft_dir,
            config.run_dir,
            word_timeline_json=config.word_timeline_json,
        )
        stage_timings["ingest_seconds"] = time.monotonic() - ingest_started
        if real_draft_result.blockers and not real_draft_result.draft_data:
            return write_boundary_block(config, real_draft_result.blockers[0])
    runtime_metadata = _run_metadata(config, real_draft_result)

    if config.mode == "write" and config.ready_run_dir is not None:
        summary = _commit_from_ready_run_dir(
            config,
            real_draft_result=real_draft_result
            if real_draft_result is not None
            else RealDraftIngestResult(
                draft_data={},
                word_timeline=[],
                subtitles=[],
                source_segments=[],
                source_materials=[],
                text_materials=[],
                text_segments=[],
                metadata={},
                blockers=[],
            ),
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )
        summary["total_seconds"] = round(time.monotonic() - total_started, 6)
        write_json(config.run_dir / "run_summary.json", summary)
        return summary

    semantic_planner, semantic_provider, semantic_planner_blocker = _semantic_decisions_planner(config)
    if semantic_planner_blocker is not None:
        return write_boundary_block(config, semantic_planner_blocker)
    engine = ArollEngine(
        deepseek_planner=semantic_planner,
        semantic_provider=semantic_provider,
        semantic_mode=normalize_semantic_mode(config.semantic_mode).value,
    )

    if config.mode == "dry-run":
        dry_started = time.monotonic()
        report = engine.run(load_run_input(config, postwrite_mode="simulated", real_draft_result=real_draft_result))
        stage_timings["planning_seconds"] = time.monotonic() - dry_started
        stage_timings["dry_run_seconds"] = stage_timings["planning_seconds"]
        report = _preflight_source_segment_templates(report, config, real_draft_result)
        if report.postwrite_report and config.input_json is None and config.draft_dir is not None and real_draft_result is not None:
            write_json(config.run_dir / "prewrite_report.json", report.postwrite_report)
        write_status = "blocked_by_prewrite_source_template_availability" if _blocked_by_source_template_availability(report) else "dry_run_no_write"
        stage_timings["total_seconds"] = time.monotonic() - total_started
        return write_operator_artifacts(
            report,
            config.run_dir,
            write_status=write_status,
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )

    if config.mode == "verify-only":
        postwrite_mode = "actual_decrypt" if config.postwrite_materials_json is not None else "unavailable"
        verify_started = time.monotonic()
        report = engine.run(load_run_input(config, postwrite_mode=postwrite_mode, real_draft_result=real_draft_result))
        stage_timings["planning_seconds"] = time.monotonic() - verify_started
        stage_timings["total_seconds"] = time.monotonic() - total_started
        write_status = "verify_only_passed" if report.status == "ok" else "verify_only_blocked"
        return write_operator_artifacts(
            report,
            config.run_dir,
            write_status=write_status,
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )

    prewrite_started = time.monotonic()
    prewrite_report = engine.run(load_run_input(config, postwrite_mode="simulated", real_draft_result=real_draft_result))
    stage_timings["planning_seconds"] = time.monotonic() - prewrite_started
    prewrite_report = _preflight_source_segment_templates(prewrite_report, config, real_draft_result)
    write_json(config.run_dir / "prewrite_report.json", prewrite_report.postwrite_report)
    if prewrite_report.status != "ok":
        write_status = (
            "blocked_by_prewrite_source_template_availability"
            if _blocked_by_source_template_availability(prewrite_report)
            else "blocked_by_prewrite_validators"
        )
        stage_timings["total_seconds"] = time.monotonic() - total_started
        return write_operator_artifacts(
            prewrite_report,
            config.run_dir,
            write_status=write_status,
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )

    if config.simulate_write:
        simulated_started = time.monotonic()
        simulated = engine.run(load_run_input(config, postwrite_mode="simulated_write", real_draft_result=real_draft_result))
        stage_timings["quality_gate_seconds"] = time.monotonic() - simulated_started
        stage_timings["total_seconds"] = time.monotonic() - total_started
        return write_operator_artifacts(
            simulated,
            config.run_dir,
            write_status="simulated_write_no_commit",
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )

    actual_postwrite_available = config.postwrite_materials_json is not None
    if not actual_postwrite_available:
        if config.allow_sacrificial_write_without_postwrite_decrypt:
            if config.draft_dir is None:
                return write_boundary_block(
                    config,
                    Blocker(
                        code="SACRIFICIAL_WRITE_REQUIRES_EXPLICIT_DRAFT_DIR",
                        message="sacrificial write override requires an explicit DraftDir and cannot run from input_json",
                        layer="operator",
                        context={"input_json": str(config.input_json) if config.input_json else ""},
                    ),
                )
            if not config.commit:
                return write_boundary_block(
                    config,
                    Blocker(
                        code="SACRIFICIAL_WRITE_REQUIRES_COMMIT_FLAG",
                        message="sacrificial write override requires explicit commit intent",
                        layer="operator",
                        context={"draft_dir": str(config.draft_dir)},
                    ),
                )
            assert real_draft_result is not None
            writeback_started = time.monotonic()
            writeback_result = RealDraftWriteback(jy_draftc=config.jy_draftc).commit(
                draft_dir=config.draft_dir,
                run_dir=config.run_dir,
                real_draft_result=real_draft_result,
                run_report=prewrite_report,
                sacrificial_write_override_used=True,
            )
            stage_timings["writeback_seconds"] = time.monotonic() - writeback_started
            stage_timings["postwrite_core_audit_seconds"] = stage_timings["writeback_seconds"]
            write_profiled_report_json(config.run_dir / "writeback_report.json", writeback_result.report, config.report_profile)
            if not writeback_result.success:
                blocked = _writeback_blocked_report(prewrite_report, writeback_result)
                _annotate_postwrite_environment(blocked, config, sacrificial_override_used=True, writeback_report=writeback_result.report)
                stage_timings["total_seconds"] = time.monotonic() - total_started
                return write_operator_artifacts(
                    blocked,
                    config.run_dir,
                    write_status="blocked_writeback_failed",
                    commit_performed=False,
                    report_profile=config.report_profile,
                    runtime_metadata=runtime_metadata,
                    stage_timings=stage_timings,
                )
            postwrite_started = time.monotonic()
            sacrificial = engine.run(
                load_run_input(
                    config,
                    postwrite_mode="skipped_for_sacrificial_draft",
                    real_draft_result=real_draft_result,
                )
            )
            stage_timings["postwrite_debug_audit_seconds"] = time.monotonic() - postwrite_started
            _annotate_postwrite_environment(
                sacrificial,
                config,
                sacrificial_override_used=True,
                writeback_report=writeback_result.report,
            )
            if sacrificial.status == "ok":
                stage_timings["total_seconds"] = time.monotonic() - total_started
                return write_operator_artifacts(
                    sacrificial,
                    config.run_dir,
                    write_status="committed_sacrificial_without_postwrite_decrypt",
                    commit_performed=True,
                    report_profile=config.report_profile,
                    runtime_metadata=runtime_metadata,
                    stage_timings=stage_timings,
                )
            stage_timings["total_seconds"] = time.monotonic() - total_started
            return write_operator_artifacts(
                sacrificial,
                config.run_dir,
                write_status="blocked_sacrificial_write_preconditions_failed",
                commit_performed=False,
                report_profile=config.report_profile,
                runtime_metadata=runtime_metadata,
                stage_timings=stage_timings,
            )
        unavailable_started = time.monotonic()
        unavailable = engine.run(load_run_input(config, postwrite_mode="unavailable", real_draft_result=real_draft_result))
        stage_timings["postwrite_core_audit_seconds"] = time.monotonic() - unavailable_started
        _annotate_postwrite_environment(unavailable, config)
        stage_timings["total_seconds"] = time.monotonic() - total_started
        return write_operator_artifacts(
            unavailable,
            config.run_dir,
            write_status="blocked_actual_decrypt_unavailable",
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )

    verified_started = time.monotonic()
    verified = engine.run(load_run_input(config, postwrite_mode="actual_decrypt", real_draft_result=real_draft_result))
    stage_timings["postwrite_core_audit_seconds"] = time.monotonic() - verified_started
    if verified.status != "ok":
        stage_timings["total_seconds"] = time.monotonic() - total_started
        return write_operator_artifacts(
            verified,
            config.run_dir,
            write_status="blocked_by_postwrite_verification",
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )
    verified = replace(
        verified,
        resolved_template_map=dict(prewrite_report.resolved_template_map or {}),
        source_binding_report=dict(prewrite_report.source_binding_report or {}),
    )

    if not config.commit:
        stage_timings["total_seconds"] = time.monotonic() - total_started
        return write_operator_artifacts(
            verified,
            config.run_dir,
            write_status="verified_no_commit_flag",
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )

    if config.draft_dir is None or real_draft_result is None:
        return write_boundary_block(
            config,
            Blocker(
                code="REAL_WRITE_REQUIRES_EXPLICIT_DRAFT_DIR",
                message="V21 real writeback requires an explicit DraftDir and real draft ingest result",
                layer="operator",
                context={"input_json": str(config.input_json) if config.input_json else ""},
            ),
        )
    writeback_started = time.monotonic()
    writeback_result = RealDraftWriteback(jy_draftc=config.jy_draftc).commit(
        draft_dir=config.draft_dir,
        run_dir=config.run_dir,
        real_draft_result=real_draft_result,
        run_report=verified,
        sacrificial_write_override_used=False,
    )
    stage_timings["writeback_seconds"] = time.monotonic() - writeback_started
    write_profiled_report_json(config.run_dir / "writeback_report.json", writeback_result.report, config.report_profile)
    if not writeback_result.success:
        blocked = _writeback_blocked_report(verified, writeback_result)
        _annotate_postwrite_environment(blocked, config, writeback_report=writeback_result.report)
        stage_timings["total_seconds"] = time.monotonic() - total_started
        return write_operator_artifacts(
            blocked,
            config.run_dir,
            write_status="blocked_writeback_failed",
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )
    _annotate_postwrite_environment(verified, config, writeback_report=writeback_result.report)
    stage_timings["total_seconds"] = time.monotonic() - total_started
    return write_operator_artifacts(
        verified,
        config.run_dir,
        write_status="committed_after_postwrite_verification",
        commit_performed=True,
        report_profile=config.report_profile,
        runtime_metadata=runtime_metadata,
        stage_timings=stage_timings,
    )

