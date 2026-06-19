from __future__ import annotations

import json
import gzip
import os
import hashlib
import time
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

from aroll_v21.decision import DeterministicBaselinePolicy, SemanticDecisionsJsonPlanner
from aroll_v21.decision.deepseek_semantic_planner import (
    deepseek_provider_from_runtime_config as deepseek_provider_from_env,
)
from aroll_v21.decision.semantic_adjudication import normalize_semantic_mode, severity_for_cluster
from aroll_v21.decision.semantic_contracts import SemanticAdjudicationMode
from aroll_v21.engine import ArollEngine, ArollRunInput, build_run_summary, write_run_artifacts
from aroll_v21.ingest.real_draft_adapter import RealDraftIngestAdapter, RealDraftIngestResult
from aroll_v21.ir.models import (
    Blocker,
    BlockerReport,
    CandidateEvidence,
    CanonicalSourceGraph,
    CanonicalWord,
    CaptionRenderUnit,
    DecisionPlan,
    EditUnit,
    FinalTimelineSegment,
    RepeatCluster,
    RunReport,
    SourceGraphInvariantReport,
    TakeDecision,
    UnitSplitPlan,
    dataclass_to_dict,
)
from aroll_v21.writeback import RealDraftWriteback, WritebackResult
from aroll_v21.writeback.dynamic_source_binding_preflight import DynamicSourceBindingPreflight


Mode = Literal["dry-run", "write", "verify-only"]
ReportProfile = Literal["minimal", "standard", "debug"]


@dataclass(frozen=True)
class ArollV21OperatorConfig:
    mode: Mode
    run_dir: Path
    input_json: Path | None = None
    draft_dir: Path | None = None
    jy_draftc: Path | None = None
    word_timeline_json: Path | None = None
    semantic_decisions_json: Path | None = None
    postwrite_materials_json: Path | None = None
    simulate_write: bool = False
    commit: bool = False
    allow_sacrificial_write_without_postwrite_decrypt: bool = False
    semantic_mode: str = "auto"
    ready_run_dir: Path | None = None
    report_profile: ReportProfile = "standard"


class DeterministicBaselineSemanticPlanner:
    """Explicit baseline planner for low-risk deterministic semantic clusters."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.policy = DeterministicBaselinePolicy()
        self.deterministic_baseline_refused_count = 0

    def decide(self, clusters) -> list[dict[str, Any]]:
        self.rows = []
        self.deterministic_baseline_refused_count = 0
        for cluster in clusters:
            keep_unit_id = cluster.variants[0].unit_id if cluster.variants else ""
            row = self.policy.decision_for_missing_cluster(
                cluster.cluster_id,
                cluster_type=str(cluster.repeat_type or ""),
                context={
                    "keep_unit_id": keep_unit_id,
                    "drop_unit_ids": [],
                    "reason": "deterministic baseline keeps low-risk semantic speech units only",
                    "severity": severity_for_cluster(cluster).value,
                    "requires_semantic_decision": any(item.requires_semantic_decision for item in cluster.evidence),
                    "confidence": max((float(item.confidence or 0.0) for item in cluster.evidence), default=0.0),
                },
            )
            if row is not None:
                self.rows.append(row)
                continue
            self.deterministic_baseline_refused_count += 1
            self.rows.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "_blocker_code": "V21_FATAL_MODIFIER_REDUNDANCY_UNRESOLVED"
                    if str(cluster.repeat_type or "") == "modifier_redundancy"
                    else "SEMANTIC_DECISION_NOT_PROVIDED",
                    "_severity": "write_blocker",
                    "_message": "deterministic baseline refused high-risk semantic issue",
                    "_decision_source": "deterministic_baseline",
                    "_semantic_mode": "deterministic_baseline",
                    "_deterministic_baseline_refused": True,
                }
            )
        return list(self.rows)


MINIMAL_ARTIFACTS = (
    "run_summary.json",
    "blocker_report.json",
    "artifact_manifest.json",
    "writeback_report.json",
)

DEBUG_ONLY_ARTIFACTS = (
    "run_report.json",
    "deepseek_batch_request.json",
    "deepseek_batch_response.json",
    "deepseek_batch_error.json",
)

REQUIRED_ARTIFACTS = (
    "source_graph.json",
    "edit_units.json",
    "repeat_clusters.json",
    "decision_plan.json",
    "semantic_request_payloads.json",
    "semantic_decisions.json",
    "semantic_decisions.resolved.json",
    "semantic_decision_cache.json",
    "semantic_adjudication_report.json",
    "deepseek_decisions.json",
    "local_policy_decisions.json",
    "final_timeline.json",
    "final_edl.json",
    "captions.json",
    "canonical_caption_template.json",
    "material_write_plan.json",
    "validator_report.json",
    "postwrite_report.json",
    "quality_gate_report.json",
    "blocker_report.json",
    "decision_trace.json",
    "run_summary.json",
    "writeback_report.json",
    "artifact_manifest.json",
)


def read_json(path: Path) -> Any:
    if path.exists():
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(path.read_text("utf-8"))
    gzip_path = path.with_suffix(path.suffix + ".gz")
    if gzip_path.exists():
        with gzip.open(gzip_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text("utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclass_to_dict(data), ensure_ascii=False, indent=2), "utf-8")


def write_profiled_report_json(path: Path, data: Any, report_profile: str) -> None:
    profile = _normalize_report_profile(report_profile)
    write_json(path, data if profile == "debug" else _compact_runtime_report(data))


def _normalize_report_profile(value: str | None) -> ReportProfile:
    profile = str(value or "standard").strip().lower()
    if profile not in {"minimal", "standard", "debug"}:
        return "standard"
    return profile  # type: ignore[return-value]


def _effective_report_profile(value: str | None, status: str | None) -> ReportProfile:
    profile = _normalize_report_profile(value)
    if profile == "standard" and str(status or "") != "ok":
        return "debug"
    return profile


COMPACT_RUNTIME_REPORT_DROP_KEYS = {
    "post_write_actual_draft_audit",
    "staged_post_write_actual_draft_audit",
    "postwrite_actual_draft_audit",
    "actual_draft_data",
    "draft_data",
}


def _compact_runtime_report(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    compact: dict[str, Any] = {}
    omitted: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if key in COMPACT_RUNTIME_REPORT_DROP_KEYS:
            omitted[key] = {
                "omitted": True,
                "reason": "debug_payload_available_only_in_debug_report_profile",
                "approx_json_bytes": len(json.dumps(dataclass_to_dict(value), ensure_ascii=False)),
            }
            continue
        compact[key] = value
    if omitted:
        compact["compact_report_omitted_debug_payloads"] = omitted
    return compact


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(dataclass_to_dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _hash_file(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        if path.exists() and path.is_file():
            return _sha256_bytes(path.read_bytes())
        gzip_path = path.with_suffix(path.suffix + ".gz")
        if gzip_path.exists() and gzip_path.is_file():
            return _sha256_bytes(gzip_path.read_bytes())
    except OSError:
        return ""
    return ""


def _safe_read_json(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _code_version_hash() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in (
        "src/aroll_v21/operator.py",
        "src/aroll_v21/engine.py",
        "src/aroll_v21/cli.py",
        "scripts/uat_fresh_draft.ps1",
    ):
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def _pipeline_config_hash(config: ArollV21OperatorConfig) -> str:
    return _hash_json(
        {
            "semantic_mode": normalize_semantic_mode(config.semantic_mode).value,
            "word_timeline_json": str(config.word_timeline_json or ""),
        }
    )


def _draft_hashes_from_real_result(result: RealDraftIngestResult | None, config: ArollV21OperatorConfig) -> dict[str, str]:
    if result is None:
        input_hash = _hash_file(config.input_json)
        fingerprint = _hash_json(
            {
                "input_json": str(config.input_json or ""),
                "input_json_hash": input_hash,
                "semantic_mode": normalize_semantic_mode(config.semantic_mode).value,
            }
        )
        return {
            "draft_fingerprint": fingerprint,
            "draft_content_hash": input_hash,
            "template_hash": "",
            "timeline_hash": input_hash,
        }
    metadata = result.metadata or {}
    draft_content_path = Path(str(metadata.get("draft_content_path") or "")) if metadata.get("draft_content_path") else None
    template_path = Path(str(metadata.get("template_path") or "")) if metadata.get("template_path") else None
    draft_content_hash = _hash_file(draft_content_path) or _hash_json(result.draft_data)
    template_hash = _hash_file(template_path) or _hash_json(
        {
            "text_materials": result.text_materials,
            "text_segments": result.text_segments,
        }
    )
    timeline_hash = _hash_json(
        {
            "timeline_id": str(metadata.get("timeline_id") or ""),
            "source_segments": result.source_segments,
            "text_segments": result.text_segments,
            "word_timeline_count": len(result.word_timeline or []),
        }
    )
    fingerprint = _hash_json(
        {
            "draft_dir": str(config.draft_dir or ""),
            "draft_content_hash": draft_content_hash,
            "template_hash": template_hash,
            "timeline_hash": timeline_hash,
            "semantic_mode": normalize_semantic_mode(config.semantic_mode).value,
        }
    )
    return {
        "draft_fingerprint": fingerprint,
        "draft_content_hash": draft_content_hash,
        "template_hash": template_hash,
        "timeline_hash": timeline_hash,
    }


def _semantic_artifact_input_hash(run_dir: Path) -> str:
    payload = {
        "semantic_adjudication_report": _safe_read_json(run_dir / "semantic_adjudication_report.json") or {},
        "semantic_decision_cache": _safe_read_json(run_dir / "semantic_decision_cache.json") or [],
        "semantic_request_payloads": _safe_read_json(run_dir / "semantic_request_payloads.json") or [],
    }
    return _hash_json(payload)


def _artifact_hashes(run_dir: Path, names: list[str] | None = None) -> dict[str, str]:
    selected = names or sorted(
        path.name
        for pattern in ("*.json", "*.json.gz")
        for path in run_dir.glob(pattern)
        if path.is_file()
    )
    hashes: dict[str, str] = {}
    for name in selected:
        path = run_dir / name
        digest = _hash_file(path)
        if digest:
            hashes[name] = digest
    return hashes


def _stage_timing_defaults(timings: dict[str, float] | None = None) -> dict[str, float]:
    base = {
        "total_seconds": 0.0,
        "dry_run_seconds": 0.0,
        "planning_seconds": 0.0,
        "semantic_adjudication_seconds": 0.0,
        "quality_gate_seconds": 0.0,
        "report_write_seconds": 0.0,
        "writeback_seconds": 0.0,
        "postwrite_core_audit_seconds": 0.0,
        "postwrite_debug_audit_seconds": 0.0,
    }
    for key, value in (timings or {}).items():
        if key in base:
            base[key] = round(float(value or 0.0), 6)
    return base


def _run_metadata(config: ArollV21OperatorConfig, real_draft_result: RealDraftIngestResult | None) -> dict[str, Any]:
    hashes = _draft_hashes_from_real_result(real_draft_result, config)
    return {
        **hashes,
        "pipeline_config_hash": _pipeline_config_hash(config),
        "code_version_hash": _code_version_hash(),
        "requested_report_profile": _normalize_report_profile(config.report_profile),
    }


def _write_artifact_manifest(
    run_dir: Path,
    *,
    report_profile: str,
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    effective_profile = _normalize_report_profile(report_profile)
    metadata_payload = dict(metadata)
    requested_profile = _normalize_report_profile(
        str(metadata_payload.pop("requested_report_profile", metadata_payload.pop("report_profile", effective_profile)))
    )
    reuse_artifacts = [
        "source_graph.json",
        "final_timeline.json",
        "captions.json",
        "material_write_plan.json",
        "semantic_decision_cache.json",
        "semantic_adjudication_report.json",
        "quality_gate_report.json",
        "prewrite_report.json",
        "writeback_report.json",
    ]
    manifest = {
        "artifact_manifest_version": 1,
        **metadata_payload,
        "report_profile": effective_profile,
        "requested_report_profile": requested_profile,
        "effective_report_profile": effective_profile,
        "run_dir": str(run_dir),
        "status": str(summary.get("status") or ""),
        "ready_for_disposable_write_pre_audit": bool(summary.get("READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT")),
        "blocker_codes": list(summary.get("blocker_codes") or []),
        "artifact_files": sorted(
            str(path.relative_to(run_dir)).replace("\\", "/")
            for pattern in ("*.json", "*.json.gz", "*.md")
            for path in run_dir.rglob(pattern)
            if path.is_file() and path.name != "artifact_manifest.json"
        ),
        "artifact_hashes": _artifact_hashes(run_dir, reuse_artifacts),
        "reuse_required_artifacts": reuse_artifacts,
        "semantic_cache_input_hash": _semantic_artifact_input_hash(run_dir),
    }
    write_json(run_dir / "artifact_manifest.json", manifest)


def _dataclass_from_dict(cls: Any, value: Any) -> Any:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"{getattr(cls, '__name__', cls)} payload must be an object")
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name in value:
            kwargs[field.name] = value[field.name]
        elif field.default is not MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            kwargs[field.name] = field.default_factory()  # type: ignore[misc]
    return cls(**kwargs)


def _words(rows: Any) -> list[CanonicalWord]:
    return [_dataclass_from_dict(CanonicalWord, row) for row in rows or [] if isinstance(row, dict)]


def _edit_units(rows: Any) -> list[EditUnit]:
    return [_dataclass_from_dict(EditUnit, row) for row in rows or [] if isinstance(row, dict)]


def _source_graph(payload: Any) -> CanonicalSourceGraph | None:
    if not isinstance(payload, dict):
        return None
    invariant = _dataclass_from_dict(SourceGraphInvariantReport, payload.get("invariant_report") or {})
    return CanonicalSourceGraph(
        words=_words(payload.get("words") or []),
        edit_units=_edit_units(payload.get("edit_units") or []),
        subtitle_rows=list(payload.get("subtitle_rows") or []),
        source_materials=list(payload.get("source_materials") or []),
        source_segments=list(payload.get("source_segments") or []),
        text_materials=list(payload.get("text_materials") or []),
        text_segments=list(payload.get("text_segments") or []),
        invariant_report=invariant,
    )


def _repeat_clusters(rows: Any) -> list[RepeatCluster]:
    clusters: list[RepeatCluster] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        clusters.append(
            RepeatCluster(
                cluster_id=str(row.get("cluster_id") or ""),
                variants=_edit_units(row.get("variants") or []),
                repeat_type=row.get("repeat_type") or "exact_repeat",
                evidence=[
                    _dataclass_from_dict(CandidateEvidence, evidence)
                    for evidence in row.get("evidence") or []
                    if isinstance(evidence, dict)
                ],
                local_recommendation=row.get("local_recommendation"),
            )
        )
    return clusters


def _decision_plan(payload: Any) -> DecisionPlan | None:
    if not isinstance(payload, dict):
        return None
    return DecisionPlan(
        decisions=[_dataclass_from_dict(TakeDecision, row) for row in payload.get("decisions") or [] if isinstance(row, dict)],
        split_decisions=[
            _dataclass_from_dict(UnitSplitPlan, row)
            for row in payload.get("split_decisions") or []
            if isinstance(row, dict)
        ],
        blocked=bool(payload.get("blocked")),
        blockers=[_dataclass_from_dict(Blocker, row) for row in payload.get("blockers") or [] if isinstance(row, dict)],
        semantic_request_payloads=list(payload.get("semantic_request_payloads") or []),
        decision_trace=list(payload.get("decision_trace") or []),
        semantic_decision_rows=list(payload.get("semantic_decision_rows") or []),
        semantic_adjudication_report=dict(payload.get("semantic_adjudication_report") or {}),
        final_target_repeat_accepted_cluster_ids=list(payload.get("final_target_repeat_accepted_cluster_ids") or []),
        final_target_repeat_unresolved_cluster_ids=list(payload.get("final_target_repeat_unresolved_cluster_ids") or []),
        modifier_redundancy_accepted_cluster_ids=list(payload.get("modifier_redundancy_accepted_cluster_ids") or []),
        modifier_redundancy_unresolved_cluster_ids=list(payload.get("modifier_redundancy_unresolved_cluster_ids") or []),
        semantic_unresolved_count=int(payload.get("semantic_unresolved_count") or 0),
        requires_human_review=bool(payload.get("requires_human_review")),
        write_allowed=bool(payload.get("write_allowed", True)),
        dry_run_continued_for_discovery=bool(payload.get("dry_run_continued_for_discovery")),
    )


def _final_timeline(rows: Any) -> list[FinalTimelineSegment]:
    return [_dataclass_from_dict(FinalTimelineSegment, row) for row in rows or [] if isinstance(row, dict)]


def _captions(rows: Any) -> list[CaptionRenderUnit]:
    return [_dataclass_from_dict(CaptionRenderUnit, row) for row in rows or [] if isinstance(row, dict)]


def _blocker_report(payload: Any) -> BlockerReport:
    if not isinstance(payload, dict):
        return BlockerReport(blocked=True, blockers=[], summary={})
    return BlockerReport(
        blocked=bool(payload.get("blocked")),
        blockers=[_dataclass_from_dict(Blocker, row) for row in payload.get("blockers") or [] if isinstance(row, dict)],
        summary=dict(payload.get("summary") or {}),
    )


def _load_ready_run_report(ready_run_dir: Path) -> RunReport:
    return RunReport(
        status=str(read_json(ready_run_dir / "run_summary.json").get("status") or "blocked"),  # type: ignore[arg-type]
        source_graph=_source_graph(read_json(ready_run_dir / "source_graph.json")),
        repeat_clusters=_repeat_clusters(read_json(ready_run_dir / "repeat_clusters.json")),
        decision_plan=_decision_plan(read_json(ready_run_dir / "decision_plan.json")),
        final_timeline=_final_timeline(read_json(ready_run_dir / "final_timeline.json")),
        captions=_captions(read_json(ready_run_dir / "captions.json")),
        material_write_plan=dict(read_json(ready_run_dir / "material_write_plan.json") or {}),
        validator_report=dict(read_json(ready_run_dir / "validator_report.json") or {}),
        postwrite_report=dict(read_json(ready_run_dir / "prewrite_report.json") or {}),
        blocker_report=_blocker_report(read_json(ready_run_dir / "blocker_report.json")),
        decision_trace=list(read_json(ready_run_dir / "decision_trace.json") or []),
        resolved_template_map=dict((read_json(ready_run_dir / "writeback_report.json") or {}).get("resolved_template_map") or {}),
        source_binding_report=dict(read_json(ready_run_dir / "writeback_report.json") or {}),
    )


def read_postwrite_materials_json(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("postwrite materials json must be a list of material objects")
    return rows


def _run_input_from_real_draft_result(
    result: RealDraftIngestResult,
    *,
    mode: Mode,
    postwrite_mode: str,
    postwrite_materials: list[dict[str, Any]] | None = None,
) -> ArollRunInput:
    return ArollRunInput(
        draft_data=result.draft_data,
        word_timeline=result.word_timeline,
        subtitles=result.subtitles,
        source_segments=result.source_segments,
        source_materials=result.source_materials,
        text_materials=result.text_materials,
        text_segments=result.text_segments,
        postwrite_materials=postwrite_materials,
        ingest_blockers=result.blockers,
        ingest_metadata=result.metadata,
        postwrite_mode=postwrite_mode,  # type: ignore[arg-type]
        mode=mode,
    )


def load_run_input(
    config: ArollV21OperatorConfig,
    *,
    postwrite_mode: str = "auto",
    real_draft_result: RealDraftIngestResult | None = None,
) -> ArollRunInput:
    use_postwrite_materials = postwrite_mode == "actual_decrypt"
    if config.input_json is None:
        if config.draft_dir is None:
            raise RuntimeError("V21_INPUT_JSON_OR_DRAFT_DIR_REQUIRED")
        result = real_draft_result or RealDraftIngestAdapter(jy_draftc=config.jy_draftc).load(
            config.draft_dir,
            config.run_dir,
            word_timeline_json=config.word_timeline_json,
        )
        postwrite_materials = (
            read_postwrite_materials_json(config.postwrite_materials_json)
            if use_postwrite_materials and config.postwrite_materials_json is not None
            else None
        )
        return _run_input_from_real_draft_result(
            result,
            mode=config.mode,
            postwrite_mode=postwrite_mode,
            postwrite_materials=list(postwrite_materials or []) if postwrite_materials is not None else None,
        )
    payload = read_json(config.input_json)
    postwrite_materials = payload.get("postwrite_materials") if "postwrite_materials" in payload else None
    if use_postwrite_materials and config.postwrite_materials_json is not None:
        postwrite_materials = read_postwrite_materials_json(config.postwrite_materials_json)
    if not use_postwrite_materials:
        postwrite_materials = None
    return ArollRunInput(
        draft_data=payload.get("draft_data") or {},
        word_timeline=list(payload.get("word_timeline") or []),
        subtitles=list(payload.get("subtitles") or []),
        source_segments=list(payload.get("source_segments") or []) if "source_segments" in payload else None,
        source_materials=list(payload.get("source_materials") or []) if "source_materials" in payload else None,
        text_materials=list(payload.get("text_materials") or []) if "text_materials" in payload else None,
        text_segments=list(payload.get("text_segments") or []) if "text_segments" in payload else None,
        postwrite_materials=list(postwrite_materials or []) if postwrite_materials is not None else None,
        postwrite_mode=postwrite_mode,  # type: ignore[arg-type]
        mode=config.mode,
    )


def write_operator_artifacts(
    report: RunReport,
    run_dir: Path,
    *,
    write_status: str,
    commit_performed: bool,
    report_profile: str = "standard",
    runtime_metadata: dict[str, Any] | None = None,
    stage_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    profile = _normalize_report_profile(report_profile)
    effective_profile = _effective_report_profile(profile, report.status)
    report_write_started = time.monotonic()
    write_run_artifacts(report, run_dir, report_profile=effective_profile)
    writeback_path = run_dir / "writeback_report.json"
    if not writeback_path.exists():
        write_json(
            writeback_path,
            {
                "writeback_attempted": False,
                "writeback_success": False,
                "WRITE_SUCCESS": False,
                "ENCRYPT_SUCCESS": False,
            },
        )
    summary = build_run_summary(report, write_status=write_status, commit_performed=commit_performed)
    semantic_report = report.decision_plan.semantic_adjudication_report if report.decision_plan else {}
    summary["deepseek_provider_config_source"] = str(semantic_report.get("deepseek_provider_config_source") or "")
    metadata = dict(runtime_metadata or {})
    summary.update(metadata)
    summary["requested_report_profile"] = profile
    summary["effective_report_profile"] = effective_profile
    summary["report_profile"] = effective_profile
    timings = dict(stage_timings or {})
    timings["report_write_seconds"] = time.monotonic() - report_write_started
    summary.update(_stage_timing_defaults(timings))
    write_json(run_dir / "run_summary.json", summary)
    manifest_metadata = {
        **metadata,
        "requested_report_profile": profile,
        "effective_report_profile": effective_profile,
    }
    _write_artifact_manifest(run_dir, report_profile=effective_profile, summary=summary, metadata=manifest_metadata)
    return summary


def _postwrite_environment(config: ArollV21OperatorConfig) -> dict[str, Any]:
    draft_content_path = ""
    if config.draft_dir is not None:
        draft_content_path = str(Path(config.draft_dir) / "draft_content.json")
    return {
        "draft_dir": str(config.draft_dir or ""),
        "jy_draftc_path": str(config.jy_draftc or ""),
        "jy_install_dir": str(os.environ.get("JY_INSTALL_DIR") or ""),
        "postwrite_decrypt_cwd": str(Path.cwd()),
        "draft_content_path": draft_content_path,
        "only_specified_draft_written": bool(config.draft_dir is not None),
    }


def _annotate_postwrite_environment(
    report: RunReport,
    config: ArollV21OperatorConfig,
    *,
    sacrificial_override_used: bool = False,
    writeback_report: dict[str, Any] | None = None,
) -> None:
    evidence = _postwrite_environment(config)
    postwrite = report.postwrite_report
    postwrite.update(evidence)
    if writeback_report:
        postwrite.update(writeback_report)
        postwrite.update(
            {
                "writeback_success": bool(writeback_report.get("writeback_success")),
                "WRITE_SUCCESS": bool(writeback_report.get("WRITE_SUCCESS")),
                "ENCRYPT_SUCCESS": bool(writeback_report.get("ENCRYPT_SUCCESS")),
            }
        )
        if writeback_report.get("writeback_success"):
            postwrite["ready_for_user_manual_qc"] = True
        _merge_prewrite_quality_gate(report, writeback_report)
    semantic_mode = normalize_semantic_mode(config.semantic_mode).value
    postwrite["semantic_mode"] = semantic_mode
    postwrite["semantic_decisions_generated_from_current_draft"] = semantic_mode == SemanticAdjudicationMode.DETERMINISTIC_BASELINE.value
    postwrite["semantic_decisions_reused_from_old_draft"] = False
    if sacrificial_override_used:
        ready_for_user_manual_qc = bool(writeback_report and writeback_report.get("writeback_success"))
        postwrite.update(
            {
                "sacrificial_write_override_used": True,
                "postwrite_decrypt_skipped_for_sacrificial_draft": True,
                "postwrite_decrypt_skip_reason": "ACTUAL_POSTWRITE_DECRYPT_UNAVAILABLE",
                "ready_for_user_manual_qc": ready_for_user_manual_qc,
            }
        )
    validator_postwrite = (report.validator_report or {}).get("postwrite_material_validator")
    if isinstance(validator_postwrite, dict):
        validator_postwrite.update(postwrite)
    if report.blocker_report and isinstance(report.blocker_report.summary, dict):
        report.blocker_report.summary.update(
            {
                "sacrificial_write_override_used": bool(sacrificial_override_used),
                "postwrite_decrypt_skipped_for_sacrificial_draft": bool(
                    postwrite.get("postwrite_decrypt_skipped_for_sacrificial_draft")
                ),
                "postwrite_decrypt_skip_reason": str(postwrite.get("postwrite_decrypt_skip_reason") or ""),
                "writeback_success": bool(postwrite.get("writeback_success")),
                "WRITE_SUCCESS": bool(postwrite.get("WRITE_SUCCESS")),
                "ENCRYPT_SUCCESS": bool(postwrite.get("ENCRYPT_SUCCESS")),
                **{key: postwrite.get(key, value) for key, value in evidence.items()},
            }
        )


def _merge_prewrite_quality_gate(report: RunReport, writeback_report: dict[str, Any]) -> None:
    quality = (report.validator_report or {}).get("quality_gate_report")
    if not isinstance(quality, dict):
        return
    speed_gate = writeback_report.get("effective_speed_gate")
    if not isinstance(speed_gate, dict):
        return
    quality["effective_speed_gate"] = dict(speed_gate)
    quality["effective_speed_gate_present"] = True
    missing = [item for item in quality.get("missing_required_gates") or [] if item != "effective_speed_gate"]
    quality["missing_required_gates"] = missing
    existing_codes = {str(code) for code in quality.get("blocker_codes") or []}
    existing_codes.update(str(code) for code in speed_gate.get("blocker_codes") or [])
    if not missing:
        existing_codes.discard("V21_QUALITY_GATE_MISSING_REQUIRED_GATE")
    quality["blocker_codes"] = sorted(code for code in existing_codes if code)
    quality["gate_passed"] = bool(quality.get("gate_passed")) and bool(speed_gate.get("gate_passed")) and not quality["blocker_codes"]
    quality["ready_for_user_manual_qc_preconditions_passed"] = (
        bool(quality.get("ready_for_user_manual_qc_preconditions_passed"))
        and bool(speed_gate.get("gate_passed"))
        and not quality["blocker_codes"]
    )


def _writeback_blocked_report(report: RunReport, writeback_result: WritebackResult) -> RunReport:
    blockers = list(report.blocker_report.blockers if report.blocker_report else []) + list(writeback_result.blockers)
    summary = dict(report.blocker_report.summary if report.blocker_report else {})
    summary.update(
        {
            "stage": "writeback",
            "writeback_success": False,
            "WRITE_SUCCESS": False,
            "ENCRYPT_SUCCESS": bool(writeback_result.report.get("ENCRYPT_SUCCESS")),
            "write_allowed": False,
            "ready_for_write": False,
            "READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT": False,
        }
    )
    postwrite = dict(report.postwrite_report or {})
    postwrite.update(writeback_result.report)
    return replace(
        report,
        status="blocked",
        postwrite_report=postwrite,
        blocker_report=BlockerReport(blocked=True, blockers=blockers, summary=summary),
    )


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


def write_boundary_block(config: ArollV21OperatorConfig, blocker: Blocker) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    profile = _normalize_report_profile(config.report_profile)
    effective_profile = _effective_report_profile(profile, "blocked")
    for name in REQUIRED_ARTIFACTS:
        if effective_profile == "minimal" and name not in MINIMAL_ARTIFACTS:
            continue
        if name in {"blocker_report.json", "run_summary.json", "postwrite_report.json"}:
            continue
        list_artifacts = {
            "edit_units.json",
            "final_edl.json",
            "semantic_decisions.resolved.json",
            "semantic_decision_cache.json",
        }
        write_json(config.run_dir / name, [] if name.endswith("s.json") or name in list_artifacts else {})
    blocker_report = BlockerReport(blocked=True, blockers=[blocker], summary={"mode": config.mode})
    postwrite_report = {
        "postwrite_mode": "unavailable",
        "postwrite_decrypt_ok": False,
        "postwrite_material_gate_ok": False,
        "block_reason": blocker.code,
    }
    summary = {
        "status": "blocked",
        "mode": config.mode,
        "write_status": "blocked",
        "commit_performed": False,
        "postwrite_mode": "unavailable",
        "postwrite_decrypt_ok": False,
        "postwrite_style_gate_ok": False,
        "commit_only_after_all_validators": True,
        "deepseek_provider_configured": False,
        "deepseek_provider_called_count": 0,
        "deepseek_provider_error": "",
        "deepseek_batch_enabled": False,
        "deepseek_batch_request_count": 0,
        "deepseek_batch_attempt_count": 0,
        "deepseek_batch_retry_count": 0,
        "deepseek_batch_issue_count": 0,
        "deepseek_batch_resolved_count": 0,
        "deepseek_batch_unresolved_count": 0,
        "deepseek_batch_missing_issue_ids": [],
        "deepseek_batch_error": "",
        "commit_reused_semantic_cache": False,
        "semantic_cache_input_hash": str(blocker.context.get("semantic_cache_input_hash") or ""),
        "semantic_cache_issue_count": 0,
        "semantic_cache_resolved_count": 0,
        "semantic_cache_unresolved_count": 0,
        "blocker_count": 1,
        "blocker_codes": [blocker.code],
        "requested_report_profile": profile,
        "effective_report_profile": effective_profile,
        "report_profile": effective_profile,
        **_stage_timing_defaults(),
    }
    write_json(config.run_dir / "blocker_report.json", blocker_report)
    if effective_profile != "minimal":
        write_json(config.run_dir / "postwrite_report.json", postwrite_report)
    write_json(config.run_dir / "writeback_report.json", postwrite_report)
    write_json(config.run_dir / "run_summary.json", summary)
    _write_artifact_manifest(
        config.run_dir,
        report_profile=effective_profile,
        summary=summary,
        metadata={
            "pipeline_config_hash": _pipeline_config_hash(config),
            "code_version_hash": _code_version_hash(),
            "requested_report_profile": profile,
            "effective_report_profile": effective_profile,
        },
    )
    return summary


def _semantic_decisions_planner(config: ArollV21OperatorConfig) -> tuple[Any | None, Any | None, Blocker | None]:
    semantic_mode = normalize_semantic_mode(config.semantic_mode)
    if semantic_mode == SemanticAdjudicationMode.DETERMINISTIC_BASELINE:
        if config.semantic_decisions_json is not None:
            return None, None, Blocker(
                code="SEMANTIC_MODE_CONFLICT",
                message="deterministic baseline semantic mode must not be combined with semantic_decisions_json",
                layer="operator",
                context={"semantic_decisions_json": str(config.semantic_decisions_json)},
            )
        return DeterministicBaselineSemanticPlanner(), None, None
    if str(config.semantic_mode or "") not in {
        "",
        "default",
        "auto",
        "semantic-requests-only",
        "semantic_requests_only",
        "deepseek",
        "fail-closed",
        "fail_closed",
    }:
        return None, None, Blocker(
            code="SEMANTIC_MODE_UNSUPPORTED",
            message="unsupported V21 semantic mode",
            layer="operator",
            context={"semantic_mode": config.semantic_mode},
        )
    cache_path = config.run_dir / "semantic_decision_cache.json"
    cache_input_hash = _semantic_cache_input_hash(config)
    if config.semantic_decisions_json is None and config.mode == "write" and cache_path.exists():
        try:
            rows = read_json(cache_path)
        except Exception as exc:
            return None, None, Blocker(
                code="SEMANTIC_DECISION_CACHE_INVALID",
                message="semantic decision cache could not be parsed",
                layer="operator",
                context={"path": str(cache_path), "error": str(exc)},
            )
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return None, None, Blocker(
                code="SEMANTIC_DECISION_CACHE_INVALID",
                message="semantic decision cache must be a list of semantic decision rows",
                layer="operator",
                context={"path": str(cache_path)},
            )
        previous_report_path = config.run_dir / "semantic_adjudication_report.json"
        previous_report = read_json(previous_report_path) if previous_report_path.exists() else {}
        if isinstance(previous_report, dict):
            previous_hash = str(previous_report.get("semantic_cache_input_hash") or "")
            if previous_hash and previous_hash != cache_input_hash:
                return None, None, Blocker(
                    code="SEMANTIC_DECISION_CACHE_INPUT_HASH_MISMATCH",
                    message="semantic decision cache does not match current run input hash",
                    layer="operator",
                    context={"path": str(cache_path), "expected_hash": cache_input_hash, "cache_hash": previous_hash},
                )
            if int(previous_report.get("semantic_cache_unresolved_count") or previous_report.get("deepseek_batch_unresolved_count") or 0) > 0:
                return None, None, Blocker(
                    code="SEMANTIC_DECISION_CACHE_UNRESOLVED",
                    message="semantic decision cache contains unresolved provider-required issues",
                    layer="operator",
                    context={"path": str(cache_path)},
                )
        planner = SemanticDecisionsJsonPlanner(rows)
        setattr(planner, "semantic_decision_cache_used", True)
        setattr(planner, "commit_reused_semantic_cache", True)
        setattr(planner, "semantic_cache_input_hash", cache_input_hash)
        setattr(planner, "semantic_cache_issue_count", len(rows))
        setattr(planner, "semantic_cache_resolved_count", len(rows))
        setattr(planner, "semantic_cache_unresolved_count", 0)
        return planner, None, None
    if (
        config.semantic_decisions_json is None
        and config.mode == "write"
        and semantic_mode in {SemanticAdjudicationMode.AUTO, SemanticAdjudicationMode.DEEPSEEK}
    ):
        return None, None, None
    if config.semantic_decisions_json is None:
        provider = deepseek_provider_from_env() if semantic_mode in {SemanticAdjudicationMode.AUTO, SemanticAdjudicationMode.DEEPSEEK} else None
        if provider is not None:
            setattr(provider, "semantic_cache_input_hash", cache_input_hash)
            setattr(provider, "semantic_cache_issue_count", 0)
            setattr(provider, "semantic_cache_resolved_count", 0)
            setattr(provider, "semantic_cache_unresolved_count", 0)
        return None, provider, None
    try:
        rows = read_json(config.semantic_decisions_json)
    except Exception as exc:
        return None, None, Blocker(
            code="SEMANTIC_DECISIONS_JSON_INVALID",
            message="semantic decisions json could not be parsed",
            layer="operator",
            context={"path": str(config.semantic_decisions_json), "error": str(exc)},
        )
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return None, None, Blocker(
            code="SEMANTIC_DECISIONS_JSON_INVALID",
            message="semantic decisions json must be a list of objects",
            layer="operator",
            context={"path": str(config.semantic_decisions_json)},
        )
    return SemanticDecisionsJsonPlanner(rows), None, None


def _semantic_cache_input_hash(config: ArollV21OperatorConfig) -> str:
    digest = hashlib.sha256()
    digest.update(str(config.semantic_mode or "").encode("utf-8"))
    if config.input_json is not None and config.input_json.exists():
        digest.update(config.input_json.read_bytes())
    elif config.draft_dir is not None:
        digest.update(str(config.draft_dir).encode("utf-8"))
    return digest.hexdigest()


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


def _summary_blocker_codes(summary: dict[str, Any]) -> list[str]:
    raw = summary.get("blocker_codes") or summary.get("BLOCKER_CODES") or []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item) for item in raw if str(item)]


def _ready_reuse_rejected(reason: str, *, context: dict[str, Any] | None = None) -> Blocker:
    return Blocker(
        code="READY_RUN_REUSE_REJECTED",
        message=f"READY dry-run cannot be reused for commit: {reason}",
        layer="operator",
        severity="fatal",
        context={"reason": reason, **dict(context or {})},
    )


def _validate_ready_run_dir(
    config: ArollV21OperatorConfig,
    *,
    current_metadata: dict[str, Any],
) -> Blocker | None:
    ready_run_dir = Path(config.ready_run_dir or "")
    if not ready_run_dir.exists() or not ready_run_dir.is_dir():
        return _ready_reuse_rejected("ready_run_dir_missing", context={"ready_run_dir": str(ready_run_dir)})
    summary_path = ready_run_dir / "run_summary.json"
    manifest_path = ready_run_dir / "artifact_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return _ready_reuse_rejected(
            "ready_run_missing_summary_or_manifest",
            context={"summary_path": str(summary_path), "manifest_path": str(manifest_path)},
        )
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    if not isinstance(summary, dict) or not isinstance(manifest, dict):
        return _ready_reuse_rejected("ready_run_summary_or_manifest_invalid")
    blocker_codes = _summary_blocker_codes(summary)
    if str(summary.get("status") or "") != "ok":
        return _ready_reuse_rejected("ready_run_status_not_ok", context={"status": str(summary.get("status") or "")})
    if not bool(summary.get("READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT")):
        return _ready_reuse_rejected("ready_run_not_ready")
    if blocker_codes:
        return _ready_reuse_rejected("ready_run_has_blockers", context={"blocker_codes": blocker_codes})
    for key in ("draft_fingerprint", "draft_content_hash", "template_hash", "timeline_hash"):
        expected = str(manifest.get(key) or summary.get(key) or "")
        current = str(current_metadata.get(key) or "")
        if expected and current and expected != current:
            return _ready_reuse_rejected(
                f"{key}_mismatch",
                context={"field": key, "ready_value": expected, "current_value": current},
            )
    for key in ("pipeline_config_hash", "code_version_hash"):
        expected = str(manifest.get(key) or summary.get(key) or "")
        current = str(current_metadata.get(key) or "")
        if expected and current and expected != current:
            return _ready_reuse_rejected(
                f"{key}_mismatch",
                context={"field": key, "ready_value": expected, "current_value": current},
            )
    recorded_semantic_hash = str(manifest.get("semantic_cache_input_hash") or summary.get("semantic_cache_input_hash") or "")
    current_semantic_hash = _semantic_artifact_input_hash(ready_run_dir)
    if recorded_semantic_hash and recorded_semantic_hash != current_semantic_hash:
        return _ready_reuse_rejected(
            "semantic_cache_input_hash_mismatch",
            context={"ready_value": recorded_semantic_hash, "current_value": current_semantic_hash},
        )
    artifact_hashes = manifest.get("artifact_hashes") or {}
    if not isinstance(artifact_hashes, dict):
        return _ready_reuse_rejected("artifact_hashes_invalid")
    for name in manifest.get("reuse_required_artifacts") or []:
        artifact = ready_run_dir / str(name)
        artifact_hash = _hash_file(artifact)
        if not artifact_hash:
            return _ready_reuse_rejected("required_reuse_artifact_missing", context={"artifact": str(artifact)})
        expected_hash = str(artifact_hashes.get(str(name)) or "")
        if expected_hash and expected_hash != artifact_hash:
            return _ready_reuse_rejected("required_reuse_artifact_hash_mismatch", context={"artifact": str(artifact)})
    final_timeline = read_json(ready_run_dir / "final_timeline.json")
    captions = read_json(ready_run_dir / "captions.json")
    material_write_plan = read_json(ready_run_dir / "material_write_plan.json")
    writeback_report = read_json(ready_run_dir / "writeback_report.json")
    if not isinstance(final_timeline, list) or not final_timeline:
        return _ready_reuse_rejected("final_timeline_missing_or_empty")
    if not isinstance(captions, list) or not captions:
        return _ready_reuse_rejected("captions_missing_or_empty")
    if not isinstance(material_write_plan, dict):
        return _ready_reuse_rejected("material_write_plan_invalid")
    plan_segments = material_write_plan.get("segments") or []
    plan_materials = material_write_plan.get("materials") or []
    if len(plan_segments) != len(captions) or len(plan_materials) != len(captions):
        return _ready_reuse_rejected(
            "material_write_plan_caption_count_mismatch",
            context={
                "caption_count": len(captions),
                "segment_count": len(plan_segments),
                "material_count": len(plan_materials),
            },
        )
    resolved_template_map = (writeback_report or {}).get("resolved_template_map") if isinstance(writeback_report, dict) else {}
    if not isinstance(resolved_template_map, dict) or len(resolved_template_map) != len(final_timeline):
        return _ready_reuse_rejected(
            "resolved_template_map_incomplete",
            context={
                "resolved_template_map_count": len(resolved_template_map) if isinstance(resolved_template_map, dict) else 0,
                "final_timeline_count": len(final_timeline),
            },
        )
    return None


def _commit_from_ready_run_dir(
    config: ArollV21OperatorConfig,
    *,
    real_draft_result: RealDraftIngestResult,
    runtime_metadata: dict[str, Any],
    stage_timings: dict[str, float],
) -> dict[str, Any]:
    if config.draft_dir is None:
        return write_boundary_block(
            config,
            _ready_reuse_rejected("commit_from_ready_run_requires_draft_dir"),
        )
    if not config.commit:
        return write_boundary_block(
            config,
            _ready_reuse_rejected("commit_from_ready_run_requires_commit_flag"),
        )
    rejection = _validate_ready_run_dir(config, current_metadata=runtime_metadata)
    if rejection is not None:
        return write_boundary_block(config, rejection)
    ready_run_dir = Path(config.ready_run_dir or "")
    try:
        ready_report = _load_ready_run_report(ready_run_dir)
    except Exception as exc:
        return write_boundary_block(
            config,
            _ready_reuse_rejected("ready_run_artifacts_could_not_be_loaded", context={"error": str(exc)}),
        )
    if ready_report.status != "ok":
        return write_boundary_block(
            config,
            _ready_reuse_rejected("ready_run_report_status_not_ok", context={"status": ready_report.status}),
        )
    ready_report.postwrite_report.update(
        {
            "commit_from_ready_run_dir": True,
            "ready_run_dir": str(ready_run_dir),
            "planning_reused_from_ready_run": True,
            "deepseek_reused_from_ready_run": True,
        }
    )
    if config.simulate_write:
        return write_operator_artifacts(
            ready_report,
            config.run_dir,
            write_status="ready_run_reused_simulated_write_no_commit",
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )
    writeback_started = time.monotonic()
    writeback_result = RealDraftWriteback(jy_draftc=config.jy_draftc).commit(
        draft_dir=config.draft_dir,
        run_dir=config.run_dir,
        real_draft_result=real_draft_result,
        run_report=ready_report,
        sacrificial_write_override_used=bool(config.allow_sacrificial_write_without_postwrite_decrypt),
    )
    stage_timings["writeback_seconds"] = time.monotonic() - writeback_started
    stage_timings["postwrite_core_audit_seconds"] = stage_timings["writeback_seconds"]
    write_profiled_report_json(config.run_dir / "writeback_report.json", writeback_result.report, config.report_profile)
    if not writeback_result.success:
        blocked = _writeback_blocked_report(ready_report, writeback_result)
        _annotate_postwrite_environment(
            blocked,
            config,
            sacrificial_override_used=bool(config.allow_sacrificial_write_without_postwrite_decrypt),
            writeback_report=writeback_result.report,
        )
        return write_operator_artifacts(
            blocked,
            config.run_dir,
            write_status="blocked_ready_run_reuse_writeback_failed",
            commit_performed=False,
            report_profile=config.report_profile,
            runtime_metadata=runtime_metadata,
            stage_timings=stage_timings,
        )
    _annotate_postwrite_environment(
        ready_report,
        config,
        sacrificial_override_used=bool(config.allow_sacrificial_write_without_postwrite_decrypt),
        writeback_report=writeback_result.report,
    )
    return write_operator_artifacts(
        ready_report,
        config.run_dir,
        write_status="committed_from_ready_run_without_replanning",
        commit_performed=True,
        report_profile=config.report_profile,
        runtime_metadata=runtime_metadata,
        stage_timings=stage_timings,
    )


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
