from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jy_bridge import DEFAULT_JY_DRAFTC
from aroll_root_mirror import root_mirror_report_from_exception, root_mirrors_timeline_id

from aroll_v21.ingest.real_draft_adapter import RealDraftIngestResult
from aroll_v21.ir.models import Blocker, RunReport
from aroll_v21.quality.effective_speed_gate import build_effective_speed_gate
from aroll_v21.writeback.dynamic_source_binder import CurrentDraftInventory, DynamicSourceBinder
from aroll_v21.writeback.effect_policy import EffectTrackPolicy
from aroll_v21.writeback.source_segment_template_resolver import SOURCE_TEMPLATE_REPORT_DEFAULTS
from aroll_v21.writeback.speed_resolver import SpeedResolutionError, SpeedResolver


RootMirrorFunc = Callable[[Path, Path, Path, str], bool]


@dataclass(frozen=True)
class PreflightResult:
    success: bool
    blockers: list[Blocker] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


class DynamicSourceBindingPreflight:
    def __init__(
        self,
        *,
        jy_draftc: Path | None = None,
        root_mirror_func: RootMirrorFunc | None = None,
        speed_resolver: SpeedResolver | None = None,
        effect_policy: EffectTrackPolicy | None = None,
        allow_preserve_unsupported_effect_tracks: bool = False,
    ) -> None:
        self.jy_draftc = Path(jy_draftc) if jy_draftc is not None else Path(DEFAULT_JY_DRAFTC)
        self.root_mirror_func = root_mirror_func or root_mirrors_timeline_id
        self.speed_resolver = speed_resolver or SpeedResolver()
        self.effect_policy = effect_policy or EffectTrackPolicy()
        self.allow_preserve_unsupported_effect_tracks = bool(allow_preserve_unsupported_effect_tracks)

    def preflight(
        self,
        *,
        draft_dir: Path | None,
        real_draft_result: RealDraftIngestResult,
        run_report: RunReport,
        run_dir: Path | None = None,
    ) -> PreflightResult:
        base_report = self._base_report(
            draft_dir=draft_dir,
            real_draft_result=real_draft_result,
            writeback_attempted=False,
        )
        inventory = CurrentDraftInventory.from_real_draft_result(real_draft_result, draft_dir=draft_dir)
        resolution = DynamicSourceBinder(inventory).bind(run_report.final_timeline, run_report.source_graph)
        if resolution.blockers:
            blocker = resolution.blockers[0]
            return self._blocked(blocker.code, blocker.message, base_report | resolution.report | blocker.context)

        try:
            video_track_id = self._select_video_track_id_from_templates(resolution.resolved_templates, run_report)
            speed_report = self._speed_preflight(real_draft_result.draft_data, resolution.resolved_templates)
            effective_speed_gate = build_effective_speed_gate(
                final_timeline=run_report.final_timeline,
                resolved_template_map=dict(resolution.report.get("resolved_template_map") or {}),
                draft_data=real_draft_result.draft_data,
                speed_resolver=SpeedResolver(real_draft_result.draft_data)
                if type(self.speed_resolver) is SpeedResolver
                else self.speed_resolver,
            )
            if not effective_speed_gate.get("gate_passed"):
                blocker_code = str((effective_speed_gate.get("blocker_codes") or ["V21_EFFECTIVE_SPEED_DRIFT"])[0])
                raise PreflightError(
                    blocker_code,
                    "effective video speed would drift after writeback source_timerange mapping",
                    {"effective_speed_gate": effective_speed_gate, **_flatten_effective_speed_gate(effective_speed_gate)},
                )
            effect_report = self._effect_preflight(real_draft_result.draft_data, run_report)
            root_report = self._root_mirror_preflight(
                draft_dir=draft_dir,
                run_dir=run_dir,
                real_draft_result=real_draft_result,
            )
        except PreflightError as exc:
            return self._blocked(exc.code, exc.message, base_report | resolution.report | exc.context)

        report = base_report | resolution.report | speed_report | effect_report | root_report | {
            "prewrite_source_segment_template_available": True,
            "selected_video_track_id": video_track_id,
            "dynamic_binding_complete": True,
            "resolved_template_map_complete": len(resolution.report.get("resolved_template_map") or {}) == len(run_report.final_timeline),
            "speed_safe": bool((speed_report.get("video_preflight") or {}).get("speed_safe")),
            "detected_speeds": list((speed_report.get("video_preflight") or {}).get("speeds") or []),
            "effective_speed_gate": effective_speed_gate,
            **_flatten_effective_speed_gate(effective_speed_gate),
            "effect_policy_safe": bool((effect_report.get("filter_preflight") or {}).get("effect_policy_safe", True)),
            "embedded_effects_preserved": bool((effect_report.get("filter_preflight") or {}).get("segment_embedded_effects_preserved")),
        }
        return PreflightResult(success=True, report=report)

    def _speed_preflight(self, draft_data: dict[str, Any], templates: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        resolver = SpeedResolver(draft_data) if type(self.speed_resolver) is SpeedResolver else self.speed_resolver
        for template in templates:
            try:
                rows.append(resolver.resolve(template, draft_data).report)
            except SpeedResolutionError as exc:
                raise PreflightError(exc.code, exc.message, {"video_preflight": {"main_video_speed_safe": False, **exc.context}}) from exc
        return {
            "video_preflight": {
                "main_video_speed_safe": all(bool(row.get("speed_safe")) for row in rows),
                "speed_safe": all(bool(row.get("speed_safe")) for row in rows),
                "speed_reports": rows,
                "speeds": [row.get("detected_speed") for row in rows],
            }
        }

    def _effect_preflight(self, draft_data: dict[str, Any], run_report: RunReport) -> dict[str, Any]:
        final_duration = max((segment.target_end_us for segment in run_report.final_timeline), default=0)
        result = self.effect_policy.inspect(
            draft_data,
            final_duration_us=final_duration,
            allow_preserve_unsupported_effect_tracks=self.allow_preserve_unsupported_effect_tracks,
        )
        if not result.safe:
            raise PreflightError(result.blocker_code, result.blocker_message, {"filter_preflight": result.report}) 
        return {"filter_preflight": result.report}

    def _root_mirror_preflight(
        self,
        *,
        draft_dir: Path | None,
        run_dir: Path | None,
        real_draft_result: RealDraftIngestResult,
    ) -> dict[str, Any]:
        metadata = real_draft_result.metadata or {}
        timeline_id = str(metadata.get("timeline_id") or "")
        if draft_dir is None or not timeline_id:
            return {
                "root_mirror_required": False,
                "root_mirror_check_failed": False,
                "root_mirror_error": "",
                "root_mirror_synced": False,
                "root_mirror_targets": [],
            }
        try:
            required = bool(self.root_mirror_func(Path(draft_dir), self.jy_draftc, Path(run_dir or ""), timeline_id))
        except Exception as exc:
            detection_report = root_mirror_report_from_exception(
                exc,
                Path(draft_dir),
                timeline_id,
            )
            self._write_root_mirror_detection_report(Path(run_dir or ""), detection_report)
            raise PreflightError(
                "V21_WRITEBACK_ROOT_MIRROR_DETECTION_FAILED",
                "root mirror requirement could not be determined safely",
                {
                    "root_mirror_required": None,
                    "root_mirror_check_failed": True,
                    "root_mirror_error": str(exc),
                    "root_mirror_synced": False,
                    "root_mirror_targets": [],
                    "root_mirror_detection_report": detection_report,
                },
            ) from exc
        return {
            "root_mirror_required": required,
            "root_mirror_check_failed": False,
            "root_mirror_error": "",
            "root_mirror_synced": False,
            "root_mirror_targets": [str(Path(draft_dir) / "draft_content.json"), str(Path(draft_dir) / "template-2.tmp")] if required else [],
        }

    def _write_root_mirror_detection_report(self, run_dir: Path, report: dict[str, Any]) -> None:
        if not str(run_dir):
            return
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "root_mirror_detection_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                "utf-8",
            )
        except OSError as exc:
            report["report_write_error"] = str(exc)

    def _select_video_track_id_from_templates(self, used_templates: list[dict[str, Any]], run_report: RunReport) -> str:
        if not used_templates or len(used_templates) != len(run_report.final_timeline):
            raise PreflightError(
                "V21_DYNAMIC_BINDING_REQUIRED",
                "resolved source template map is incomplete",
                {"final_timeline_segment_count": len(run_report.final_timeline), "resolved_source_segment_template_count": len(used_templates)},
            )
        track_ids = {str(template.get("track_id") or "") for template in used_templates}
        if "" in track_ids:
            raise PreflightError(
                "V21_WRITEBACK_MAIN_VIDEO_TRACK_NOT_FOUND",
                "source segment templates do not include track_id",
                {"used_source_segment_ids": sorted(str(template.get("id") or "") for template in used_templates)},
            )
        if len(track_ids) != 1:
            raise PreflightError(
                "V21_DYNAMIC_BINDING_PRIMARY_VIDEO_AMBIGUOUS",
                "final timeline resolved to multiple video tracks",
                {"used_video_track_ids": sorted(track_ids)},
            )
        return next(iter(track_ids))

    def _base_report(
        self,
        *,
        draft_dir: Path | None,
        real_draft_result: RealDraftIngestResult,
        writeback_attempted: bool,
    ) -> dict[str, Any]:
        metadata = real_draft_result.metadata or {}
        return {
            "writeback_attempted": bool(writeback_attempted),
            "writeback_success": False,
            "commit_performed": False,
            "encrypt_success": False,
            "WRITE_SUCCESS": False,
            "ENCRYPT_SUCCESS": False,
            "draft_dir": str(draft_dir or ""),
            "timeline_id": str(metadata.get("timeline_id") or ""),
            "timeline_dir": str(metadata.get("timeline_dir") or ""),
            "draft_content_path": str(metadata.get("draft_content_path") or ""),
            "template_path": str(metadata.get("template_path") or ""),
            "target_writes": {},
            "selected_text_track_id": None,
            "selected_video_track_id": None,
            "sacrificial_write_override_used": False,
            "postwrite_decrypt_skipped_for_sacrificial_draft": False,
            "borrowed_v20_low_level_io_reference": True,
            **SOURCE_TEMPLATE_REPORT_DEFAULTS,
        }

    def _blocked(self, code: str, message: str, report: dict[str, Any]) -> PreflightResult:
        return PreflightResult(
            success=False,
            blockers=[Blocker(code=code, message=message, layer="writeback", context=report)],
            report=report | {"writeback_success": False, "block_reason": code},
        )


class PreflightError(RuntimeError):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


def _flatten_effective_speed_gate(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "effective_speed_gate_passed": bool(report.get("gate_passed")),
        "effective_speed_min": report.get("effective_speed_min"),
        "effective_speed_max": report.get("effective_speed_max"),
        "effective_speed_drift_count": int(report.get("effective_speed_drift_count") or 0),
        "effective_speed_projected_row_missing_count": int(report.get("effective_speed_projected_row_missing_count") or 0),
        "effective_speed_projected_row_count": int(report.get("effective_speed_projected_row_count") or len(report.get("segment_reports") or [])),
        "expected_speeds": list(report.get("expected_speeds") or []),
        "segment_reports": list(report.get("segment_reports") or []),
    }
