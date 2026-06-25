from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jy_bridge import (
    DEFAULT_JY_DRAFTC,
    assert_all_project_timeline_files_match_folder_ids,
    assert_layout_has_no_duplicate_timeline_ids,
    decrypt,
    assert_timeline_content_id,
    encrypt,
)
from aroll_root_mirror import root_mirror_report_from_exception, root_mirrors_timeline_id

from aroll_v21.ingest.real_draft_adapter import RealDraftIngestResult
from aroll_v21.ir.models import Blocker, CaptionRenderUnit, FinalTimelineSegment, RunReport
from aroll_v21.quality.caption_alignment import build_caption_alignment_report
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.validate.rough_cut_quality import build_rough_cut_quality_metrics
from aroll_v21.writeback import actual_timeline_audit as actual_timeline_audit_helpers
from aroll_v21.writeback import snapshot_guard as snapshot_guard_helpers
from aroll_v21.writeback import video_projection as video_projection_helpers
from aroll_v21.writeback import writeback_time_utils as writeback_time_utils_helpers
from aroll_v21.writeback.effect_policy import EffectTrackPolicy
from aroll_v21.writeback import postwrite_audit as postwrite_audit_helpers
from aroll_v21.writeback.source_segment_template_resolver import (
    SOURCE_TEMPLATE_REPORT_DEFAULTS,
    SourceSegmentTemplateResolution,
)
from aroll_v21.writeback.speed_resolver import SpeedResolutionError, SpeedResolver
from aroll_v21.writeback import track_selector
from aroll_v21.writeback.video_write_plan_projector import (
    project_video_segment_from_template,
    safe_handle_report_from_projected_segments,
)
from aroll_v21.writeback.writeback_reports import (
    _base_report as build_base_report,
    _assert_writeback_rough_cut_quality as assert_writeback_rough_cut_quality_report,
    _finalize_post_write_actual_draft_audit,
    _flatten_post_write_actual_draft_audit,
    _overlap_count as overlap_count_report,
    _unique_strings,
    _writeback_rough_cut_quality as build_writeback_rough_cut_quality,
)


EncryptFunc = Callable[[Path, Path, Path], None]
DecryptFunc = Callable[[Path, Path, Path], None]
RootMirrorFunc = Callable[[Path, Path, Path, str], bool]
TimelineContentCheckFunc = Callable[[dict[str, Any], str, Path], None]
LayoutCheckFunc = Callable[[Path], None]
ProjectFolderCheckFunc = Callable[[Path, Path, Path], None]

EPSILON = 0.0001
EFFECTIVE_SPEED_TOLERANCE = 0.01


@dataclass(frozen=True)
class WritebackResult:
    success: bool
    blockers: list[Blocker] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


class RealDraftWriteback:
    """V21-only real draft writeback backend.

    This module performs deterministic IO for the already-compiled V21
    timeline. It only borrows low-level draft JSON/encrypt/write conventions
    and must not import legacy orchestration or patch modules.
    """

    def __init__(
        self,
        *,
        jy_draftc: Path | None = None,
        encrypt_func: EncryptFunc | None = None,
        decrypt_func: DecryptFunc | None = None,
        root_mirror_func: RootMirrorFunc | None = None,
        timeline_content_check_func: TimelineContentCheckFunc | None = None,
        layout_check_func: LayoutCheckFunc | None = None,
        project_folder_check_func: ProjectFolderCheckFunc | None = None,
    ) -> None:
        self.jy_draftc = Path(jy_draftc) if jy_draftc is not None else Path(DEFAULT_JY_DRAFTC)
        self.encrypt_func = encrypt_func or encrypt
        self.decrypt_func = decrypt_func or decrypt
        self.root_mirror_func = root_mirror_func or root_mirrors_timeline_id
        self.timeline_content_check_func = timeline_content_check_func or assert_timeline_content_id
        self.layout_check_func = layout_check_func or assert_layout_has_no_duplicate_timeline_ids
        self.project_folder_check_func = project_folder_check_func or assert_all_project_timeline_files_match_folder_ids

    def commit(
        self,
        *,
        draft_dir: Path,
        run_dir: Path,
        real_draft_result: RealDraftIngestResult,
        run_report: RunReport,
        sacrificial_write_override_used: bool = False,
    ) -> WritebackResult:
        run_dir = Path(run_dir)
        draft_dir = Path(draft_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = real_draft_result.metadata or {}
        timeline_id = str(metadata.get("timeline_id") or "")
        timeline_dir = Path(str(metadata.get("timeline_dir") or "")) if metadata.get("timeline_dir") else None
        draft_content_path = Path(str(metadata.get("draft_content_path") or "")) if metadata.get("draft_content_path") else None
        template_path = Path(str(metadata.get("template_path") or "")) if metadata.get("template_path") else None
        base_report = self._base_report(
            draft_dir=draft_dir,
            real_draft_result=real_draft_result,
            writeback_attempted=True,
            sacrificial_write_override_used=sacrificial_write_override_used,
        )
        missing_metadata = [
            name
            for name, value in (
                ("timeline_id", timeline_id),
                ("timeline_dir", timeline_dir),
                ("draft_content_path", draft_content_path),
                ("template_path", template_path),
            )
            if not value
        ]
        if missing_metadata:
            return self._blocked(
                "V21_WRITEBACK_TIMELINE_METADATA_MISSING",
                "real draft ingest metadata does not include active timeline write targets",
                base_report | {"missing_metadata": missing_metadata},
            )
        assert timeline_dir is not None and draft_content_path is not None and template_path is not None
        if not self._is_within(draft_content_path, draft_dir) or not self._is_within(template_path, draft_dir):
            return self._blocked(
                "V21_WRITEBACK_TARGET_OUTSIDE_DRAFT_DIR",
                "writeback target is outside the explicitly provided DraftDir",
                base_report,
            )

        try:
            integrity_report = self._timeline_integrity_checks(
                real_draft_result.draft_data,
                draft_dir=draft_dir,
                run_dir=run_dir,
                timeline_id=timeline_id,
                draft_content_path=draft_content_path,
            )
            modified, mutation_report = self._modified_draft_data(real_draft_result, run_report)
            expected_video_segments = list(mutation_report.pop("_post_write_expected_video_segments", []))
            expected_text_segments = list(mutation_report.pop("_post_write_expected_text_segments", []))
        except WritebackError as exc:
            return self._blocked(exc.code, exc.message, base_report | exc.context)

        plain_modified = run_dir / "draft_content.v21.modified.dec.json"
        encrypted_out = run_dir / "draft_content.v21.modified.enc.json"
        try:
            plain_modified.write_text(json.dumps(modified, ensure_ascii=False, indent=2), "utf-8")
            self.encrypt_func(self.jy_draftc, plain_modified, encrypted_out)
        except Exception as exc:
            return self._blocked(
                "V21_WRITEBACK_ENCRYPT_FAILED",
                "jy-draftc encrypt failed for V21 modified draft_content",
                base_report
                | {
                    "plain_modified_path": str(plain_modified),
                    "encrypted_out_path": str(encrypted_out),
                    "error": str(exc),
                },
            )
        if not encrypted_out.exists():
            return self._blocked(
                "V21_WRITEBACK_ENCRYPT_FAILED",
                "encrypt completed without producing encrypted output",
                base_report | {"plain_modified_path": str(plain_modified), "encrypted_out_path": str(encrypted_out)},
            )

        root_mirror_required = False
        root_mirror_check_failed = False
        root_mirror_error = ""
        try:
            root_mirror_required = bool(self.root_mirror_func(draft_dir, self.jy_draftc, run_dir, timeline_id))
        except Exception as exc:
            root_mirror_check_failed = True
            root_mirror_error = str(exc)
            detection_report = root_mirror_report_from_exception(
                exc,
                draft_dir,
                timeline_id,
            )
            self._write_root_mirror_detection_report(run_dir, detection_report)
            return self._blocked(
                "V21_WRITEBACK_ROOT_MIRROR_DETECTION_FAILED",
                "root mirror requirement could not be determined safely",
                base_report
                | {
                    "root_mirror_required": None,
                    "root_mirror_check_failed": root_mirror_check_failed,
                    "root_mirror_error": root_mirror_error,
                    "root_mirror_synced": False,
                    "root_mirror_targets": [],
                    "root_mirror_detection_report": detection_report,
                },
            )
        targets = [draft_content_path, template_path]
        root_draft_content = draft_dir / "draft_content.json"
        root_template = draft_dir / "template-2.tmp"
        if root_mirror_required:
            targets.extend([root_draft_content, root_template])

        duration_us = max((segment.target_end_us for segment in run_report.final_timeline), default=0)
        pending_target_writes = {str(target): False for target in targets}
        report = base_report | {
            **mutation_report,
            "timeline_integrity_checks": integrity_report,
            "writeback_success": False,
            "commit_performed": False,
            "encrypt_success": True,
            "WRITE_SUCCESS": False,
            "ENCRYPT_SUCCESS": True,
            "plain_modified_path": str(plain_modified),
            "encrypted_out_path": str(encrypted_out),
            "target_writes": pending_target_writes,
            "root_mirror_required": root_mirror_required,
            "root_mirror_written": False,
            "root_mirror_check_failed": root_mirror_check_failed,
            "root_mirror_check_error": root_mirror_error,
            "root_mirror_synced": False,
            "root_mirror_targets": [str(root_draft_content), str(root_template)] if root_mirror_required else [],
            "final_timeline_segment_count": len(run_report.final_timeline),
            "caption_segment_count": len(run_report.material_write_plan.get("segments") or []),
            "caption_material_count": len(run_report.material_write_plan.get("materials") or []),
            "duration_us": duration_us,
            "transactional_writeback_enabled": True,
            "staged_audit_before_target_replace": True,
        }
        staged_audit = self._post_write_actual_draft_audit(
            draft_dir=draft_dir,
            actual_draft_content_path=encrypted_out,
            run_dir=run_dir,
            run_report=run_report,
            expected_video_track_id=str(mutation_report.get("selected_video_track_id") or ""),
            expected_text_track_id=str(mutation_report.get("selected_text_track_id") or ""),
            expected_video_segments=expected_video_segments,
            expected_text_segments=expected_text_segments,
            target_writes=pending_target_writes,
            require_target_writes_committed=False,
        )
        report["staged_post_write_actual_draft_audit"] = staged_audit
        if not staged_audit.get("gate_passed"):
            report["post_write_actual_draft_audit"] = staged_audit
            report.update(_flatten_post_write_actual_draft_audit(staged_audit))
            report.update(
                {
                    "ready_for_user_manual_qc": False,
                    "block_reason": "V21_POST_WRITE_ACTUAL_DRAFT_AUDIT_FAILED",
                    "target_writes": pending_target_writes,
                    "rollback_performed": False,
                    "rollback_required": False,
                }
            )
            return WritebackResult(
                success=False,
                blockers=[
                    Blocker(
                        code="V21_POST_WRITE_ACTUAL_DRAFT_AUDIT_FAILED",
                        message="staged V21 draft failed actual audit before replacing target files",
                        layer="writeback",
                        context={"post_write_actual_draft_audit": staged_audit},
                    )
                ],
                report=report,
            )

        try:
            snapshots = self._snapshot_targets(targets, run_dir)
        except Exception as exc:
            report.update(
                {
                    "target_writes": {str(target): False for target in targets},
                    "rollback_required": False,
                    "rollback_performed": False,
                    "target_snapshot_failed": True,
                    "error": str(exc),
                }
            )
            return self._blocked(
                "V21_WRITEBACK_TARGET_WRITE_FAILED",
                "failed to snapshot target files before transactional V21 writeback",
                report,
            )
        target_write_attempts: dict[str, bool] = {}
        rollback_report: dict[str, Any] = {"rollback_performed": False, "rollback_success": None, "rollback_errors": []}
        try:
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(encrypted_out, target)
                target_write_attempts[str(target)] = target.exists() and target.stat().st_size == encrypted_out.stat().st_size
        except Exception as exc:
            rollback_report = self._restore_target_snapshots(snapshots)
            report.update(
                {
                    "target_write_attempt_results": target_write_attempts,
                    "target_writes": {str(target): False for target in targets},
                    "rollback_required": True,
                    **rollback_report,
                    "error": str(exc),
                }
            )
            return self._blocked(
                "V21_WRITEBACK_TARGET_WRITE_FAILED",
                "failed to write encrypted V21 draft_content to target timeline files",
                report,
            )
        if not all(target_write_attempts.values()):
            rollback_report = self._restore_target_snapshots(snapshots)
            report.update(
                {
                    "target_write_attempt_results": target_write_attempts,
                    "target_writes": {str(target): False for target in targets},
                    "rollback_required": True,
                    **rollback_report,
                }
            )
            return self._blocked(
                "V21_WRITEBACK_TARGET_WRITE_FAILED",
                "one or more target timeline files were not written",
                report,
            )

        audit = self._post_write_actual_draft_audit(
            draft_dir=draft_dir,
            actual_draft_content_path=draft_content_path,
            run_dir=run_dir,
            run_report=run_report,
            expected_video_track_id=str(mutation_report.get("selected_video_track_id") or ""),
            expected_text_track_id=str(mutation_report.get("selected_text_track_id") or ""),
            expected_video_segments=expected_video_segments,
            expected_text_segments=expected_text_segments,
            target_writes=target_write_attempts,
            require_target_writes_committed=True,
        )
        report["post_write_actual_draft_audit"] = audit
        report.update(_flatten_post_write_actual_draft_audit(audit))
        if not audit.get("gate_passed"):
            rollback_report = self._restore_target_snapshots(snapshots)
            report.update(
                {
                    "writeback_success": False,
                    "commit_performed": False,
                    "WRITE_SUCCESS": False,
                    "ready_for_user_manual_qc": False,
                    "block_reason": "V21_POST_WRITE_ACTUAL_DRAFT_AUDIT_FAILED",
                    "target_write_attempt_results": target_write_attempts,
                    "target_writes": {str(target): False for target in targets},
                    "rollback_required": True,
                    **rollback_report,
                }
            )
            return WritebackResult(
                success=False,
                blockers=[
                    Blocker(
                        code="V21_POST_WRITE_ACTUAL_DRAFT_AUDIT_FAILED",
                        message="post-write actual draft audit failed or did not execute",
                        layer="writeback",
                        context={"post_write_actual_draft_audit": audit},
                    )
                ],
                report=report,
            )
        report.update(
            {
                "writeback_success": True,
                "commit_performed": True,
                "WRITE_SUCCESS": True,
                "target_writes": target_write_attempts,
                "root_mirror_written": bool(root_mirror_required and target_write_attempts.get(str(root_draft_content)) and target_write_attempts.get(str(root_template))),
                "root_mirror_synced": bool(root_mirror_required and target_write_attempts.get(str(root_draft_content)) and target_write_attempts.get(str(root_template))),
                "rollback_required": False,
                **rollback_report,
            }
        )
        report["ready_for_user_manual_qc"] = True
        return WritebackResult(success=True, report=report)

    def _modified_draft_data(self, real_draft_result: RealDraftIngestResult, run_report: RunReport) -> tuple[dict[str, Any], dict[str, Any]]:
        data = deepcopy(real_draft_result.draft_data)
        materials = data.setdefault("materials", {})
        if not isinstance(materials, dict):
            raise WritebackError("V21_WRITEBACK_SCHEMA_UNSUPPORTED", "draft materials root is not an object")
        text_materials = [deepcopy(row) for row in run_report.material_write_plan.get("materials") or []]
        text_segments = [deepcopy(row) for row in run_report.material_write_plan.get("segments") or []]
        if not text_materials or not text_segments:
            raise WritebackError("V21_WRITEBACK_MATERIAL_PLAN_EMPTY", "material_write_plan has no caption materials or segments")
        source_resolution = self._source_resolution_from_run_report(run_report)
        gapless_projection_plan = self._gapless_caption_video_projection_plan(run_report)
        self._apply_gapless_caption_ranges(
            text_segments,
            run_report.captions,
            gapless_projection_plan["caption_target_ranges"],
        )

        template_material_ids = self._template_candidate_material_ids(run_report)
        subtitle_track_ids = self._subtitle_bound_track_ids(real_draft_result.text_segments, template_material_ids)
        text_track_id = self._select_subtitle_track_id(data, real_draft_result.text_segments, template_material_ids)
        text_track = self._track_by_id(data, text_track_id)
        if text_track is None:
            raise WritebackError(
                "V21_WRITEBACK_SUBTITLE_TRACK_NOT_FOUND",
                "could not locate selected subtitle text track in draft",
                {"selected_text_track_id": text_track_id},
            )
        text_material_by_id = {
            str(row.get("id") or ""): row
            for row in materials.get("texts") or []
            if isinstance(row, dict) and str(row.get("id") or "")
        }
        visible_text_rows = self._visible_text_rows(data)
        classified_existing_text_rows = self._classified_actual_text_rows(
            visible_text_rows,
            expected_segment_ids=set(),
            expected_material_ids=set(),
            template_material_ids=template_material_ids,
        )
        old_subtitle_segments_by_track: dict[str, list[dict[str, Any]]] = {}
        for row in classified_existing_text_rows:
            if row["classification"] == "confirmed_non_subtitle":
                continue
            track_id = str(row.get("track_id") or "")
            if track_id:
                old_subtitle_segments_by_track.setdefault(track_id, []).append(row["segment"])
        confirmed_non_subtitle_segments = [
            row["segment"]
            for row in classified_existing_text_rows
            if row["classification"] == "confirmed_non_subtitle"
        ]
        old_subtitle_segments = [row for rows in old_subtitle_segments_by_track.values() for row in rows]
        old_text_segment_ids = {str(row.get("id") or "") for row in old_subtitle_segments if row.get("id")}
        old_text_material_ids = {
            str(row.get("material_id") or row.get("materialId") or "")
            for row in old_subtitle_segments
            if str(row.get("material_id") or row.get("materialId") or "")
        }
        if old_subtitle_segments and not old_text_segment_ids:
            raise WritebackError("V21_WRITEBACK_OLD_SUBTITLE_SEGMENTS_UNBOUND", "old subtitle text segments cannot be safely identified")

        for subtitle_track_id, old_rows in old_subtitle_segments_by_track.items():
            track = self._track_by_id(data, subtitle_track_id)
            if track is None:
                raise WritebackError(
                    "V21_WRITEBACK_SUBTITLE_TRACK_NOT_FOUND",
                    "subtitle-bound track could not be located in draft",
                    {"selected_text_track_id": subtitle_track_id},
                )
            remove_ids = {
                str(row.get("id") or "")
                for row in old_rows
                if str(row.get("id") or "")
            }
            track["segments"] = [
                segment
                for segment in track.get("segments") or []
                if str(segment.get("id") or "") not in remove_ids
            ]
        for segment in text_segments:
            segment.setdefault("track_id", text_track_id)
        preserved_selected_segments = [segment for segment in text_track.get("segments") or [] if isinstance(segment, dict)]
        text_track["segments"] = preserved_selected_segments + deepcopy(text_segments)
        preserved_text_material_ids = {
            str(row["segment"].get("material_id") or row["segment"].get("materialId") or "")
            for row in classified_existing_text_rows
            if row["classification"] == "confirmed_non_subtitle"
            and str(row["segment"].get("material_id") or row["segment"].get("materialId") or "")
        }
        removable_old_text_material_ids = old_text_material_ids - preserved_text_material_ids
        materials["texts"] = [row for row in materials.get("texts") or [] if str(row.get("id") or "") not in removable_old_text_material_ids]
        materials["texts"].extend(deepcopy(text_materials))

        video_track_id = self._select_video_track_id_from_templates(source_resolution.resolved_templates, run_report)
        video_track = self._track_by_id(data, video_track_id)
        if video_track is None or not self._is_video_track(video_track):
            raise WritebackError(
                "V21_WRITEBACK_MAIN_VIDEO_TRACK_NOT_FOUND",
                "could not locate selected source video track in draft",
                {"selected_video_track_id": video_track_id},
            )
        preflight_report = self._preflight_tracks(
            data,
            run_report,
            source_resolution.resolved_templates,
            selected_video_track_id=video_track_id,
        )
        if not preflight_report["video_preflight"]["main_video_speed_safe"]:
            raise WritebackError(
                "V21_WRITEBACK_UNSUPPORTED_SPEED_MAPPING",
                "main video source/material time mapping is not safe for V21 writeback",
                preflight_report,
            )

        video_segments: list[dict[str, Any]] = []
        video_projection_units = list(gapless_projection_plan["video_units"])
        for index, final_segment in enumerate(video_projection_units, start=1):
            template = source_resolution.templates_by_final_segment_id.get(final_segment.segment_id)
            if template is None:
                raise WritebackError(
                    "V21_WRITEBACK_SOURCE_SEGMENT_TEMPLATE_MISSING",
                    "could not locate source segment template for final timeline segment",
                    {"segment_id": final_segment.segment_id, "source_segment_id": final_segment.source_segment_id},
                )
            video_segments.append(self._video_segment_from_template(template, final_segment, index, data))
        video_track["segments"] = video_segments
        effective_speed_report = self._effective_speed_report(video_segments)

        duration_us = max(
            (
                self._timerange_start(segment.get("target_timerange")) + self._timerange_duration(segment.get("target_timerange"))
                for segment in video_segments
            ),
            default=0,
        )
        for key in ("duration", "duration_us"):
            if key in data:
                data[key] = duration_us
        if "duration" not in data and "duration_us" not in data:
            data["duration"] = duration_us
        rough_cut_quality = self._writeback_rough_cut_quality(
            data,
            run_report,
            selected_text_track_id=text_track_id,
            subtitle_track_ids=subtitle_track_ids | set(old_subtitle_segments_by_track),
            old_subtitle_material_ids=removable_old_text_material_ids,
            preserved_non_subtitle_text_segment_count=len(confirmed_non_subtitle_segments),
        )
        self._assert_writeback_rough_cut_quality(rough_cut_quality)
        return data, {
            "selected_text_track_id": text_track_id,
            "selected_video_track_id": video_track_id,
            "subtitle_bound_track_ids": sorted(subtitle_track_ids | set(old_subtitle_segments_by_track)),
            "old_subtitle_segment_count": len(old_subtitle_segments),
            "old_subtitle_material_count": len(removable_old_text_material_ids),
            "new_caption_segment_count": len(text_segments),
            "new_caption_material_count": len(text_materials),
            "gapless_video_write_plan_enabled": True,
            "gapless_video_row_count": len(video_segments),
            "gapless_caption_repacked_count": int(gapless_projection_plan.get("caption_repacked_count") or 0),
            "gapless_final_video_end_us": duration_us,
            "non_subtitle_text_tracks_preserved": True,
            "non_subtitle_text_segments_preserved": True,
            "non_subtitle_text_materials_preserved": True,
            "preserved_non_subtitle_text_segment_count": len(confirmed_non_subtitle_segments),
            "canonical_caption_segment_count": rough_cut_quality["canonical_caption_segment_count"],
            "visible_caption_track_count": rough_cut_quality["visible_caption_track_count"],
            "old_subtitle_residue_track_count": rough_cut_quality["old_subtitle_residue_track_count"],
            "overlapping_caption_segments_count": rough_cut_quality["overlapping_caption_segments_count"],
            "effective_speed_min": effective_speed_report["effective_speed_min"],
            "effective_speed_max": effective_speed_report["effective_speed_max"],
            "effective_speed_drift_count": effective_speed_report["effective_speed_drift_count"],
            "safe_handle_policy_enabled": effective_speed_report["safe_handle_policy_enabled"],
            "lead_handle_requested_count": effective_speed_report["lead_handle_requested_count"],
            "tail_handle_requested_count": effective_speed_report["tail_handle_requested_count"],
            "lead_handle_applied_count": effective_speed_report["lead_handle_applied_count"],
            "tail_handle_applied_count": effective_speed_report["tail_handle_applied_count"],
            "segments_with_no_lead_handle": effective_speed_report["segments_with_no_lead_handle"],
            "segments_with_no_tail_handle": effective_speed_report["segments_with_no_tail_handle"],
            "handle_blocked_count": effective_speed_report["handle_blocked_count"],
            "handle_blocked_reasons": effective_speed_report["handle_blocked_reasons"],
            "source_mapping_mode": "dynamic_source_binding",
            **source_resolution.report,
            "rough_cut_quality": rough_cut_quality,
            **preflight_report,
            "speed_safe": bool((preflight_report.get("video_preflight") or {}).get("speed_safe")),
            "detected_speeds": list((preflight_report.get("video_preflight") or {}).get("speeds") or []),
            "effect_policy_safe": bool((preflight_report.get("filter_preflight") or {}).get("effect_policy_safe", True)),
            "embedded_effects_preserved": bool((preflight_report.get("filter_preflight") or {}).get("segment_embedded_effects_preserved")),
            "_post_write_expected_video_segments": deepcopy(video_segments),
            "_post_write_expected_text_segments": deepcopy(text_segments),
        }

    def _load_actual_draft_content(self, actual_draft_content_path: Path, run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        report: dict[str, Any] = {
            "actual_draft_loaded": False,
            "actual_draft_source": "",
            "actual_draft_decrypted_path": "",
            "actual_draft_load_error": "",
        }
        if not actual_draft_content_path.exists():
            report["actual_draft_load_error"] = "actual draft content path does not exist"
            return None, report
        try:
            data = json.loads(actual_draft_content_path.read_text("utf-8"))
            if isinstance(data, dict):
                report.update({"actual_draft_loaded": True, "actual_draft_source": "direct_json"})
                return data, report
        except Exception as direct_exc:
            report["actual_draft_load_error"] = str(direct_exc)

        decrypted_path = run_dir / "post_write_actual_draft_audit.dec.json"
        try:
            self.decrypt_func(self.jy_draftc, actual_draft_content_path, decrypted_path)
            data = json.loads(decrypted_path.read_text("utf-8"))
            if isinstance(data, dict):
                report.update(
                    {
                        "actual_draft_loaded": True,
                        "actual_draft_source": "decrypted_json",
                        "actual_draft_decrypted_path": str(decrypted_path),
                        "actual_draft_load_error": "",
                    }
                )
                return data, report
            report["actual_draft_load_error"] = "decrypted draft content is not a JSON object"
        except Exception as decrypt_exc:
            previous_error = str(report.get("actual_draft_load_error") or "")
            report["actual_draft_decrypted_path"] = str(decrypted_path)
            report["actual_draft_load_error"] = f"direct_json_failed={previous_error}; decrypt_failed={decrypt_exc}"
        return None, report

    def _video_segments_match_plan(self, actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
        return postwrite_audit_helpers._video_segments_match_plan(
            actual,
            expected,
            segment_timerange_signature=self._segment_timerange_signature,
        )

    def _expected_caption_segments_present(self, actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
        return postwrite_audit_helpers._expected_caption_segments_present(
            actual,
            expected,
            segment_timerange_signature=self._segment_timerange_signature,
        )

    def _visible_text_rows(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        return postwrite_audit_helpers._visible_text_rows(
            data,
            is_text_track=self._is_text_track,
            text_segment_text=self._text_segment_text,
            timerange_start=self._timerange_start,
            timerange_duration=self._timerange_duration,
        )

    def _is_visible_timeline_row(self, row: dict[str, Any]) -> bool:
        return postwrite_audit_helpers._is_visible_timeline_row(row)

    def _text_segment_text(self, segment: dict[str, Any], material: dict[str, Any]) -> str:
        return postwrite_audit_helpers._text_segment_text(
            segment,
            material,
            text_material_text=self._text_material_text,
        )

    def _classified_actual_text_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        expected_segment_ids: set[str],
        expected_material_ids: set[str],
        template_material_ids: set[str],
    ) -> list[dict[str, Any]]:
        return postwrite_audit_helpers._classified_actual_text_rows(
            rows,
            expected_segment_ids=expected_segment_ids,
            expected_material_ids=expected_material_ids,
            template_material_ids=template_material_ids,
            is_confirmed_non_subtitle_text=self._is_confirmed_non_subtitle_text,
        )

    def _actual_text_residue_report(
        self,
        *,
        visible_text_rows: list[dict[str, Any]],
        actual_video_segments: list[dict[str, Any]],
        expected_text_segments: list[dict[str, Any]],
        run_report: RunReport,
    ) -> dict[str, Any]:
        return postwrite_audit_helpers._actual_text_residue_report(
            visible_text_rows=visible_text_rows,
            actual_video_segments=actual_video_segments,
            expected_text_segments=expected_text_segments,
            run_report=run_report,
            classified_actual_text_rows=self._classified_actual_text_rows,
            template_candidate_material_ids=self._template_candidate_material_ids,
            expected_caption_segments_present=self._expected_caption_segments_present,
            has_containing_video_segment=self._has_containing_video_segment,
            timerange_start=self._timerange_start,
            timerange_duration=self._timerange_duration,
        )
    def _has_containing_video_segment(self, text_row: dict[str, Any], video_segments: list[dict[str, Any]]) -> bool:
        start = int(text_row.get("target_start_us") or 0)
        end = int(text_row.get("target_end_us") or 0)
        for segment in video_segments:
            target_start = self._timerange_start(segment.get("target_timerange"))
            target_end = target_start + self._timerange_duration(segment.get("target_timerange"))
            if target_start <= start and end <= target_end:
                return True
        caption_id = str(text_row.get("caption_id") or "")
        if caption_id:
            split_rows = [
                segment
                for segment in video_segments
                if isinstance(segment.get("_v21_audio_coverage"), dict)
                and str((segment.get("_v21_audio_coverage") or {}).get("caption_id") or "") == caption_id
            ]
            if split_rows:
                split_start = min(self._timerange_start(segment.get("target_timerange")) for segment in split_rows)
                split_end = max(
                    self._timerange_start(segment.get("target_timerange")) + self._timerange_duration(segment.get("target_timerange"))
                    for segment in split_rows
                )
                if split_start <= start and end <= split_end:
                    return True
        if self._video_rows_covering_target_range(video_segments, start, end):
            return True
        return False

    def _select_subtitle_track_id(self, data: dict[str, Any], text_segments: list[dict[str, Any]], template_material_ids: set[str]) -> str:
        return track_selector._select_subtitle_track_id(
            data,
            text_segments,
            template_material_ids,
            error_cls=WritebackError,
        )

    def _template_candidate_material_ids(self, run_report: RunReport) -> set[str]:
        return track_selector._template_candidate_material_ids(run_report)

    def _subtitle_bound_track_ids(self, text_segments: list[dict[str, Any]], template_material_ids: set[str]) -> set[str]:
        return track_selector._subtitle_bound_track_ids(
            text_segments,
            template_material_ids,
            error_cls=WritebackError,
        )

    def _classified_text_segments_by_track(
        self,
        text_segments: list[dict[str, Any]],
        subtitle_track_ids: set[str],
        template_material_ids: set[str],
        text_material_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        return track_selector._classified_text_segments_by_track(
            text_segments,
            subtitle_track_ids,
            template_material_ids,
            text_material_by_id,
            error_cls=WritebackError,
        )

    def _classify_text_segment(
        self,
        segment: dict[str, Any],
        template_material_ids: set[str],
        text_material_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return track_selector._classify_text_segment(segment, template_material_ids, text_material_by_id)

    def _is_confirmed_non_subtitle_text(self, segment: dict[str, Any], material: dict[str, Any]) -> bool:
        return track_selector._is_confirmed_non_subtitle_text(segment, material)

    def _metadata_token_matches(self, text: str, token: str) -> bool:
        return track_selector._metadata_token_matches(text, token)

    def _text_material_text(self, material: dict[str, Any]) -> str:
        return track_selector._text_material_text(material)

    def _old_subtitle_segments_by_track(
        self,
        text_segments: list[dict[str, Any]],
        subtitle_track_ids: set[str],
        template_material_ids: set[str],
    ) -> dict[str, list[dict[str, Any]]]:
        return track_selector._old_subtitle_segments_by_track(
            text_segments,
            subtitle_track_ids,
            template_material_ids,
            error_cls=WritebackError,
        )

    def _select_video_track_id_from_templates(self, used_templates: list[dict[str, Any]], run_report: RunReport) -> str:
        return track_selector._select_video_track_id_from_templates(
            used_templates,
            run_report,
            error_cls=WritebackError,
        )

    def _track_by_id(self, data: dict[str, Any], track_id: str) -> dict[str, Any] | None:
        return track_selector._track_by_id(data, track_id)

    def _is_text_track(self, track: dict[str, Any]) -> bool:
        return track_selector._is_text_track(track)

    def _is_video_track(self, track: dict[str, Any]) -> bool:
        return track_selector._is_video_track(track)
    def _timeline_integrity_checks(
        self,
        data: dict[str, Any],
        *,
        draft_dir: Path,
        run_dir: Path,
        timeline_id: str,
        draft_content_path: Path,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "timeline_content_id_ok": False,
            "layout_duplicate_ids_ok": False,
            "project_timeline_folder_ids_ok": False,
        }
        try:
            self.timeline_content_check_func(data, timeline_id, draft_content_path)
            report["timeline_content_id_ok"] = True
        except Exception as exc:
            report["timeline_content_id_error"] = str(exc)
            raise WritebackError("V21_TIMELINE_CONTENT_ID_MISMATCH", "draft_content timeline id does not match active timeline folder", {"timeline_integrity_checks": report}) from exc
        try:
            self.layout_check_func(draft_dir)
            report["layout_duplicate_ids_ok"] = True
        except Exception as exc:
            report["layout_duplicate_ids_error"] = str(exc)
            raise WritebackError("V21_TIMELINE_LAYOUT_DUPLICATE_IDS", "timeline layout contains duplicate timeline ids", {"timeline_integrity_checks": report}) from exc
        try:
            self.project_folder_check_func(draft_dir, self.jy_draftc, run_dir)
            report["project_timeline_folder_ids_ok"] = True
        except Exception as exc:
            report["project_timeline_folder_ids_error"] = str(exc)
            raise WritebackError("V21_PROJECT_TIMELINE_FOLDER_ID_MISMATCH", "project timeline files do not match folder ids", {"timeline_integrity_checks": report}) from exc
        return report

    def _preflight_tracks(
        self,
        data: dict[str, Any],
        run_report: RunReport,
        used_templates: list[dict[str, Any]],
        *,
        selected_video_track_id: str,
    ) -> dict[str, Any]:
        speed_reports = []
        resolver = SpeedResolver(data)
        try:
            for template in used_templates:
                speed_reports.append(resolver.resolve(template).report)
        except SpeedResolutionError as exc:
            raise WritebackError(exc.code, exc.message, {"video_preflight": {"main_video_speed_safe": False, **exc.context}}) from exc
        speed_safe = all(bool(row.get("speed_safe")) for row in speed_reports)
        audio_tracks = [track for track in data.get("tracks") or [] if self._track_type_contains(track, "audio")]
        effect_result = EffectTrackPolicy().inspect(
            data,
            final_duration_us=max((segment.target_end_us for segment in run_report.final_timeline), default=0),
        )
        if not effect_result.safe:
            raise WritebackError(effect_result.blocker_code, effect_result.blocker_message, {"filter_preflight": effect_result.report})
        return {
            "video_preflight": {
                "selected_video_track_id": selected_video_track_id,
                "main_video_speed_safe": speed_safe,
                "speed_safe": speed_safe,
                "speed_reports": speed_reports,
                "source_segment_count": len(used_templates),
                "speeds": [row.get("detected_speed") for row in speed_reports],
            },
            "audio_preflight": {
                "independent_audio_track_detected": bool(audio_tracks),
                "has_complex_audio": bool(audio_tracks),
                "audio_track_count": len(audio_tracks),
            },
            "filter_preflight": effect_result.report,
        }

    def _track_type_contains(self, track: dict[str, Any], token: str) -> bool:
        return track_selector._track_type_contains(track, token)
    def _blocked(self, code: str, message: str, report: dict[str, Any]) -> WritebackResult:
        return WritebackResult(
            success=False,
            blockers=[Blocker(code=code, message=message, layer="writeback", context=report)],
            report=report | {"writeback_success": False, "block_reason": code},
        )

    def _write_root_mirror_detection_report(self, run_dir: Path, report: dict[str, Any]) -> None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "root_mirror_detection_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                "utf-8",
            )
        except OSError as exc:
            report["report_write_error"] = str(exc)

    def _base_report(
        self,
        *,
        draft_dir: Path | None,
        real_draft_result: RealDraftIngestResult,
        writeback_attempted: bool,
        sacrificial_write_override_used: bool,
    ) -> dict[str, Any]:
        return build_base_report(
            draft_dir=draft_dir,
            real_draft_result=real_draft_result,
            writeback_attempted=writeback_attempted,
            sacrificial_write_override_used=sacrificial_write_override_used,
        )
    def _source_resolution_from_run_report(self, run_report: RunReport) -> SourceSegmentTemplateResolution:
        resolved_template_map = dict(run_report.resolved_template_map or {})
        if len(resolved_template_map) != len(run_report.final_timeline):
            raise WritebackError(
                "V21_DYNAMIC_BINDING_REQUIRED",
                "RealDraftWriteback requires a complete current-draft resolved_template_map",
                {
                    "resolved_template_count": len(resolved_template_map),
                    "final_timeline_segment_count": len(run_report.final_timeline),
                },
            )
        templates_by_final_id: dict[str, dict[str, Any]] = {}
        resolved_templates: list[dict[str, Any]] = []
        for segment in run_report.final_timeline:
            binding = resolved_template_map.get(segment.segment_id) or {}
            template = binding.get("current_video_segment_template")
            if not isinstance(template, dict):
                raise WritebackError(
                    "V21_DYNAMIC_BINDING_REQUIRED",
                    "resolved_template_map entry is missing current_video_segment_template",
                    {"segment_id": segment.segment_id},
                )
            resolved_template = dict(template)
            material_template = binding.get("current_material_template")
            if isinstance(material_template, dict):
                resolved_template["_resolved_current_material_template"] = dict(material_template)
            templates_by_final_id[segment.segment_id] = resolved_template
            resolved_templates.append(resolved_template)
        excluded_report_keys = {
            "writeback_attempted",
            "writeback_success",
            "commit_performed",
            "encrypt_success",
            "WRITE_SUCCESS",
            "ENCRYPT_SUCCESS",
            "draft_dir",
            "timeline_id",
            "timeline_dir",
            "draft_content_path",
            "template_path",
            "target_writes",
            "selected_text_track_id",
            "selected_video_track_id",
            "sacrificial_write_override_used",
            "postwrite_decrypt_skipped_for_sacrificial_draft",
            "borrowed_v20_low_level_io_reference",
        }
        report = {
            key: value
            for key, value in dict(run_report.source_binding_report or {}).items()
            if key not in excluded_report_keys
        }
        report["resolved_template_map"] = resolved_template_map
        return SourceSegmentTemplateResolution(
            success=True,
            templates_by_final_segment_id=templates_by_final_id,
            resolved_templates=resolved_templates,
            blockers=[],
            report=report,
        )

    def _writeback_rough_cut_quality(
        self,
        data: dict[str, Any],
        run_report: RunReport,
        *,
        selected_text_track_id: str,
        subtitle_track_ids: set[str],
        old_subtitle_material_ids: set[str],
        preserved_non_subtitle_text_segment_count: int,
    ) -> dict[str, Any]:
        return build_writeback_rough_cut_quality(
            data,
            run_report,
            selected_text_track_id=selected_text_track_id,
            subtitle_track_ids=subtitle_track_ids,
            old_subtitle_material_ids=old_subtitle_material_ids,
            preserved_non_subtitle_text_segment_count=preserved_non_subtitle_text_segment_count,
            visible_text_rows=self._visible_text_rows,
            classified_actual_text_rows=self._classified_actual_text_rows,
            template_candidate_material_ids=self._template_candidate_material_ids,
            is_text_track=self._is_text_track,
            overlap_count=self._overlap_count,
        )
    def _assert_writeback_rough_cut_quality(self, metrics: dict[str, Any]) -> None:
        assert_writeback_rough_cut_quality_report(metrics, error_cls=WritebackError)
    def _overlap_count(self, segments: list[dict[str, Any]]) -> int:
        return overlap_count_report(
            segments,
            timerange_start=self._timerange_start,
            timerange_duration=self._timerange_duration,
        )
    def _is_within(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False


class WritebackError(RuntimeError):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

def _bind_real_draft_writeback_helpers() -> None:
    dependencies = globals()
    video_projection_helpers.configure_writeback_dependencies(dependencies)
    actual_timeline_audit_helpers.configure_writeback_dependencies(dependencies)
    snapshot_guard_helpers.configure_writeback_dependencies(dependencies)
    writeback_time_utils_helpers.configure_writeback_dependencies(dependencies)
    RealDraftWriteback._gapless_caption_video_projection_plan = video_projection_helpers._gapless_caption_video_projection_plan  # type: ignore[method-assign]
    RealDraftWriteback._video_projection_units_from_caption_spans = video_projection_helpers._video_projection_units_from_caption_spans  # type: ignore[method-assign]
    RealDraftWriteback._apply_gapless_caption_ranges = video_projection_helpers._apply_gapless_caption_ranges  # type: ignore[method-assign]
    RealDraftWriteback._caption_final_segment_ids = video_projection_helpers._caption_final_segment_ids  # type: ignore[method-assign]
    RealDraftWriteback._final_segment_video_projection_groups = video_projection_helpers._final_segment_video_projection_groups  # type: ignore[method-assign]
    RealDraftWriteback._gapless_projection_group_handles = video_projection_helpers._gapless_projection_group_handles  # type: ignore[method-assign]
    RealDraftWriteback._caption_ids_for_word_group = video_projection_helpers._caption_ids_for_word_group  # type: ignore[method-assign]
    RealDraftWriteback._apply_caption_ranges_for_projected_segment = video_projection_helpers._apply_caption_ranges_for_projected_segment  # type: ignore[method-assign]
    RealDraftWriteback._project_caption_word_span_to_group = video_projection_helpers._project_caption_word_span_to_group  # type: ignore[method-assign]
    RealDraftWriteback._merge_caption_target_range = video_projection_helpers._merge_caption_target_range  # type: ignore[method-assign]
    RealDraftWriteback._caption_spoken_source_span = video_projection_helpers._caption_spoken_source_span  # type: ignore[method-assign]
    RealDraftWriteback._caption_video_source_groups = video_projection_helpers._caption_video_source_groups  # type: ignore[method-assign]
    RealDraftWriteback._video_segment_from_template = video_projection_helpers._video_segment_from_template  # type: ignore[method-assign]
    RealDraftWriteback._effective_speed_report = video_projection_helpers._effective_speed_report  # type: ignore[method-assign]
    RealDraftWriteback._effective_speed_from_row = video_projection_helpers._effective_speed_from_row  # type: ignore[method-assign]
    RealDraftWriteback._effective_speed_supported_or_baseline = video_projection_helpers._effective_speed_supported_or_baseline  # type: ignore[method-assign]
    RealDraftWriteback._post_write_actual_draft_audit = actual_timeline_audit_helpers._post_write_actual_draft_audit  # type: ignore[method-assign]
    RealDraftWriteback._actual_final_timeline_from_video_rows = actual_timeline_audit_helpers._actual_final_timeline_from_video_rows  # type: ignore[method-assign]
    RealDraftWriteback._actual_timeline_with_caption_split_containers = actual_timeline_audit_helpers._actual_timeline_with_caption_split_containers  # type: ignore[method-assign]
    RealDraftWriteback._actual_caption_units_from_text_rows = actual_timeline_audit_helpers._actual_caption_units_from_text_rows  # type: ignore[method-assign]
    RealDraftWriteback._containing_actual_timeline_segment = actual_timeline_audit_helpers._containing_actual_timeline_segment  # type: ignore[method-assign]
    RealDraftWriteback._timeline_segments_covering_target_range = actual_timeline_audit_helpers._timeline_segments_covering_target_range  # type: ignore[method-assign]
    RealDraftWriteback._synthetic_caption_container_from_segments = actual_timeline_audit_helpers._synthetic_caption_container_from_segments  # type: ignore[method-assign]
    RealDraftWriteback._actual_audio_coverage_report = actual_timeline_audit_helpers._actual_audio_coverage_report  # type: ignore[method-assign]
    RealDraftWriteback._jianying_canonical_timeline_sync_report = actual_timeline_audit_helpers._jianying_canonical_timeline_sync_report  # type: ignore[method-assign]
    RealDraftWriteback._video_rows_by_caption_id = actual_timeline_audit_helpers._video_rows_by_caption_id  # type: ignore[method-assign]
    RealDraftWriteback._video_rows_by_timeline_segment_id = actual_timeline_audit_helpers._video_rows_by_timeline_segment_id  # type: ignore[method-assign]
    RealDraftWriteback._video_rows_covering_target_range = actual_timeline_audit_helpers._video_rows_covering_target_range  # type: ignore[method-assign]
    RealDraftWriteback._video_rows_overlapping_target_range = actual_timeline_audit_helpers._video_rows_overlapping_target_range  # type: ignore[method-assign]
    RealDraftWriteback._caption_split_gap_row = actual_timeline_audit_helpers._caption_split_gap_row  # type: ignore[method-assign]
    RealDraftWriteback._actual_source_interval_rows = actual_timeline_audit_helpers._actual_source_interval_rows  # type: ignore[method-assign]
    RealDraftWriteback._actual_source_interval_for_video_row = actual_timeline_audit_helpers._actual_source_interval_for_video_row  # type: ignore[method-assign]
    RealDraftWriteback._ranges_overlap = actual_timeline_audit_helpers._ranges_overlap  # type: ignore[method-assign]
    RealDraftWriteback._segment_timerange_signature = actual_timeline_audit_helpers._segment_timerange_signature  # type: ignore[method-assign]
    RealDraftWriteback._plan_quality_audit = actual_timeline_audit_helpers._plan_quality_audit  # type: ignore[method-assign]
    RealDraftWriteback._snapshot_targets = snapshot_guard_helpers._snapshot_targets  # type: ignore[method-assign]
    RealDraftWriteback._restore_target_snapshots = snapshot_guard_helpers._restore_target_snapshots  # type: ignore[method-assign]
    RealDraftWriteback._segment_speed = writeback_time_utils_helpers._segment_speed  # type: ignore[method-assign]
    RealDraftWriteback._timerange_start = writeback_time_utils_helpers._timerange_start  # type: ignore[method-assign]
    RealDraftWriteback._timerange_duration = writeback_time_utils_helpers._timerange_duration  # type: ignore[method-assign]
    RealDraftWriteback._display_to_material_delta = writeback_time_utils_helpers._display_to_material_delta  # type: ignore[method-assign]
    RealDraftWriteback._source_timeline_to_material_time = writeback_time_utils_helpers._source_timeline_to_material_time  # type: ignore[method-assign]


_bind_real_draft_writeback_helpers()
