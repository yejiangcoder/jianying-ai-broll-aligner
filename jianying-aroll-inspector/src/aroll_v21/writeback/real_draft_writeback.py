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
from aroll_v21.writeback.effect_policy import EffectTrackPolicy
from aroll_v21.writeback.source_segment_template_resolver import (
    SOURCE_TEMPLATE_REPORT_DEFAULTS,
    SourceSegmentTemplateResolution,
)
from aroll_v21.writeback.speed_resolver import SpeedResolutionError, SpeedResolver
from aroll_v21.writeback.video_write_plan_projector import (
    project_video_segment_from_template,
    safe_handle_report_from_projected_segments,
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


def _finalize_post_write_actual_draft_audit(audit: dict[str, Any]) -> dict[str, Any]:
    failures = list(audit.get("failure_reasons") or [])
    passed = bool(audit.get("executed")) and not failures
    audit["gate_passed"] = passed
    blocker_codes = list(audit.get("blocker_codes") or [])
    if not passed:
        blocker_codes.append("V21_POST_WRITE_ACTUAL_DRAFT_AUDIT_FAILED")
    audit["blocker_codes"] = [] if passed else _unique_strings(blocker_codes)
    audit["post_write_actual_draft_audit_executed"] = bool(audit.get("executed"))
    audit["post_write_actual_draft_audit_gate_passed"] = passed
    audit["post_write_actual_draft_audit_blocker_codes"] = list(audit.get("blocker_codes") or [])
    return audit


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


def _flatten_post_write_actual_draft_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "post_write_actual_draft_audit_required_on_commit": True,
        "post_write_actual_draft_audit_executed": bool(audit.get("executed")),
        "post_write_actual_draft_audit_gate_passed": bool(audit.get("gate_passed")),
        "post_write_actual_draft_audit_blocker_codes": list(audit.get("blocker_codes") or []),
        "post_write_actual_draft_audit_failure_reasons": list(audit.get("failure_reasons") or []),
        "post_write_actual_draft_loaded": bool(audit.get("actual_draft_loaded")),
        "post_write_actual_draft_source": str(audit.get("actual_draft_source") or ""),
        "post_write_actual_video_rows_match_plan": bool(audit.get("actual_video_rows_match_plan")),
        "post_write_actual_caption_rows_match_plan": bool(audit.get("actual_caption_rows_match_plan")),
        "post_write_expected_caption_rows_present": bool(audit.get("expected_caption_rows_present")),
        "post_write_actual_has_no_extra_caption_like_text_segments": bool(audit.get("actual_has_no_extra_caption_like_text_segments")),
        "post_write_actual_caption_rows_exact_match_plan": bool(audit.get("actual_caption_rows_exact_match_plan")),
        "post_write_actual_text_residue_gate_passed": bool(audit.get("actual_text_residue_gate_passed")),
        "post_write_actual_audio_coverage_gate_passed": bool(audit.get("actual_audio_coverage_gate_passed")),
        "post_write_actual_visible_text_repeat_gate_passed": bool(audit.get("actual_visible_text_repeat_gate_passed")),
        "post_write_actual_text_segment_count": int(audit.get("actual_text_segment_count") or 0),
        "post_write_generated_caption_segment_count": int(audit.get("generated_caption_segment_count") or 0),
        "post_write_preserved_non_subtitle_count": int(audit.get("preserved_non_subtitle_count") or 0),
        "post_write_old_subtitle_residue_count": int(audit.get("old_subtitle_residue_count") or 0),
        "post_write_orphan_text_segment_count": int(audit.get("orphan_text_segment_count") or 0),
        "post_write_text_after_final_video_end_count": int(audit.get("text_after_final_video_end_count") or 0),
        "post_write_floating_caption_count": int(audit.get("floating_caption_count") or 0),
        "post_write_audio_coverage_failure_count": int(audit.get("audio_coverage_failure_count") or 0),
        "post_write_heard_but_uncaptioned_word_count": int(audit.get("heard_but_uncaptioned_word_count") or 0),
        "post_write_dropped_but_reintroduced_word_count": int(audit.get("dropped_but_reintroduced_word_count") or 0),
        "post_write_actual_visible_repeat_candidate_count": int(audit.get("actual_visible_repeat_candidate_count") or 0),
        "final_video_end_us": int(audit.get("final_video_end_us") or 0),
        "max_caption_end_us": int(audit.get("max_caption_end_us") or 0),
        "captions_after_final_video_end_count": int(audit.get("captions_after_final_video_end_count") or 0),
        "post_write_video_target_gap_count_gt_300ms": int(audit.get("post_write_video_target_gap_count_gt_300ms") or 0),
        "post_write_total_video_target_gap_us": int(audit.get("post_write_total_video_target_gap_us") or 0),
        "caption_video_drift_count": int(audit.get("caption_video_drift_count") or 0),
        "max_caption_video_drift_us": int(audit.get("max_caption_video_drift_us") or 0),
        "split_caption_container_mismatch_count": int(audit.get("split_caption_container_mismatch_count") or 0),
        "caption_crosses_video_split_gap_count": int(audit.get("caption_crosses_video_split_gap_count") or 0),
        "caption_words_not_covered_by_actual_video_count": int(audit.get("caption_words_not_covered_by_actual_video_count") or 0),
        "jianying_canonical_timeline_sync_gate_passed": bool(audit.get("jianying_canonical_timeline_sync_gate_passed")),
        "post_write_actual_effective_speed_gate_passed": bool(audit.get("actual_effective_speed_gate_passed")),
        "post_write_actual_visual_pacing_gate_passed": bool(audit.get("actual_visual_pacing_gate_passed")),
        "post_write_actual_caption_gui_readability_gate_passed": bool(audit.get("actual_caption_gui_readability_gate_passed")),
        "post_write_actual_final_caption_visible_repeat_gate_passed": bool(audit.get("actual_final_caption_visible_repeat_gate_passed")),
        "post_write_actual_caption_alignment_gate_passed": bool(audit.get("actual_caption_alignment_gate_passed")),
        "post_write_actual_only_specified_draft_written": bool(audit.get("only_specified_draft_written")),
    }


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

    def _gapless_caption_video_projection_plan(self, run_report: RunReport) -> dict[str, Any]:
        final_by_id = {segment.segment_id: segment for segment in run_report.final_timeline}
        word_by_id = {
            str(word.word_id): word
            for word in (run_report.source_graph.words if run_report.source_graph is not None else [])
        }
        units: list[FinalTimelineSegment] = []
        caption_target_ranges: dict[str, dict[str, int]] = {}
        missing_caption_segment_ids: list[str] = []
        target_cursor = 0
        for caption in sorted(run_report.captions, key=lambda row: (int(row.target_start_us), int(row.target_end_us), str(row.caption_id))):
            final_id = str(caption.containing_video_segment_id or (caption.timeline_segment_ids[0] if caption.timeline_segment_ids else ""))
            final_segment = final_by_id.get(final_id)
            if final_segment is None:
                missing_caption_segment_ids.append(str(caption.caption_id))
                continue
            spoken_start, spoken_end = self._caption_spoken_source_span(caption, word_by_id)
            if spoken_start is None or spoken_end is None or spoken_end <= spoken_start:
                raise WritebackError(
                    "V21_WRITEBACK_CAPTION_SOURCE_SPAN_MISSING",
                    "caption cannot be projected to video because its spoken source span is missing",
                    {"caption_id": caption.caption_id, "timeline_segment_ids": list(caption.timeline_segment_ids)},
                )
            target_start = int(caption.target_start_us)
            target_end = int(caption.target_end_us)
            if target_end <= target_start:
                raise WritebackError(
                    "V21_WRITEBACK_CAPTION_SOURCE_SPAN_MISSING",
                    "caption cannot be projected to video because its target span is empty",
                    {"caption_id": caption.caption_id, "target_start_us": target_start, "target_end_us": target_end},
                )
            source_groups = self._caption_video_source_groups(caption, list(word_by_id.values()), source_start=spoken_start, source_end=spoken_end)
            caption_range_start: int | None = None
            caption_range_end: int | None = None
            for group_index, group in enumerate(source_groups, start=1):
                group_source_start = int(group["source_start_us"])
                group_source_end = int(group["source_end_us"])
                group_duration = max(0, group_source_end - group_source_start)
                if group_duration <= 0:
                    continue
                group_target_start = target_cursor
                group_target_end = group_target_start + group_duration
                target_cursor = group_target_end
                if caption_range_start is None:
                    caption_range_start = group_target_start
                caption_range_end = group_target_end
                debug_hints = dict(final_segment.debug_hints)
                debug_hints.update(
                    {
                        "safe_handle_policy_enabled": False,
                        "writeback_caption_span_projection": True,
                        "writeback_caption_id": caption.caption_id,
                        "writeback_caption_split_index": group_index,
                        "writeback_caption_split_count": len(source_groups),
                        "writeback_original_final_segment_id": final_segment.segment_id,
                    }
                )
                units.append(
                    FinalTimelineSegment(
                        segment_id=final_segment.segment_id,
                        source_material_id=final_segment.source_material_id,
                        source_segment_id=final_segment.source_segment_id,
                        source_start_us=group_source_start,
                        source_end_us=group_source_end,
                        target_start_us=group_target_start,
                        target_end_us=group_target_end,
                        word_ids=list(group["word_ids"]),
                        text=caption.text,
                        decision_ids=list(final_segment.decision_ids),
                        spoken_source_start_us=group_source_start,
                        spoken_source_end_us=group_source_end,
                        clip_source_start_us=group_source_start,
                        clip_source_end_us=group_source_end,
                        lead_handle_us=0,
                        tail_handle_us=0,
                        debug_hints=debug_hints,
                    )
                )
            if caption_range_start is None or caption_range_end is None or caption_range_end <= caption_range_start:
                raise WritebackError(
                    "V21_WRITEBACK_CAPTION_SOURCE_SPAN_MISSING",
                    "caption cannot be projected to a non-empty gapless video interval",
                    {"caption_id": caption.caption_id, "timeline_segment_ids": list(caption.timeline_segment_ids)},
                )
            caption_target_ranges[str(caption.caption_id)] = {
                "target_start_us": int(caption_range_start),
                "target_end_us": int(caption_range_end),
            }
        if missing_caption_segment_ids:
            raise WritebackError(
                "V21_WRITEBACK_CAPTION_SOURCE_SPAN_MISSING",
                "one or more captions do not have a containing final video segment",
                {"caption_ids": missing_caption_segment_ids[:20], "missing_caption_count": len(missing_caption_segment_ids)},
            )
        if not units and run_report.final_timeline:
            raise WritebackError(
                "V21_WRITEBACK_CAPTION_SOURCE_SPAN_MISSING",
                "no caption-backed video projection units were available",
                {"final_timeline_segment_count": len(run_report.final_timeline), "caption_count": len(run_report.captions)},
            )
        return {
            "video_units": units,
            "caption_target_ranges": caption_target_ranges,
            "caption_repacked_count": len(caption_target_ranges),
            "final_video_end_us": int(target_cursor),
            "video_write_plan_gapless": True,
        }

    def _video_projection_units_from_caption_spans(self, run_report: RunReport) -> list[FinalTimelineSegment]:
        return list(self._gapless_caption_video_projection_plan(run_report)["video_units"])

    def _apply_gapless_caption_ranges(
        self,
        text_segments: list[dict[str, Any]],
        captions: list[CaptionRenderUnit],
        caption_target_ranges: dict[str, dict[str, int]],
    ) -> None:
        missing_caption_ids: list[str] = []
        for caption, segment in zip(captions, text_segments):
            caption_range = caption_target_ranges.get(str(caption.caption_id))
            if caption_range is None:
                missing_caption_ids.append(str(caption.caption_id))
                continue
            target_start = int(caption_range["target_start_us"])
            target_end = int(caption_range["target_end_us"])
            if target_end <= target_start:
                raise WritebackError(
                    "V21_WRITEBACK_CAPTION_SOURCE_SPAN_MISSING",
                    "caption gapless target range is empty",
                    {"caption_id": caption.caption_id, "target_start_us": target_start, "target_end_us": target_end},
                )
            segment["target_timerange"] = {"start": target_start, "duration": target_end - target_start}
        if missing_caption_ids or len(text_segments) != len(captions):
            raise WritebackError(
                "V21_WRITEBACK_CAPTION_SOURCE_SPAN_MISSING",
                "caption write segments cannot be matched to gapless projection ranges",
                {
                    "missing_caption_ids": missing_caption_ids[:20],
                    "missing_caption_count": len(missing_caption_ids),
                    "caption_count": len(captions),
                    "text_segment_count": len(text_segments),
                },
            )

    def _caption_spoken_source_span(
        self,
        caption: CaptionRenderUnit,
        word_by_id: dict[str, Any],
    ) -> tuple[int | None, int | None]:
        explicit_start = int(caption.spoken_source_start_us) if caption.spoken_source_start_us is not None else None
        explicit_end = int(caption.spoken_source_end_us) if caption.spoken_source_end_us is not None else None
        words = [word_by_id[str(word_id)] for word_id in caption.word_ids if str(word_id) in word_by_id]
        if words:
            word_start = min(int(word.source_start_us) for word in words)
            word_end = max(int(word.source_end_us) for word in words)
            if explicit_start is not None and explicit_end is not None and not (explicit_start <= word_start and word_end <= explicit_end):
                return explicit_start, explicit_end
            return word_start, word_end
        if explicit_start is not None and explicit_end is not None:
            return explicit_start, explicit_end
        return None, None

    def _caption_video_source_groups(
        self,
        caption: CaptionRenderUnit,
        words: list[Any],
        *,
        source_start: int,
        source_end: int,
    ) -> list[dict[str, Any]]:
        caption_word_ids = {str(word_id) for word_id in caption.word_ids}
        kept_words = sorted(
            [
                word
                for word in words
                if str(word.word_id) in caption_word_ids
                and self._ranges_overlap(int(word.source_start_us), int(word.source_end_us), int(source_start), int(source_end))
            ],
            key=lambda word: (int(word.source_start_us), int(word.source_end_us), str(word.word_id)),
        )
        if not kept_words:
            return [{"source_start_us": int(source_start), "source_end_us": int(source_end), "word_ids": list(caption.word_ids)}]
        groups: list[list[Any]] = [[kept_words[0]]]
        for previous, current in zip(kept_words, kept_words[1:]):
            gap_start = int(previous.source_end_us)
            gap_end = int(current.source_start_us)
            split_for_dropped_speech = any(
                str(word.word_id) not in caption_word_ids
                and self._ranges_overlap(int(word.source_start_us), int(word.source_end_us), gap_start, gap_end)
                for word in words
            )
            if split_for_dropped_speech:
                groups.append([current])
            else:
                groups[-1].append(current)
        return [
            {
                "source_start_us": min(int(word.source_start_us) for word in group),
                "source_end_us": max(int(word.source_end_us) for word in group),
                "word_ids": [str(word.word_id) for word in group],
            }
            for group in groups
        ]

    def _video_segment_from_template(self, template: dict[str, Any], final_segment, index: int, draft_data: dict[str, Any]) -> dict[str, Any]:
        speed = self._segment_speed(template, draft_data)
        row = project_video_segment_from_template(template, final_segment, index, speed)
        row["_v21_audio_coverage"] = {
            "timeline_segment_id": str(final_segment.segment_id),
            "caption_id": str((final_segment.debug_hints or {}).get("writeback_caption_id") or ""),
            "source_start_us": int(final_segment.source_start_us),
            "source_end_us": int(final_segment.source_end_us),
            "spoken_source_start_us": int(
                final_segment.spoken_source_start_us if final_segment.spoken_source_start_us is not None else final_segment.source_start_us
            ),
            "spoken_source_end_us": int(
                final_segment.spoken_source_end_us if final_segment.spoken_source_end_us is not None else final_segment.source_end_us
            ),
            "word_ids": list(final_segment.word_ids),
        }
        return row

    def _effective_speed_report(self, video_segments: list[dict[str, Any]]) -> dict[str, Any]:
        speeds: list[float] = []
        drift_count = 0
        for row in video_segments:
            speed = self._effective_speed_from_row(row)
            if speed is None:
                continue
            speeds.append(speed)
            if not self._effective_speed_supported_or_baseline(speed):
                drift_count += 1
        report = {
            "effective_speed_min": min(speeds, default=None),
            "effective_speed_max": max(speeds, default=None),
            "effective_speed_drift_count": drift_count,
        }
        report.update(safe_handle_report_from_projected_segments(video_segments))
        return report

    def _effective_speed_from_row(self, row: dict[str, Any]) -> float | None:
        source_timerange = row.get("source_timerange")
        target_timerange = row.get("target_timerange")
        source_duration = self._timerange_duration(source_timerange)
        target_duration = self._timerange_duration(target_timerange)
        if source_duration <= 0 or target_duration <= 0:
            return None
        return round(source_duration / target_duration, 6)

    def _effective_speed_supported_or_baseline(self, speed: float) -> bool:
        return any(abs(speed - candidate) <= EFFECTIVE_SPEED_TOLERANCE for candidate in (1.0, 1.2))

    def _post_write_actual_draft_audit(
        self,
        *,
        draft_dir: Path,
        actual_draft_content_path: Path,
        run_dir: Path,
        run_report: RunReport,
        expected_video_track_id: str,
        expected_text_track_id: str,
        expected_video_segments: list[dict[str, Any]],
        expected_text_segments: list[dict[str, Any]],
        target_writes: dict[str, bool],
        require_target_writes_committed: bool = True,
    ) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        audit: dict[str, Any] = {
            "post_write_actual_draft_audit_required_on_commit": True,
            "executed": False,
            "gate_passed": False,
            "actual_draft_loaded": False,
            "actual_draft_path": str(actual_draft_content_path),
            "actual_draft_source": "",
            "only_specified_draft_written": False,
            "actual_video_rows_match_plan": False,
            "actual_caption_rows_match_plan": False,
            "expected_caption_rows_present": False,
            "actual_has_no_extra_caption_like_text_segments": False,
            "actual_caption_rows_exact_match_plan": False,
            "actual_text_residue_gate_passed": False,
            "actual_audio_coverage_gate_passed": False,
            "actual_visible_text_repeat_gate_passed": False,
            "actual_effective_speed_gate_passed": False,
            "actual_visual_pacing_gate_passed": False,
            "actual_caption_gui_readability_gate_passed": False,
            "actual_final_caption_visible_repeat_gate_passed": False,
            "actual_caption_alignment_gate_passed": False,
            "failure_reasons": failures,
            "blocker_codes": [],
        }

        def fail(reason: str, **context: Any) -> None:
            failures.append({"reason": reason, **context})

        target_paths_within_draft = bool(target_writes) and all(
            self._is_within(Path(path), draft_dir) for path in target_writes
        )
        target_commit_state_ok = (not require_target_writes_committed) or all(bool(value) for value in target_writes.values())
        only_specified = target_paths_within_draft and target_commit_state_ok
        audit["only_specified_draft_written"] = only_specified
        audit["target_writes_committed_required"] = bool(require_target_writes_committed)
        audit["target_paths_within_draft"] = target_paths_within_draft
        if not only_specified:
            fail("only_specified_draft_written_failed", target_writes=dict(target_writes))

        actual_data, load_report = self._load_actual_draft_content(actual_draft_content_path, run_dir)
        audit.update(load_report)
        if actual_data is None:
            fail("actual_draft_missing_or_unreadable", error=load_report.get("actual_draft_load_error", ""))
            return _finalize_post_write_actual_draft_audit(audit)

        audit["executed"] = True
        video_track = self._track_by_id(actual_data, expected_video_track_id)
        text_track = self._track_by_id(actual_data, expected_text_track_id)
        if video_track is None:
            fail("actual_video_track_missing", expected_video_track_id=expected_video_track_id)
            actual_video_segments: list[dict[str, Any]] = []
        else:
            actual_video_segments = [row for row in video_track.get("segments") or [] if isinstance(row, dict)]
        if text_track is None:
            fail("actual_caption_track_missing", expected_text_track_id=expected_text_track_id)
            actual_text_segments: list[dict[str, Any]] = []
        else:
            actual_text_segments = [row for row in text_track.get("segments") or [] if isinstance(row, dict)]

        video_match = self._video_segments_match_plan(actual_video_segments, expected_video_segments)
        audit["actual_video_rows_match_plan"] = video_match
        audit["actual_video_segment_count"] = len(actual_video_segments)
        audit["expected_video_segment_count"] = len(expected_video_segments)
        if not video_match:
            fail(
                "actual_video_rows_differ_from_projection",
                actual_video_segment_count=len(actual_video_segments),
                expected_video_segment_count=len(expected_video_segments),
            )

        visible_text_rows = self._visible_text_rows(actual_data)
        text_residue_report = self._actual_text_residue_report(
            visible_text_rows=visible_text_rows,
            actual_video_segments=actual_video_segments,
            expected_text_segments=expected_text_segments,
            run_report=run_report,
        )
        audit["actual_text_residue_report"] = text_residue_report
        audit.update(
            {
                key: text_residue_report[key]
                for key in (
                    "actual_text_residue_gate_passed",
                    "expected_caption_rows_present",
                    "actual_has_no_extra_caption_like_text_segments",
                    "actual_caption_rows_exact_match_plan",
                    "actual_caption_rows_match_plan",
                    "actual_text_segment_count",
                    "generated_caption_segment_count",
                    "preserved_non_subtitle_count",
                    "old_subtitle_residue_count",
                    "orphan_text_segment_count",
                    "text_after_final_video_end_count",
                    "floating_caption_count",
                )
            }
        )
        caption_match = bool(text_residue_report.get("actual_caption_rows_exact_match_plan"))
        audit["actual_caption_segment_count"] = int(text_residue_report.get("generated_caption_segment_count") or 0)
        audit["expected_caption_segment_count"] = len(expected_text_segments)
        audit["selected_text_track_segment_count"] = len(actual_text_segments)
        if not caption_match:
            fail(
                "actual_caption_rows_differ_from_material_write_plan",
                actual_caption_segment_count=int(text_residue_report.get("generated_caption_segment_count") or 0),
                actual_text_segment_count=int(text_residue_report.get("actual_text_segment_count") or 0),
                expected_caption_segment_count=len(expected_text_segments),
                expected_caption_rows_present=bool(text_residue_report.get("expected_caption_rows_present")),
                actual_has_no_extra_caption_like_text_segments=bool(text_residue_report.get("actual_has_no_extra_caption_like_text_segments")),
            )
        if not bool(text_residue_report.get("actual_text_residue_gate_passed")):
            fail("actual_text_residue_gate_failed", actual_text_residue_report=text_residue_report)

        speed_report = self._effective_speed_report(actual_video_segments)
        audit["actual_effective_speed_report"] = speed_report
        speed_passed = bool(actual_video_segments) and int(speed_report.get("effective_speed_drift_count") or 0) == 0
        audit["actual_effective_speed_gate_passed"] = speed_passed
        if not speed_passed:
            fail("actual_effective_speed_gate_failed", effective_speed_report=speed_report)

        plan_quality = self._plan_quality_audit(run_report)
        audit.update(plan_quality)
        for reason in plan_quality.get("plan_quality_failure_reasons") or []:
            fail(str(reason))

        actual_timeline = self._actual_final_timeline_from_video_rows(actual_video_segments, run_report.final_timeline)
        actual_alignment_timeline = self._actual_timeline_with_caption_split_containers(
            actual_timeline,
            text_residue_report.get("generated_caption_rows") or [],
        )
        actual_caption_units = self._actual_caption_units_from_text_rows(
            text_residue_report.get("generated_caption_rows") or [],
            actual_timeline=actual_alignment_timeline,
            run_report=run_report,
        )
        actual_caption_alignment = build_caption_alignment_report(
            final_timeline=actual_alignment_timeline,
            captions=actual_caption_units,
            visible_caption_track_count=1 if actual_caption_units else 0,
            caption_lane_count=1 if actual_caption_units else 0,
        )
        audit["actual_caption_alignment_report"] = actual_caption_alignment
        audit["actual_caption_alignment_gate_passed"] = bool(actual_caption_alignment.get("gate_passed"))
        audit["actual_caption_gui_readability_gate_passed"] = bool(actual_caption_alignment.get("caption_gui_track_gate_passed")) and bool(
            actual_caption_alignment.get("subtitle_readability_gate_passed")
        )
        if not bool(actual_caption_alignment.get("gate_passed")):
            fail("actual_caption_alignment_gate_failed", actual_caption_alignment_report=actual_caption_alignment)

        caption_like_rows = list(text_residue_report.get("generated_caption_rows") or []) + list(text_residue_report.get("old_subtitle_residue_segments") or [])
        actual_visible_text_units = self._actual_caption_units_from_text_rows(
            caption_like_rows,
            actual_timeline=actual_alignment_timeline,
            run_report=run_report,
        )
        actual_visible_text_repeat = build_final_caption_visible_repeat_gate(actual_visible_text_units)
        audit["actual_visible_text_repeat_report"] = actual_visible_text_repeat
        audit["actual_visible_text_repeat_gate_passed"] = bool(actual_visible_text_repeat.get("gate_passed"))
        audit["actual_final_caption_visible_repeat_gate_passed"] = bool(actual_visible_text_repeat.get("gate_passed"))
        audit["actual_visible_repeat_candidate_count"] = int(actual_visible_text_repeat.get("visible_repeat_candidate_count") or 0)
        if not bool(actual_visible_text_repeat.get("gate_passed")):
            fail("actual_visible_text_repeat_gate_failed", actual_visible_text_repeat_report=actual_visible_text_repeat)

        audio_coverage_report = self._actual_audio_coverage_report(
            actual_video_segments=actual_video_segments,
            run_report=run_report,
        )
        audit["actual_audio_coverage_report"] = audio_coverage_report
        audit["actual_audio_coverage_gate_passed"] = bool(audio_coverage_report.get("gate_passed"))
        audit["audio_coverage_failure_count"] = int(audio_coverage_report.get("audio_coverage_failure_count") or 0)
        audit["heard_but_uncaptioned_word_count"] = int(audio_coverage_report.get("heard_but_uncaptioned_word_count") or 0)
        audit["dropped_but_reintroduced_word_count"] = int(audio_coverage_report.get("dropped_but_reintroduced_word_count") or 0)
        if not bool(audio_coverage_report.get("gate_passed")):
            fail("actual_audio_coverage_gate_failed", actual_audio_coverage_report=audio_coverage_report)

        canonical_sync_report = self._jianying_canonical_timeline_sync_report(
            actual_video_segments=actual_video_segments,
            generated_caption_rows=text_residue_report.get("generated_caption_rows") or [],
            run_report=run_report,
        )
        audit["jianying_canonical_timeline_sync_report"] = canonical_sync_report
        for key in (
            "final_video_end_us",
            "max_caption_end_us",
            "captions_after_final_video_end_count",
            "post_write_video_target_gap_count_gt_300ms",
            "post_write_total_video_target_gap_us",
            "caption_video_drift_count",
            "max_caption_video_drift_us",
            "split_caption_container_mismatch_count",
            "caption_crosses_video_split_gap_count",
            "caption_words_not_covered_by_actual_video_count",
            "jianying_canonical_timeline_sync_gate_passed",
        ):
            audit[key] = canonical_sync_report.get(key)
        if not bool(canonical_sync_report.get("jianying_canonical_timeline_sync_gate_passed")):
            audit["blocker_codes"] = _unique_strings(
                list(audit.get("blocker_codes") or []) + list(canonical_sync_report.get("blocker_codes") or [])
            )
            fail("jianying_canonical_timeline_sync_failed", jianying_canonical_timeline_sync_report=canonical_sync_report)

        return _finalize_post_write_actual_draft_audit(audit)

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
        if len(actual) != len(expected) or not expected:
            return False
        for actual_row, expected_row in zip(actual, expected):
            if str(actual_row.get("id") or "") != str(expected_row.get("id") or ""):
                return False
            if self._segment_timerange_signature(actual_row) != self._segment_timerange_signature(expected_row):
                return False
        return True

    def _expected_caption_segments_present(self, actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
        if not expected:
            return False
        actual_by_id = {str(row.get("id") or ""): row for row in actual if str(row.get("id") or "")}
        for expected_row in expected:
            expected_id = str(expected_row.get("id") or "")
            actual_row = actual_by_id.get(expected_id)
            if actual_row is None:
                return False
            if self._segment_timerange_signature(actual_row, include_source=False) != self._segment_timerange_signature(
                expected_row,
                include_source=False,
            ):
                return False
            if str(actual_row.get("material_id") or actual_row.get("materialId") or "") != str(
                expected_row.get("material_id") or expected_row.get("materialId") or ""
            ):
                return False
        return True

    def _visible_text_rows(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        materials = data.get("materials") if isinstance(data.get("materials"), dict) else {}
        text_material_by_id = {
            str(row.get("id") or ""): row
            for row in materials.get("texts") or []
            if isinstance(row, dict) and str(row.get("id") or "")
        }
        rows: list[dict[str, Any]] = []
        for track in data.get("tracks") or []:
            if not isinstance(track, dict) or not self._is_text_track(track) or not self._is_visible_timeline_row(track):
                continue
            track_id = str(track.get("id") or "")
            track_type = str(track.get("type") or track.get("track_type") or "")
            for segment in track.get("segments") or []:
                if not isinstance(segment, dict) or not self._is_visible_timeline_row(segment):
                    continue
                material_id = str(segment.get("material_id") or segment.get("materialId") or "")
                material = text_material_by_id.get(material_id) or {}
                text = self._text_segment_text(segment, material)
                if not text.strip():
                    continue
                target_start = self._timerange_start(segment.get("target_timerange"))
                duration = self._timerange_duration(segment.get("target_timerange"))
                segment_for_classification = dict(segment)
                segment_for_classification.setdefault("track_id", track_id)
                segment_for_classification.setdefault("track_type", track_type)
                rows.append(
                    {
                        "track_id": track_id,
                        "track_type": track_type,
                        "segment_id": str(segment.get("id") or ""),
                        "material_id": material_id,
                        "segment": segment_for_classification,
                        "material": material,
                        "text": text,
                        "target_start_us": target_start,
                        "target_end_us": target_start + duration,
                        "duration_us": duration,
                    }
                )
        return rows

    def _is_visible_timeline_row(self, row: dict[str, Any]) -> bool:
        for key in ("visible", "is_visible", "enable", "enabled"):
            if row.get(key) is False:
                return False
        for key in ("hidden", "disabled", "is_hidden"):
            if row.get(key) is True:
                return False
        return True

    def _text_segment_text(self, segment: dict[str, Any], material: dict[str, Any]) -> str:
        for key in ("text", "recognize_text"):
            value = segment.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return self._text_material_text(material)

    def _classified_actual_text_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        expected_segment_ids: set[str],
        expected_material_ids: set[str],
        template_material_ids: set[str],
    ) -> list[dict[str, Any]]:
        classified_rows: list[dict[str, Any]] = []
        for row in rows:
            segment = row["segment"]
            material = row["material"]
            segment_id = str(row.get("segment_id") or "")
            material_id = str(row.get("material_id") or "")
            if segment_id in expected_segment_ids or material_id in expected_material_ids:
                classification = "generated_caption"
                reason = "segment_or_material_id_matches_v21_caption_plan"
                generated_caption_id = True
            elif self._is_confirmed_non_subtitle_text(segment, material):
                classification = "confirmed_non_subtitle"
                reason = "segment_or_material_metadata_marks_non_subtitle_text"
                generated_caption_id = False
            elif material_id in template_material_ids:
                classification = "old_subtitle_residue"
                reason = "material_id_matches_old_subtitle_template"
                generated_caption_id = False
            else:
                classification = "old_subtitle_residue"
                reason = "visible_caption_like_text_without_confirmed_non_subtitle_metadata"
                generated_caption_id = False
            classified = dict(row)
            classified.update(
                {
                    "classification": classification,
                    "classification_reason": reason,
                    "generated_caption_id": generated_caption_id,
                }
            )
            classified_rows.append(classified)
        return classified_rows

    def _actual_text_residue_report(
        self,
        *,
        visible_text_rows: list[dict[str, Any]],
        actual_video_segments: list[dict[str, Any]],
        expected_text_segments: list[dict[str, Any]],
        run_report: RunReport,
    ) -> dict[str, Any]:
        expected_segment_ids = {str(row.get("id") or "") for row in expected_text_segments if str(row.get("id") or "")}
        expected_material_ids = {
            str(row.get("material_id") or row.get("materialId") or "")
            for row in expected_text_segments
            if str(row.get("material_id") or row.get("materialId") or "")
        }
        expected_material_ids.update(
            str(row.get("id") or "")
            for row in run_report.material_write_plan.get("materials") or []
            if isinstance(row, dict) and str(row.get("id") or "")
        )
        classified_rows = self._classified_actual_text_rows(
            visible_text_rows,
            expected_segment_ids=expected_segment_ids,
            expected_material_ids=expected_material_ids,
            template_material_ids=self._template_candidate_material_ids(run_report),
        )
        caption_by_segment_id: dict[str, str] = {}
        caption_by_material_id: dict[str, str] = {}
        for caption, segment in zip(run_report.captions, expected_text_segments):
            segment_id = str(segment.get("id") or "")
            material_id = str(segment.get("material_id") or segment.get("materialId") or "")
            if segment_id:
                caption_by_segment_id[segment_id] = caption.caption_id
            if material_id:
                caption_by_material_id[material_id] = caption.caption_id
        for row in classified_rows:
            if row["classification"] == "generated_caption":
                row["caption_id"] = caption_by_segment_id.get(str(row.get("segment_id") or "")) or caption_by_material_id.get(str(row.get("material_id") or "")) or ""
        generated_rows = [row for row in classified_rows if row["classification"] == "generated_caption"]
        residue_rows = [row for row in classified_rows if row["classification"] == "old_subtitle_residue"]
        preserved_rows = [row for row in classified_rows if row["classification"] == "confirmed_non_subtitle"]
        caption_like_rows = [*generated_rows, *residue_rows]
        final_video_end = max(
            (
                self._timerange_start(row.get("target_timerange")) + self._timerange_duration(row.get("target_timerange"))
                for row in actual_video_segments
            ),
            default=0,
        )
        orphan_rows = [
            row
            for row in caption_like_rows
            if not self._has_containing_video_segment(row, actual_video_segments)
        ]
        text_after_rows = [
            row
            for row in caption_like_rows
            if final_video_end > 0 and int(row.get("target_end_us") or 0) > final_video_end
        ]
        floating_rows = list(orphan_rows)
        expected_present = self._expected_caption_segments_present([row["segment"] for row in visible_text_rows], expected_text_segments)
        no_extra_caption_like = not residue_rows and not orphan_rows and not text_after_rows and not floating_rows
        exact_match = expected_present and no_extra_caption_like and len(generated_rows) == len(expected_text_segments)
        gate_passed = exact_match
        return {
            "gate_passed": gate_passed,
            "actual_text_residue_gate_passed": gate_passed,
            "expected_caption_rows_present": expected_present,
            "actual_has_no_extra_caption_like_text_segments": no_extra_caption_like,
            "actual_caption_rows_exact_match_plan": exact_match,
            "actual_caption_rows_match_plan": exact_match,
            "actual_text_segment_count": len(visible_text_rows),
            "generated_caption_segment_count": len(generated_rows),
            "preserved_non_subtitle_count": len(preserved_rows),
            "old_subtitle_residue_count": len(residue_rows),
            "orphan_text_segment_count": len(orphan_rows),
            "text_after_final_video_end_count": len(text_after_rows),
            "floating_caption_count": len(floating_rows),
            "final_video_end_us": final_video_end,
            "generated_caption_rows": generated_rows,
            "old_subtitle_residue_segments": residue_rows,
            "orphan_text_segments": orphan_rows,
            "text_after_final_video_end_segments": text_after_rows,
            "floating_caption_segments": floating_rows,
            "preserved_non_subtitle_segments": preserved_rows,
        }

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
        return False

    def _actual_final_timeline_from_video_rows(
        self,
        actual_video_segments: list[dict[str, Any]],
        planned_segments: list[FinalTimelineSegment],
    ) -> list[FinalTimelineSegment]:
        timeline: list[FinalTimelineSegment] = []
        for index, row in enumerate(actual_video_segments):
            planned = planned_segments[index] if index < len(planned_segments) else None
            coverage = row.get("_v21_audio_coverage") if isinstance(row.get("_v21_audio_coverage"), dict) else {}
            target_start = self._timerange_start(row.get("target_timerange"))
            target_end = target_start + self._timerange_duration(row.get("target_timerange"))
            source_start, source_end, spoken_start, spoken_end = self._actual_source_interval_for_video_row(row, planned)
            segment_id = str(row.get("id") or f"actual_video_{index + 1:06d}")
            debug_hints = dict(planned.debug_hints) if planned is not None else {}
            if coverage:
                debug_hints.update(
                    {
                        "writeback_caption_id": str(coverage.get("caption_id") or ""),
                        "writeback_original_final_segment_id": str(coverage.get("timeline_segment_id") or ""),
                    }
                )
            timeline.append(
                FinalTimelineSegment(
                    segment_id=segment_id,
                    source_material_id=planned.source_material_id if planned is not None else str(row.get("material_id") or row.get("materialId") or ""),
                    source_segment_id=planned.source_segment_id if planned is not None else None,
                    source_start_us=source_start,
                    source_end_us=source_end,
                    target_start_us=target_start,
                    target_end_us=target_end,
                    word_ids=list(coverage.get("word_ids") or (planned.word_ids if planned is not None else [])),
                    text=planned.text if planned is not None else "",
                    decision_ids=list(planned.decision_ids) if planned is not None else [],
                    spoken_source_start_us=spoken_start,
                    spoken_source_end_us=spoken_end,
                    clip_source_start_us=source_start,
                    clip_source_end_us=source_end,
                    lead_handle_us=max(0, spoken_start - source_start),
                    tail_handle_us=max(0, source_end - spoken_end),
                    debug_hints=debug_hints,
                )
            )
        return timeline

    def _actual_timeline_with_caption_split_containers(
        self,
        actual_timeline: list[FinalTimelineSegment],
        caption_rows: list[dict[str, Any]],
    ) -> list[FinalTimelineSegment]:
        timeline = list(actual_timeline)
        existing_ids = {segment.segment_id for segment in timeline}
        for row in caption_rows:
            caption_id = str(row.get("caption_id") or "")
            if not caption_id:
                continue
            start = int(row.get("target_start_us") or 0)
            end = int(row.get("target_end_us") or 0)
            if any(int(segment.target_start_us) <= start and end <= int(segment.target_end_us) for segment in timeline):
                continue
            split_segments = [
                segment
                for segment in actual_timeline
                if str((segment.debug_hints or {}).get("writeback_caption_id") or "") == caption_id
            ]
            if not split_segments:
                continue
            split_start = min(int(segment.target_start_us) for segment in split_segments)
            split_end = max(int(segment.target_end_us) for segment in split_segments)
            if not (split_start <= start and end <= split_end):
                continue
            word_ids: list[str] = []
            for segment in split_segments:
                for word_id in segment.word_ids:
                    if str(word_id) not in word_ids:
                        word_ids.append(str(word_id))
            source_start = min(int(segment.source_start_us) for segment in split_segments)
            source_end = max(int(segment.source_end_us) for segment in split_segments)
            segment_id = f"actual_caption_container_{caption_id}"
            if segment_id in existing_ids:
                continue
            existing_ids.add(segment_id)
            timeline.append(
                FinalTimelineSegment(
                    segment_id=segment_id,
                    source_material_id=split_segments[0].source_material_id,
                    source_segment_id=split_segments[0].source_segment_id,
                    source_start_us=source_start,
                    source_end_us=source_end,
                    target_start_us=start,
                    target_end_us=end,
                    word_ids=word_ids,
                    text=str(row.get("text") or ""),
                    decision_ids=[],
                    spoken_source_start_us=source_start,
                    spoken_source_end_us=source_end,
                    clip_source_start_us=source_start,
                    clip_source_end_us=source_end,
                    lead_handle_us=0,
                    tail_handle_us=0,
                    debug_hints={"writeback_caption_id": caption_id, "synthetic_split_caption_container": True},
                )
            )
        return timeline

    def _actual_caption_units_from_text_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        actual_timeline: list[FinalTimelineSegment],
        run_report: RunReport,
    ) -> list[CaptionRenderUnit]:
        caption_by_segment_id: dict[str, CaptionRenderUnit] = {}
        caption_by_material_id: dict[str, CaptionRenderUnit] = {}
        for caption, segment in zip(run_report.captions, run_report.material_write_plan.get("segments") or []):
            if not isinstance(segment, dict):
                continue
            segment_id = str(segment.get("id") or "")
            material_id = str(segment.get("material_id") or segment.get("materialId") or "")
            if segment_id:
                caption_by_segment_id[segment_id] = caption
            if material_id:
                caption_by_material_id[material_id] = caption
        units: list[CaptionRenderUnit] = []
        for index, row in enumerate(sorted(rows, key=lambda item: (int(item.get("target_start_us") or 0), int(item.get("target_end_us") or 0), str(item.get("segment_id") or ""))), start=1):
            planned = caption_by_segment_id.get(str(row.get("segment_id") or "")) or caption_by_material_id.get(str(row.get("material_id") or ""))
            container = self._containing_actual_timeline_segment(row, actual_timeline)
            timeline_segment_ids = [container.segment_id] if container is not None else (list(planned.timeline_segment_ids) if planned is not None else [])
            units.append(
                CaptionRenderUnit(
                    caption_id=planned.caption_id if planned is not None else f"actual_text_{str(row.get('segment_id') or index)}",
                    timeline_segment_ids=timeline_segment_ids,
                    word_ids=list(planned.word_ids) if planned is not None else [],
                    text=str(row.get("text") or ""),
                    target_start_us=int(row.get("target_start_us") or 0),
                    target_end_us=int(row.get("target_end_us") or 0),
                    source_subtitle_uids=list(planned.source_subtitle_uids) if planned is not None else [],
                    style_template_id=planned.style_template_id if planned is not None else "actual_visible_text",
                    spoken_source_start_us=planned.spoken_source_start_us if planned is not None else None,
                    spoken_source_end_us=planned.spoken_source_end_us if planned is not None else None,
                    containing_video_segment_id=container.segment_id if container is not None else None,
                )
            )
        return units

    def _containing_actual_timeline_segment(
        self,
        text_row: dict[str, Any],
        actual_timeline: list[FinalTimelineSegment],
    ) -> FinalTimelineSegment | None:
        start = int(text_row.get("target_start_us") or 0)
        end = int(text_row.get("target_end_us") or 0)
        for segment in actual_timeline:
            if int(segment.target_start_us) <= start and end <= int(segment.target_end_us):
                return segment
        caption_id = str(text_row.get("caption_id") or "")
        if caption_id:
            split_segments = [
                segment
                for segment in actual_timeline
                if str((segment.debug_hints or {}).get("writeback_caption_id") or "") == caption_id
            ]
            if split_segments:
                target_start = min(int(segment.target_start_us) for segment in split_segments)
                target_end = max(int(segment.target_end_us) for segment in split_segments)
                if target_start <= start and end <= target_end:
                    word_ids: list[str] = []
                    for segment in split_segments:
                        for word_id in segment.word_ids:
                            if str(word_id) not in word_ids:
                                word_ids.append(str(word_id))
                    source_start = min(int(segment.source_start_us) for segment in split_segments)
                    source_end = max(int(segment.source_end_us) for segment in split_segments)
                    return FinalTimelineSegment(
                        segment_id=f"actual_caption_container_{caption_id}",
                        source_material_id=split_segments[0].source_material_id,
                        source_segment_id=split_segments[0].source_segment_id,
                        source_start_us=source_start,
                        source_end_us=source_end,
                        target_start_us=start,
                        target_end_us=end,
                        word_ids=word_ids,
                        text=str(text_row.get("text") or ""),
                        decision_ids=[],
                        spoken_source_start_us=source_start,
                        spoken_source_end_us=source_end,
                        clip_source_start_us=source_start,
                        clip_source_end_us=source_end,
                        lead_handle_us=0,
                        tail_handle_us=0,
                        debug_hints={"writeback_caption_id": caption_id, "synthetic_split_caption_container": True},
                    )
        return None

    def _actual_audio_coverage_report(
        self,
        *,
        actual_video_segments: list[dict[str, Any]],
        run_report: RunReport,
    ) -> dict[str, Any]:
        if run_report.source_graph is None:
            return {
                "gate_passed": False,
                "actual_audio_coverage_gate_passed": False,
                "audio_coverage_failure_count": 1,
                "heard_but_uncaptioned_word_count": 0,
                "dropped_but_reintroduced_word_count": 0,
                "failure_reasons": ["source_graph_missing"],
                "audio_segment_reports": [],
            }
        words = list(run_report.source_graph.words)
        captioned_word_ids = {str(word_id) for caption in run_report.captions for word_id in caption.word_ids}
        planned_word_ids = {str(word_id) for segment in run_report.final_timeline for word_id in segment.word_ids}
        segment_reports: list[dict[str, Any]] = []
        failure_count = 0
        heard_but_uncaptioned_count = 0
        dropped_but_reintroduced_count = 0
        allowed_gap_count = 0
        for interval in self._actual_source_interval_rows(actual_video_segments, run_report.final_timeline):
            heard_words = [
                word
                for word in words
                if self._ranges_overlap(int(word.source_start_us), int(word.source_end_us), interval["source_start_us"], interval["source_end_us"])
            ]
            missing_words: list[dict[str, Any]] = []
            allowed_words: list[dict[str, Any]] = []
            dropped_words: list[dict[str, Any]] = []
            for word in heard_words:
                word_id = str(word.word_id)
                if word_id in captioned_word_ids:
                    continue
                word_row = {
                    "word_id": word_id,
                    "text": word.text,
                    "source_start_us": int(word.source_start_us),
                    "source_end_us": int(word.source_end_us),
                }
                overlaps_spoken = self._ranges_overlap(
                    int(word.source_start_us),
                    int(word.source_end_us),
                    interval["spoken_source_start_us"],
                    interval["spoken_source_end_us"],
                )
                if not overlaps_spoken:
                    allowed_words.append(word_row | {"allowed_reason": "explicit_handle_gap"})
                    continue
                missing_words.append(word_row)
                if word_id not in planned_word_ids:
                    dropped_words.append(word_row)
            heard_but_uncaptioned_count += len(missing_words)
            dropped_but_reintroduced_count += len(dropped_words)
            allowed_gap_count += len(allowed_words)
            if missing_words or dropped_words:
                failure_count += 1
            segment_reports.append(
                {
                    "video_segment_id": interval["video_segment_id"],
                    "timeline_segment_id": interval["timeline_segment_id"],
                    "caption_id": interval.get("caption_id", ""),
                    "source_start_us": interval["source_start_us"],
                    "source_end_us": interval["source_end_us"],
                    "spoken_source_start_us": interval["spoken_source_start_us"],
                    "spoken_source_end_us": interval["spoken_source_end_us"],
                    "heard_word_count": len(heard_words),
                    "heard_but_uncaptioned_word_count": len(missing_words),
                    "dropped_but_reintroduced_word_count": len(dropped_words),
                    "allowed_handle_gap_word_count": len(allowed_words),
                    "heard_but_uncaptioned_words": missing_words,
                    "dropped_but_reintroduced_words": dropped_words,
                    "allowed_handle_gap_words": allowed_words,
                }
            )
        gate_passed = failure_count == 0 and heard_but_uncaptioned_count == 0 and dropped_but_reintroduced_count == 0
        return {
            "gate_passed": gate_passed,
            "actual_audio_coverage_gate_passed": gate_passed,
            "audio_coverage_failure_count": failure_count,
            "heard_but_uncaptioned_word_count": heard_but_uncaptioned_count,
            "dropped_but_reintroduced_word_count": dropped_but_reintroduced_count,
            "allowed_handle_gap_word_count": allowed_gap_count,
            "final_video_segment_count": len(actual_video_segments),
            "captioned_word_count": len(captioned_word_ids),
            "audio_segment_reports": segment_reports,
        }

    def _jianying_canonical_timeline_sync_report(
        self,
        *,
        actual_video_segments: list[dict[str, Any]],
        generated_caption_rows: list[dict[str, Any]],
        run_report: RunReport,
    ) -> dict[str, Any]:
        canonical_video_segments: list[dict[str, Any]] = []
        original_video_rows: list[dict[str, Any]] = []
        target_cursor = 0
        previous_target_end = 0
        gap_count = 0
        total_gap_us = 0
        for index, row in enumerate(actual_video_segments, start=1):
            target_start = self._timerange_start(row.get("target_timerange"))
            target_duration = self._timerange_duration(row.get("target_timerange"))
            target_end = target_start + target_duration
            gap_us = target_start - previous_target_end
            if gap_us > 0:
                total_gap_us += gap_us
            if gap_us > 300_000:
                gap_count += 1
            original_video_rows.append(
                {
                    "index": index,
                    "segment_id": str(row.get("id") or f"actual_video_{index:06d}"),
                    "target_start_us": target_start,
                    "target_end_us": target_end,
                    "duration_us": target_duration,
                    "caption_id": str((row.get("_v21_audio_coverage") or {}).get("caption_id") or "")
                    if isinstance(row.get("_v21_audio_coverage"), dict)
                    else "",
                }
            )
            canonical_row = deepcopy(row)
            canonical_row["target_timerange"] = {"start": target_cursor, "duration": max(0, target_duration)}
            canonical_row["_v21_canonical_save"] = {
                "original_target_start_us": target_start,
                "original_target_end_us": target_end,
                "canonical_target_start_us": target_cursor,
                "canonical_target_end_us": target_cursor + max(0, target_duration),
            }
            canonical_video_segments.append(canonical_row)
            target_cursor += max(0, target_duration)
            previous_target_end = target_end

        final_video_end_us = int(target_cursor)
        caption_rows = sorted(
            generated_caption_rows,
            key=lambda row: (int(row.get("target_start_us") or 0), int(row.get("target_end_us") or 0), str(row.get("caption_id") or "")),
        )
        max_caption_end_us = max((int(row.get("target_end_us") or 0) for row in caption_rows), default=0)
        captions_after_final_video_end = [
            row
            for row in caption_rows
            if final_video_end_us > 0 and int(row.get("target_end_us") or 0) > final_video_end_us
        ]
        canonical_rows_by_caption = self._video_rows_by_caption_id(canonical_video_segments)
        original_rows_by_caption = self._video_rows_by_caption_id(actual_video_segments)
        caption_by_id = {str(caption.caption_id): caption for caption in run_report.captions}

        caption_video_drift_rows: list[dict[str, Any]] = []
        split_caption_mismatch_rows: list[dict[str, Any]] = []
        split_gap_rows: list[dict[str, Any]] = []
        uncovered_word_rows: list[dict[str, Any]] = []
        for row in caption_rows:
            caption_id = str(row.get("caption_id") or "")
            caption_start = int(row.get("target_start_us") or 0)
            caption_end = int(row.get("target_end_us") or 0)
            canonical_group = canonical_rows_by_caption.get(caption_id) or []
            if canonical_group:
                group_start = min(self._timerange_start(segment.get("target_timerange")) for segment in canonical_group)
                group_end = max(
                    self._timerange_start(segment.get("target_timerange")) + self._timerange_duration(segment.get("target_timerange"))
                    for segment in canonical_group
                )
            else:
                group_start = 0
                group_end = 0
            drift_us = max(abs(caption_start - group_start), abs(caption_end - group_end)) if canonical_group else max(caption_end - caption_start, 0)
            if drift_us:
                caption_video_drift_rows.append(
                    {
                        "caption_id": caption_id,
                        "caption_start_us": caption_start,
                        "caption_end_us": caption_end,
                        "canonical_video_start_us": group_start,
                        "canonical_video_end_us": group_end,
                        "drift_us": drift_us,
                    }
                )
            if not canonical_group or caption_start != group_start or caption_end != group_end:
                split_caption_mismatch_rows.append(
                    {
                        "caption_id": caption_id,
                        "caption_start_us": caption_start,
                        "caption_end_us": caption_end,
                        "canonical_video_start_us": group_start,
                        "canonical_video_end_us": group_end,
                        "canonical_video_row_count": len(canonical_group),
                    }
                )

            original_group = original_rows_by_caption.get(caption_id) or []
            split_gap = self._caption_split_gap_row(row, original_group)
            if split_gap is not None:
                split_gap_rows.append(split_gap)

            planned_caption = caption_by_id.get(caption_id)
            expected_word_ids = {str(word_id) for word_id in (planned_caption.word_ids if planned_caption is not None else [])}
            covered_word_ids = {
                str(word_id)
                for segment in canonical_group
                for word_id in ((segment.get("_v21_audio_coverage") or {}).get("word_ids") or [])
                if isinstance(segment.get("_v21_audio_coverage"), dict)
            }
            missing_word_ids = sorted(expected_word_ids - covered_word_ids)
            if missing_word_ids:
                uncovered_word_rows.append(
                    {
                        "caption_id": caption_id,
                        "missing_word_ids": missing_word_ids,
                        "missing_word_count": len(missing_word_ids),
                    }
                )

        canonical_timeline = self._actual_final_timeline_from_video_rows(canonical_video_segments, run_report.final_timeline)
        canonical_alignment_timeline = self._actual_timeline_with_caption_split_containers(canonical_timeline, caption_rows)
        canonical_caption_units = self._actual_caption_units_from_text_rows(
            caption_rows,
            actual_timeline=canonical_alignment_timeline,
            run_report=run_report,
        )
        canonical_caption_alignment = build_caption_alignment_report(
            final_timeline=canonical_alignment_timeline,
            captions=canonical_caption_units,
            visible_caption_track_count=1 if canonical_caption_units else 0,
            caption_lane_count=1 if canonical_caption_units else 0,
        )
        caption_words_not_covered_count = sum(int(row["missing_word_count"]) for row in uncovered_word_rows)
        failure_reasons: list[str] = []
        if captions_after_final_video_end:
            failure_reasons.append("captions_after_final_video_end")
        if gap_count:
            failure_reasons.append("post_write_video_target_gaps")
        if caption_video_drift_rows:
            failure_reasons.append("caption_video_drift_after_canonical_save")
        if split_caption_mismatch_rows:
            failure_reasons.append("split_caption_container_mismatch")
        if split_gap_rows:
            failure_reasons.append("caption_crosses_video_split_gap")
        if caption_words_not_covered_count:
            failure_reasons.append("caption_words_not_covered_by_actual_video")
        if not bool(canonical_caption_alignment.get("gate_passed")):
            failure_reasons.append("canonical_caption_alignment_gate_failed")
        gate_passed = not failure_reasons
        return {
            "gate_passed": gate_passed,
            "jianying_canonical_timeline_sync_gate_passed": gate_passed,
            "blocker_codes": [] if gate_passed else ["V21_JIANYING_CANONICAL_TIMELINE_SYNC_FAILED"],
            "failure_reasons": failure_reasons,
            "final_video_end_us": final_video_end_us,
            "max_caption_end_us": max_caption_end_us,
            "captions_after_final_video_end_count": len(captions_after_final_video_end),
            "captions_after_final_video_end": captions_after_final_video_end[:20],
            "post_write_video_target_gap_count_gt_300ms": gap_count,
            "post_write_total_video_target_gap_us": total_gap_us,
            "post_write_video_target_gap_threshold_us": 300_000,
            "caption_video_drift_count": len(caption_video_drift_rows),
            "max_caption_video_drift_us": max((int(row["drift_us"]) for row in caption_video_drift_rows), default=0),
            "caption_video_drift_rows": caption_video_drift_rows[:20],
            "split_caption_container_mismatch_count": len(split_caption_mismatch_rows),
            "split_caption_container_mismatch_rows": split_caption_mismatch_rows[:20],
            "caption_crosses_video_split_gap_count": len(split_gap_rows),
            "caption_crosses_video_split_gap_rows": split_gap_rows[:20],
            "caption_words_not_covered_by_actual_video_count": caption_words_not_covered_count,
            "caption_words_not_covered_by_actual_video_rows": uncovered_word_rows[:20],
            "canonical_caption_alignment_gate_passed": bool(canonical_caption_alignment.get("gate_passed")),
            "canonical_caption_alignment_report": canonical_caption_alignment,
            "canonical_video_row_count": len(canonical_video_segments),
            "canonical_caption_row_count": len(caption_rows),
            "original_video_rows": original_video_rows[:20],
        }

    def _video_rows_by_caption_id(self, video_segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        rows_by_caption: dict[str, list[dict[str, Any]]] = {}
        for segment in video_segments:
            coverage = segment.get("_v21_audio_coverage") if isinstance(segment.get("_v21_audio_coverage"), dict) else {}
            caption_id = str(coverage.get("caption_id") or "")
            if not caption_id:
                continue
            rows_by_caption.setdefault(caption_id, []).append(segment)
        return rows_by_caption

    def _caption_split_gap_row(
        self,
        caption_row: dict[str, Any],
        video_segments: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if len(video_segments) <= 1:
            return None
        ordered = sorted(
            video_segments,
            key=lambda segment: (
                self._timerange_start(segment.get("target_timerange")),
                self._timerange_start(segment.get("target_timerange")) + self._timerange_duration(segment.get("target_timerange")),
            ),
        )
        caption_start = int(caption_row.get("target_start_us") or 0)
        caption_end = int(caption_row.get("target_end_us") or 0)
        previous_end = self._timerange_start(ordered[0].get("target_timerange")) + self._timerange_duration(ordered[0].get("target_timerange"))
        for segment in ordered[1:]:
            current_start = self._timerange_start(segment.get("target_timerange"))
            current_end = current_start + self._timerange_duration(segment.get("target_timerange"))
            if current_start > previous_end and self._ranges_overlap(caption_start, caption_end, previous_end, current_start):
                return {
                    "caption_id": str(caption_row.get("caption_id") or ""),
                    "caption_start_us": caption_start,
                    "caption_end_us": caption_end,
                    "gap_start_us": previous_end,
                    "gap_end_us": current_start,
                    "gap_us": current_start - previous_end,
                }
            previous_end = max(previous_end, current_end)
        return None

    def _actual_source_interval_rows(
        self,
        actual_video_segments: list[dict[str, Any]],
        planned_segments: list[FinalTimelineSegment],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(actual_video_segments):
            planned = planned_segments[index] if index < len(planned_segments) else None
            coverage = row.get("_v21_audio_coverage") if isinstance(row.get("_v21_audio_coverage"), dict) else {}
            source_start, source_end, spoken_start, spoken_end = self._actual_source_interval_for_video_row(row, planned)
            rows.append(
                {
                    "video_segment_id": str(row.get("id") or f"actual_video_{index + 1:06d}"),
                    "timeline_segment_id": str(coverage.get("timeline_segment_id") or (planned.segment_id if planned is not None else "")),
                    "caption_id": str(coverage.get("caption_id") or ""),
                    "source_start_us": source_start,
                    "source_end_us": source_end,
                    "spoken_source_start_us": spoken_start,
                    "spoken_source_end_us": spoken_end,
                }
            )
        return rows

    def _actual_source_interval_for_video_row(
        self,
        row: dict[str, Any],
        planned: FinalTimelineSegment | None,
    ) -> tuple[int, int, int, int]:
        coverage = row.get("_v21_audio_coverage") if isinstance(row.get("_v21_audio_coverage"), dict) else {}
        if coverage:
            source_start = int(coverage.get("source_start_us") or 0)
            source_end = int(coverage.get("source_end_us") or source_start)
            spoken_start = int(coverage.get("spoken_source_start_us") or source_start)
            spoken_end = int(coverage.get("spoken_source_end_us") or source_end)
            return source_start, source_end, spoken_start, spoken_end
        if planned is None:
            start = self._timerange_start(row.get("source_timerange"))
            end = start + self._timerange_duration(row.get("source_timerange"))
            return start, end, start, end
        spoken_start = int(planned.spoken_source_start_us if planned.spoken_source_start_us is not None else planned.source_start_us)
        spoken_end = int(planned.spoken_source_end_us if planned.spoken_source_end_us is not None else planned.source_end_us)
        clip_start = int(planned.clip_source_start_us if planned.clip_source_start_us is not None else planned.source_start_us)
        clip_end = int(planned.clip_source_end_us if planned.clip_source_end_us is not None else planned.source_end_us)
        projection = row.get("_safe_handle_projection") if isinstance(row.get("_safe_handle_projection"), dict) else {}
        if projection and bool(projection.get("safe_handle_policy_enabled")):
            source_start = clip_start if int(projection.get("lead_handle_applied_us") or 0) > 0 else spoken_start
            source_end = clip_end if int(projection.get("tail_handle_applied_us") or 0) > 0 else spoken_end
            return source_start, source_end, spoken_start, spoken_end
        return int(planned.source_start_us), int(planned.source_end_us), spoken_start, spoken_end

    def _ranges_overlap(self, left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
        return int(left_start) < int(right_end) and int(left_end) > int(right_start)

    def _segment_timerange_signature(self, row: dict[str, Any], *, include_source: bool = True) -> tuple[int, ...]:
        target = row.get("target_timerange")
        values = [
            self._timerange_start(target),
            self._timerange_duration(target),
        ]
        if include_source:
            source = row.get("source_timerange")
            values.extend([self._timerange_start(source), self._timerange_duration(source)])
        return tuple(values)

    def _plan_quality_audit(self, run_report: RunReport) -> dict[str, Any]:
        validator = run_report.validator_report or {}
        quality = validator.get("quality_gate_report") if isinstance(validator.get("quality_gate_report"), dict) else {}
        visual = validator.get("visual_pacing_gate") if isinstance(validator.get("visual_pacing_gate"), dict) else {}
        caption = validator.get("caption_alignment_gate") if isinstance(validator.get("caption_alignment_gate"), dict) else {}
        caption_repeat = validator.get("final_caption_visible_repeat_gate") if isinstance(validator.get("final_caption_visible_repeat_gate"), dict) else {}
        if not caption_repeat and isinstance(quality.get("final_caption_visible_repeat_gate"), dict):
            caption_repeat = quality.get("final_caption_visible_repeat_gate") or {}
        failures: list[str] = []
        quality_passed = bool(quality.get("gate_passed"))
        visual_passed = bool(visual.get("gate_passed")) and bool(visual.get("visual_pacing_executed")) and bool(
            visual.get("visual_merge_safety_gate_passed", True)
        )
        caption_passed = bool(caption.get("gate_passed")) and bool(caption.get("caption_gui_track_gate_passed", True)) and bool(
            caption.get("subtitle_readability_gate_passed", True)
        )
        caption_repeat_passed = bool(caption_repeat.get("gate_passed")) if caption_repeat else False
        if not quality_passed:
            failures.append("prewrite_quality_gate_not_passed")
        if not visual_passed:
            failures.append("prewrite_visual_pacing_gate_not_passed")
        if not caption_passed:
            failures.append("prewrite_caption_gui_readability_or_alignment_gate_not_passed")
        if not caption_repeat_passed:
            failures.append("prewrite_final_caption_visible_repeat_gate_not_passed")
        return {
            "prewrite_quality_gate_passed": quality_passed,
            "actual_visual_pacing_gate_passed": visual_passed,
            "actual_caption_gui_readability_gate_passed": caption_passed,
            "actual_caption_alignment_gate_passed": bool(caption.get("gate_passed")),
            "actual_final_caption_visible_repeat_gate_passed": caption_repeat_passed,
            "plan_quality_failure_reasons": failures,
        }

    def _select_subtitle_track_id(self, data: dict[str, Any], text_segments: list[dict[str, Any]], template_material_ids: set[str]) -> str:
        if not template_material_ids:
            raise WritebackError("V21_WRITEBACK_SUBTITLE_TRACK_NOT_FOUND", "canonical template report has no subtitle candidate material ids")
        counts: dict[str, int] = {}
        for segment in text_segments:
            material_id = str(segment.get("material_id") or segment.get("materialId") or "")
            if material_id not in template_material_ids:
                continue
            track_id = str(segment.get("track_id") or "")
            segment_id = str(segment.get("id") or "")
            if not track_id or not segment_id or not material_id:
                continue
            counts[track_id] = counts.get(track_id, 0) + 1
        if not counts:
            raise WritebackError(
                "V21_WRITEBACK_SUBTITLE_TRACK_NOT_FOUND",
                "subtitle candidate materials are not bound to a text track",
                {"candidate_material_ids": sorted(template_material_ids)},
            )
        max_count = max(counts.values())
        winners = sorted(track_id for track_id, count in counts.items() if count == max_count)
        if len(winners) != 1:
            raise WritebackError(
                "V21_WRITEBACK_SUBTITLE_TRACK_NOT_UNIQUE",
                "subtitle-bound text segments map to multiple equally likely text tracks",
                {"candidate_track_counts": counts},
            )
        track_id = winners[0]
        track = self._track_by_id(data, track_id)
        if track is None or not self._is_text_track(track):
            raise WritebackError(
                "V21_WRITEBACK_SUBTITLE_TRACK_NOT_FOUND",
                "selected subtitle-bound track_id is not a text/subtitle track",
                {"selected_text_track_id": track_id},
            )
        return track_id

    def _template_candidate_material_ids(self, run_report: RunReport) -> set[str]:
        template_report = (run_report.material_write_plan or {}).get("template_report") or {}
        candidate_ids = {
            str(value)
            for value in template_report.get("candidate_material_ids") or []
            if str(value or "")
        }
        representative = str(template_report.get("representative_material_id") or "")
        canonical = str((run_report.material_write_plan or {}).get("canonical_caption_template_id") or "")
        candidate_ids.update(value for value in (representative, canonical) if value)
        return candidate_ids

    def _subtitle_bound_track_ids(self, text_segments: list[dict[str, Any]], template_material_ids: set[str]) -> set[str]:
        track_ids = {
            str(row.get("track_id") or "")
            for row in text_segments
            if str(row.get("material_id") or row.get("materialId") or "") in template_material_ids and str(row.get("track_id") or "")
        }
        if not track_ids:
            raise WritebackError(
                "V21_WRITEBACK_SUBTITLE_TRACK_NOT_FOUND",
                "subtitle candidate materials are not bound to any text track",
                {"candidate_material_ids": sorted(template_material_ids)},
            )
        return track_ids

    def _classified_text_segments_by_track(
        self,
        text_segments: list[dict[str, Any]],
        subtitle_track_ids: set[str],
        template_material_ids: set[str],
        text_material_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        by_track: dict[str, list[dict[str, Any]]] = {track_id: [] for track_id in subtitle_track_ids}
        for row in text_segments:
            track_id = str(row.get("track_id") or "")
            if track_id not in subtitle_track_ids:
                continue
            classified = self._classify_text_segment(row, template_material_ids, text_material_by_id)
            by_track.setdefault(track_id, []).append(classified)
        if not any(
            classified["classification"] == "confirmed_subtitle_bound"
            for rows in by_track.values()
            for classified in rows
        ):
            raise WritebackError(
                "V21_WRITEBACK_OLD_SUBTITLE_SEGMENTS_UNBOUND",
                "no old subtitle-bound text segments matched canonical template candidates",
                {"subtitle_track_ids": sorted(subtitle_track_ids), "candidate_material_ids": sorted(template_material_ids)},
            )
        return by_track

    def _classify_text_segment(
        self,
        segment: dict[str, Any],
        template_material_ids: set[str],
        text_material_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        material_id = str(segment.get("material_id") or segment.get("materialId") or "")
        material = text_material_by_id.get(material_id) or {}
        text = self._text_material_text(material)
        if not material_id:
            classification = "unknown_unsafe"
            reason = "text segment has no material_id"
        elif material_id in template_material_ids:
            classification = "confirmed_subtitle_bound"
            reason = "material_id_matches_canonical_subtitle_template"
        elif self._is_confirmed_non_subtitle_text(segment, material):
            classification = "confirmed_non_subtitle"
            reason = "segment_or_material_metadata_marks_non_subtitle_text"
        else:
            classification = "unknown_unsafe"
            reason = "text segment is on a subtitle-bound track but lacks subtitle or non-subtitle metadata"
        return {
            "classification": classification,
            "reason": reason,
            "segment": segment,
            "material": material,
            "text": text,
        }

    def _is_confirmed_non_subtitle_text(self, segment: dict[str, Any], material: dict[str, Any]) -> bool:
        non_subtitle_tokens = ("title", "callout", "overlay", "note", "sticker", "label")
        explicit_false_keys = ("is_subtitle", "is_caption", "subtitle", "caption")
        for row in (segment, material):
            for key in explicit_false_keys:
                if row.get(key) is False:
                    return True
        metadata_values = [
            segment.get("id"),
            segment.get("track_id"),
            segment.get("type"),
            segment.get("role"),
            segment.get("name"),
            segment.get("category"),
            material.get("id"),
            material.get("type"),
            material.get("role"),
            material.get("name"),
            material.get("category"),
        ]
        for value in metadata_values:
            text = str(value or "").lower()
            if any(self._metadata_token_matches(text, token) for token in non_subtitle_tokens):
                return True
        return False

    def _metadata_token_matches(self, text: str, token: str) -> bool:
        normalized = "".join(char if char.isalnum() else "_" for char in text.lower())
        parts = [part for part in normalized.split("_") if part]
        return token in parts

    def _text_material_text(self, material: dict[str, Any]) -> str:
        for key in ("text", "recognize_text"):
            value = material.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for key in ("content", "base_content"):
            value = material.get(key)
            if isinstance(value, str):
                try:
                    payload = json.loads(value)
                except json.JSONDecodeError:
                    continue
            elif isinstance(value, dict):
                payload = value
            else:
                continue
            if isinstance(payload, dict):
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    return text
        return ""

    def _old_subtitle_segments_by_track(
        self,
        text_segments: list[dict[str, Any]],
        subtitle_track_ids: set[str],
        template_material_ids: set[str],
    ) -> dict[str, list[dict[str, Any]]]:
        by_track: dict[str, list[dict[str, Any]]] = {track_id: [] for track_id in subtitle_track_ids}
        for row in text_segments:
            track_id = str(row.get("track_id") or "")
            if track_id not in subtitle_track_ids:
                continue
            material_id = str(row.get("material_id") or row.get("materialId") or "")
            if not material_id:
                raise WritebackError(
                    "V21_WRITEBACK_OLD_SUBTITLE_SEGMENTS_UNBOUND",
                    "subtitle-bound text segment has no material_id",
                    {"selected_text_track_id": track_id, "segment_id": str(row.get("id") or "")},
                )
            if material_id in template_material_ids:
                by_track.setdefault(track_id, []).append(row)
        if not any(by_track.values()):
            raise WritebackError(
                "V21_WRITEBACK_OLD_SUBTITLE_SEGMENTS_UNBOUND",
                "no old subtitle-bound text segments matched canonical template candidates",
                {"subtitle_track_ids": sorted(subtitle_track_ids), "candidate_material_ids": sorted(template_material_ids)},
            )
        return by_track

    def _select_video_track_id_from_templates(self, used_templates: list[dict[str, Any]], run_report: RunReport) -> str:
        if not used_templates or len(used_templates) != len(run_report.final_timeline):
            raise WritebackError(
                "V21_WRITEBACK_SOURCE_SEGMENT_TEMPLATE_MISSING",
                "final_timeline source templates could not be fully resolved",
                {"final_timeline_segment_count": len(run_report.final_timeline), "resolved_source_segment_template_count": len(used_templates)},
            )
        track_ids = {
            str(template.get("track_id") or "")
            for template in used_templates
        }
        if "" in track_ids:
            raise WritebackError(
                "V21_WRITEBACK_MAIN_VIDEO_TRACK_NOT_FOUND",
                "source segment templates do not include track_id",
                {"used_source_segment_ids": sorted(str(template.get("id") or "") for template in used_templates)},
            )
        if len(track_ids) != 1:
            raise WritebackError(
                "V21_WRITEBACK_MULTIPLE_SOURCE_VIDEO_TRACKS_UNSUPPORTED",
                "final_timeline uses source segments from multiple video tracks",
                {"used_video_track_ids": sorted(track_ids)},
            )
        return next(iter(track_ids))

    def _track_by_id(self, data: dict[str, Any], track_id: str) -> dict[str, Any] | None:
        for track in data.get("tracks") or []:
            if not isinstance(track, dict):
                continue
            if str(track.get("id") or "") == track_id:
                return track
        return None

    def _is_text_track(self, track: dict[str, Any]) -> bool:
        track_type = str(track.get("type") or track.get("track_type") or "").lower()
        return "text" in track_type or "subtitle" in track_type

    def _is_video_track(self, track: dict[str, Any]) -> bool:
        track_type = str(track.get("type") or track.get("track_type") or "").lower()
        return "video" in track_type

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
        track_type = str(track.get("type") or track.get("track_type") or "").lower()
        return token in track_type

    def _segment_speed(self, segment: dict[str, Any], draft_data: dict[str, Any]) -> float:
        speed_report = segment.get("_resolved_speed_report")
        if isinstance(speed_report, dict) and speed_report.get("speed_safe") and speed_report.get("detected_speed") is not None:
            return float(speed_report.get("detected_speed") or 1.0)
        return SpeedResolver(draft_data).resolve(segment).speed

    def _timerange_start(self, value: Any) -> int:
        return int(value.get("start") or 0) if isinstance(value, dict) else 0

    def _timerange_duration(self, value: Any) -> int:
        return int(value.get("duration") or 0) if isinstance(value, dict) else 0

    def _display_to_material_delta(self, display_delta_us: int, speed: float | int | str | None) -> int:
        return int(round(int(display_delta_us) * float(speed or 1.0)))

    def _source_timeline_to_material_time(
        self,
        source_timeline_time_us: int,
        segment_target_start_us: int,
        segment_source_start_us: int,
        speed: float | int | str | None,
    ) -> int:
        display_offset = int(source_timeline_time_us) - int(segment_target_start_us)
        return int(segment_source_start_us) + self._display_to_material_delta(display_offset, speed)

    def _snapshot_targets(self, targets: list[Path], run_dir: Path) -> list[dict[str, Any]]:
        snapshot_dir = run_dir / "writeback_target_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshots: list[dict[str, Any]] = []
        for index, target in enumerate(targets, start=1):
            target_path = Path(target)
            existed = target_path.exists()
            snapshot_path = snapshot_dir / f"target_{index:02d}.bak"
            if existed:
                shutil.copyfile(target_path, snapshot_path)
            snapshots.append(
                {
                    "target": target_path,
                    "existed": existed,
                    "snapshot": snapshot_path if existed else None,
                }
            )
        return snapshots

    def _restore_target_snapshots(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        restored: dict[str, bool] = {}
        for row in snapshots:
            target = Path(row["target"])
            try:
                if bool(row.get("existed")):
                    snapshot = Path(row["snapshot"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(snapshot, target)
                    restored[str(target)] = target.exists() and target.stat().st_size == snapshot.stat().st_size
                else:
                    if target.exists():
                        target.unlink()
                    restored[str(target)] = not target.exists()
            except Exception as exc:
                restored[str(target)] = False
                errors.append({"target": str(target), "error": str(exc)})
        return {
            "rollback_performed": True,
            "rollback_success": bool(restored) and all(restored.values()) and not errors,
            "rollback_target_results": restored,
            "rollback_errors": errors,
        }

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
        metadata = real_draft_result.metadata or {}
        timeline_dir = Path(str(metadata.get("timeline_dir") or "")) if metadata.get("timeline_dir") else None
        draft_content_path = Path(str(metadata.get("draft_content_path") or "")) if metadata.get("draft_content_path") else None
        template_path = Path(str(metadata.get("template_path") or "")) if metadata.get("template_path") else None
        return {
            "writeback_attempted": bool(writeback_attempted),
            "writeback_success": False,
            "commit_performed": False,
            "encrypt_success": False,
            "WRITE_SUCCESS": False,
            "ENCRYPT_SUCCESS": False,
            "draft_dir": str(draft_dir or ""),
            "timeline_id": str(metadata.get("timeline_id") or ""),
            "timeline_dir": str(timeline_dir or ""),
            "draft_content_path": str(draft_content_path or ""),
            "template_path": str(template_path or ""),
            "target_writes": {},
            "selected_text_track_id": None,
            "selected_video_track_id": None,
            "sacrificial_write_override_used": bool(sacrificial_write_override_used),
            "postwrite_decrypt_skipped_for_sacrificial_draft": bool(sacrificial_write_override_used),
            "borrowed_v20_low_level_io_reference": True,
            **SOURCE_TEMPLATE_REPORT_DEFAULTS,
        }

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
        new_caption_material_ids = {
            str(row.get("id") or "")
            for row in run_report.material_write_plan.get("materials") or []
            if str(row.get("id") or "")
        }
        visible_caption_track_count = 0
        old_residue_track_ids: set[str] = set()
        overlapping_caption_segments_count = 0
        selected_track_total_segment_count = 0
        selected_canonical_caption_segment_count = 0
        canonical_caption_segment_count = 0
        visible_text_rows = self._visible_text_rows(data)
        classified_text_rows = self._classified_actual_text_rows(
            visible_text_rows,
            expected_segment_ids={
                str(row.get("id") or "")
                for row in run_report.material_write_plan.get("segments") or []
                if isinstance(row, dict) and str(row.get("id") or "")
            },
            expected_material_ids=new_caption_material_ids,
            template_material_ids=self._template_candidate_material_ids(run_report),
        )
        residue_text_rows = [row for row in classified_text_rows if row["classification"] == "old_subtitle_residue"]
        for track in data.get("tracks") or []:
            if not isinstance(track, dict) or not self._is_text_track(track):
                continue
            track_id = str(track.get("id") or "")
            track_segments = [segment for segment in track.get("segments") or [] if isinstance(segment, dict)]
            caption_segments = [
                segment
                for segment in track_segments
                if str(segment.get("material_id") or segment.get("materialId") or "") in new_caption_material_ids
            ]
            if caption_segments:
                visible_caption_track_count += 1
            canonical_caption_segment_count += len(caption_segments)
            if track_id == selected_text_track_id:
                selected_track_total_segment_count = len(track_segments)
                selected_canonical_caption_segment_count = len(caption_segments)
            if any(str(segment.get("material_id") or segment.get("materialId") or "") in old_subtitle_material_ids for segment in track_segments) or any(
                str(row.get("track_id") or "") == track_id for row in residue_text_rows
            ):
                old_residue_track_ids.add(track_id)
            overlapping_caption_segments_count += self._overlap_count(caption_segments)
        metrics = build_rough_cut_quality_metrics(
            final_timeline=run_report.final_timeline,
            captions=run_report.captions,
            material_write_plan=run_report.material_write_plan,
            visible_caption_track_count=visible_caption_track_count,
            old_subtitle_residue_track_count=len(old_residue_track_ids),
            overlapping_caption_segments_count=overlapping_caption_segments_count,
        )
        metrics["canonical_caption_segment_count"] = canonical_caption_segment_count
        metrics["selected_canonical_caption_segment_count"] = selected_canonical_caption_segment_count
        metrics["selected_text_track_total_segment_count"] = selected_track_total_segment_count
        metrics["preserved_non_subtitle_text_segment_count"] = int(preserved_non_subtitle_text_segment_count)
        metrics["old_subtitle_residue_count"] = len(residue_text_rows)
        metrics["non_subtitle_text_tracks_preserved"] = True
        metrics["non_subtitle_text_segments_preserved"] = True
        metrics["selected_canonical_caption_segments_match_captions"] = selected_canonical_caption_segment_count == metrics["caption_count"]
        metrics["canonical_caption_segments_match_captions"] = canonical_caption_segment_count == metrics["caption_count"]
        metrics["selected_track_total_segment_count_allows_non_subtitle"] = selected_track_total_segment_count >= selected_canonical_caption_segment_count
        metrics["selected_canonical_subtitle_track_segment_count"] = selected_canonical_caption_segment_count
        metrics["selected_canonical_subtitle_track_matches_captions"] = metrics["selected_canonical_caption_segments_match_captions"]
        return metrics

    def _assert_writeback_rough_cut_quality(self, metrics: dict[str, Any]) -> None:
        failed_checks: list[str] = []
        final_timeline_count = int(metrics.get("final_timeline_count") or 0)
        caption_count = int(metrics.get("caption_count") or 0)
        material_count = int(metrics.get("material_count") or 0)
        segment_count = int(metrics.get("segment_count") or 0)
        if final_timeline_count <= 0 or caption_count <= 0 or caption_count < final_timeline_count:
            failed_checks.append("video_caption_count_contract")
        if len({caption_count, material_count, segment_count}) != 1:
            failed_checks.append("caption_material_segment_count_mismatch")
        if int(metrics.get("visible_caption_track_count") or 0) != 1:
            failed_checks.append("visible_caption_track_count")
        if int(metrics.get("old_subtitle_residue_track_count") or 0) != 0:
            failed_checks.append("old_subtitle_residue_track_count")
        if int(metrics.get("old_subtitle_residue_count") or 0) != 0:
            failed_checks.append("old_subtitle_residue_count")
        if int(metrics.get("overlapping_caption_segments_count") or 0) != 0:
            failed_checks.append("overlapping_caption_segments_count")
        if int(metrics.get("canonical_caption_segment_count") or 0) != int(metrics.get("caption_count") or 0):
            failed_checks.append("canonical_caption_segment_count")
        if int(metrics.get("selected_canonical_caption_segment_count") or 0) != int(metrics.get("caption_count") or 0):
            failed_checks.append("selected_canonical_caption_segment_count")
        if int(metrics.get("target_gap_count") or 0) != 0:
            failed_checks.append("target_gap_count")
        if int(metrics.get("target_overlap_count") or 0) != 0:
            failed_checks.append("target_overlap_count")
        if failed_checks:
            raise WritebackError(
                "V21_WRITEBACK_ROUGH_CUT_QC_FAILED",
                "post-mutation writeback rough-cut QC failed",
                {"rough_cut_quality": metrics, "failed_checks": failed_checks},
            )

    def _overlap_count(self, segments: list[dict[str, Any]]) -> int:
        ordered = sorted(
            (
                {
                    "start": self._timerange_start(segment.get("target_timerange")),
                    "end": self._timerange_start(segment.get("target_timerange")) + self._timerange_duration(segment.get("target_timerange")),
                }
                for segment in segments
            ),
            key=lambda row: (row["start"], row["end"]),
        )
        count = 0
        previous_end = None
        for row in ordered:
            if previous_end is not None and row["start"] < previous_end:
                count += 1
            previous_end = row["end"]
        return count

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
