from __future__ import annotations

from copy import deepcopy
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_final_repeat_gate import build_final_repeat_gate_report
from aroll_hidden_audio_repeat_gate import build_hidden_audio_repeat_report
from aroll_safe_cut_boundary_gate import audit_safe_cut_boundaries
from aroll_subtitle_style_integrity_gate import audit_subtitle_style_integrity, text_content_schema_issues
from aroll_v21.ir.models import CanonicalSourceGraph, CaptionRenderUnit, DecisionPlan, FinalTimelineSegment
from aroll_v21.quality import (
    build_caption_alignment_report,
    build_final_repeat_convergence_report,
    build_quality_gate_report,
    build_visual_pacing_report,
)
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.final_timeline_quality_guard import build_final_timeline_quality_guard_report
from aroll_v21.validate.projected_write_view import ProjectedWriteViewError, build_projected_write_view
from aroll_v21.validate.rough_cut_quality import build_rough_cut_quality_metrics


PREWRITE_PROJECTED_WRITE_VIEW_BLOCKER_CODE = "V21_PREWRITE_PROJECTED_WRITE_VIEW_FAILED"


def _with_prewrite_projection_gate(report: dict[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    applied = bool(payload.get("prewrite_projected_write_view_applied"))
    gapless = bool(payload.get("prewrite_projection_video_write_plan_gapless")) if applied else False
    passed = bool(applied and gapless and not payload.get("prewrite_projection_error_code"))
    payload["prewrite_projected_write_view_gate_passed"] = passed
    payload["prewrite_projected_write_view_blocker_codes"] = [] if passed else [PREWRITE_PROJECTED_WRITE_VIEW_BLOCKER_CODE]
    return payload


def _caption_rows(captions: list[CaptionRenderUnit]) -> list[dict[str, Any]]:
    return [
        {
            "fragment_id": caption.caption_id,
            "fragment_text": caption.text,
            "text": caption.text,
            "word_ids": caption.word_ids,
            "target_start_us": caption.target_start_us,
            "target_duration_us": caption.target_end_us - caption.target_start_us,
            "source_subtitle_uids": caption.source_subtitle_uids,
        }
        for caption in captions
    ]


def _word_rows(source_graph: CanonicalSourceGraph, final_timeline: list[FinalTimelineSegment]) -> list[dict[str, Any]]:
    final_word_ids = {word_id for segment in final_timeline for word_id in segment.word_ids}
    return [
        {
            "word_id": word.word_id,
            "word_text": word.text,
            "start_us": word.source_start_us,
            "end_us": word.source_end_us,
            "subtitle_uid": word.subtitle_uid,
            "subtitle_index": word.subtitle_index,
        }
        for word in source_graph.words
        if word.word_id in final_word_ids
    ]


def _edl_rows(final_timeline: list[FinalTimelineSegment]) -> list[dict[str, Any]]:
    return [
        {
            "clip_id": segment.segment_id,
            "source_material_id": segment.source_material_id,
            "source_segment_id": segment.source_segment_id,
            "source_start_us": segment.source_start_us,
            "source_end_us": segment.source_end_us,
            "target_start_us": segment.target_start_us,
            "target_duration_us": segment.target_end_us - segment.target_start_us,
            "word_ids": segment.word_ids,
        }
        for segment in final_timeline
    ]


class ReadOnlyValidators:
    def run(
        self,
        *,
        source_graph: CanonicalSourceGraph,
        decision_plan: DecisionPlan,
        final_timeline: list[FinalTimelineSegment],
        captions: list[CaptionRenderUnit],
        material_write_plan: dict[str, Any],
        visual_pacing_report: dict[str, Any] | None = None,
        postwrite_materials: list[dict[str, Any]] | None = None,
        postwrite_mode: str = "auto",
    ) -> dict[str, Any]:
        before = {
            "final_timeline": deepcopy(final_timeline),
            "captions": deepcopy(captions),
            "material_write_plan": deepcopy(material_write_plan),
        }
        caption_rows = _caption_rows(captions)
        word_rows = _word_rows(source_graph, final_timeline)
        edl_rows = _edl_rows(final_timeline)
        projected_final_timeline = final_timeline
        projected_captions = captions
        projected_material_write_plan = material_write_plan
        prewrite_projection_report: dict[str, Any] = _with_prewrite_projection_gate(
            {"prewrite_projected_write_view_applied": False}
        )
        try:
            projected_view = build_projected_write_view(
                source_graph=source_graph,
                decision_plan=decision_plan,
                final_timeline=final_timeline,
                captions=captions,
                material_write_plan=material_write_plan,
            )
            projected_final_timeline = projected_view.final_timeline
            projected_captions = projected_view.captions
            projected_material_write_plan = projected_view.material_write_plan
            prewrite_projection_report = _with_prewrite_projection_gate(dict(projected_view.report))
        except ProjectedWriteViewError as exc:
            prewrite_projection_report = _with_prewrite_projection_gate(
                {
                    "prewrite_projected_write_view_applied": False,
                    "prewrite_projection_error_code": exc.code,
                    "prewrite_projection_error_message": exc.message,
                    "prewrite_projection_error_context": exc.context,
                }
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            prewrite_projection_report = _with_prewrite_projection_gate(
                {
                    "prewrite_projected_write_view_applied": False,
                    "prewrite_projection_error_code": type(exc).__name__,
                    "prewrite_projection_error_message": str(exc),
                }
            )
        residual_audit = {"issues": []}
        final_repeat = build_final_repeat_gate_report(residual_audit, caption_rows)
        final_repeat = self._final_repeat_semantic_status(final_repeat, decision_plan)
        hidden_repeat = build_hidden_audio_repeat_report(residual_audit, caption_rows, word_rows)
        hidden_repeat = self._modifier_redundancy_semantic_status(
            hidden_repeat,
            decision_plan,
            candidate_key="adjacent_modifier_semantic_redundancy_samples",
        )
        hidden_repeat["hidden_audio_repeat_gate_passed"] = len(hidden_repeat.get("blocking_issues") or []) == 0
        safe_cut = audit_safe_cut_boundaries(word_rows, final_edl=edl_rows)
        safe_cut["partial_multichar_cut_count"] = sum(
            1
            for row in safe_cut.get("unsafe_boundary_samples") or []
            if len(str(row.get("word_text") or "")) > 1
        )
        coverage = self._coverage(final_timeline, captions)
        style = self._style(source_graph, material_write_plan)
        rough_cut = self._rough_cut_quality(projected_final_timeline, projected_captions, projected_material_write_plan)
        rough_cut.update(prewrite_projection_report)
        postwrite = self._postwrite(material_write_plan, postwrite_materials, postwrite_mode=postwrite_mode)
        semantic = self._semantic(decision_plan)
        final_repeat_convergence = build_final_repeat_convergence_report(
            decision_trace=decision_plan.decision_trace,
            final_repeat_report=final_repeat,
        )
        visual_pacing = build_visual_pacing_report(
            final_timeline=final_timeline,
            captions=captions,
            executed=bool((visual_pacing_report or {}).get("visual_pacing_executed")),
            merge_report=visual_pacing_report,
        )
        final_caption_visible_repeat = build_final_caption_visible_repeat_gate(list(captions))
        caption_alignment = build_caption_alignment_report(
            final_timeline=projected_final_timeline,
            captions=projected_captions,
        )
        caption_alignment.update(prewrite_projection_report)
        final_timeline_quality_guard = build_final_timeline_quality_guard_report(
            source_graph=source_graph,
            final_timeline=final_timeline,
            captions=captions,
        )
        read_only_ok = before == {
            "final_timeline": final_timeline,
            "captions": captions,
            "material_write_plan": material_write_plan,
        }
        base_ok = all(
            [
                final_repeat.get("final_repeat_gate_passed"),
                hidden_repeat.get("hidden_audio_repeat_gate_passed"),
                safe_cut.get("safe_cut_boundary_gate_passed"),
                coverage.get("subtitle_coverage_gate_passed"),
                style.get("prewrite_style_gate_ok"),
                rough_cut.get("rough_cut_quality_gate_passed"),
                postwrite.get("postwrite_material_gate_ok"),
                semantic.get("semantic_final_review_validator_passed"),
                final_caption_visible_repeat.get("gate_passed"),
                caption_alignment.get("gate_passed"),
                final_timeline_quality_guard.get("gate_passed"),
                prewrite_projection_report.get("prewrite_projected_write_view_gate_passed"),
                read_only_ok,
            ]
        )
        quality_gate = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": [], "prewrite_pending": True},
            final_repeat_convergence_gate=final_repeat_convergence,
            final_caption_visible_repeat_gate=final_caption_visible_repeat,
            semantic_adjudication_gate=semantic,
            visual_pacing_gate=visual_pacing,
            caption_alignment_gate=caption_alignment,
            final_timeline_quality_guard_gate=final_timeline_quality_guard,
            prewrite_projection_gate=prewrite_projection_report,
            ready_for_user_manual_qc_preconditions_passed=base_ok,
        )
        ok = bool(base_ok and quality_gate.get("gate_passed"))
        return {
            "validators_read_only": read_only_ok,
            "validator_report_ok": bool(ok),
            "final_repeat_validator": final_repeat,
            "final_repeat_convergence_gate": final_repeat_convergence,
            "hidden_audio_repeat_validator": hidden_repeat,
            "safe_cut_validator": safe_cut,
            "subtitle_coverage_validator": coverage,
            "visual_pacing_gate": visual_pacing,
            "final_caption_visible_repeat_gate": final_caption_visible_repeat,
            "caption_alignment_gate": caption_alignment,
            "final_timeline_quality_guard_report": final_timeline_quality_guard,
            "subtitle_style_validator": style,
            "rough_cut_quality_validator": rough_cut,
            "prewrite_projected_write_view": prewrite_projection_report,
            "postwrite_material_validator": postwrite,
            "semantic_final_review_validator": semantic,
            "quality_gate_report": quality_gate,
        }

    def _coverage(self, final_timeline: list[FinalTimelineSegment], captions: list[CaptionRenderUnit]) -> dict[str, Any]:
        segment_ids = {segment.segment_id for segment in final_timeline}
        caption_segment_ids = {segment_id for caption in captions for segment_id in caption.timeline_segment_ids}
        final_word_ids = {str(word_id) for segment in final_timeline for word_id in segment.word_ids}
        caption_word_ids = {str(word_id) for caption in captions for word_id in caption.word_ids}
        missing_final_word_ids = sorted(final_word_ids - caption_word_ids)
        captions_have_words = all(bool(caption.word_ids) for caption in captions)
        return {
            "all_final_segments_have_word_ids": all(bool(segment.word_ids) for segment in final_timeline),
            "all_captions_derived_from_final_timeline": caption_segment_ids <= segment_ids and captions_have_words,
            "all_final_timeline_words_captioned": not missing_final_word_ids,
            "missing_caption_segment_ids": sorted(caption_segment_ids - segment_ids),
            "missing_final_timeline_caption_word_ids": missing_final_word_ids[:50],
            "missing_final_timeline_caption_word_count": len(missing_final_word_ids),
            "subtitle_coverage_gate_passed": all(bool(segment.word_ids) for segment in final_timeline)
            and caption_segment_ids <= segment_ids
            and captions_have_words
            and not missing_final_word_ids,
        }

    def _style(self, source_graph: CanonicalSourceGraph, material_write_plan: dict[str, Any]) -> dict[str, Any]:
        source_rows = [
            {"material": material, "segment": segment}
            for material in source_graph.text_materials
            for segment in (source_graph.text_segments or [{}])
            if not segment or str(segment.get("material_id") or "") == str(material.get("id") or "")
        ]
        report = audit_subtitle_style_integrity(
            source_rows,
            list(material_write_plan.get("segments") or []),
            list(material_write_plan.get("materials") or []),
        )
        return report | {
            "prewrite_style_gate_ok": bool(report.get("style_integrity_gate_passed")) and bool(material_write_plan.get("no_writer_fallback")),
            "no_writer_fallback": bool(material_write_plan.get("no_writer_fallback")),
            "writer_fallback_count": int(material_write_plan.get("writer_fallback_count") or 0),
            "giant_subtitle_count": int(report.get("style_safety_violation_count") or 0),
            "output_material_fingerprint_count": len(set(material_write_plan.get("output_material_fingerprints") or [])),
        }

    def _rough_cut_quality(
        self,
        final_timeline: list[FinalTimelineSegment],
        captions: list[CaptionRenderUnit],
        material_write_plan: dict[str, Any],
    ) -> dict[str, Any]:
        return build_rough_cut_quality_metrics(
            final_timeline=final_timeline,
            captions=captions,
            material_write_plan=material_write_plan,
            visible_caption_track_count=1,
            old_subtitle_residue_track_count=0,
            overlapping_caption_segments_count=0,
        )

    def _postwrite(
        self,
        material_write_plan: dict[str, Any],
        postwrite_materials: list[dict[str, Any]] | None,
        *,
        postwrite_mode: str = "auto",
    ) -> dict[str, Any]:
        if postwrite_mode == "auto":
            effective_mode = "actual_decrypt" if postwrite_materials is not None else "simulated"
        else:
            effective_mode = postwrite_mode
        if effective_mode == "unavailable":
            return {
                "postwrite_mode": "unavailable",
                "postwrite_decrypt_ok": False,
                "postwrite_verification_source": "unavailable",
                "real_uat_verified": False,
                "content_schema_error_count": 0,
                "content_schema_errors": [],
                "postwrite_material_gate_ok": False,
                "block_reason": "ACTUAL_POSTWRITE_DECRYPT_UNAVAILABLE",
            }
        if effective_mode == "skipped_for_sacrificial_draft":
            return {
                "postwrite_mode": "skipped_for_sacrificial_draft",
                "postwrite_decrypt_ok": False,
                "postwrite_verification_source": "skipped_for_sacrificial_draft",
                "real_uat_verified": False,
                "content_schema_error_count": 0,
                "content_schema_errors": [],
                "postwrite_material_gate_ok": True,
                "sacrificial_write_override_used": True,
                "postwrite_decrypt_skipped_for_sacrificial_draft": True,
                "postwrite_decrypt_skip_reason": "ACTUAL_POSTWRITE_DECRYPT_UNAVAILABLE",
                "ready_for_user_manual_qc": True,
            }
        materials = postwrite_materials if postwrite_materials is not None else list(material_write_plan.get("materials") or [])
        schema_issues = [
            {"material_id": row.get("id"), "issues": text_content_schema_issues(row)}
            for row in materials
            if text_content_schema_issues(row)
        ]
        return {
            "postwrite_mode": effective_mode,
            "postwrite_decrypt_ok": effective_mode == "actual_decrypt",
            "postwrite_verification_source": "actual_postwrite_materials"
            if effective_mode == "actual_decrypt"
            else f"{effective_mode}_plan_materials",
            "real_uat_verified": bool(effective_mode == "actual_decrypt"),
            "content_schema_error_count": len(schema_issues),
            "content_schema_errors": schema_issues[:100],
            "postwrite_material_gate_ok": len(schema_issues) == 0,
        }

    def _semantic(self, decision_plan: DecisionPlan) -> dict[str, Any]:
        human_review = [decision for decision in decision_plan.decisions if decision.requires_human_review]
        fatal_blockers = [blocker for blocker in decision_plan.blockers if blocker.severity == "fatal"]
        write_blockers = [blocker for blocker in decision_plan.blockers if blocker.severity == "write_blocker"]
        adjudication = dict(decision_plan.semantic_adjudication_report or {})
        semantic_request_count = int(adjudication.get("semantic_request_count") or len(decision_plan.semantic_request_payloads))
        semantic_request_unresolved_count = int(adjudication.get("semantic_request_unresolved_count") or decision_plan.semantic_unresolved_count)
        fatal_semantic_issue_count = int(adjudication.get("fatal_semantic_issue_count") or 0)
        semantic_gate_passed = bool(
            semantic_request_unresolved_count == 0
            and fatal_semantic_issue_count == 0
            and not human_review
            and not fatal_blockers
            and not write_blockers
            and decision_plan.write_allowed
        )
        return {
            "semantic_final_review_validator_passed": semantic_gate_passed,
            "semantic_adjudication_gate_passed": semantic_gate_passed,
            "semantic_request_count": semantic_request_count,
            "semantic_request_unresolved_count": semantic_request_unresolved_count,
            "fatal_semantic_issue_count": fatal_semantic_issue_count,
            "deepseek_provider_configured": bool(adjudication.get("deepseek_provider_configured")),
            "deepseek_provider_called_count": int(adjudication.get("deepseek_provider_called_count") or 0),
            "deterministic_baseline_refused_count": int(adjudication.get("deterministic_baseline_refused_count") or 0),
            "blocker_codes": list(adjudication.get("blocker_codes") or []),
            "unresolved_issue_ids": list(adjudication.get("unresolved_issue_ids") or []),
            "semantic_review_blocker_count": len(human_review) + len(write_blockers) + len(fatal_blockers),
            "semantic_unresolved_count": decision_plan.semantic_unresolved_count,
            "requires_human_review": decision_plan.requires_human_review,
            "write_allowed": semantic_gate_passed,
            "dry_run_continued_for_discovery": decision_plan.dry_run_continued_for_discovery,
            "human_review_decision_ids": [decision.decision_id for decision in human_review],
            "write_blocker_codes": [blocker.code for blocker in write_blockers],
            "fatal_blocker_codes": [blocker.code for blocker in fatal_blockers],
        }

    def _final_repeat_semantic_status(self, report: dict[str, Any], decision_plan: DecisionPlan) -> dict[str, Any]:
        report = self._modifier_redundancy_semantic_status(
            report,
            decision_plan,
            candidate_key="adjacent_modifier_semantic_redundancy_candidates",
        )
        candidates = list(report.get("final_target_repeat_candidates") or [])
        accepted = set(decision_plan.final_target_repeat_accepted_cluster_ids)
        unresolved = set(decision_plan.final_target_repeat_unresolved_cluster_ids)
        protected_text_pairs = self._protected_final_target_text_pairs(decision_plan)
        accepted_count = 0
        semantic_unresolved_count = 0
        blocking_medium_count = 0
        blocking_high_count = 0
        annotated = []
        for candidate in candidates:
            row = dict(candidate)
            cluster_id = self._final_target_cluster_id(row)
            confidence = str(row.get("confidence") or "")
            resolution = ""
            candidate_pair = self._final_target_candidate_text_pair(row)
            if cluster_id in accepted:
                accepted_count += 1
                resolution = "accepted_by_semantic_decision"
            elif candidate_pair is not None and candidate_pair in protected_text_pairs:
                accepted_count += 1
                resolution = "accepted_by_protected_semantic_bridge"
            elif cluster_id in unresolved:
                semantic_unresolved_count += 1
                resolution = "semantic_unresolved_write_blocker"
            elif confidence == "high":
                blocking_high_count += 1
                resolution = "fatal_unresolved_high"
            elif confidence == "medium":
                blocking_medium_count += 1
                resolution = "fatal_uncovered_medium"
            annotated.append(self._candidate_with_fields(row, {"v21_resolution": resolution} if resolution else {}))

        other_blocking_count = (
            int(report.get("final_text_repeat_high_count") or 0)
            + int(report.get("final_text_repeat_medium_count") or 0)
            + int(report.get("final_semantic_repeat_high_count") or 0)
            + int(report.get("final_hidden_word_repeat_high_count") or 0)
            + int(report.get("final_cjk_short_repeat_fatal_count") or 0)
            + int(report.get("adjacent_modifier_semantic_redundancy_fatal_count") or 0)
        )
        updated = dict(report)
        updated["final_target_repeat_candidates"] = annotated[:100]
        updated["final_target_repeat_accepted_count"] = accepted_count
        updated["final_target_repeat_semantic_unresolved_count"] = semantic_unresolved_count
        updated["final_target_repeat_high_count"] = blocking_high_count
        updated["final_target_repeat_medium_count"] = blocking_medium_count
        updated["final_repeat_gate_passed"] = (other_blocking_count + blocking_high_count + blocking_medium_count) == 0
        return updated

    def _final_target_cluster_id(self, candidate: dict[str, Any]) -> str:
        raw = str(candidate.get("cluster_id") or "")
        return raw if raw.startswith("final_target_repeat_") else f"final_target_repeat_{raw}"

    def _protected_final_target_text_pairs(self, decision_plan: DecisionPlan) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for row in decision_plan.decision_trace:
            if not isinstance(row, dict):
                continue
            if row.get("route") != "final_target_repeat":
                continue
            if row.get("decision") != "keep_all_protected_semantic_bridge":
                continue
            pair = self._text_pair_from_row(row)
            if pair is None:
                continue
            pairs.add(pair)
            pairs.add((pair[1], pair[0]))
        return pairs

    def _final_target_candidate_text_pair(self, candidate: dict[str, Any]) -> tuple[str, str] | None:
        items = [row for row in list(candidate.get("items") or []) if isinstance(row, dict)]
        if len(items) >= 2:
            left = normalize_text(str(items[0].get("text") or ""))
            right = normalize_text(str(items[1].get("text") or ""))
        else:
            candidates = [row for row in list(candidate.get("candidates") or []) if isinstance(row, dict)]
            if len(candidates) < 2:
                no_pair: tuple[str, str] | None = None
                return no_pair
            left = normalize_text(str(candidates[0].get("text") or ""))
            right = normalize_text(str(candidates[1].get("text") or ""))
        if not left or not right:
            no_pair: tuple[str, str] | None = None
            return no_pair
        return left, right

    def _text_pair_from_row(self, row: dict[str, Any]) -> tuple[str, str] | None:
        text_pair = [normalize_text(str(value or "")) for value in list(row.get("text_pair") or [])]
        if len(text_pair) >= 2 and text_pair[0] and text_pair[1]:
            return text_pair[0], text_pair[1]
        left = normalize_text(str(row.get("left_text") or ""))
        right = normalize_text(str(row.get("right_text") or ""))
        if not left or not right:
            no_pair: tuple[str, str] | None = None
            return no_pair
        return left, right

    def _modifier_redundancy_semantic_status(
        self,
        report: dict[str, Any],
        decision_plan: DecisionPlan,
        *,
        candidate_key: str,
    ) -> dict[str, Any]:
        accepted = set(decision_plan.modifier_redundancy_accepted_cluster_ids)
        unresolved = set(decision_plan.modifier_redundancy_unresolved_cluster_ids)
        candidates = list(report.get(candidate_key) or [])
        annotated = []
        accepted_count = 0
        semantic_unresolved_count = 0
        uncovered_fatal_count = 0
        resolved_texts: set[str] = set()
        fatal_index = 0
        for candidate in candidates:
            row = dict(candidate)
            is_modifier = str(row.get("type") or row.get("issue_type") or "") == "adjacent_modifier_semantic_redundancy"
            is_fatal = str(row.get("severity") or "fatal") == "fatal"
            if not is_modifier:
                annotated.append(row)
                continue
            cluster_id = self._modifier_cluster_id(row, fatal_index)
            fatal_index += 1
            resolution = ""
            if cluster_id in accepted:
                accepted_count += 1
                resolution = "accepted_by_semantic_decision"
                resolved_texts.update(self._modifier_issue_texts(row))
            elif cluster_id in unresolved:
                semantic_unresolved_count += 1
                resolution = "semantic_unresolved_write_blocker"
                resolved_texts.update(self._modifier_issue_texts(row))
            elif is_fatal:
                uncovered_fatal_count += 1
                resolution = "fatal_uncovered_modifier_redundancy"
            fields = {"cluster_id": cluster_id}
            if resolution:
                fields["v21_resolution"] = resolution
            annotated.append(self._candidate_with_fields(row, fields))

        filtered_blocking_issues = []
        for issue in report.get("blocking_issues") or []:
            if not isinstance(issue, dict):
                filtered_blocking_issues.append(issue)
                continue
            issue_type = str(issue.get("type") or issue.get("issue_type") or "")
            if issue_type in {"adjacent_modifier_semantic_redundancy", "word_timeline_repeated_island"} and self._modifier_issue_texts(issue) & resolved_texts:
                continue
            filtered_blocking_issues.append(issue)

        updated = dict(report)
        updated[candidate_key] = annotated[:100]
        updated["blocking_issues"] = filtered_blocking_issues[:100]
        if "issues" in updated:
            filtered_issues = []
            for issue in updated.get("issues") or []:
                if not isinstance(issue, dict):
                    filtered_issues.append(issue)
                    continue
                issue_type = str(issue.get("type") or issue.get("issue_type") or "")
                if issue_type in {"adjacent_modifier_semantic_redundancy", "word_timeline_repeated_island"} and self._modifier_issue_texts(issue) & resolved_texts:
                    continue
                filtered_issues.append(issue)
            updated["issues"] = filtered_issues[:100]
        if "word_timeline_repeated_island_count" in updated:
            updated["word_timeline_repeated_island_count"] = sum(
                1
                for issue in filtered_blocking_issues
                if isinstance(issue, dict) and str(issue.get("type") or issue.get("issue_type") or "") == "word_timeline_repeated_island"
            )
        updated["adjacent_modifier_semantic_redundancy_accepted_count"] = accepted_count
        updated["adjacent_modifier_semantic_redundancy_semantic_unresolved_count"] = semantic_unresolved_count
        updated["adjacent_modifier_semantic_redundancy_fatal_count"] = uncovered_fatal_count
        return updated

    def _modifier_cluster_id(self, candidate: dict[str, Any], offset: int) -> str:
        raw = str(candidate.get("cluster_id") or "")
        if raw:
            return raw
        return f"repeat_{2000 + offset:06d}"

    def _modifier_issue_texts(self, issue: dict[str, Any]) -> set[str]:
        values = {
            str(issue.get("text") or ""),
            str(issue.get("phrase") or ""),
            str(issue.get("fragment_text") or ""),
            str(issue.get("left_text") or "") + str(issue.get("right_text") or ""),
        }
        return {normalize_text(value) for value in values if len(normalize_text(value)) >= 2}

    def _candidate_with_fields(self, candidate: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
        return dict(candidate) | dict(fields)
