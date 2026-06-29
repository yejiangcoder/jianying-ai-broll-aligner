from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from aroll_v21.ir.models import Blocker
from aroll_v21.quality.final_timeline_repair_apply import recompute_final_timeline_safe_handles


@dataclass(frozen=True)
class QualityPipelineHooks:
    render_captions: Callable[[list[Any], Any], list[Any]]
    visual_pacing_normalize: Callable[[list[Any], Any], tuple[list[Any], dict[str, Any]]]
    repair_final_visible_caption_issues: Callable[..., Any]
    drop_deterministic_self_repair_aborted_segments: Callable[[list[Any], Any], list[Any]]
    drop_final_target_aborted_caption_restarts: Callable[[list[Any], list[Any], Any, Any], list[Any]]
    record_quality_mutation: Callable[..., dict[str, Any] | None]
    accept_pending_visual_pacing_recheck: Callable[[dict[str, Any] | None], None]
    combined_final_visible_repair_report: Callable[..., dict[str, Any]]
    reconcile_late_final_target_repeat_semantics: Callable[
        [list[Any], list[Any], Any, Any, list[dict[str, Any]]],
        tuple[list[Any], list[Any], list[Blocker]],
    ]
    quality_mutation_report_fields: Callable[[list[dict[str, Any]]], dict[str, Any]]
    final_timeline_state_signature: Callable[[list[Any]], tuple[Any, ...]]
    final_visible_state_signature: Callable[[list[Any], list[Any]], tuple[Any, ...]]
    sync_semantic_gate_with_final_output: Callable[[Any, list[Any], list[Any]], None]
    refresh_semantic_adjudication_report: Callable[[Any], None]


@dataclass(frozen=True)
class QualityPipelineResult:
    final_timeline: list[Any]
    captions: list[Any]
    visual_pacing_report: dict[str, Any]
    final_visible_repair_report: dict[str, Any]
    quality_mutations: list[dict[str, Any]]


class QualityPipeline:
    def __init__(self, hooks: QualityPipelineHooks) -> None:
        self.hooks = hooks

    def run(
        self,
        *,
        final_timeline: list[Any],
        source_graph: Any,
        decision_plan: Any,
        blockers: list[Blocker],
    ) -> QualityPipelineResult:
        final_timeline = self.hooks.drop_deterministic_self_repair_aborted_segments(final_timeline, decision_plan)
        quality_mutations: list[dict[str, Any]] = []
        initial_visual_before_timeline = list(final_timeline)
        initial_visual_before_captions = self.hooks.render_captions(final_timeline, source_graph)
        final_timeline, visual_pacing_report = self.hooks.visual_pacing_normalize(final_timeline, source_graph)
        captions = self.hooks.render_captions(final_timeline, source_graph)
        self.hooks.record_quality_mutation(
            quality_mutations,
            phase="visual_pacing.initial",
            rule_name="visual_pacing.normalize",
            before_timeline=initial_visual_before_timeline,
            before_captions=initial_visual_before_captions,
            after_timeline=final_timeline,
            after_captions=captions,
            source_graph=source_graph,
            action={"visual_pacing_executed": True},
            after_visual_pacing_report=visual_pacing_report,
            enforce_regression_guard=False,
        )
        before_final_target_cleanup_signature = self.hooks.final_timeline_state_signature(final_timeline)
        before_final_target_cleanup_timeline = list(final_timeline)
        before_final_target_cleanup_captions = list(captions)
        final_timeline = self.hooks.drop_final_target_aborted_caption_restarts(
            final_timeline,
            captions,
            source_graph,
            decision_plan,
        )
        if self.hooks.final_timeline_state_signature(final_timeline) != before_final_target_cleanup_signature:
            captions_after_final_target_cleanup = self.hooks.render_captions(final_timeline, source_graph)
            self.hooks.record_quality_mutation(
                quality_mutations,
                phase="final_target_cleanup.pre_repair",
                rule_name="_drop_final_target_aborted_caption_restarts",
                before_timeline=before_final_target_cleanup_timeline,
                before_captions=before_final_target_cleanup_captions,
                after_timeline=final_timeline,
                after_captions=captions_after_final_target_cleanup,
                source_graph=source_graph,
                action={"cleanup": "drop_final_target_aborted_caption_restarts"},
                enforce_regression_guard=False,
            )
            before_cleanup_visual_timeline = list(final_timeline)
            before_cleanup_visual_captions = list(captions_after_final_target_cleanup)
            final_timeline, visual_pacing_report = self.hooks.visual_pacing_normalize(final_timeline, source_graph)
            captions = self.hooks.render_captions(final_timeline, source_graph)
            self.hooks.record_quality_mutation(
                quality_mutations,
                phase="visual_pacing.after_final_target_cleanup",
                rule_name="visual_pacing.normalize",
                before_timeline=before_cleanup_visual_timeline,
                before_captions=before_cleanup_visual_captions,
                after_timeline=final_timeline,
                after_captions=captions,
                source_graph=source_graph,
                action={"visual_pacing_executed": True},
                after_visual_pacing_report=visual_pacing_report,
                enforce_regression_guard=False,
            )
        final_visible_repair_reports: list[dict[str, Any]] = []
        visual_pacing_rerun_after_final_repair_count = 0
        final_visible_repair_max_cycle_exhausted = False
        final_visible_repair_cycle_stop_reason = ""
        seen_final_visible_cycle_signatures: set[tuple[Any, ...]] = {
            self.hooks.final_visible_state_signature(final_timeline, captions)
        }
        max_final_visible_repair_cycles = 8
        for cycle_index in range(max_final_visible_repair_cycles):
            state_before_repair = self.hooks.final_visible_state_signature(final_timeline, captions)
            seen_final_visible_cycle_signatures.add(state_before_repair)
            timeline_before_repair = list(final_timeline)
            captions_before_repair = list(captions)
            final_visible_repair = self.hooks.repair_final_visible_caption_issues(
                final_timeline=final_timeline,
                captions=captions,
                source_graph=source_graph,
                render_captions=lambda timeline: self.hooks.render_captions(timeline, source_graph),
            )
            final_visible_repair_reports.append(final_visible_repair.report)
            final_timeline = final_visible_repair.final_timeline
            captions = final_visible_repair.captions
            state_after_repair = self.hooks.final_visible_state_signature(final_timeline, captions)
            repair_action_count = int(final_visible_repair.report.get("final_visible_repair_action_count") or 0)
            repair_mutation = self.hooks.record_quality_mutation(
                quality_mutations,
                phase="final_visible_repair.cycle",
                rule_name="repair_final_visible_caption_issues",
                before_timeline=timeline_before_repair,
                before_captions=captions_before_repair,
                after_timeline=final_timeline,
                after_captions=captions,
                source_graph=source_graph,
                action={
                    "cycle_index": cycle_index + 1,
                    "action_count": repair_action_count,
                    "stop_reason": str(final_visible_repair.report.get("final_visible_repair_stop_reason") or ""),
                },
            )
            self.hooks.accept_pending_visual_pacing_recheck(repair_mutation)
            if repair_mutation is not None and not bool(repair_mutation.get("accepted")):
                final_visible_repair_max_cycle_exhausted = True
                final_visible_repair_cycle_stop_reason = "quality_mutation_regression_detected"
                break
            if repair_action_count <= 0:
                break
            if state_after_repair in seen_final_visible_cycle_signatures:
                final_visible_repair_max_cycle_exhausted = True
                final_visible_repair_cycle_stop_reason = "repair_cycle_state_repeated"
                break
            seen_final_visible_cycle_signatures.add(state_after_repair)
            if cycle_index + 1 >= max_final_visible_repair_cycles:
                final_visible_repair_max_cycle_exhausted = True
                final_visible_repair_cycle_stop_reason = "max_repair_cycles_exhausted"
                break
            timeline_before_pacing_signature = self.hooks.final_timeline_state_signature(final_timeline)
            timeline_before_pacing = list(final_timeline)
            captions_before_pacing = list(captions)
            visual_pacing_report_before = dict(visual_pacing_report or {})
            final_timeline, visual_pacing_report = self.hooks.visual_pacing_normalize(final_timeline, source_graph)
            if self.hooks.final_timeline_state_signature(final_timeline) == timeline_before_pacing_signature:
                captions = captions_before_pacing
            else:
                captions = self.hooks.render_captions(final_timeline, source_graph)
            visual_pacing_rerun_after_final_repair_count += 1
            pacing_mutation = self.hooks.record_quality_mutation(
                quality_mutations,
                phase="visual_pacing.after_final_visible_repair",
                rule_name="visual_pacing.normalize",
                before_timeline=timeline_before_pacing,
                before_captions=captions_before_pacing,
                after_timeline=final_timeline,
                after_captions=captions,
                source_graph=source_graph,
                action={"cycle_index": cycle_index + 1, "visual_pacing_executed": True},
                after_visual_pacing_report=visual_pacing_report,
            )
            if pacing_mutation is not None and not bool(pacing_mutation.get("accepted")):
                final_timeline = timeline_before_pacing
                captions = captions_before_pacing
                visual_pacing_report = visual_pacing_report_before
                final_visible_repair_cycle_stop_reason = "visual_pacing_regression_reverted_after_final_visible_repair"
                break
            state_after_pacing = self.hooks.final_visible_state_signature(final_timeline, captions)
            if state_after_pacing != state_after_repair and state_after_pacing in seen_final_visible_cycle_signatures:
                final_visible_repair_max_cycle_exhausted = True
                final_visible_repair_cycle_stop_reason = "visual_pacing_reintroduced_seen_repair_state"
                break
            seen_final_visible_cycle_signatures.add(state_after_pacing)
        final_visible_repair_report = self.hooks.combined_final_visible_repair_report(
            final_visible_repair_reports,
            visual_pacing_rerun_after_final_repair_count=visual_pacing_rerun_after_final_repair_count,
            max_cycle_exhausted=final_visible_repair_max_cycle_exhausted,
            cycle_stop_reason=final_visible_repair_cycle_stop_reason,
            quality_mutations=quality_mutations,
        )
        if not final_visible_repair_max_cycle_exhausted:
            before_post_repair_final_target_cleanup_signature = self.hooks.final_timeline_state_signature(final_timeline)
            before_post_repair_final_target_cleanup_timeline = list(final_timeline)
            before_post_repair_final_target_cleanup_captions = list(captions)
            final_timeline = self.hooks.drop_final_target_aborted_caption_restarts(
                final_timeline,
                captions,
                source_graph,
                decision_plan,
            )
            if self.hooks.final_timeline_state_signature(final_timeline) != before_post_repair_final_target_cleanup_signature:
                captions_after_post_repair_cleanup = self.hooks.render_captions(final_timeline, source_graph)
                cleanup_mutation = self.hooks.record_quality_mutation(
                    quality_mutations,
                    phase="post_cleanup.final_target_restart",
                    rule_name="_drop_final_target_aborted_caption_restarts",
                    before_timeline=before_post_repair_final_target_cleanup_timeline,
                    before_captions=before_post_repair_final_target_cleanup_captions,
                    after_timeline=final_timeline,
                    after_captions=captions_after_post_repair_cleanup,
                    source_graph=source_graph,
                    action={"cleanup": "drop_final_target_aborted_caption_restarts"},
                )
                if cleanup_mutation is not None and not bool(cleanup_mutation.get("accepted")):
                    final_visible_repair_max_cycle_exhausted = True
                    final_visible_repair_cycle_stop_reason = "quality_mutation_regression_detected"
                before_post_cleanup_visual_timeline = list(final_timeline)
                before_post_cleanup_visual_captions = list(captions_after_post_repair_cleanup)
                final_timeline, visual_pacing_report = self.hooks.visual_pacing_normalize(final_timeline, source_graph)
                captions = self.hooks.render_captions(final_timeline, source_graph)
                visual_cleanup_mutation = self.hooks.record_quality_mutation(
                    quality_mutations,
                    phase="visual_pacing.post_cleanup",
                    rule_name="visual_pacing.normalize",
                    before_timeline=before_post_cleanup_visual_timeline,
                    before_captions=before_post_cleanup_visual_captions,
                    after_timeline=final_timeline,
                    after_captions=captions,
                    source_graph=source_graph,
                    action={"visual_pacing_executed": True},
                    after_visual_pacing_report=visual_pacing_report,
                )
                if visual_cleanup_mutation is not None and not bool(visual_cleanup_mutation.get("accepted")):
                    final_visible_repair_max_cycle_exhausted = True
                    final_visible_repair_cycle_stop_reason = "quality_mutation_regression_detected"
                post_cleanup_state = self.hooks.final_visible_state_signature(final_timeline, captions)
                if post_cleanup_state in seen_final_visible_cycle_signatures:
                    final_visible_repair_max_cycle_exhausted = True
                    final_visible_repair_cycle_stop_reason = "post_cleanup_reintroduced_seen_repair_state"
                else:
                    seen_final_visible_cycle_signatures.add(post_cleanup_state)
                if final_visible_repair_max_cycle_exhausted:
                    final_visible_repair_report = self.hooks.combined_final_visible_repair_report(
                        final_visible_repair_reports,
                        visual_pacing_rerun_after_final_repair_count=visual_pacing_rerun_after_final_repair_count,
                        max_cycle_exhausted=final_visible_repair_max_cycle_exhausted,
                        cycle_stop_reason=final_visible_repair_cycle_stop_reason,
                        quality_mutations=quality_mutations,
                    )
                else:
                    timeline_before_post_cleanup_repair = list(final_timeline)
                    captions_before_post_cleanup_repair = list(captions)
                    post_cleanup_repair = self.hooks.repair_final_visible_caption_issues(
                        final_timeline=final_timeline,
                        captions=captions,
                        source_graph=source_graph,
                        render_captions=lambda timeline: self.hooks.render_captions(timeline, source_graph),
                    )
                    final_visible_repair_reports.append(post_cleanup_repair.report)
                    final_timeline = post_cleanup_repair.final_timeline
                    captions = post_cleanup_repair.captions
                    post_cleanup_repair_action_count = int(
                        post_cleanup_repair.report.get("final_visible_repair_action_count") or 0
                    )
                    post_cleanup_repair_mutation = self.hooks.record_quality_mutation(
                        quality_mutations,
                        phase="final_visible_repair.post_cleanup",
                        rule_name="repair_final_visible_caption_issues",
                        before_timeline=timeline_before_post_cleanup_repair,
                        before_captions=captions_before_post_cleanup_repair,
                        after_timeline=final_timeline,
                        after_captions=captions,
                        source_graph=source_graph,
                        action={
                            "action_count": post_cleanup_repair_action_count,
                            "stop_reason": str(post_cleanup_repair.report.get("final_visible_repair_stop_reason") or ""),
                        },
                    )
                    self.hooks.accept_pending_visual_pacing_recheck(post_cleanup_repair_mutation)
                    if post_cleanup_repair_mutation is not None and not bool(post_cleanup_repair_mutation.get("accepted")):
                        final_visible_repair_max_cycle_exhausted = True
                        final_visible_repair_cycle_stop_reason = "quality_mutation_regression_detected"
                    post_cleanup_repair_state = self.hooks.final_visible_state_signature(final_timeline, captions)
                    if (
                        not final_visible_repair_max_cycle_exhausted
                        and post_cleanup_repair_action_count > 0
                        and post_cleanup_repair_state in seen_final_visible_cycle_signatures
                    ):
                        final_visible_repair_max_cycle_exhausted = True
                        final_visible_repair_cycle_stop_reason = "post_cleanup_repair_state_repeated"
                    else:
                        seen_final_visible_cycle_signatures.add(post_cleanup_repair_state)
                    if not final_visible_repair_max_cycle_exhausted and post_cleanup_repair_action_count > 0:
                        timeline_before_pacing_signature = self.hooks.final_timeline_state_signature(final_timeline)
                        timeline_before_post_cleanup_pacing = list(final_timeline)
                        state_before_post_cleanup_pacing = self.hooks.final_visible_state_signature(final_timeline, captions)
                        captions_before_pacing = list(captions)
                        final_timeline, visual_pacing_report = self.hooks.visual_pacing_normalize(final_timeline, source_graph)
                        if self.hooks.final_timeline_state_signature(final_timeline) == timeline_before_pacing_signature:
                            captions = captions_before_pacing
                        else:
                            captions = self.hooks.render_captions(final_timeline, source_graph)
                        visual_pacing_rerun_after_final_repair_count += 1
                        post_cleanup_pacing_mutation = self.hooks.record_quality_mutation(
                            quality_mutations,
                            phase="visual_pacing.after_post_cleanup_repair",
                            rule_name="visual_pacing.normalize",
                            before_timeline=timeline_before_post_cleanup_pacing,
                            before_captions=captions_before_pacing,
                            after_timeline=final_timeline,
                            after_captions=captions,
                            source_graph=source_graph,
                            action={"visual_pacing_executed": True},
                            after_visual_pacing_report=visual_pacing_report,
                        )
                        if post_cleanup_pacing_mutation is not None and not bool(post_cleanup_pacing_mutation.get("accepted")):
                            final_visible_repair_max_cycle_exhausted = True
                            final_visible_repair_cycle_stop_reason = "quality_mutation_regression_detected"
                        state_after_post_cleanup_pacing = self.hooks.final_visible_state_signature(final_timeline, captions)
                        if (
                            state_after_post_cleanup_pacing != state_before_post_cleanup_pacing
                            and state_after_post_cleanup_pacing in seen_final_visible_cycle_signatures
                        ):
                            final_visible_repair_max_cycle_exhausted = True
                            final_visible_repair_cycle_stop_reason = "post_cleanup_visual_pacing_reintroduced_seen_repair_state"
                        else:
                            seen_final_visible_cycle_signatures.add(state_after_post_cleanup_pacing)
                    final_visible_repair_report = self.hooks.combined_final_visible_repair_report(
                        final_visible_repair_reports,
                        visual_pacing_rerun_after_final_repair_count=visual_pacing_rerun_after_final_repair_count,
                        max_cycle_exhausted=final_visible_repair_max_cycle_exhausted,
                        cycle_stop_reason=final_visible_repair_cycle_stop_reason,
                        quality_mutations=quality_mutations,
                    )
        final_timeline, captions, late_final_target_blockers = self.hooks.reconcile_late_final_target_repeat_semantics(
            final_timeline,
            captions,
            source_graph,
            decision_plan,
            quality_mutations,
        )
        blockers.extend(late_final_target_blockers)
        if not final_visible_repair_max_cycle_exhausted:
            late_final_visible_repair = self.hooks.repair_final_visible_caption_issues(
                final_timeline=final_timeline,
                captions=captions,
                source_graph=source_graph,
                render_captions=lambda timeline: self.hooks.render_captions(timeline, source_graph),
            )
            late_repair_action_count = int(
                late_final_visible_repair.report.get("final_visible_repair_action_count") or 0
            )
            if late_repair_action_count > 0:
                timeline_before_late_repair = list(final_timeline)
                captions_before_late_repair = list(captions)
                final_visible_repair_reports.append(late_final_visible_repair.report)
                final_timeline = late_final_visible_repair.final_timeline
                captions = late_final_visible_repair.captions
                late_repair_mutation = self.hooks.record_quality_mutation(
                    quality_mutations,
                    phase="final_visible_repair.after_late_semantic_reconcile",
                    rule_name="repair_final_visible_caption_issues",
                    before_timeline=timeline_before_late_repair,
                    before_captions=captions_before_late_repair,
                    after_timeline=final_timeline,
                    after_captions=captions,
                    source_graph=source_graph,
                    action={
                        "action_count": late_repair_action_count,
                        "stop_reason": str(late_final_visible_repair.report.get("final_visible_repair_stop_reason") or ""),
                    },
                )
                self.hooks.accept_pending_visual_pacing_recheck(late_repair_mutation)
                if late_repair_mutation is not None and not bool(late_repair_mutation.get("accepted")):
                    final_visible_repair_max_cycle_exhausted = True
                    final_visible_repair_cycle_stop_reason = "quality_mutation_regression_detected"
                if not final_visible_repair_max_cycle_exhausted:
                    timeline_before_late_pacing_signature = self.hooks.final_timeline_state_signature(final_timeline)
                    timeline_before_late_pacing = list(final_timeline)
                    captions_before_late_pacing = list(captions)
                    visual_pacing_report_before_late = dict(visual_pacing_report or {})
                    final_timeline, visual_pacing_report = self.hooks.visual_pacing_normalize(final_timeline, source_graph)
                    if self.hooks.final_timeline_state_signature(final_timeline) == timeline_before_late_pacing_signature:
                        captions = captions_before_late_pacing
                    else:
                        captions = self.hooks.render_captions(final_timeline, source_graph)
                    visual_pacing_rerun_after_final_repair_count += 1
                    late_pacing_mutation = self.hooks.record_quality_mutation(
                        quality_mutations,
                        phase="visual_pacing.after_late_semantic_reconcile_repair",
                        rule_name="visual_pacing.normalize",
                        before_timeline=timeline_before_late_pacing,
                        before_captions=captions_before_late_pacing,
                        after_timeline=final_timeline,
                        after_captions=captions,
                        source_graph=source_graph,
                        action={"visual_pacing_executed": True},
                        after_visual_pacing_report=visual_pacing_report,
                    )
                    if late_pacing_mutation is not None and not bool(late_pacing_mutation.get("accepted")):
                        final_timeline = timeline_before_late_pacing
                        captions = captions_before_late_pacing
                        visual_pacing_report = visual_pacing_report_before_late
                        final_visible_repair_cycle_stop_reason = (
                            "visual_pacing_regression_reverted_after_late_semantic_reconcile_repair"
                        )
                final_visible_repair_report = self.hooks.combined_final_visible_repair_report(
                    final_visible_repair_reports,
                    visual_pacing_rerun_after_final_repair_count=visual_pacing_rerun_after_final_repair_count,
                    max_cycle_exhausted=final_visible_repair_max_cycle_exhausted,
                    cycle_stop_reason=final_visible_repair_cycle_stop_reason,
                    quality_mutations=quality_mutations,
                )
        final_visible_repair_report.update(self.hooks.quality_mutation_report_fields(quality_mutations))
        final_safe_handle_result = recompute_final_timeline_safe_handles(
            final_timeline=final_timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda timeline: self.hooks.render_captions(timeline, source_graph),
            pass_index=int(final_visible_repair_report.get("final_timeline_repair_intent_action_count") or 0) + 1,
        )
        if final_safe_handle_result is not None:
            final_timeline = final_safe_handle_result.final_timeline
            captions = final_safe_handle_result.captions
            engine_safe_actions = list(final_visible_repair_report.get("engine_final_timeline_repair_actions") or [])
            engine_safe_actions.append(final_safe_handle_result.action)
            final_visible_repair_report["engine_final_timeline_repair_actions"] = engine_safe_actions
            final_visible_repair_report["engine_final_timeline_repair_action_count"] = len(engine_safe_actions)
            final_visible_repair_report["engine_final_safe_handle_recompute_applied"] = True
        self.hooks.sync_semantic_gate_with_final_output(decision_plan, final_timeline, captions)
        self.hooks.refresh_semantic_adjudication_report(decision_plan)
        blockers.extend(decision_plan.blockers)
        return QualityPipelineResult(
            final_timeline=final_timeline,
            captions=captions,
            visual_pacing_report=visual_pacing_report,
            final_visible_repair_report=final_visible_repair_report,
            quality_mutations=quality_mutations,
        )
