from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.engine import build_run_summary
from aroll_v21.operator import ArollV21OperatorConfig, run_operator
from aroll_v21.ir import BlockerReport, CaptionRenderUnit, DecisionPlan, FinalTimelineSegment, RunReport
from aroll_v21.ingest.real_draft_adapter import RealDraftIngestResult
from aroll_v21.quality.caption_alignment import build_caption_alignment_report
from aroll_v21.quality import final_visible_caption_repair as final_visible_repair_module
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
from aroll_v21.quality.final_repeat_convergence import build_final_repeat_convergence_report
from aroll_v21.quality.quality_gate import build_quality_gate_report
from aroll_v21.quality.visual_pacing import VisualPacingNormalizer, build_visual_pacing_report
from aroll_v21.quality.visual_pacing.intra_segment_gap import split_large_intra_segment_gaps
from aroll_v21.render.subtitle_renderer import SubtitleRenderer, _cleanup_caption_units
from aroll_v21.validate import ReadOnlyValidators
from aroll_v21.validate.projected_write_view import ProjectedWriteViewError
from aroll_v21.writeback import real_draft_writeback as real_writeback_module
from aroll_v21.writeback.dynamic_source_binding_preflight import DynamicSourceBindingPreflight
from aroll_v21.writeback.real_draft_writeback import RealDraftWriteback
from tests.test_aroll_v21_captions_after_prefix_drop import _template_rows
from tests.test_aroll_v21_real_writeback_backend import run_report_from_result
from tests.test_aroll_v21_sacrificial_write_override import (
    FakeAdapter,
    create_disposable_draft,
    fake_encrypt,
    fake_integrity_ok,
    fake_real_draft_result,
    fake_root_mirror_not_required,
)


def _segment(index: int, start: int, end: int) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=f"v21_seg_{index:06d}",
        source_material_id="",
        source_segment_id=None,
        source_start_us=start,
        source_end_us=end,
        target_start_us=start,
        target_end_us=end,
        word_ids=[f"w{index}"],
        text=f"字幕{index}",
        decision_ids=[],
    )


def _caption(index: int, segment_id: str, start: int, end: int, text: str = "字幕") -> CaptionRenderUnit:
    return CaptionRenderUnit(
        caption_id=f"v21_cap_{index:06d}",
        timeline_segment_ids=[segment_id],
        word_ids=[f"w{index}"],
        text=text,
        target_start_us=start,
        target_end_us=end,
        source_subtitle_uids=[f"s{index}"],
        style_template_id="canonical_caption_template",
    )


def _semantic_gate_ok() -> dict[str, object]:
    return {
        "semantic_adjudication_gate_passed": True,
        "semantic_request_count": 0,
        "semantic_request_unresolved_count": 0,
        "fatal_semantic_issue_count": 0,
        "blocker_codes": [],
    }


def _final_visible_gate_ok() -> dict[str, object]:
    return {"gate_passed": True, "blocker_codes": []}


def _final_timeline_guard_ok() -> dict[str, object]:
    return {
        "gate_passed": True,
        "write_gate_passed": True,
        "quality_guard_passed": True,
        "blocker_codes": [],
        "blocking_candidate_count": 0,
    }


def _multi_caption_fake_result(*, root: Path | None = None) -> RealDraftIngestResult:
    result = fake_real_draft_result(root=root)
    draft_data = deepcopy(result.draft_data)
    source_segment = deepcopy(result.source_segments[0])
    source_segment.update(
        {
            "source_timerange": {"start": 0, "duration": 1_800_000},
            "target_timerange": {"start": 0, "duration": 1_800_000},
            "source_start_us": 0,
            "source_end_us": 1_800_000,
            "target_start_us": 0,
            "target_end_us": 1_800_000,
        }
    )
    draft_data["duration"] = 1_800_000
    draft_data["tracks"][0]["segments"][0] = source_segment
    draft_data["materials"]["videos"][0]["duration"] = 1_800_000
    return RealDraftIngestResult(
        draft_data=draft_data,
        source_segments=[source_segment],
        source_materials=[{"source_material_id": "main_video_a", "type": "video", "duration_us": 1_800_000}],
        word_timeline=[
            {"word_id": "w001", "word_text": "这个问题需要确认", "start_us": 0, "end_us": 500_000, "subtitle_index": 1, "subtitle_uid": "s001"},
            {"word_id": "w002", "word_text": "那个答案继续保留", "start_us": 600_000, "end_us": 1_100_000, "subtitle_index": 1, "subtitle_uid": "s001"},
            {"word_id": "w003", "word_text": "然后继续推进测试", "start_us": 1_200_000, "end_us": 1_700_000, "subtitle_index": 1, "subtitle_uid": "s001"},
        ],
        subtitles=[
            {
                "subtitle_uid": "s001",
                "subtitle_index": 1,
                "text": "这个问题需要确认那个答案继续保留然后继续推进测试",
                "word_ids": ["w001", "w002", "w003"],
            },
        ],
        text_materials=deepcopy(result.text_materials),
        text_segments=deepcopy(result.text_segments),
        metadata=deepcopy(result.metadata),
    )


def _graph_for_visual_merge_rows(rows: list[tuple[str, str, int, int]]) :
    materials, text_segments = _template_rows()
    words = [
        {
            "word_id": word_id,
            "word_text": text,
            "start_us": start,
            "end_us": end,
            "subtitle_index": index,
            "subtitle_uid": f"s{index:03d}",
        }
        for index, (word_id, text, start, end) in enumerate(rows, start=1)
    ]
    return ArollEngine().ingest.build_source_graph(
        word_timeline=words,
        subtitles=[
            {
                "subtitle_uid": row["subtitle_uid"],
                "subtitle_index": row["subtitle_index"],
                "text": row["word_text"],
                "word_ids": [row["word_id"]],
            }
            for row in words
        ],
        source_segments=[
            {
                "id": "primary_window",
                "material_id": "main",
                "type": "video",
                "source_start_us": 0,
                "source_end_us": max(end for _word_id, _text, _start, end in rows) + 500_000,
            }
        ],
        text_materials=materials,
        text_segments=text_segments,
    )


def _graph_for_single_subtitle_words(rows: list[tuple[str, str, int, int]]):
    materials, text_segments = _template_rows()
    words = [
        {
            "word_id": word_id,
            "word_text": text,
            "start_us": start,
            "end_us": end,
            "subtitle_index": 1,
            "subtitle_uid": "s001",
        }
        for word_id, text, start, end in rows
    ]
    return ArollEngine().ingest.build_source_graph(
        word_timeline=words,
        subtitles=[
            {
                "subtitle_uid": "s001",
                "subtitle_index": 1,
                "text": "".join(row["word_text"] for row in words),
                "word_ids": [row["word_id"] for row in words],
            }
        ],
        source_segments=[
            {
                "id": "primary_window",
                "material_id": "main",
                "type": "video",
                "source_start_us": 0,
                "source_end_us": max(end for _word_id, _text, _start, end in rows) + 500_000,
            }
        ],
        text_materials=materials,
        text_segments=text_segments,
    )


def _timeline_from_word_rows(rows: list[tuple[str, str, int, int]]) -> list[FinalTimelineSegment]:
    return [
        replace(
            _segment(index, start, end),
            source_material_id="main",
            source_segment_id="primary_window",
            source_start_us=start,
            source_end_us=end,
            target_start_us=start,
            target_end_us=end,
            word_ids=[word_id],
            text=text,
        )
        for index, (word_id, text, start, end) in enumerate(rows, start=1)
    ]


def _timeline_with_source_gap_dangling_pair(
    *,
    target_gap_us: int = 0,
    previous_text: str = "我们反对",
    current_text: str = "的是公共问题",
) -> tuple[object, list[FinalTimelineSegment]]:
    rows = [
        ("w001", previous_text, 0, 800_000),
        ("w002", current_text, 2_320_000, 3_120_000),
    ]
    source_graph = _graph_for_single_subtitle_words(rows)
    timeline = [
        replace(
            _segment(1, 0, 800_000),
            source_material_id="main_a",
            source_segment_id="source_a",
            source_start_us=0,
            source_end_us=800_000,
            target_start_us=0,
            target_end_us=800_000,
            word_ids=["w001"],
            text=previous_text,
        ),
        replace(
            _segment(2, 800_000 + target_gap_us, 1_600_000 + target_gap_us),
            source_material_id="main_b",
            source_segment_id="source_b",
            source_start_us=2_320_000,
            source_end_us=3_120_000,
            target_start_us=800_000 + target_gap_us,
            target_end_us=1_600_000 + target_gap_us,
            word_ids=["w002"],
            text=current_text,
        ),
    ]
    return source_graph, timeline


def _timeline_with_partial_tail_dangling_pair() -> tuple[object, list[FinalTimelineSegment]]:
    materials, text_segments = _template_rows()
    words = [
        {
            "word_id": "w001",
            "word_text": "前文铺垫",
            "start_us": 0,
            "end_us": 300_000,
            "subtitle_index": 1,
            "subtitle_uid": "s001",
        },
        {
            "word_id": "w002",
            "word_text": "你嘲笑嘉豪",
            "start_us": 300_000,
            "end_us": 800_000,
            "subtitle_index": 2,
            "subtitle_uid": "s002",
        },
        {
            "word_id": "w003",
            "word_text": "的是对自己人的规训",
            "start_us": 2_320_000,
            "end_us": 3_120_000,
            "subtitle_index": 3,
            "subtitle_uid": "s003",
        },
    ]
    source_graph = ArollEngine().ingest.build_source_graph(
        word_timeline=words,
        subtitles=[
            {
                "subtitle_uid": row["subtitle_uid"],
                "subtitle_index": row["subtitle_index"],
                "text": row["word_text"],
                "word_ids": [row["word_id"]],
            }
            for row in words
        ],
        source_segments=[
            {
                "id": "primary_window",
                "material_id": "main",
                "type": "video",
                "source_start_us": 0,
                "source_end_us": 3_620_000,
            }
        ],
        text_materials=materials,
        text_segments=text_segments,
    )
    timeline = [
        replace(
            _segment(1, 0, 800_000),
            source_material_id="main_a",
            source_segment_id="source_a",
            source_start_us=0,
            source_end_us=800_000,
            target_start_us=0,
            target_end_us=800_000,
            word_ids=["w001", "w002"],
            text="前文铺垫你嘲笑嘉豪",
        ),
        replace(
            _segment(2, 800_000, 1_600_000),
            source_material_id="main_b",
            source_segment_id="source_b",
            source_start_us=2_320_000,
            source_end_us=3_120_000,
            target_start_us=800_000,
            target_end_us=1_600_000,
            word_ids=["w003"],
            text="的是对自己人的规训",
        ),
    ]
    return source_graph, timeline


def _audit_writeback(encrypt_func=fake_encrypt, decrypt_func=None) -> RealDraftWriteback:
    return RealDraftWriteback(
        jy_draftc=Path("jy-draftc.exe"),
        encrypt_func=encrypt_func,
        decrypt_func=decrypt_func,
        root_mirror_func=fake_root_mirror_not_required,
        timeline_content_check_func=fake_integrity_ok,
        layout_check_func=fake_integrity_ok,
        project_folder_check_func=fake_integrity_ok,
    )


def _append_actual_text_segment(
    data: dict,
    *,
    segment_id: str,
    material_id: str,
    text: str,
    start_us: int,
    duration_us: int,
    track_id: str = "text_track",
    role: str | None = None,
) -> None:
    material = {"id": material_id, "type": "text", "text": text}
    if role is not None:
        material["role"] = role
    data.setdefault("materials", {}).setdefault("texts", []).append(material)
    track = next(track for track in data["tracks"] if track["id"] == track_id)
    track.setdefault("segments", []).append(
        {
            "id": segment_id,
            "type": "text",
            "material_id": material_id,
            "target_timerange": {"start": start_us, "duration": duration_us},
        }
    )


def _caption_trim_report(result: RealDraftIngestResult):
    report = run_report_from_result(result)
    first_word = report.source_graph.words[0]
    dropped_word = replace(
        first_word,
        word_id="w_drop",
        text="删掉",
        normalized_text="删掉",
        source_start_us=100_000,
        source_end_us=400_000,
    )
    kept_word = replace(
        first_word,
        word_id="w_keep",
        text="保留",
        normalized_text="保留",
        source_start_us=500_000,
        source_end_us=800_000,
    )
    final_segment = replace(
        report.final_timeline[0],
        source_start_us=100_000,
        source_end_us=800_000,
        target_start_us=0,
        target_end_us=700_000,
        word_ids=["w_drop", "w_keep"],
        text="删掉保留",
        spoken_source_start_us=None,
        spoken_source_end_us=None,
        clip_source_start_us=None,
        clip_source_end_us=None,
        lead_handle_us=0,
        tail_handle_us=0,
    )
    caption = replace(
        report.captions[0],
        word_ids=["w_keep"],
        text="保留",
        target_start_us=300_000,
        target_end_us=600_000,
        spoken_source_start_us=500_000,
        spoken_source_end_us=800_000,
        containing_video_segment_id=final_segment.segment_id,
        timeline_segment_ids=[final_segment.segment_id],
    )
    material_write_plan = deepcopy(report.material_write_plan)
    material_write_plan["segments"][0]["target_timerange"] = {"start": 300_000, "duration": 300_000}
    return replace(
        report,
        source_graph=replace(report.source_graph, words=[dropped_word, kept_word]),
        final_timeline=[final_segment],
        captions=[caption],
        material_write_plan=material_write_plan,
    )


def _dropped_reintroduced_report(result: RealDraftIngestResult):
    report = _caption_trim_report(result)
    final_segment = replace(
        report.final_timeline[0],
        source_start_us=500_000,
        source_end_us=800_000,
        target_start_us=300_000,
        target_end_us=600_000,
        word_ids=["w_keep"],
        text="保留",
    )
    return replace(report, final_timeline=[final_segment])


def _inside_dropped_report(result: RealDraftIngestResult):
    report = run_report_from_result(result)
    base_word = report.source_graph.words[0]
    before_word = replace(
        base_word,
        word_id="w_before",
        text="前",
        normalized_text="前",
        source_start_us=100_000,
        source_end_us=300_000,
    )
    dropped_word = replace(
        base_word,
        word_id="w_drop_inside",
        text="删掉",
        normalized_text="删掉",
        source_start_us=400_000,
        source_end_us=500_000,
    )
    after_word = replace(
        base_word,
        word_id="w_after",
        text="后",
        normalized_text="后",
        source_start_us=600_000,
        source_end_us=800_000,
    )
    final_segment = replace(
        report.final_timeline[0],
        source_start_us=100_000,
        source_end_us=800_000,
        target_start_us=0,
        target_end_us=700_000,
        word_ids=["w_before", "w_after"],
        text="前后",
        spoken_source_start_us=None,
        spoken_source_end_us=None,
        clip_source_start_us=None,
        clip_source_end_us=None,
        lead_handle_us=0,
        tail_handle_us=0,
    )
    caption = replace(
        report.captions[0],
        word_ids=["w_before", "w_after"],
        text="前后",
        target_start_us=0,
        target_end_us=700_000,
        spoken_source_start_us=100_000,
        spoken_source_end_us=800_000,
        containing_video_segment_id=final_segment.segment_id,
        timeline_segment_ids=[final_segment.segment_id],
    )
    material_write_plan = deepcopy(report.material_write_plan)
    material_write_plan["segments"][0]["target_timerange"] = {"start": 0, "duration": 700_000}
    return replace(
        report,
        source_graph=replace(report.source_graph, words=[before_word, dropped_word, after_word]),
        final_timeline=[final_segment],
        captions=[caption],
        material_write_plan=material_write_plan,
    )


class ArollV21QualityGateTests(unittest.TestCase):
    def test_final_repeat_convergence_reduces_high_count_to_zero(self) -> None:
        report = build_final_repeat_convergence_report(
            decision_trace=[
                {
                    "route": "final_target_repeat",
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "drop_recommended",
                    "drop_index": 1,
                    "convergence_iteration": 1,
                    "applied": True,
                }
            ],
            final_repeat_report={"final_target_repeat_high_count": 0, "final_target_repeat_candidates": []},
        )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["final_repeat_high_count_after"], 0)
        self.assertEqual(report["dropped_cluster_ids"], ["final_target_repeat_tc_0001"])
        self.assertEqual(report["dropped_segment_indices"], [1])
        self.assertEqual(report["iterations"], 1)

    def test_final_repeat_convergence_report_deduplicates_dropped_clusters(self) -> None:
        report = build_final_repeat_convergence_report(
            decision_trace=[
                {
                    "route": "final_target_repeat",
                    "cluster_id": "final_target_repeat_tc_0011",
                    "decision": "drop_recommended",
                    "dropped_segment_indices": [4],
                    "convergence_iteration": 1,
                    "applied": True,
                },
                {
                    "route": "final_target_repeat",
                    "cluster_id": "final_target_repeat_tc_0011",
                    "decision": "drop_recommended",
                    "dropped_segment_indices": [4],
                    "convergence_iteration": 1,
                    "applied": True,
                },
                {
                    "route": "final_target_repeat",
                    "cluster_id": "final_target_repeat_tc_0012",
                    "decision": "auto_drop_high_confidence_exact_repeat",
                    "dropped_indices": [4],
                    "convergence_iteration": 2,
                    "applied": True,
                },
            ],
            final_repeat_report={"final_target_repeat_high_count": 0, "final_target_repeat_candidates": []},
        )

        self.assertEqual(report["dropped_cluster_ids"], ["final_target_repeat_tc_0011", "final_target_repeat_tc_0012"])
        self.assertEqual(report["dropped_cluster_count"], 2)
        self.assertEqual(report["dropped_segment_indices"], [4])
        self.assertEqual(report["dropped_segment_count"], 1)
        self.assertEqual(report["final_repeat_dropped_segment_count"], 1)
        self.assertEqual(report["clusters_per_dropped_segment"], {"4": ["final_target_repeat_tc_0011", "final_target_repeat_tc_0012"]})
        self.assertEqual(report["iterations"], 2)

    def test_unresolved_high_repeat_blocks_ready(self) -> None:
        report = build_final_repeat_convergence_report(
            decision_trace=[],
            final_repeat_report={
                "final_target_repeat_high_count": 1,
                "final_target_repeat_candidates": [
                    {
                        "cluster_id": "final_target_repeat_tc_0001",
                        "confidence": "high",
                        "v21_resolution": "fatal_unresolved_high",
                    }
                ],
            },
        )

        self.assertFalse(report["gate_passed"])
        self.assertIn("V21_FINAL_REPEAT_UNRESOLVED_AFTER_CONVERGENCE", report["blocker_codes"])

    def test_final_caption_visible_repeat_classifies_distant_containment_without_blocking(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 700_000, text="就国南"),
                _caption(2, "v21_seg_000002", 10_000_000, 11_000_000, text="就国南就只会内斗"),
            ]
        )

        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["containment_repeat_count"], 0)
        self.assertEqual(gate["containment_repeat_raw_count"], 1)
        self.assertEqual(gate["repeat_classification_candidates"][0]["classification"], "short_concept_reuse")
        self.assertNotIn("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED", gate["blocker_codes"])

    def test_final_caption_visible_repeat_blocks_prefix_suffix_overlap(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 900_000, text="你应该相信自己"),
                _caption(2, "v21_seg_000002", 900_000, 1_800_000, text="相信自己能做到"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["prefix_suffix_overlap_count"], 1)
        self.assertEqual(gate["prefix_suffix_overlap_candidates"][0]["overlap_text"], "相信自己")

    def test_final_caption_visible_repeat_warns_on_shared_ngram_without_boundary_restart(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 900_000, text="我们重新开始吧"),
                _caption(2, "v21_seg_000002", 3_000_000, 3_900_000, text="大家都要重新开始"),
            ]
        )

        self.assertTrue(gate["gate_passed"], gate)
        self.assertEqual(gate["ngram_repeat_count"], 0)
        self.assertGreaterEqual(gate["ngram_repeat_raw_count"], 1)

    def test_final_caption_visible_repeat_allows_adjacent_middle_shared_phrase(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_600_000, text="这个订单编号已经填好"),
                _caption(2, "v21_seg_000002", 2_800_000, 4_200_000, text="你把客户订单编号再核对"),
            ]
        )

        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["ngram_repeat_count"], 0)

    def test_final_visible_blocks_dangling_de_prefix(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 900_000, text="你嘲笑嘉豪"),
                _caption(2, "v21_seg_000002", 900_000, 1_800_000, text="的是对自己人的规训"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["dangling_prefix_suffix_count"], 1)
        self.assertIn("V21_FINAL_VISIBLE_DANGLING_PREFIX_SUFFIX", gate["blocker_codes"])
        self.assertEqual(gate["dangling_prefix_suffix_candidates"][0]["type"], "dangling_prefix_or_suffix")

    def test_final_visible_blocks_dangling_suffix_caption(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 900_000, text="我们还没有说完"),
                _caption(2, "v21_seg_000002", 900_000, 1_800_000, text="了"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["dangling_prefix_suffix_count"], 1)
        self.assertIn("V21_FINAL_VISIBLE_DANGLING_PREFIX_SUFFIX", gate["blocker_codes"])

    def test_final_visible_rechecks_deepseek_retained_side_quality(self) -> None:
        caption_gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_200_000, text="看就看到有人想爬出粪坑"),
            ]
        )
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "detector_report_present": True},
            final_caption_visible_repeat_gate=caption_gate,
            final_timeline_quality_guard_gate=_final_timeline_guard_ok(),
            semantic_adjudication_gate=_semantic_gate_ok(),
            visual_pacing_gate={"gate_passed": True, "visual_pacing_executed": True, "visual_merge_safety_gate_passed": True, "blocker_codes": []},
            caption_alignment_gate={"gate_passed": True, "caption_gui_track_gate_passed": True, "subtitle_readability_gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(caption_gate["gate_passed"])
        self.assertFalse(quality["gate_passed"])
        self.assertFalse(quality["ready_for_user_manual_qc_preconditions_passed"])
        self.assertEqual(caption_gate["semantic_garbage_or_asr_suspect_count"], 1)
        self.assertIn("V21_FINAL_VISIBLE_SEMANTIC_GARBAGE_OR_ASR_SUSPECT", quality["blocker_codes"])
        self.assertEqual(
            caption_gate["semantic_garbage_or_asr_suspect_candidates"][0]["allowed_recheck_decisions"],
            ["drop_bad_fragment", "trim_repeated_prefix", "keep_if_coherent", "requires_human_review"],
        )

    def test_quality_gate_report_keeps_final_visible_semantic_arbitration_payloads(self) -> None:
        caption_gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_600_000, text="他的购物车全是投资和享受"),
                _caption(2, "v21_seg_000002", 1_620_000, 2_300_000, text="你的购物车"),
            ]
        )
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "detector_report_present": True},
            final_caption_visible_repeat_gate=caption_gate,
            final_timeline_quality_guard_gate=_final_timeline_guard_ok(),
            semantic_adjudication_gate=_semantic_gate_ok(),
            visual_pacing_gate={"gate_passed": True, "visual_pacing_executed": True, "visual_merge_safety_gate_passed": True, "blocker_codes": []},
            caption_alignment_gate={"gate_passed": True, "caption_gui_track_gate_passed": True, "subtitle_readability_gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        visible = quality["final_caption_visible_repeat_gate"]
        self.assertTrue(quality["gate_passed"])
        self.assertEqual(visible["repeat_semantic_arbitration_candidate_count"], 1)
        self.assertEqual(visible["repeat_semantic_arbitration_request_count"], 1)
        self.assertEqual(
            visible["repeat_semantic_arbitration_request_payloads"][0]["local_context"]["semantic_repeat_evidence"]["shared_text"],
            "的购物车",
        )

    def test_final_visible_detects_cross_caption_semantic_containment_window(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 900_000, text="在舞台中央大声说"),
                _caption(2, "v21_seg_000002", 900_000, 1_800_000, text="把那个敢于在舞台"),
                _caption(3, "v21_seg_000003", 1_800_000, 2_700_000, text="中央大声说话的自己"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["cross_caption_semantic_containment_count"], 1)
        self.assertIn("V21_FINAL_VISIBLE_CROSS_CAPTION_SEMANTIC_CONTAINMENT", gate["blocker_codes"])
        self.assertEqual(gate["cross_caption_semantic_containment_candidates"][0]["window_caption_ids"], ["v21_cap_000002", "v21_cap_000003"])

    def test_final_visible_detects_restart_repeat_across_2_3_captions(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 900_000, text="在舞台中央大声说"),
                _caption(2, "v21_seg_000002", 900_000, 1_800_000, text="把那个敢于在舞台"),
                _caption(3, "v21_seg_000003", 1_800_000, 2_700_000, text="中央大声说话的自己"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertGreaterEqual(gate["restart_repeat_visible_count"], 1)
        self.assertIn("V21_FINAL_VISIBLE_RESTART_REPEAT", gate["blocker_codes"])

    def test_final_visible_detects_internal_pivot_restart_repeat(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 900_000, text="你是你们是极度恐慌"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["restart_repeat_visible_count"], 1)
        self.assertEqual(gate["restart_repeat_visible_candidates"][0]["drop_text"], "你是")
        self.assertIn("V21_FINAL_VISIBLE_RESTART_REPEAT", gate["blocker_codes"])

    def test_final_visible_repairs_dangling_prefix_by_merge_or_recheck(self) -> None:
        rows = [
            ("w001", "我们反对", 0, 800_000),
            ("w002", "的是公共问题", 800_000, 1_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = _timeline_from_word_rows(rows)
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_final_counts"]["dangling_prefix_suffix_count"], 0)
        self.assertEqual([segment.text for segment in result.final_timeline], ["我们反对的是公共问题"])
        self.assertEqual([caption.text for caption in result.captions], ["我们反对的是公共问题"])
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "merge_with_previous_segment")

    def test_final_visible_segment_merge_recomputes_target_end_for_speed_invariant(self) -> None:
        rows = [
            ("w001", "我们反对", 0, 800_000),
            ("w002", "的是公共问题", 800_000, 1_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        timeline = [
            replace(
                _segment(1, 0, 666_667),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=800_000,
                target_start_us=0,
                target_end_us=666_667,
                word_ids=["w001"],
                text="我们反对",
                clip_source_end_us=800_000,
            ),
            replace(
                _segment(2, 666_667, 1_333_334),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=800_000,
                source_end_us=1_600_000,
                target_start_us=666_667,
                target_end_us=1_333_334,
                word_ids=["w002"],
                text="的是公共问题",
                clip_source_end_us=1_820_000,
                tail_handle_us=220_000,
            ),
        ]
        captions = [
            CaptionRenderUnit("v21_cap_000001", ["v21_seg_000001"], ["w001"], "我们反对", 0, 666_667, ["s001"], "canonical_caption_template", containing_video_segment_id="v21_seg_000001"),
            CaptionRenderUnit("v21_cap_000002", ["v21_seg_000002"], ["w002"], "的是公共问题", 666_667, 1_333_334, ["s001"], "canonical_caption_template", containing_video_segment_id="v21_seg_000002"),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: SubtitleRenderer().render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertEqual(len(result.final_timeline), 1)
        self.assertEqual(result.final_timeline[0].target_end_us, 1_600_000)
        self.assertEqual(result.final_timeline[0].clip_source_end_us, 1_820_000)
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "merge_with_previous_segment")

    def test_final_visible_caption_only_merge_normalizes_de_shi_prepositional_boundary(self) -> None:
        source_graph, timeline = _timeline_with_partial_tail_dangling_pair()
        renderer = SubtitleRenderer()
        captions = [
            CaptionRenderUnit(
                caption_id="v21_cap_000001",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w001"],
                text="前文铺垫",
                target_start_us=0,
                target_end_us=300_000,
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=300_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000002",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w002"],
                text="你嘲笑嘉豪",
                target_start_us=300_000,
                target_end_us=800_000,
                source_subtitle_uids=["s002"],
                style_template_id="canonical_caption_template",
                spoken_source_start_us=300_000,
                spoken_source_end_us=800_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000003",
                timeline_segment_ids=["v21_seg_000002"],
                word_ids=["w003"],
                text="的是对自己人的规训",
                target_start_us=800_000,
                target_end_us=1_600_000,
                source_subtitle_uids=["s003"],
                style_template_id="canonical_caption_template",
                spoken_source_start_us=2_320_000,
                spoken_source_end_us=3_120_000,
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([caption.text for caption in result.captions], ["前文铺垫", "你嘲笑嘉豪是对自己人的规训"])
        self.assertEqual(result.report["caption_only_materialized_merge_count"], 1)

    def test_final_visible_trims_leading_filler_before_long_source_gap(self) -> None:
        rows = [
            ("w001", "咳", 0, 700_000),
            ("w002", "立刻", 1_900_000, 2_200_000),
            ("w003", "给", 2_200_000, 2_360_000),
            ("w004", "我", 2_360_000, 2_520_000),
            ("w005", "关了", 2_520_000, 2_900_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_900_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_900_000,
                target_start_us=0,
                target_end_us=2_900_000,
                word_ids=[row[0] for row in rows],
                text="咳立刻给我关了",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertEqual([segment.text for segment in result.final_timeline], ["立刻给我关了"])
        self.assertEqual(result.final_timeline[0].source_start_us, 1_900_000)
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "trim_leading_filler_gap")
        self.assertEqual(result.report["final_visible_repair_actions"][0]["repair_transaction_rule_name"], "leading_filler_gap")
        self.assertEqual(result.report["final_visible_repair_transaction_count"], 1)

    def test_final_visible_trims_single_word_intrusion_before_connector(self) -> None:
        rows = [
            ("w001", "底盘", 0, 360_000),
            ("w002", "不行", 360_000, 760_000),
            ("w003", "手", 1_160_000, 1_360_000),
            ("w004", "所以", 1_700_000, 2_060_000),
            ("w005", "你", 2_060_000, 2_220_000),
            ("w006", "手里", 2_220_000, 2_620_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_620_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_620_000,
                target_start_us=0,
                target_end_us=2_620_000,
                word_ids=[row[0] for row in rows],
                text="底盘不行手所以你手里",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "底盘不行所以你手里")
        self.assertNotIn("w003", [word_id for segment in result.final_timeline for word_id in segment.word_ids])
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "trim_single_word_intrusion_before_connector")
        self.assertEqual(
            result.report["final_visible_repair_actions"][0]["repair_transaction_rule_name"],
            "connector_single_word_intrusion",
        )

    def test_final_visible_trims_repeated_object_head_before_tail_repeat(self) -> None:
        rows = [
            ("w001", "心形", 0, 700_000),
            ("w002", "用", 1_000_000, 1_200_000),
            ("w003", "个", 1_200_000, 1_360_000),
            ("w004", "玫瑰花", 1_360_000, 1_800_000),
            ("w005", "摆", 1_880_000, 2_040_000),
            ("w006", "个", 2_040_000, 2_200_000),
            ("w007", "土", 2_240_000, 2_320_000),
            ("w008", "到", 2_400_000, 2_560_000),
            ("w009", "爆", 2_560_000, 2_640_000),
            ("w010", "的", 2_680_000, 2_840_000),
            ("w011", "心", 2_840_000, 3_200_000),
        ]
        materials, text_segments = _template_rows()
        words = [
            {
                "word_id": word_id,
                "word_text": text,
                "start_us": start,
                "end_us": end,
                "subtitle_index": 1 if word_id == "w001" else 2,
                "subtitle_uid": "s001" if word_id == "w001" else "s002",
            }
            for word_id, text, start, end in rows
        ]
        source_graph = ArollEngine().ingest.build_source_graph(
            word_timeline=words,
            subtitles=[
                {"subtitle_uid": "s001", "subtitle_index": 1, "text": "心形", "word_ids": ["w001"]},
                {
                    "subtitle_uid": "s002",
                    "subtitle_index": 2,
                    "text": "用个玫瑰花摆个土到爆的心",
                    "word_ids": [row[0] for row in rows[1:]],
                },
            ],
            source_segments=[
                {
                    "id": "primary_window",
                    "material_id": "main",
                    "type": "video",
                    "source_start_us": 0,
                    "source_end_us": 3_700_000,
                }
            ],
            text_materials=materials,
            text_segments=text_segments,
        )
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 3_200_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=3_200_000,
                target_start_us=0,
                target_end_us=3_200_000,
                word_ids=[row[0] for row in rows],
                text="心形用个玫瑰花摆个土到爆的心",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertEqual([segment.text for segment in result.final_timeline], ["用个玫瑰花摆个土到爆的心"])
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "trim_repeated_object_head")
        self.assertEqual(
            result.report["final_visible_repair_actions"][0]["repair_transaction_rule_name"],
            "leading_object_head_repeated_as_tail",
        )

    def test_final_visible_gate_allows_parallel_condition_discourse_opener(self) -> None:
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001"],
                "但凡任何一个普通人",
                0,
                800_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=800_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w002"],
                "但凡给你一点反馈",
                800_000,
                1_500_000,
                ["s002"],
                "canonical_caption_template",
                spoken_source_start_us=860_000,
                spoken_source_end_us=1_500_000,
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        gate = build_final_caption_visible_repeat_gate(captions)

        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["restart_repeat_visible_count"], 0)

    def test_final_visible_gate_detects_adjacent_repeated_short_discourse_opener(self) -> None:
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001"],
                "然后开始整理",
                0,
                800_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=800_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w002"],
                "然后继续处理",
                800_000,
                1_500_000,
                ["s002"],
                "canonical_caption_template",
                spoken_source_start_us=860_000,
                spoken_source_end_us=1_500_000,
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        gate = build_final_caption_visible_repeat_gate(captions)

        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["restart_repeat_visible_count"], 1)
        self.assertEqual(gate["restart_repeat_visible_candidates"][0]["pattern"], "repeated_discourse_opener")
        self.assertEqual(gate["restart_repeat_visible_candidates"][0]["drop_text"], "然后")

    def test_final_visible_gate_keeps_short_is_clause_before_nominal_predicate(self) -> None:
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002", "w003", "w004"],
                "女人就永远是",
                0,
                700_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=700_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w005", "w006", "w007"],
                "薛定谔的集美",
                700_000,
                1_500_000,
                ["s002"],
                "canonical_caption_template",
                spoken_source_start_us=700_000,
                spoken_source_end_us=1_500_000,
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        gate = build_final_caption_visible_repeat_gate(captions)

        self.assertTrue(gate["gate_passed"], gate)
        self.assertEqual(gate["semantic_integrity_count"], 0)

    def test_final_visible_gate_keeps_passive_tail_when_next_caption_completes_predicate(self) -> None:
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002", "w003", "w004"],
                "连最基础的下颌线都没有",
                0,
                900_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=900_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w005", "w006", "w007"],
                "你的下颌线如果都被",
                900_000,
                1_500_000,
                ["s002"],
                "canonical_caption_template",
                spoken_source_start_us=900_000,
                spoken_source_end_us=1_500_000,
                containing_video_segment_id="v21_seg_000002",
            ),
            CaptionRenderUnit(
                "v21_cap_000003",
                ["v21_seg_000003"],
                ["w008", "w009"],
                "脂肪吃的干干净净",
                1_500_000,
                2_300_000,
                ["s003"],
                "canonical_caption_template",
                spoken_source_start_us=1_500_000,
                spoken_source_end_us=2_300_000,
                containing_video_segment_id="v21_seg_000003",
            ),
        ]

        gate = build_final_caption_visible_repeat_gate(captions)

        self.assertTrue(gate["gate_passed"], gate)
        self.assertEqual(gate["semantic_integrity_count"], 0)

    def test_final_visible_gate_allows_short_ascii_head_before_immediate_predicate_continuation(self) -> None:
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002", "w003"],
                "Pro从来就",
                0,
                760_000,
                ["s001"],
                "canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000001"],
                ["w004", "w005", "w006"],
                "不是靠包装就能够得来的",
                760_000,
                2_500_000,
                ["s002"],
                "canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000003",
                ["v21_seg_000002"],
                ["w007"],
                "Pro",
                2_500_000,
                3_000_000,
                ["s003"],
                "canonical_caption_template",
                containing_video_segment_id="v21_seg_000002",
            ),
            CaptionRenderUnit(
                "v21_cap_000004",
                ["v21_seg_000003"],
                ["w008", "w009"],
                "只是把你本来就存在的",
                3_000_000,
                4_000_000,
                ["s004"],
                "canonical_caption_template",
                containing_video_segment_id="v21_seg_000003",
            ),
        ]

        gate = build_final_caption_visible_repeat_gate(captions)

        self.assertTrue(gate["gate_passed"], gate)
        self.assertEqual(gate["visible_repeat_candidate_count"], 0)

    def test_final_visible_repair_keeps_parallel_condition_discourse_opener(self) -> None:
        rows = [
            ("w001", "但凡", 0, 260_000),
            ("w002", "任何", 260_000, 520_000),
            ("w003", "一个", 520_000, 760_000),
            ("w004", "普通人", 760_000, 1_120_000),
            ("w005", "但凡", 1_180_000, 1_440_000),
            ("w006", "给你", 1_440_000, 1_720_000),
            ("w007", "一点", 1_720_000, 1_960_000),
            ("w008", "反馈", 1_960_000, 2_260_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_120_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_120_000,
                target_start_us=0,
                target_end_us=1_120_000,
                word_ids=["w001", "w002", "w003", "w004"],
                text="但凡任何一个普通人",
            ),
            replace(
                _segment(2, 1_120_000, 2_200_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_180_000,
                source_end_us=2_260_000,
                target_start_us=1_120_000,
                target_end_us=2_200_000,
                word_ids=["w005", "w006", "w007", "w008"],
                text="但凡给你一点反馈",
            ),
        ]
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002", "w003", "w004"],
                "但凡任何一个普通人",
                0,
                1_120_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=1_120_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w005", "w006", "w007", "w008"],
                "但凡给你一点反馈",
                1_120_000,
                2_200_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=1_180_000,
                spoken_source_end_us=2_260_000,
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertEqual([segment.text for segment in result.final_timeline], ["但凡任何一个普通人", "但凡给你一点反馈"])
        self.assertEqual([caption.text for caption in result.captions], ["但凡任何一个普通人", "但凡给你一点反馈"])

    def test_final_visible_repair_keeps_progressive_semantic_expansion(self) -> None:
        rows = [
            ("w001", "你", 0, 160_000),
            ("w002", "更", 160_000, 320_000),
            ("w003", "没钱", 320_000, 640_000),
            ("w004", "去", 640_000, 840_000),
            ("w005", "健身", 840_000, 1_120_000),
            ("w006", "了", 1_120_000, 1_360_000),
            ("w007", "也", 1_360_000, 1_560_000),
            ("w008", "没钱", 1_560_000, 1_840_000),
            ("w009", "去", 1_840_000, 2_000_000),
            ("w010", "做", 2_000_000, 2_120_000),
            ("w011", "YM", 2_120_000, 2_480_000),
            ("w012", "了", 2_480_000, 2_680_000),
            ("w013", "也", 2_680_000, 2_880_000),
            ("w014", "没钱", 2_880_000, 3_160_000),
            ("w015", "去", 3_160_000, 3_320_000),
            ("w016", "投资", 3_320_000, 3_600_000),
            ("w017", "自己", 3_600_000, 3_840_000),
            ("w018", "了", 3_840_000, 4_120_000),
            ("w019", "你", 4_120_000, 4_320_000),
            ("w020", "变得", 4_320_000, 4_520_000),
            ("w021", "更", 4_520_000, 4_680_000),
            ("w022", "丑", 4_680_000, 4_920_000),
            ("w023", "了", 4_920_000, 5_240_000),
            ("w024", "哈", 5_240_000, 5_480_000),
            ("w025", "你", 5_480_000, 5_680_000),
            ("w026", "更", 5_680_000, 5_880_000),
            ("w027", "没有", 5_880_000, 6_120_000),
            ("w028", "认识", 6_120_000, 6_400_000),
            ("w029", "你", 6_400_000, 6_600_000),
            ("w030", "更", 6_600_000, 6_800_000),
            ("w031", "没有", 6_800_000, 7_200_000),
            ("w032", "你", 7_200_000, 7_440_000),
            ("w033", "变得", 7_440_000, 7_720_000),
            ("w034", "你", 7_720_000, 7_880_000),
            ("w035", "更", 7_880_000, 8_040_000),
            ("w036", "没有", 8_040_000, 8_280_000),
            ("w037", "认识", 8_280_000, 8_480_000),
            ("w038", "异性", 8_480_000, 8_720_000),
            ("w039", "的", 8_720_000, 8_920_000),
            ("w040", "渠道", 8_920_000, 9_160_000),
            ("w041", "了", 9_160_000, 9_500_000),
            ("w042", "就", 9_500_000, 9_620_000),
            ("w043", "变得", 9_620_000, 9_820_000),
            ("w044", "更", 9_820_000, 10_020_000),
            ("w045", "压抑", 10_020_000, 10_500_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        segment_specs = [
            ("你更没钱去健身了", ["w001", "w002", "w003", "w004", "w005", "w006"]),
            ("也没钱去做YM了", ["w007", "w008", "w009", "w010", "w011", "w012"]),
            ("也没钱去投资自己了", ["w013", "w014", "w015", "w016", "w017", "w018"]),
            ("你变得更丑了", ["w019", "w020", "w021", "w022", "w023"]),
            ("哈你更没有认识", ["w024", "w025", "w026", "w027", "w028"]),
            ("你更没有你变得", ["w029", "w030", "w031", "w032", "w033"]),
            ("你更没有认识异性的渠道了", ["w034", "w035", "w036", "w037", "w038", "w039", "w040", "w041"]),
            ("就变得更压抑", ["w042", "w043", "w044", "w045"]),
        ]
        word_times = {word_id: (start, end) for word_id, _text, start, end in rows}
        timeline = [
            replace(
                _segment(index, word_times[word_ids[0]][0], word_times[word_ids[-1]][1]),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=word_times[word_ids[0]][0],
                source_end_us=word_times[word_ids[-1]][1],
                target_start_us=word_times[word_ids[0]][0],
                target_end_us=word_times[word_ids[-1]][1],
                word_ids=word_ids,
                text=text,
            )
            for index, (text, word_ids) in enumerate(segment_specs, start=1)
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertEqual([segment.text for segment in result.final_timeline], [text for text, _word_ids in segment_specs])
        self.assertEqual([caption.text for caption in result.captions], [text for text, _word_ids in segment_specs])

    def test_final_visible_repair_keeps_parallel_condition_after_aborted_condition_opener(self) -> None:
        rows = [
            ("w001", "但凡", 0, 260_000),
            ("w002", "给你", 260_000, 520_000),
            ("w003", "四分", 520_000, 760_000),
            ("w004", "但凡", 1_000_000, 1_260_000),
            ("w005", "任何", 1_260_000, 1_520_000),
            ("w006", "一个", 1_520_000, 1_760_000),
            ("w007", "四分女", 1_760_000, 2_160_000),
            ("w008", "但凡", 2_360_000, 2_620_000),
            ("w009", "给你", 2_620_000, 2_880_000),
            ("w010", "释放", 2_880_000, 3_160_000),
            ("w011", "一丁点", 3_160_000, 3_480_000),
            ("w012", "好感", 3_480_000, 3_780_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 760_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=760_000,
                target_start_us=0,
                target_end_us=760_000,
                word_ids=["w001", "w002", "w003"],
                text="但凡给你四分",
            ),
            replace(
                _segment(2, 760_000, 1_920_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_000_000,
                source_end_us=2_160_000,
                target_start_us=760_000,
                target_end_us=1_920_000,
                word_ids=["w004", "w005", "w006", "w007"],
                text="但凡任何一个四分女",
            ),
            replace(
                _segment(3, 1_920_000, 3_340_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=2_360_000,
                source_end_us=3_780_000,
                target_start_us=1_920_000,
                target_end_us=3_340_000,
                word_ids=["w008", "w009", "w010", "w011", "w012"],
                text="但凡给你释放一丁点好感",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        visible = "".join(caption.text for caption in result.captions)
        self.assertNotIn("但凡给你四分", visible)
        self.assertIn("但凡任何一个四分女", visible)
        self.assertIn("但凡给你释放一丁点好感", visible)

    def test_final_visible_repair_trims_repeated_short_discourse_opener(self) -> None:
        rows = [
            ("w001", "然后", 0, 260_000),
            ("w002", "开始", 260_000, 520_000),
            ("w003", "整理", 520_000, 760_000),
            ("w004", "然后", 820_000, 1_080_000),
            ("w005", "继续", 1_080_000, 1_340_000),
            ("w006", "处理", 1_340_000, 1_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 760_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=760_000,
                target_start_us=0,
                target_end_us=760_000,
                word_ids=["w001", "w002", "w003"],
                text="然后开始整理",
            ),
            replace(
                _segment(2, 760_000, 1_540_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=820_000,
                source_end_us=1_600_000,
                target_start_us=760_000,
                target_end_us=1_540_000,
                word_ids=["w004", "w005", "w006"],
                text="然后继续处理",
            ),
        ]
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002", "w003"],
                "然后开始整理",
                0,
                760_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=760_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w004", "w005", "w006"],
                "然后继续处理",
                760_000,
                1_540_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=820_000,
                spoken_source_end_us=1_600_000,
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertEqual([segment.text for segment in result.final_timeline], ["然后开始整理", "继续处理"])
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "trim_restart_prefix")

    def test_final_visible_repair_trims_repeated_short_discourse_opener_inside_merged_segment(self) -> None:
        rows = [
            ("w001", "然后", 0, 260_000),
            ("w002", "开始", 260_000, 520_000),
            ("w003", "整理", 520_000, 760_000),
            ("w004", "然后", 820_000, 1_080_000),
            ("w005", "继续", 1_080_000, 1_340_000),
            ("w006", "处理", 1_340_000, 1_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_260_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_260_000,
                target_start_us=0,
                target_end_us=2_260_000,
                word_ids=["w001", "w002", "w003", "w004", "w005", "w006"],
                text="然后开始整理然后继续处理",
            )
        ]
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002", "w003"],
                "然后开始整理",
                0,
                760_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=760_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000001"],
                ["w004", "w005", "w006"],
                "然后继续处理",
                760_000,
                1_540_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=820_000,
                spoken_source_end_us=1_600_000,
                containing_video_segment_id="v21_seg_000001",
            ),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        final_text = "".join(segment.text for segment in result.final_timeline)
        self.assertEqual(final_text, "然后开始整理继续处理")
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "trim_restart_prefix")

    def test_final_visible_repair_keeps_restart_lead_after_internal_prefix_fragment(self) -> None:
        rows = [
            ("w001", "普通", 0, 240_000),
            ("w002", "社交", 240_000, 520_000),
            ("w003", "的", 520_000, 680_000),
            ("w004", "状", 680_000, 840_000),
            ("w005", "她", 840_000, 1_020_000),
            ("w006", "会", 1_020_000, 1_180_000),
            ("w007", "在", 1_180_000, 1_340_000),
            ("w008", "普通", 1_340_000, 1_580_000),
            ("w009", "社交", 1_580_000, 1_840_000),
            ("w010", "的", 1_840_000, 2_000_000),
            ("w011", "叠加态下", 2_000_000, 2_420_000),
            ("w012", "对你笑", 2_420_000, 2_900_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_900_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_900_000,
                target_start_us=0,
                target_end_us=2_900_000,
                word_ids=[row[0] for row in rows],
                text="普通社交的状她会在普通社交的叠加态下对你笑",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        final_text = "".join(segment.text for segment in result.final_timeline)
        self.assertEqual(final_text, "她会在普通社交的叠加态下对你笑")
        self.assertIn("她会在", final_text)
        restart_actions = [
            action
            for action in result.report["final_visible_repair_actions"]
            if action.get("issue_type") == "restart_repeat_visible"
        ]
        self.assertTrue(restart_actions)
        self.assertEqual(restart_actions[0].get("drop_text"), "普通社交的状")

    def test_final_visible_gate_allows_label_reuse_before_definition(self) -> None:
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002"],
                "都叫表白",
                0,
                700_000,
                ["s001"],
                "canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w003", "w004"],
                "表白等于释放信号",
                700_000,
                1_700_000,
                ["s002"],
                "canonical_caption_template",
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        gate = build_final_caption_visible_repeat_gate(captions)

        self.assertTrue(gate["gate_passed"], gate)
        self.assertEqual(gate["prefix_suffix_overlap_count"], 0)

    def test_final_visible_gate_allows_explanatory_term_reuse(self) -> None:
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001"],
                "核心概念",
                0,
                700_000,
                ["s001"],
                "canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w002", "w003"],
                "什么叫核心概念",
                1_200_000,
                2_300_000,
                ["s002"],
                "canonical_caption_template",
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        gate = build_final_caption_visible_repeat_gate(captions)

        self.assertTrue(gate["gate_passed"], gate)
        self.assertEqual(gate["containment_repeat_count"], 0)
        self.assertEqual(gate["ngram_repeat_count"], 0)

    def test_subtitle_renderer_does_not_drop_disjoint_explanatory_term_caption(self) -> None:
        rows = [
            ("w001", "核心概念", 0, 600_000),
            ("w002", "什么叫", 1_200_000, 1_760_000),
            ("w003", "核心概念", 1_760_000, 2_360_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        timeline = [
            replace(
                _segment(1, 0, 600_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=600_000,
                target_start_us=0,
                target_end_us=600_000,
                word_ids=["w001"],
                text="核心概念",
            ),
            replace(
                _segment(2, 1_200_000, 2_360_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_200_000,
                source_end_us=2_360_000,
                target_start_us=600_000,
                target_end_us=1_760_000,
                word_ids=["w002", "w003"],
                text="什么叫核心概念",
            ),
        ]

        captions = SubtitleRenderer().render(timeline, source_graph)

        self.assertEqual([caption.text for caption in captions], ["核心概念", "什么叫核心概念"])
        self.assertEqual([caption.word_ids for caption in captions], [["w001"], ["w002", "w003"]])

    def test_subtitle_renderer_repeat_cleanup_preserves_disjoint_word_coverage(self) -> None:
        rows = [
            ("w001", "重复表达", 0, 600_000),
            ("w002", "重复表达", 1_200_000, 1_800_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        timeline = [
            replace(
                _segment(1, 0, 600_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=600_000,
                target_start_us=0,
                target_end_us=600_000,
                word_ids=["w001"],
                text="重复表达",
            ),
            replace(
                _segment(2, 1_200_000, 1_800_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_200_000,
                source_end_us=1_800_000,
                target_start_us=600_000,
                target_end_us=1_200_000,
                word_ids=["w002"],
                text="重复表达",
            ),
        ]

        captions = SubtitleRenderer().render(timeline, source_graph)

        self.assertEqual([caption.text for caption in captions], ["重复表达", "重复表达"])
        self.assertEqual([caption.word_ids for caption in captions], [["w001"], ["w002"]])

    def test_final_visible_repairs_dangling_prefix_by_caption_only_merge_when_source_gap_slightly_exceeds_limit(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair()
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_unresolved"], [])
        self.assertEqual(result.report["final_visible_repair_final_counts"]["dangling_prefix_suffix_count"], 0)
        self.assertEqual(result.report["final_visible_repair_final_timeline_counts"]["dangling_prefix_suffix_count"], 0)
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "caption_only_merge_with_previous")
        self.assertEqual([caption.text for caption in result.captions], ["我们反对的是公共问题"])
        self.assertEqual(result.captions[0].timeline_segment_ids, ["v21_seg_000001", "v21_seg_000002"])

    def test_caption_only_merge_does_not_merge_video_segments(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair()
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )
        alignment = build_caption_alignment_report(final_timeline=result.final_timeline, captions=result.captions)

        self.assertEqual([segment.segment_id for segment in result.final_timeline], ["v21_seg_000001", "v21_seg_000002"])
        self.assertEqual([segment.source_material_id for segment in result.final_timeline], ["main_a", "main_b"])
        self.assertIsNone(result.captions[0].containing_video_segment_id)
        self.assertTrue(alignment["gate_passed"])
        self.assertEqual(alignment["caption_outside_video_count"], 0)
        self.assertEqual(alignment["caption_cross_primary_window_count"], 0)

    def test_final_visible_caption_only_merges_same_subtitle_short_tail(self) -> None:
        rows = [
            ("w001", "马上", 0, 240_000),
            ("w002", "关掉", 240_000, 520_000),
            ("w003", "这个", 520_000, 760_000),
            ("w004", "窗口", 760_000, 1_100_000),
            ("w005", "明白", 1_500_000, 1_900_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        timeline = [
            replace(
                _segment(1, 0, 1_100_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_100_000,
                target_start_us=0,
                target_end_us=1_100_000,
                word_ids=["w001", "w002", "w003", "w004"],
                text="马上关掉这个窗口",
            ),
            replace(
                _segment(2, 1_100_000, 1_500_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_500_000,
                source_end_us=1_900_000,
                target_start_us=1_100_000,
                target_end_us=1_500_000,
                word_ids=["w005"],
                text="明白",
            ),
        ]
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002", "w003", "w004"],
                "马上关掉这个窗口",
                0,
                1_100_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=1_100_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000002"],
                ["w005"],
                "明白",
                1_100_000,
                1_500_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=1_500_000,
                spoken_source_end_us=1_900_000,
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda _repaired: captions,
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertEqual([caption.text for caption in result.captions], ["马上关掉这个窗口明白"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["马上关掉这个窗口", "明白"])
        self.assertEqual(
            result.report["final_visible_repair_actions"][-1]["decision"],
            "caption_only_merge_same_subtitle_short_tail",
        )

    def test_caption_only_merge_requires_target_adjacency(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair(target_gap_us=200_000)
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertFalse(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_final_counts"]["dangling_prefix_suffix_count"], 1)
        self.assertEqual(result.report["final_visible_repair_unresolved"][0]["reason"], "no_safe_deterministic_repair_available")

    def test_caption_only_merge_still_blocks_when_combined_caption_unreadable(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair(
            previous_text="这是一个非常非常长的前文内容",
            current_text="的是另一个非常非常长的后文内容",
        )
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertFalse(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_final_counts"]["dangling_prefix_suffix_count"], 1)
        self.assertEqual(result.report["final_visible_repair_unresolved"][0]["reason"], "no_safe_deterministic_repair_available")

    def test_final_visible_rebalances_leading_de_when_full_merge_would_be_too_long(self) -> None:
        source_graph = _graph_for_single_subtitle_words(
            [
                ("w001", "所以你只能", 0, 600_000),
                ("w002", "逮住身边的仅有", 600_000, 1_500_000),
                ("w003", "的", 1_500_000, 1_700_000),
                ("w004", "一个普通的女同学女同事", 1_700_000, 3_000_000),
            ]
        )
        timeline = [
            replace(
                _segment(1, 0, 3_000_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=3_000_000,
                target_start_us=0,
                target_end_us=3_000_000,
                word_ids=["w001", "w002", "w003", "w004"],
                text="所以你只能逮住身边的仅有的一个普通的女同学女同事",
            )
        ]
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002"],
                "所以你只能逮住身边的仅有",
                0,
                1_500_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=1_500_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000001"],
                ["w003", "w004"],
                "的一个普通的女同学女同事",
                1_500_000,
                3_000_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=1_500_000,
                spoken_source_end_us=3_000_000,
                containing_video_segment_id="v21_seg_000001",
            ),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: captions,
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report)
        self.assertIn(
            "transfer_leading_function_prefix_to_previous_caption",
            [action["decision"] for action in result.report["final_visible_repair_actions"]],
        )
        self.assertEqual([caption.text for caption in result.captions], ["所以你只能逮住身边的仅有的", "一个普通的女同学女同事"])
        self.assertEqual([segment.segment_id for segment in result.final_timeline], ["v21_seg_000001"])
        self.assertTrue(all(len(caption.text) <= 20 for caption in result.captions))
        self.assertTrue(build_final_caption_visible_repeat_gate(result.captions)["gate_passed"])

    def test_caption_only_finalize_can_rebalance_leading_de_when_merge_too_long(self) -> None:
        source_graph = _graph_for_single_subtitle_words(
            [
                ("w001", "所以你只能", 0, 600_000),
                ("w002", "逮住身边的仅有", 600_000, 1_500_000),
                ("w003", "的", 1_500_000, 1_700_000),
                ("w004", "一个普通的女同学女同事", 1_700_000, 3_000_000),
            ]
        )
        captions = [
            CaptionRenderUnit(
                "v21_cap_000001",
                ["v21_seg_000001"],
                ["w001", "w002"],
                "所以你只能逮住身边的仅有",
                0,
                1_500_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=1_500_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                "v21_cap_000002",
                ["v21_seg_000001"],
                ["w003", "w004"],
                "的一个普通的女同学女同事",
                1_500_000,
                3_000_000,
                ["s001"],
                "canonical_caption_template",
                spoken_source_start_us=1_500_000,
                spoken_source_end_us=3_000_000,
                containing_video_segment_id="v21_seg_000001",
            ),
        ]

        repaired, actions = final_visible_repair_module._finalize_caption_only_dangling_merges(
            captions,
            source_graph=source_graph,
            pass_index_start=1,
        )

        self.assertEqual([caption.text for caption in repaired], ["所以你只能逮住身边的仅有的", "一个普通的女同学女同事"])
        self.assertEqual(actions[0]["decision"], "transfer_leading_function_prefix_to_previous_caption")

    def test_caption_only_merge_materializes_consumed_caption_state(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair()
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        action = result.report["final_visible_repair_actions"][0]
        self.assertTrue(action["caption_only_merge_materialized"])
        self.assertEqual(action["consumed_caption_state"], "consumed_by_caption_only_merge")
        self.assertEqual(action["merged_into_caption_id"], "v21_cap_000001")
        self.assertEqual(action["consumed_caption_id"], "v21_cap_000002")
        self.assertEqual(result.report["caption_only_materialized_merge_count"], 1)
        self.assertEqual(result.report["caption_only_materialized_merges"][0]["state"], "materialized_caption_only_merge")
        self.assertEqual(result.report["caption_only_materialized_merges"][0]["consumed_timeline_segment_ids"], ["v21_seg_000002"])

    def test_caption_only_merge_removes_dangling_from_effective_visible_captions(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair()
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertEqual([caption.text for caption in result.captions], ["我们反对的是公共问题"])
        self.assertEqual(result.report["final_visible_effective_caption_count"], 1)
        self.assertEqual(result.report["caption_only_consumed_caption_ids"], ["v21_timeline_cap_000002"])
        self.assertNotIn("V21_FINAL_VISIBLE_DANGLING_PREFIX_SUFFIX", result.report["final_visible_repair_final_timeline_blocker_codes"])

    def test_caption_only_merge_makes_final_timeline_counts_zero(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair()
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertEqual(
            result.report["final_visible_repair_final_timeline_counts"],
            {
                "dangling_prefix_suffix_count": 0,
                "semantic_garbage_or_asr_suspect_count": 0,
                "semantic_integrity_count": 0,
                "cross_caption_semantic_containment_count": 0,
                "restart_repeat_visible_count": 0,
            },
        )
        attached = ArollEngine()._attach_final_caption_visible_repeat_gate(
            {
                "validator_report_ok": True,
                "final_visible_caption_repair_report": result.report,
            },
            result.captions,
        )
        attached_gate = attached["final_caption_visible_repeat_gate"]
        self.assertTrue(attached["validator_report_ok"])
        self.assertTrue(attached_gate["final_visible_repair_success"])
        self.assertEqual(attached_gate["final_visible_repair_unresolved_count"], 0)
        self.assertEqual(attached_gate["final_visible_repair_final_timeline_counts"]["dangling_prefix_suffix_count"], 0)

    def test_caption_only_merge_does_not_change_video_segments(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair()
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertEqual(len(result.final_timeline), len(timeline))
        self.assertEqual([segment.segment_id for segment in result.final_timeline], [segment.segment_id for segment in timeline])
        self.assertEqual([segment.source_material_id for segment in result.final_timeline], [segment.source_material_id for segment in timeline])
        self.assertEqual([segment.source_segment_id for segment in result.final_timeline], [segment.source_segment_id for segment in timeline])
        self.assertEqual([segment.source_start_us for segment in result.final_timeline], [segment.source_start_us for segment in timeline])
        self.assertEqual([segment.source_end_us for segment in result.final_timeline], [segment.source_end_us for segment in timeline])

    def test_final_visible_repair_unresolved_zero_after_materialized_caption_merge(self) -> None:
        source_graph, timeline = _timeline_with_source_gap_dangling_pair()
        renderer = SubtitleRenderer()
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_unresolved"], [])
        self.assertEqual(result.report["final_visible_repair_final_counts"]["dangling_prefix_suffix_count"], 0)
        self.assertEqual(result.report["final_visible_repair_final_timeline_counts"]["dangling_prefix_suffix_count"], 0)

    def test_caption_only_materialization_supports_partial_previous_segment_tail(self) -> None:
        source_graph, timeline = _timeline_with_partial_tail_dangling_pair()
        renderer = SubtitleRenderer()
        captions = [
            CaptionRenderUnit(
                caption_id="v21_cap_000001",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w001"],
                text="前文铺垫",
                target_start_us=0,
                target_end_us=300_000,
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                spoken_source_start_us=0,
                spoken_source_end_us=300_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000002",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w002"],
                text="你嘲笑嘉豪",
                target_start_us=300_000,
                target_end_us=800_000,
                source_subtitle_uids=["s002"],
                style_template_id="canonical_caption_template",
                spoken_source_start_us=300_000,
                spoken_source_end_us=800_000,
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000003",
                timeline_segment_ids=["v21_seg_000002"],
                word_ids=["w003"],
                text="的是对自己人的规训",
                target_start_us=800_000,
                target_end_us=1_600_000,
                source_subtitle_uids=["s003"],
                style_template_id="canonical_caption_template",
                spoken_source_start_us=2_320_000,
                spoken_source_end_us=3_120_000,
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["caption_only_materialized_merge_count"], 1)
        self.assertEqual(result.report["final_visible_repair_unresolved"], [])
        self.assertEqual(result.report["final_visible_repair_final_timeline_counts"]["dangling_prefix_suffix_count"], 0)
        self.assertEqual([caption.text for caption in result.captions], ["前文铺垫", "你嘲笑嘉豪是对自己人的规训"])
        self.assertEqual(result.report["caption_only_materialized_merges"][0]["materialization_type"], "partial_previous_segment_tail")
        self.assertEqual(result.report["caption_only_materialized_merges"][0]["covered_previous_tail_word_ids"], ["w002"])
        self.assertEqual(result.report["caption_only_materialized_merges"][0]["preserved_previous_prefix_word_ids"], ["w001"])
        self.assertEqual(len(result.final_timeline), 2)
        self.assertEqual([segment.source_start_us for segment in result.final_timeline], [0, 2_320_000])
        self.assertEqual([segment.source_end_us for segment in result.final_timeline], [800_000, 3_120_000])

    def test_caption_only_materialization_survives_later_timeline_repair(self) -> None:
        materials, text_segments = _template_rows()
        words = [
            {"word_id": "w001", "word_text": "前文铺垫", "start_us": 0, "end_us": 300_000, "subtitle_index": 1, "subtitle_uid": "s001"},
            {"word_id": "w002", "word_text": "你嘲笑嘉豪", "start_us": 300_000, "end_us": 800_000, "subtitle_index": 2, "subtitle_uid": "s002"},
            {"word_id": "w003", "word_text": "的是对自己人的规训", "start_us": 2_320_000, "end_us": 3_120_000, "subtitle_index": 3, "subtitle_uid": "s003"},
            {"word_id": "w004", "word_text": "你", "start_us": 3_300_000, "end_us": 3_420_000, "subtitle_index": 4, "subtitle_uid": "s004"},
            {"word_id": "w005", "word_text": "是", "start_us": 3_420_000, "end_us": 3_540_000, "subtitle_index": 4, "subtitle_uid": "s004"},
            {"word_id": "w006", "word_text": "你们", "start_us": 3_540_000, "end_us": 3_720_000, "subtitle_index": 4, "subtitle_uid": "s004"},
            {"word_id": "w007", "word_text": "是", "start_us": 3_720_000, "end_us": 3_840_000, "subtitle_index": 4, "subtitle_uid": "s004"},
            {"word_id": "w008", "word_text": "极度恐慌", "start_us": 3_840_000, "end_us": 4_300_000, "subtitle_index": 4, "subtitle_uid": "s004"},
        ]
        source_graph = ArollEngine().ingest.build_source_graph(
            word_timeline=words,
            subtitles=[
                {
                    "subtitle_uid": uid,
                    "subtitle_index": index,
                    "text": "".join(row["word_text"] for row in words if row["subtitle_uid"] == uid),
                    "word_ids": [row["word_id"] for row in words if row["subtitle_uid"] == uid],
                }
                for index, uid in [(1, "s001"), (2, "s002"), (3, "s003"), (4, "s004")]
            ],
            source_segments=[{"id": "primary_window", "material_id": "main", "type": "video", "source_start_us": 0, "source_end_us": 4_800_000}],
            text_materials=materials,
            text_segments=text_segments,
        )
        timeline = [
            replace(
                _segment(1, 0, 800_000),
                source_material_id="main_a",
                source_segment_id="source_a",
                source_start_us=0,
                source_end_us=800_000,
                target_start_us=0,
                target_end_us=800_000,
                word_ids=["w001", "w002"],
                text="前文铺垫你嘲笑嘉豪",
            ),
            replace(
                _segment(2, 800_000, 1_600_000),
                source_material_id="main_b",
                source_segment_id="source_b",
                source_start_us=2_320_000,
                source_end_us=3_120_000,
                target_start_us=800_000,
                target_end_us=1_600_000,
                word_ids=["w003"],
                text="的是对自己人的规训",
            ),
            replace(
                _segment(3, 1_600_000, 2_600_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=3_300_000,
                source_end_us=4_300_000,
                target_start_us=1_600_000,
                target_end_us=2_600_000,
                word_ids=["w004", "w005", "w006", "w007", "w008"],
                text="你是你们是极度恐慌",
            ),
        ]
        captions = [
            CaptionRenderUnit("v21_cap_000001", ["v21_seg_000001"], ["w001"], "前文铺垫", 0, 300_000, ["s001"], "canonical_caption_template", spoken_source_start_us=0, spoken_source_end_us=300_000, containing_video_segment_id="v21_seg_000001"),
            CaptionRenderUnit("v21_cap_000002", ["v21_seg_000001"], ["w002"], "你嘲笑嘉豪", 300_000, 800_000, ["s002"], "canonical_caption_template", spoken_source_start_us=300_000, spoken_source_end_us=800_000, containing_video_segment_id="v21_seg_000001"),
            CaptionRenderUnit("v21_cap_000003", ["v21_seg_000002"], ["w003"], "的是对自己人的规训", 800_000, 1_600_000, ["s003"], "canonical_caption_template", spoken_source_start_us=2_320_000, spoken_source_end_us=3_120_000, containing_video_segment_id="v21_seg_000002"),
            CaptionRenderUnit("v21_cap_000004", ["v21_seg_000003"], ["w004", "w005", "w006", "w007", "w008"], "你是你们是极度恐慌", 1_600_000, 2_600_000, ["s004"], "canonical_caption_template", spoken_source_start_us=3_300_000, spoken_source_end_us=4_300_000, containing_video_segment_id="v21_seg_000003"),
        ]
        renderer = SubtitleRenderer()

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["caption_only_materialized_merge_count"], 1)
        self.assertEqual(result.report["final_visible_repair_final_timeline_counts"]["dangling_prefix_suffix_count"], 0)
        self.assertEqual(
            [
                action["decision"]
                for action in result.report["final_visible_repair_actions"]
                if action["decision"] == "caption_only_merge_with_previous"
            ],
            ["caption_only_merge_with_previous"],
        )
        self.assertIn("你嘲笑嘉豪是对自己人的规训", [caption.text for caption in result.captions])
        self.assertNotIn("的是对自己人的规训", [caption.text for caption in result.captions])
        self.assertIn("你们是极度恐慌", [segment.text for segment in result.final_timeline])

    def test_final_visible_repairs_source_boundary_function_prefix(self) -> None:
        rows = [
            ("w001", "前面内容", 0, 700_000),
            ("w002", "就", 700_000, 820_000),
            ("w003", "有了继续表达的底气", 820_000, 1_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 820_000, 1_600_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=820_000,
                source_end_us=1_600_000,
                target_start_us=0,
                target_end_us=780_000,
                word_ids=["w003"],
                text="有了继续表达的底气",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["source_boundary_prefix_repair_count"], 1)
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "prepend_source_boundary_prefix")
        self.assertEqual(
            result.report["final_visible_repair_actions"][0]["repair_transaction_rule_name"],
            "source_boundary_prefix_gap",
        )
        self.assertEqual(result.final_timeline[0].word_ids, ["w002", "w003"])
        self.assertEqual(result.final_timeline[0].text, "就有了继续表达的底气")
        self.assertEqual([caption.text for caption in result.captions], ["就有了继续表达的底气"])

    def test_final_visible_prefers_later_complete_take_before_de_shi_duplicate(self) -> None:
        rows = [
            ("w001", "你", 0, 200_000),
            ("w002", "嘲笑", 240_000, 480_000),
            ("w003", "嘉豪", 560_000, 840_000),
            ("w004", "是", 1_000_000, 1_160_000),
            ("w005", "对", 1_160_000, 1_320_000),
            ("w006", "自己", 1_320_000, 1_520_000),
            ("w007", "人", 1_520_000, 1_640_000),
            ("w008", "的", 1_640_000, 1_800_000),
            ("w009", "是", 2_000_000, 2_160_000),
            ("w010", "对", 2_160_000, 2_320_000),
            ("w011", "自己", 2_320_000, 2_520_000),
            ("w012", "人", 2_520_000, 2_640_000),
            ("w013", "的", 2_640_000, 2_800_000),
            ("w014", "规训", 2_840_000, 3_120_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 840_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=840_000,
                target_start_us=0,
                target_end_us=840_000,
                word_ids=["w001", "w002", "w003"],
                text="你嘲笑嘉豪",
            ),
            replace(
                _segment(2, 840_000, 2_320_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_640_000,
                source_end_us=3_120_000,
                target_start_us=840_000,
                target_end_us=2_320_000,
                word_ids=["w008", "w009", "w010", "w011", "w012", "w013", "w014"],
                text="的是对自己人的规训",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertIn(
            "keep_later_complete_take_for_de_shi_duplicate",
            [action["decision"] for action in result.report["final_visible_repair_actions"]],
        )
        self.assertNotIn(
            "bridge_omitted_source_tail_and_trim_de_shi_duplicate",
            [action["decision"] for action in result.report["final_visible_repair_actions"]],
        )
        self.assertEqual(result.final_timeline[0].text, "你嘲笑嘉豪")
        self.assertEqual(result.final_timeline[1].text, "是对自己人的规训")
        self.assertEqual(result.final_timeline[0].word_ids, ["w001", "w002", "w003"])
        self.assertEqual(result.final_timeline[1].word_ids, ["w009", "w010", "w011", "w012", "w013", "w014"])
        self.assertEqual([caption.text for caption in result.captions], ["你嘲笑嘉豪", "是对自己人的规训"])

    def test_de_shi_bridge_suffix_is_not_dropped_as_isolated_junk_before_next_caption(self) -> None:
        rows = [
            ("w001", "你", 0, 200_000),
            ("w002", "嘲笑", 240_000, 480_000),
            ("w003", "嘉豪", 560_000, 840_000),
            ("w004", "是", 1_000_000, 1_160_000),
            ("w005", "对", 1_160_000, 1_320_000),
            ("w006", "自己", 1_320_000, 1_520_000),
            ("w007", "人", 1_520_000, 1_640_000),
            ("w008", "的", 1_640_000, 1_800_000),
            ("w009", "是", 2_000_000, 2_160_000),
            ("w010", "对", 2_160_000, 2_320_000),
            ("w011", "自己", 2_320_000, 2_520_000),
            ("w012", "人", 2_520_000, 2_640_000),
            ("w013", "的", 2_640_000, 2_800_000),
            ("w014", "规训", 2_840_000, 3_120_000),
            ("w015", "你其实是在嘲笑年少时期的自己", 4_200_000, 5_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 840_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=840_000,
                target_start_us=0,
                target_end_us=840_000,
                word_ids=["w001", "w002", "w003"],
                text="你嘲笑嘉豪",
            ),
            replace(
                _segment(2, 840_000, 2_320_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_640_000,
                source_end_us=3_120_000,
                target_start_us=840_000,
                target_end_us=2_320_000,
                word_ids=["w008", "w009", "w010", "w011", "w012", "w013", "w014"],
                text="的是对自己人的规训",
            ),
            replace(
                _segment(3, 2_320_000, 3_720_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=4_200_000,
                source_end_us=5_600_000,
                target_start_us=2_320_000,
                target_end_us=3_720_000,
                word_ids=["w015"],
                text="你其实是在嘲笑年少时期的自己",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        actions = result.report["final_visible_repair_actions"]
        self.assertIn("keep_later_complete_take_for_de_shi_duplicate", [action["decision"] for action in actions])
        self.assertNotIn(
            "的规训",
            [
                action.get("junk_text")
                for action in actions
                if action["decision"] == "drop_isolated_junk_segment"
            ],
        )
        self.assertEqual([segment.text for segment in result.final_timeline], ["你嘲笑嘉豪", "是对自己人的规训", "你其实是在嘲笑年少时期的自己"])
        self.assertEqual([caption.text for caption in result.captions], ["你嘲笑嘉豪", "是对自己人的规训", "你其实是在嘲笑年少时期的自己"])

    def test_final_visible_drops_isolated_semantic_junk_caption(self) -> None:
        rows = [
            ("w001", "出现在周围所有人的视线里", 0, 1_200_000),
            ("w002", "交配", 1_600_000, 1_840_000),
            ("w003", "权", 1_960_000, 2_120_000),
            ("w004", "跟着老子把输掉的", 2_600_000, 3_800_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(_segment(1, 0, 1_200_000), source_material_id="main", source_segment_id="primary_window", source_start_us=0, source_end_us=1_200_000, word_ids=["w001"], text="出现在周围所有人的视线里"),
            replace(_segment(2, 1_200_000, 1_720_000), source_material_id="main", source_segment_id="primary_window", source_start_us=1_600_000, source_end_us=2_120_000, word_ids=["w002", "w003"], text="交配权"),
            replace(_segment(3, 1_720_000, 2_920_000), source_material_id="main", source_segment_id="primary_window", source_start_us=2_600_000, source_end_us=3_800_000, word_ids=["w004"], text="跟着老子把输掉的"),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertIn("drop_isolated_junk_segment", [action["decision"] for action in result.report["final_visible_repair_actions"]])
        self.assertIn(
            "isolated_semantic_junk_caption",
            result.report["final_visible_repair_transaction_rule_names"],
        )
        self.assertNotIn("交配权", [segment.text for segment in result.final_timeline])
        self.assertNotIn("交配权", [caption.text for caption in result.captions])

    def test_final_visible_keeps_four_char_isolated_content_phrase(self) -> None:
        rows = [
            ("w001", "才导致国男在婚恋市场上", 0, 1_200_000),
            ("w002", "举步维艰", 1_600_000, 2_280_000),
            ("w003", "最后只能像一个小丑", 2_720_000, 4_000_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_200_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_200_000,
                word_ids=["w001"],
                text="才导致国男在婚恋市场上",
            ),
            replace(
                _segment(2, 1_200_000, 1_880_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_600_000,
                source_end_us=2_280_000,
                word_ids=["w002"],
                text="举步维艰",
            ),
            replace(
                _segment(3, 1_880_000, 3_160_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=2_720_000,
                source_end_us=4_000_000,
                word_ids=["w003"],
                text="最后只能像一个小丑",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertNotIn(
            "举步维艰",
            [
                action.get("junk_text")
                for action in result.report["final_visible_repair_actions"]
                if action["decision"] in {"drop_isolated_junk_segment", "trim_isolated_junk_words"}
            ],
        )
        self.assertIn("举步维艰", [segment.text for segment in result.final_timeline])
        self.assertIn("举步维艰", [caption.text for caption in result.captions])

    def test_final_visible_drops_four_char_low_information_isolated_fragment(self) -> None:
        rows = [
            ("w001", "她根本就不在乎别人评价", 0, 1_200_000),
            ("w002", "就", 4_400_000, 4_680_000),
            ("w003", "根本", 4_680_000, 4_960_000),
            ("w004", "就", 4_960_000, 5_400_000),
            ("w005", "集美她发那种精修图", 7_000_000, 8_200_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_200_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_200_000,
                target_start_us=0,
                target_end_us=1_200_000,
                word_ids=["w001"],
                text="她根本就不在乎别人评价",
            ),
            replace(
                _segment(2, 1_200_000, 2_200_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=4_400_000,
                source_end_us=5_400_000,
                target_start_us=1_200_000,
                target_end_us=2_200_000,
                word_ids=["w002", "w003", "w004"],
                text="就根本就",
            ),
            replace(
                _segment(3, 2_200_000, 3_400_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=7_000_000,
                source_end_us=8_200_000,
                target_start_us=2_200_000,
                target_end_us=3_400_000,
                word_ids=["w005"],
                text="集美她发那种精修图",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertIn("drop_isolated_junk_segment", [action["decision"] for action in result.report["final_visible_repair_actions"]])
        self.assertNotIn("就根本就", [segment.text for segment in result.final_timeline])
        self.assertNotIn("就根本就", [caption.text for caption in result.captions])

    def test_final_visible_keeps_action_fragment_before_dependent_object(self) -> None:
        rows = [
            ("w001", "你根本不知道什么叫爽", 0, 1_200_000),
            ("w002", "抱着", 1_600_000, 2_000_000),
            ("w003", "个破游戏就当享受了", 2_360_000, 3_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(_segment(1, 0, 1_200_000), source_material_id="main", source_segment_id="primary_window", source_start_us=0, source_end_us=1_200_000, word_ids=["w001"], text="你根本不知道什么叫爽"),
            replace(_segment(2, 1_200_000, 1_600_000), source_material_id="main", source_segment_id="primary_window", source_start_us=1_600_000, source_end_us=2_000_000, word_ids=["w002"], text="抱着"),
            replace(_segment(3, 1_600_000, 2_840_000), source_material_id="main", source_segment_id="primary_window", source_start_us=2_360_000, source_end_us=3_600_000, word_ids=["w003"], text="个破游戏就当享受了"),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertIn("抱着", [segment.text for segment in result.final_timeline])
        self.assertNotIn(
            "抱着",
            [
                action.get("junk_text")
                for action in result.report["final_visible_repair_actions"]
                if action["decision"] in {"drop_isolated_junk_segment", "trim_isolated_junk_words"}
            ],
        )

    def test_final_visible_keeps_expressive_tail_after_reaction_phrase(self) -> None:
        rows = [
            ("w001", "然后在心里冷笑一声哼", 0, 1_200_000),
            ("w002", "闹麻了", 1_720_000, 2_160_000),
            ("w003", "然后晚上就直接去下一个派对了", 2_680_000, 4_000_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_200_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_200_000,
                target_start_us=0,
                target_end_us=1_200_000,
                word_ids=["w001"],
                text="然后在心里冷笑一声哼",
            ),
            replace(
                _segment(2, 1_200_000, 1_640_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_720_000,
                source_end_us=2_160_000,
                target_start_us=1_200_000,
                target_end_us=1_640_000,
                word_ids=["w002"],
                text="闹麻了",
            ),
            replace(
                _segment(3, 1_640_000, 2_960_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=2_680_000,
                source_end_us=4_000_000,
                target_start_us=1_640_000,
                target_end_us=2_960_000,
                word_ids=["w003"],
                text="然后晚上就直接去下一个派对了",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertIn("闹麻了", [segment.text for segment in result.final_timeline])
        self.assertNotIn(
            "闹麻了",
            [
                action.get("junk_text")
                for action in result.report["final_visible_repair_actions"]
                if action["decision"] in {"drop_isolated_junk_segment", "trim_isolated_junk_words"}
            ],
        )

    def test_final_visible_trims_isolated_junk_suffix_from_shared_segment(self) -> None:
        rows = [
            ("w001", "出现在周围所有人的视线里", 0, 1_200_000),
            ("w002", "交配", 1_600_000, 1_840_000),
            ("w003", "权", 1_960_000, 2_120_000),
            ("w004", "跟着老子把输掉的", 2_600_000, 3_800_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_720_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_120_000,
                word_ids=["w001", "w002", "w003"],
                text="出现在周围所有人的视线里交配权",
            ),
            replace(
                _segment(2, 1_720_000, 2_920_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=2_600_000,
                source_end_us=3_800_000,
                word_ids=["w004"],
                text="跟着老子把输掉的",
            ),
        ]
        captions = [
            CaptionRenderUnit(
                caption_id="v21_cap_000001",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w001"],
                text="出现在周围所有人的视线里",
                target_start_us=0,
                target_end_us=1_200_000,
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000002",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w002", "w003"],
                text="交配权",
                target_start_us=1_200_000,
                target_end_us=1_720_000,
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000003",
                timeline_segment_ids=["v21_seg_000002"],
                word_ids=["w004"],
                text="跟着老子把输掉的",
                target_start_us=1_720_000,
                target_end_us=2_920_000,
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000002",
            ),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        action = result.report["final_visible_repair_actions"][0]
        self.assertEqual(action["decision"], "trim_isolated_junk_words")
        self.assertEqual(action["trimmed_segment_ids"], ["v21_seg_000001"])
        self.assertEqual(result.final_timeline[0].word_ids, ["w001"])
        self.assertNotIn("交配权", [segment.text for segment in result.final_timeline])

    def test_final_visible_trims_same_segment_de_duplicate_prefix(self) -> None:
        rows = [
            ("w001", "全", 0, 200_000),
            ("w002", "是", 200_000, 360_000),
            ("w003", "provider", 440_000, 900_000),
            ("w004", "的", 900_000, 940_000),
            ("w005", "全", 1_200_000, 1_360_000),
            ("w006", "是", 1_360_000, 1_520_000),
            ("w007", "provider", 1_640_000, 2_000_000),
            ("w008", "的", 2_040_000, 2_120_000),
            ("w009", "流水账", 2_200_000, 2_800_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_800_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_800_000,
                target_start_us=0,
                target_end_us=2_800_000,
                word_ids=[row[0] for row in rows],
                text="全是provider的全是provider的流水账",
            )
        ]
        captions = [
            CaptionRenderUnit(
                caption_id="v21_cap_000001",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w004", "w005", "w006", "w007", "w008", "w009"],
                text="的全是provider的流水账",
                target_start_us=900_000,
                target_end_us=2_800_000,
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            )
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertIn(
            "trim_same_segment_de_duplicate_prefix",
            [action["decision"] for action in result.report["final_visible_repair_actions"]],
        )
        self.assertEqual([segment.text for segment in result.final_timeline], ["全是provider", "的流水账"])
        self.assertEqual(result.final_timeline[0].word_ids, ["w001", "w002", "w003"])
        self.assertEqual(result.final_timeline[1].word_ids, ["w008", "w009"])
        self.assertEqual([caption.text for caption in result.captions], ["全是provider的流水账"])
        self.assertEqual(result.captions[0].timeline_segment_ids, ["v21_seg_000001", "v21_seg_000001_split_001"])
        _, visual = VisualPacingNormalizer().normalize(result.final_timeline, source_graph)
        self.assertTrue(visual["visual_merge_safety_gate_passed"])
        self.assertEqual(visual["unsafe_merge_group_count"], 0)
        self.assertEqual(visual["dropped_content_reintroduced_count"], 0)

    def test_final_visible_moves_selected_source_boundary_prefix_from_previous_segment(self) -> None:
        rows = [
            ("w001", "让一个普通的四分女", 0, 1_280_000),
            ("w002", "就", 1_280_000, 1_480_000),
            ("w003", "有了敢张口管你要38万彩礼的底气", 1_506_666, 4_340_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_480_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_480_000,
                target_start_us=0,
                target_end_us=1_480_000,
                word_ids=["w001", "w002"],
                text="让一个普通的四分女就",
            ),
            replace(
                _segment(2, 1_480_000, 4_313_334),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_506_666,
                source_end_us=4_340_000,
                target_start_us=1_480_000,
                target_end_us=4_313_334,
                word_ids=["w003"],
                text="有了敢张口管你要38万彩礼的底气",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["source_boundary_prefix_repair_count"], 1)
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "prepend_source_boundary_prefix")
        self.assertEqual(result.report["final_visible_repair_actions"][0]["transferred_from_segment_id"], "v21_seg_000001")
        self.assertEqual(result.final_timeline[0].word_ids, ["w001"])
        self.assertEqual(result.final_timeline[1].word_ids, ["w002", "w003"])
        self.assertEqual([caption.text for caption in result.captions], ["让一个普通的四分女", "就有了敢张口管你要38万彩礼的底气"])

    def test_final_visible_repairs_internal_pivot_restart_prefix(self) -> None:
        rows = [
            ("w001", "你", 0, 120_000),
            ("w002", "是", 120_000, 240_000),
            ("w003", "你们", 240_000, 420_000),
            ("w004", "是", 420_000, 540_000),
            ("w005", "极度恐慌", 540_000, 1_100_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_100_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_100_000,
                target_start_us=0,
                target_end_us=1_100_000,
                word_ids=["w001", "w002", "w003", "w004", "w005"],
                text="你是你们是极度恐慌",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "trim_restart_prefix")
        self.assertTrue(
            result.report["final_visible_repair_actions"][0]["repair_transaction_rule_name"].startswith("gate_candidate."),
            result.report["final_visible_repair_actions"][0],
        )
        self.assertEqual(result.final_timeline[0].word_ids, ["w003", "w004", "w005"])
        self.assertEqual(result.final_timeline[0].text, "你们是极度恐慌")
        self.assertEqual(result.report["final_visible_repair_final_counts"]["restart_repeat_visible_count"], 0)

    def test_final_visible_repairs_fuzzy_internal_prefix_restart(self) -> None:
        rows = [
            ("w001", "你", 0, 120_000),
            ("w002", "去找", 120_000, 360_000),
            ("w003", "那种", 360_000, 600_000),
            ("w004", "漂亮", 600_000, 900_000),
            ("w005", "你", 980_000, 1_100_000),
            ("w006", "找", 1_100_000, 1_240_000),
            ("w007", "那种", 1_240_000, 1_480_000),
            ("w008", "漂亮", 1_480_000, 1_720_000),
            ("w009", "女生的后台私信去看看", 1_720_000, 2_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_600_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_600_000,
                target_start_us=0,
                target_end_us=2_600_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows],
                text="你去找那种漂亮你找那种漂亮女生的后台私信去看看",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.final_timeline[0].word_ids[:5], ["w005", "w006", "w007", "w008", "w009"])
        self.assertEqual(result.final_timeline[0].text, "你找那种漂亮女生的后台私信去看看")
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "trim_restart_prefix")

    def test_final_visible_repairs_partial_phrase_tail_mismatch_restart(self) -> None:
        rows = [
            ("w001", "那为什么你验资", 0, 560_000),
            ("w002", "失败后", 560_000, 920_000),
            ("w003", "马上", 1_000_000, 1_160_000),
            ("w004", "立刻", 1_160_000, 1_340_000),
            ("w005", "滚", 1_340_000, 1_500_000),
            ("w006", "去", 1_500_000, 1_740_000),
            ("w007", "马上", 2_000_000, 2_160_000),
            ("w008", "立刻", 2_160_000, 2_340_000),
            ("w009", "滚", 2_340_000, 2_500_000),
            ("w010", "下桌呢", 2_500_000, 3_000_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_840_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_740_000,
                target_start_us=0,
                target_end_us=1_740_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[:6]],
                text="那为什么你验资失败后马上立刻滚去",
            ),
            replace(
                _segment(2, 1_740_000, 2_740_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=2_000_000,
                source_end_us=3_000_000,
                target_start_us=1_740_000,
                target_end_us=2_740_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[6:]],
                text="马上立刻滚下桌呢",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["那为什么你验资失败后", "马上立刻滚下桌呢"])
        self.assertNotIn("马上立刻滚去", [caption.text for caption in result.captions])
        self.assertIn("drop_partial_phrase_restart_span", [action["decision"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_repairs_negative_filler_restart_inside_segment(self) -> None:
        rows = [
            ("w001", "你", 0, 120_000),
            ("w002", "根本", 120_000, 320_000),
            ("w003", "不", 320_000, 440_000),
            ("w004", "就", 440_000, 560_000),
            ("w005", "不", 560_000, 680_000),
            ("w006", "知道", 680_000, 900_000),
            ("w007", "什么", 900_000, 1_100_000),
            ("w008", "叫", 1_100_000, 1_220_000),
            ("w009", "爽", 1_220_000, 1_420_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_420_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_420_000,
                target_start_us=0,
                target_end_us=1_420_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows],
                text="你根本不就不知道什么叫爽",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "你根本不知道什么叫爽")
        self.assertEqual(
            [word_id for segment in result.final_timeline for word_id in segment.word_ids],
            ["w001", "w002", "w005", "w006", "w007", "w008", "w009"],
        )
        self.assertIn("drop_negative_predicate_restart_span", [action["decision"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_repair_short_merge_refreshes_spoken_and_clip_bounds(self) -> None:
        rows = [
            ("w001", "这辈子没见过", 0, 1_500_000),
            ("w002", "你", 1_500_000, 1_620_000),
            ("w003", "根本", 1_620_000, 1_820_000),
            ("w004", "不", 1_820_000, 1_940_000),
            ("w005", "就", 1_940_000, 2_060_000),
            ("w006", "不", 2_060_000, 2_180_000),
            ("w007", "知道", 2_180_000, 2_400_000),
            ("w008", "什么", 2_400_000, 2_600_000),
            ("w009", "叫", 2_600_000, 2_720_000),
            ("w010", "爽", 2_720_000, 3_120_000),
            ("w011", "抱着", 3_420_000, 3_740_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_500_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_500_000,
                target_start_us=0,
                target_end_us=1_500_000,
                word_ids=["w001"],
                text="这辈子没见过",
                spoken_source_start_us=0,
                spoken_source_end_us=1_500_000,
                clip_source_start_us=0,
                clip_source_end_us=1_500_000,
            ),
            replace(
                _segment(2, 1_500_000, 3_120_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_500_000,
                source_end_us=3_120_000,
                target_start_us=1_500_000,
                target_end_us=3_120_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[1:10]],
                text="你根本不就不知道什么叫爽",
                spoken_source_start_us=1_500_000,
                spoken_source_end_us=3_120_000,
                clip_source_start_us=1_500_000,
                clip_source_end_us=3_120_000,
            ),
            replace(
                _segment(3, 3_120_000, 3_440_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=3_420_000,
                source_end_us=3_740_000,
                target_start_us=3_120_000,
                target_end_us=3_440_000,
                word_ids=["w011"],
                text="抱着",
                spoken_source_start_us=None,
                spoken_source_end_us=None,
                clip_source_start_us=None,
                clip_source_end_us=None,
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["这辈子没见过你根本", "不知道什么叫爽抱着"])
        merged_suffix = result.final_timeline[1]
        self.assertEqual(merged_suffix.source_start_us, 2_060_000)
        self.assertEqual(merged_suffix.source_end_us, 3_740_000)
        self.assertEqual(merged_suffix.spoken_source_start_us, 2_060_000)
        self.assertEqual(merged_suffix.spoken_source_end_us, 3_740_000)
        self.assertEqual(merged_suffix.clip_source_start_us, 2_060_000)
        self.assertEqual(merged_suffix.clip_source_end_us, 3_740_000)
        self.assertEqual(merged_suffix.target_end_us - merged_suffix.target_start_us, 1_680_000)
        self.assertIn("drop_negative_predicate_restart_span", [action["decision"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_repairs_partial_boundary_restart_tail(self) -> None:
        rows = [
            ("w001", "任何", 0, 240_000),
            ("w002", "一个", 240_000, 480_000),
            ("w003", "四分", 480_000, 760_000),
            ("w004", "女", 760_000, 960_000),
            ("w005", "给", 1_000_000, 1_160_000),
            ("w006", "你", 1_160_000, 1_300_000),
            ("w007", "一", 1_300_000, 1_520_000),
            ("w008", "给", 1_560_000, 1_720_000),
            ("w009", "你", 1_720_000, 1_860_000),
            ("w010", "四分", 1_940_000, 2_260_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_520_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_520_000,
                target_start_us=0,
                target_end_us=1_520_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[:7]],
                text="任何一个四分女给你一",
            ),
            replace(
                _segment(2, 1_520_000, 2_220_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_560_000,
                source_end_us=2_260_000,
                target_start_us=1_520_000,
                target_end_us=2_220_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[7:]],
                text="给你四分",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["任何一个四分女", "给你四分"])
        self.assertIn("boundary_restart", [action["issue_type"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_repairs_source_tail_restart_before_unselected_completion(self) -> None:
        rows = [
            ("w001", "你", 0, 160_000),
            ("w002", "都", 160_000, 320_000),
            ("w003", "不知道", 320_000, 560_000),
            ("w004", "是", 560_000, 720_000),
            ("w005", "怎么", 720_000, 920_000),
            ("w006", "输掉", 920_000, 1_240_000),
            ("w007", "的", 1_240_000, 1_440_000),
            ("w008", "你", 1_520_000, 1_680_000),
            ("w009", "连", 1_680_000, 1_820_000),
            ("w010", "死", 1_820_000, 1_980_000),
            ("w011", "的", 2_060_000, 2_220_000),
            ("w012", "你", 2_820_000, 2_980_000),
            ("w013", "连", 2_980_000, 3_120_000),
            ("w014", "怎么", 3_120_000, 3_360_000),
            ("w015", "死", 3_360_000, 3_520_000),
            ("w016", "都", 3_640_000, 3_920_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_220_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_220_000,
                target_start_us=0,
                target_end_us=2_220_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[:11]],
                text="你都不知道是怎么输掉的你连死的",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "你都不知道是怎么输掉的")
        self.assertIn("boundary_restart", [action["issue_type"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_source_tail_restart_lookahead_uses_adjacent_word_gaps(self) -> None:
        rows = [
            ("w001", "主句", 0, 420_000),
            ("w002", "你", 500_000, 660_000),
            ("w003", "连", 660_000, 800_000),
            ("w004", "错", 800_000, 960_000),
            ("w005", "的", 1_080_000, 1_240_000),
            ("w006", "你", 2_120_000, 2_280_000),
            ("w007", "连", 2_280_000, 2_420_000),
            ("w008", "怎么", 2_520_000, 2_760_000),
            ("w009", "错", 2_760_000, 2_920_000),
            ("w010", "都", 3_040_000, 3_260_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_240_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_240_000,
                target_start_us=0,
                target_end_us=1_240_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[:5]],
                text="主句你连错的",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "主句")
        self.assertIn("boundary_restart", [action["issue_type"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_drops_short_whole_segment_abandoned_before_source_completion(self) -> None:
        rows = [
            ("w001", "保留", 0, 420_000),
            ("w002", "甲", 700_000, 860_000),
            ("w003", "乙", 860_000, 1_020_000),
            ("w004", "丙", 1_020_000, 1_180_000),
            ("w005", "甲", 1_300_000, 1_460_000),
            ("w006", "丁", 1_620_000, 1_820_000),
            ("w007", "甲", 2_100_000, 2_260_000),
            ("w008", "乙", 2_260_000, 2_420_000),
            ("w009", "丙", 2_420_000, 2_580_000),
            ("w010", "完成", 2_620_000, 2_980_000),
            ("w011", "继续", 3_240_000, 3_560_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 420_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=420_000,
                target_start_us=0,
                target_end_us=420_000,
                word_ids=["w001"],
                text="保留",
            ),
            replace(
                _segment(2, 420_000, 1_540_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=700_000,
                source_end_us=1_820_000,
                target_start_us=420_000,
                target_end_us=1_540_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[1:6]],
                text="甲乙丙甲丁",
            ),
            replace(
                _segment(3, 1_540_000, 1_860_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=3_240_000,
                source_end_us=3_560_000,
                target_start_us=1_540_000,
                target_end_us=1_860_000,
                word_ids=["w011"],
                text="继续",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["保留", "继续"])
        actions = result.report["timeline_repair_proposal_actions"]
        self.assertEqual(actions[0]["target_segment_id"], "v21_seg_000002")
        self.assertEqual(actions[0]["target_word_ids"], ["w002", "w003", "w004", "w005", "w006"])

    def test_final_visible_repairs_next_prefix_duplicate_after_elided_restart_tail(self) -> None:
        rows = [
            ("w001", "前", 0, 160_000),
            ("w002", "半", 160_000, 320_000),
            ("w003", "尾", 320_000, 480_000),
            ("w004", "甲", 480_000, 620_000),
            ("w005", "前", 1_040_000, 1_200_000),
            ("w006", "半", 1_200_000, 1_360_000),
            ("w007", "尾", 1_360_000, 1_520_000),
            ("w008", "甲", 1_520_000, 1_660_000),
            ("w009", "继续", 1_860_000, 2_160_000),
            ("w010", "推进", 2_160_000, 2_460_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 620_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=620_000,
                target_start_us=0,
                target_end_us=620_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[:4]],
                text="前半尾甲",
            ),
            replace(
                _segment(2, 620_000, 1_700_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_520_000,
                source_end_us=2_460_000,
                target_start_us=620_000,
                target_end_us=1_700_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[7:]],
                text="甲继续推进",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["前半尾甲", "继续推进"])
        actions = result.report["timeline_repair_proposal_actions"]
        self.assertEqual(actions[0]["decision"], "suffix_trim")
        self.assertEqual(actions[0]["target_segment_id"], "v21_seg_000002")
        self.assertEqual(actions[0]["target_word_ids"], ["w008"])

    def test_final_visible_repairs_command_shape_restart_after_elided_marker(self) -> None:
        rows = [
            ("w001", "从此", 0, 300_000),
            ("w002", "拿", 320_000, 460_000),
            ("w003", "甲", 460_000, 660_000),
            ("w004", "让", 820_000, 980_000),
            ("w005", "让", 1_060_000, 1_220_000),
            ("w006", "甲", 1_220_000, 1_420_000),
            ("w007", "拿", 1_420_000, 1_560_000),
            ("w008", "任务", 1_560_000, 1_920_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 660_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=660_000,
                target_start_us=0,
                target_end_us=660_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[:3]],
                text="从此拿甲",
            ),
            replace(
                _segment(2, 660_000, 1_520_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_060_000,
                source_end_us=1_920_000,
                target_start_us=660_000,
                target_end_us=1_520_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[4:]],
                text="让甲拿任务",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["从此", "让甲拿任务"])
        actions = result.report["timeline_repair_proposal_actions"]
        self.assertEqual(actions[0]["target_segment_id"], "v21_seg_000001")
        self.assertEqual(actions[0]["target_word_ids"], ["w002", "w003"])

    def test_final_visible_appends_truncated_compound_tail_from_source(self) -> None:
        rows = [
            ("w001", "那么", 0, 220_000),
            ("w002", "就", 220_000, 340_000),
            ("w003", "会", 340_000, 500_000),
            ("w004", "瞬间", 500_000, 740_000),
            ("w005", "触发", 820_000, 1_020_000),
            ("w006", "波", 1_100_000, 1_140_000),
            ("w007", "函数", 1_240_000, 1_440_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_140_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_140_000,
                target_start_us=0,
                target_end_us=1_140_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[:6]],
                text="那么就会瞬间触发波",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "那么就会瞬间触发波函数")
        self.assertIn("append_source_boundary_compound_tail", [action["decision"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_repairs_open_pronoun_preposition_tail(self) -> None:
        rows = [
            ("w001", "他", 0, 120_000),
            ("w002", "发", 120_000, 260_000),
            ("w003", "个", 260_000, 380_000),
            ("w004", "精修", 420_000, 620_000),
            ("w005", "的", 660_000, 780_000),
            ("w006", "自拍", 780_000, 1_120_000),
            ("w007", "你", 1_140_000, 1_300_000),
            ("w008", "就", 1_300_000, 1_440_000),
            ("w009", "躲", 1_480_000, 1_600_000),
            ("w010", "在", 1_600_000, 1_860_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_860_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_860_000,
                target_start_us=0,
                target_end_us=1_860_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows],
                text="他发个精修的自拍你就躲在",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "他发个精修的自拍")
        self.assertIn("trim_dangling_suffix_tail", [action["decision"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_drops_short_weak_pronoun_fragment(self) -> None:
        rows = [
            ("w001", "就", 0, 220_000),
            ("w002", "你", 220_000, 360_000),
            ("w003", "以为", 360_000, 600_000),
            ("w004", "你", 760_000, 960_000),
            ("w005", "没人", 1_240_000, 1_520_000),
            ("w006", "会", 1_520_000, 1_660_000),
            ("w007", "在乎", 1_660_000, 1_860_000),
            ("w008", "你", 1_900_000, 2_020_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 960_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=960_000,
                target_start_us=0,
                target_end_us=960_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[:4]],
                text="就你以为你",
            ),
            replace(
                _segment(2, 960_000, 1_740_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_240_000,
                source_end_us=2_020_000,
                target_start_us=960_000,
                target_end_us=1_740_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows[4:]],
                text="没人会在乎你",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["没人会在乎你"])
        self.assertIn("drop_dangling_weak_pronoun_fragment", [action["decision"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_repairs_strengthened_negative_restart_inside_segment(self) -> None:
        rows = [
            ("w001", "绝对", 0, 180_000),
            ("w002", "不会", 180_000, 380_000),
            ("w003", "绝不", 380_000, 600_000),
            ("w004", "给", 600_000, 720_000),
            ("w005", "任何", 720_000, 920_000),
            ("w006", "一个", 920_000, 1_100_000),
            ("w007", "看不上", 1_100_000, 1_460_000),
            ("w008", "你", 1_460_000, 1_580_000),
            ("w009", "的人", 1_580_000, 1_760_000),
            ("w010", "买礼物", 1_760_000, 2_180_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_180_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_180_000,
                target_start_us=0,
                target_end_us=2_180_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows],
                text="绝对不会绝不给任何一个看不上你的人买礼物",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "绝不给任何一个看不上你的人买礼物")
        self.assertEqual(
            [word_id for segment in result.final_timeline for word_id in segment.word_ids],
            ["w003", "w004", "w005", "w006", "w007", "w008", "w009", "w010"],
        )

    def test_final_visible_accepts_intraword_duplicate_after_source_normalization(self) -> None:
        rows = [
            ("w001", "一一", 0, 220_000),
            ("w002", "节", 220_000, 340_000),
            ("w003", "课", 340_000, 460_000),
            ("w004", "600", 460_000, 720_000),
            ("w005", "块钱", 720_000, 980_000),
            ("w006", "的", 980_000, 1_080_000),
            ("w007", "普拉提", 1_080_000, 1_520_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_520_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_520_000,
                target_start_us=0,
                target_end_us=1_520_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows],
                text="一一节课600块钱的普拉提",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(caption.text for caption in result.captions), "一节课600块钱的普拉提")
        self.assertEqual(result.report["final_visible_repair_final_counts"]["semantic_garbage_or_asr_suspect_count"], 0)

    def test_final_visible_repairs_short_prefix_self_correction_inside_segment(self) -> None:
        rows = [
            ("w001", "把", 0, 120_000),
            ("w002", "生杀", 120_000, 340_000),
            ("w003", "大权", 340_000, 560_000),
            ("w004", "双手", 560_000, 780_000),
            ("w005", "乖乖", 780_000, 1_000_000),
            ("w006", "奉", 1_000_000, 1_120_000),
            ("w007", "双手", 1_120_000, 1_340_000),
            ("w008", "奉上", 1_340_000, 1_620_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 1_620_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_620_000,
                target_start_us=0,
                target_end_us=1_620_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows],
                text="把生杀大权双手乖乖奉双手奉上",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "把生杀大权双手奉上")
        self.assertEqual(
            [word_id for segment in result.final_timeline for word_id in segment.word_ids],
            ["w001", "w002", "w003", "w007", "w008"],
        )

    def test_final_visible_repairs_abandoned_clause_before_restarted_prefix(self) -> None:
        rows = [
            ("w001", "你", 0, 120_000),
            ("w002", "被", 120_000, 240_000),
            ("w003", "爆", 240_000, 360_000),
            ("w004", "完", 360_000, 480_000),
            ("w005", "了", 480_000, 600_000),
            ("w006", "金币", 600_000, 880_000),
            ("w007", "那", 880_000, 1_000_000),
            ("w008", "导师", 1_000_000, 1_240_000),
            ("w009", "呢", 1_240_000, 1_360_000),
            ("w010", "你", 1_360_000, 1_480_000),
            ("w011", "被", 1_480_000, 1_600_000),
            ("w012", "爆", 1_600_000, 1_720_000),
            ("w013", "完", 1_720_000, 1_840_000),
            ("w014", "的", 1_840_000, 1_960_000),
            ("w015", "金币", 1_960_000, 2_240_000),
            ("w016", "那", 2_240_000, 2_360_000),
            ("w017", "导致", 2_360_000, 2_600_000),
            ("w018", "呢", 2_600_000, 2_720_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 2_720_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_720_000,
                target_start_us=0,
                target_end_us=2_720_000,
                word_ids=[word_id for word_id, _text, _start, _end in rows],
                text="你被爆完了金币那导师呢你被爆完的金币那导致呢",
            )
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "你被爆完的金币那导致呢")
        self.assertEqual(
            [word_id for segment in result.final_timeline for word_id in segment.word_ids],
            ["w010", "w011", "w012", "w013", "w014", "w015", "w016", "w017", "w018"],
        )

    def test_final_visible_repairs_cross_caption_containment_by_dropping_shorter_repeat(self) -> None:
        rows = [
            ("w001", "站在高处大声说", 0, 800_000),
            ("w002", "提醒大家站在高处", 800_000, 1_600_000),
            ("w003", "大声说出真相", 1_600_000, 2_400_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = _timeline_from_word_rows(rows)
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_final_counts"]["cross_caption_semantic_containment_count"], 0)
        self.assertEqual([segment.text for segment in result.final_timeline], ["提醒大家站在高处", "大声说出真相"])
        self.assertEqual(result.report["final_visible_repair_actions"][0]["decision"], "drop_shorter_repeated_segment")

    def test_final_visible_repairs_restart_repeat_across_caption_window(self) -> None:
        rows = [
            ("w001", "重新开始表达观点", 0, 800_000),
            ("w002", "先重新开始表达观点的部分", 800_000, 1_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = _timeline_from_word_rows(rows)
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_final_counts"]["restart_repeat_visible_count"], 0)
        self.assertEqual([segment.text for segment in result.final_timeline], ["先重新开始表达观点的部分"])
        self.assertEqual(result.report["final_visible_repair_actions"][0]["issue_type"], "restart_repeat_visible")

    def test_final_visible_recheck_can_block_unrepairable_asr_suspect(self) -> None:
        rows = [("w001", "想就想到还有问题", 0, 1_200_000)]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = _timeline_from_word_rows(rows)
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertFalse(result.report["final_visible_repair_success"])
        self.assertEqual(result.report["final_visible_repair_final_counts"]["semantic_garbage_or_asr_suspect_count"], 1)
        self.assertEqual(result.report["final_visible_recheck_required_count"], 1)
        self.assertEqual(result.report["final_visible_repair_unresolved"][0]["reason"], "no_safe_deterministic_repair_available")

    def test_final_visible_repair_failure_blocks_even_when_rendered_gate_is_clean(self) -> None:
        report = ArollEngine()._attach_final_caption_visible_repeat_gate(
            {
                "validator_report_ok": True,
                "final_visible_caption_repair_report": {
                    "final_visible_repair_attempted": True,
                    "final_visible_repair_success": False,
                    "final_visible_repair_unresolved": [{"reason": "timeline_repeat_unresolved"}],
                    "final_visible_repair_final_timeline_counts": {
                        "dangling_prefix_suffix_count": 0,
                        "semantic_garbage_or_asr_suspect_count": 0,
                        "cross_caption_semantic_containment_count": 1,
                        "restart_repeat_visible_count": 0,
                    },
                },
            },
            [_caption(1, "v21_seg_000001", 0, 900_000, text="正常表达")],
        )

        gate = report["final_caption_visible_repeat_gate"]
        self.assertFalse(report["validator_report_ok"])
        self.assertFalse(gate["gate_passed"])
        self.assertIn("V21_FINAL_VISIBLE_REPAIR_UNRESOLVED", gate["blocker_codes"])

    def test_final_visible_counts_zero_after_successful_repair(self) -> None:
        rows = [
            ("w001", "我们讨论", 0, 500_000),
            ("w002", "的是公共问题", 500_000, 1_000_000),
            ("w003", "走到前面认真说", 1_000_000, 1_500_000),
            ("w004", "有人走到前面", 1_500_000, 2_000_000),
            ("w005", "认真说出想法", 2_000_000, 2_500_000),
            ("w006", "重新整理这个观点", 2_500_000, 3_000_000),
            ("w007", "先重新整理这个观点", 3_000_000, 3_500_000),
            ("w008", "想", 3_500_000, 3_650_000),
            ("w009", "就", 3_650_000, 3_800_000),
            ("w010", "想", 3_800_000, 3_950_000),
            ("w011", "到", 3_950_000, 4_100_000),
            ("w012", "还有", 4_100_000, 4_350_000),
            ("w013", "问题", 4_350_000, 4_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(_segment(1, 0, 500_000), source_material_id="main", source_segment_id="primary_window", source_start_us=0, source_end_us=500_000, word_ids=["w001"], text="我们讨论"),
            replace(_segment(2, 500_000, 1_000_000), source_material_id="main", source_segment_id="primary_window", source_start_us=500_000, source_end_us=1_000_000, word_ids=["w002"], text="的是公共问题"),
            replace(_segment(3, 1_000_000, 1_500_000), source_material_id="main", source_segment_id="primary_window", source_start_us=1_000_000, source_end_us=1_500_000, word_ids=["w003"], text="走到前面认真说"),
            replace(_segment(4, 1_500_000, 2_000_000), source_material_id="main", source_segment_id="primary_window", source_start_us=1_500_000, source_end_us=2_000_000, word_ids=["w004"], text="有人走到前面"),
            replace(_segment(5, 2_000_000, 2_500_000), source_material_id="main", source_segment_id="primary_window", source_start_us=2_000_000, source_end_us=2_500_000, word_ids=["w005"], text="认真说出想法"),
            replace(_segment(6, 2_500_000, 3_000_000), source_material_id="main", source_segment_id="primary_window", source_start_us=2_500_000, source_end_us=3_000_000, word_ids=["w006"], text="重新整理这个观点"),
            replace(_segment(7, 3_000_000, 3_500_000), source_material_id="main", source_segment_id="primary_window", source_start_us=3_000_000, source_end_us=3_500_000, word_ids=["w007"], text="先重新整理这个观点"),
            replace(
                _segment(8, 3_500_000, 4_600_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=3_500_000,
                source_end_us=4_600_000,
                word_ids=["w008", "w009", "w010", "w011", "w012", "w013"],
                text="想就想到还有问题",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual(
            result.report["final_visible_repair_final_counts"],
            {
                "dangling_prefix_suffix_count": 0,
                "semantic_garbage_or_asr_suspect_count": 0,
                "semantic_integrity_count": 0,
                "cross_caption_semantic_containment_count": 0,
                "restart_repeat_visible_count": 0,
            },
        )
        self.assertGreaterEqual(result.report["final_visible_repair_action_count"], 4)

    def test_renderer_preserves_generic_disjoint_visible_repeats_for_timeline_repair(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "就国南", 0, 640_000),
                ("w002", "就国南就只会内斗", 700_000, 1_700_000),
            ]
        )
        segments = [
            replace(_segment(1, 0, 640_000), source_end_us=640_000, word_ids=["w001"], text="就国南"),
            replace(
                _segment(2, 700_000, 1_700_000),
                source_start_us=700_000,
                source_end_us=1_700_000,
                word_ids=["w002"],
                text="就国南就只会内斗",
            ),
        ]

        captions = SubtitleRenderer().render(segments, source_graph)
        gate = build_final_caption_visible_repeat_gate(captions)

        self.assertEqual([caption.text for caption in captions], ["就国南", "就国南就只会内斗"])
        self.assertEqual([caption.word_ids for caption in captions], [["w001"], ["w002"]])
        self.assertGreaterEqual(gate["visible_repeat_candidate_count"], 1)
        self.assertFalse(gate["gate_passed"], gate)

    def test_final_caption_repeat_gate_classifies_distant_short_concept_without_blocking(self) -> None:
        caption_gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 700_000, text="就是在"),
                _caption(2, "v21_seg_000002", 10_000_000, 11_000_000, text="她们就是在集体做多啊"),
            ]
        )
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "final_repeat_high_count_after": 0,
                "detector_report_present": True,
            },
            final_caption_visible_repeat_gate=caption_gate,
            final_timeline_quality_guard_gate=_final_timeline_guard_ok(),
            visual_pacing_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
            },
            semantic_adjudication_gate=_semantic_gate_ok(),
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertTrue(quality["gate_passed"])
        self.assertEqual(caption_gate["containment_repeat_count"], 0)
        self.assertEqual(caption_gate["containment_repeat_raw_count"], 1)
        self.assertEqual(caption_gate["visible_repeat_allow_candidate_count"], 1)
        self.assertEqual(caption_gate["repeat_classification_candidates"][0]["classification"], "short_concept_reuse")
        self.assertNotIn("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED", quality["blocker_codes"])
        self.assertEqual(quality["final_repeat_convergence_gate"]["final_repeat_high_count_after"], 0)

    def test_failure_sample_visible_repeat_candidates_are_classified_without_blocking_ready(self) -> None:
        caption_gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 640_000, text="就国南"),
                _caption(51, "v21_seg_000051", 69_239_998, 70_799_998, text="就国南就只会内斗"),
                _caption(59, "v21_seg_000059", 81_320_000, 83_186_666, text="她们就是在集体做多啊"),
                _caption(75, "v21_seg_000075", 106_406_666, 106_886_666, text="就是在"),
            ]
        )
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "final_repeat_high_count_after": 0},
            final_caption_visible_repeat_gate=caption_gate,
            final_timeline_quality_guard_gate=_final_timeline_guard_ok(),
            visual_pacing_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
            },
            semantic_adjudication_gate=_semantic_gate_ok(),
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertTrue(quality["gate_passed"])
        self.assertEqual(caption_gate["containment_repeat_count"], 0)
        self.assertEqual(caption_gate["containment_repeat_raw_count"], 2)
        self.assertEqual(caption_gate["visible_repeat_candidate_count"], 0)
        self.assertGreaterEqual(caption_gate["visible_repeat_allow_candidate_count"], 2)
        self.assertGreaterEqual(caption_gate["visible_repeat_warning_candidate_count"], 1)
        self.assertNotIn("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED", quality["blocker_codes"])
        self.assertTrue(
            all(
                row["distance_kind"] == "distant" and row["severity"] != "fatal"
                for row in caption_gate["repeat_classification_candidates"]
            )
        )

    def test_repeat_convergence_does_not_zero_report_without_detector(self) -> None:
        report = build_final_repeat_convergence_report(
            decision_trace=[],
            final_repeat_report={},
        )

        self.assertFalse(report["gate_passed"])
        self.assertFalse(report["detector_report_present"])
        self.assertIn("V21_FINAL_REPEAT_DETECTOR_REPORT_MISSING", report["blocker_codes"])

    def test_visual_pacing_report_fields(self) -> None:
        report = build_visual_pacing_report(
            final_timeline=[_segment(1, 0, 500_000), _segment(2, 500_000, 2_000_000)],
            captions=[_caption(1, "v21_seg_000001", 0, 500_000), _caption(2, "v21_seg_000002", 500_000, 2_000_000)],
            executed=True,
            merge_report={
                "visual_pacing_executed": True,
                "visual_pacing_merge_attempted_count": 0,
                "visual_pacing_merged_count": 0,
                "visual_short_segment_count_lt_1200ms_before": 1,
                "visual_short_segment_count_lt_1200ms_after": 1,
                "semantic_bridge_short_segment_count": 1,
                "visual_pacing_blocker_codes": [],
            },
        )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["final_video_segment_count"], 2)
        self.assertEqual(report["caption_count"], 2)
        self.assertEqual(report["visual_short_segment_count_lt_1200ms"], 1)
        self.assertGreater(report["median_segment_duration_us"], 0)
        self.assertGreater(report["p10_segment_duration_us"], 0)
        self.assertTrue(report["visual_pacing_executed"])

    def test_visual_pacing_report_false_when_not_executed(self) -> None:
        report = build_visual_pacing_report(
            final_timeline=[_segment(1, 0, 500_000), _segment(2, 500_000, 2_000_000)],
            captions=[_caption(1, "v21_seg_000001", 0, 500_000), _caption(2, "v21_seg_000002", 500_000, 2_000_000)],
        )

        self.assertFalse(report["gate_passed"])
        self.assertIn("V21_VISUAL_PACING_NOT_EXECUTED", report["blocker_codes"])

    def test_visual_pacing_executes_merge_not_report_only(self) -> None:
        result = _multi_caption_fake_result()
        source_graph = ArollEngine().ingest.build_source_graph(
            draft_data=result.draft_data,
            word_timeline=result.word_timeline,
            subtitles=result.subtitles,
            source_segments=result.source_segments,
            source_materials=result.source_materials,
            text_materials=result.text_materials,
            text_segments=result.text_segments,
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="这个", word_ids=["w001"]),
            replace(_segment(2, 350_000, 650_000), text="那个", word_ids=["w002"]),
            replace(_segment(3, 700_000, 1_200_000), text="然后", word_ids=["w003"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertTrue(visual["visual_pacing_executed"])
        self.assertGreater(visual["visual_pacing_merged_count"], 0)
        self.assertLess(len(normalized), len(initial))
        self.assertTrue(visual["gate_passed"])
        self.assertTrue(visual["visual_merge_safety_gate_passed"])

    def test_visual_pacing_splits_large_intra_segment_word_gap(self) -> None:
        rows = [
            ("w001", "前半句", 0, 1_300_000),
            ("w002", "后半句", 1_800_000, 3_100_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 3_100_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=3_100_000,
                target_start_us=0,
                target_end_us=3_100_000,
                word_ids=["w001", "w002"],
                text="前半句后半句",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual([segment.text for segment in normalized], ["前半句", "后半句"])
        self.assertEqual([(segment.source_start_us, segment.source_end_us) for segment in normalized], [(0, 1_300_000), (1_800_000, 3_100_000)])
        self.assertEqual([(segment.target_start_us, segment.target_end_us) for segment in normalized], [(0, 1_300_000), (1_300_000, 2_600_000)])
        self.assertTrue(visual["gate_passed"])
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_max_us"], 500_000)
        self.assertEqual(visual["large_intra_segment_gap_candidates"][0]["reason"], "large_intra_segment_gap_split")

    def test_visual_pacing_splits_visible_320ms_intra_segment_word_gap(self) -> None:
        rows = [
            ("w001", "前半句", 0, 1_300_000),
            ("w002", "后半句", 1_620_000, 2_920_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 2_920_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_920_000,
                target_start_us=0,
                target_end_us=2_920_000,
                word_ids=["w001", "w002"],
                text="前半句后半句",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual([segment.text for segment in normalized], ["前半句", "后半句"])
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_max_us"], 320_000)
        self.assertEqual(visual["large_intra_segment_gap_candidates"][0]["reason"], "large_intra_segment_gap_split")

    def test_visual_pacing_splits_visible_240ms_intra_segment_word_gap(self) -> None:
        rows = [
            ("w001", "前半句", 0, 1_300_000),
            ("w002", "后半句", 1_540_000, 2_840_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 2_840_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_840_000,
                target_start_us=0,
                target_end_us=2_840_000,
                word_ids=["w001", "w002"],
                text="前半句后半句",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual([segment.text for segment in normalized], ["前半句", "后半句"])
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_max_us"], 240_000)
        self.assertEqual(visual["large_intra_segment_gap_candidates"][0]["reason"], "large_intra_segment_gap_split")

    def test_visual_pacing_drops_boundary_filler_isolated_by_large_intra_segment_gap(self) -> None:
        rows = [
            ("w001", "咳", 0, 800_000),
            ("w002", "立刻给老子关了", 2_066_666, 3_226_666),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 3_226_666),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=3_226_666,
                target_start_us=0,
                target_end_us=3_226_666,
                word_ids=["w001", "w002"],
                text="咳立刻给老子关了",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual([segment.text for segment in normalized], ["立刻给老子关了"])
        self.assertEqual(normalized[0].source_start_us, 2_066_666)
        self.assertEqual(normalized[0].target_start_us, 0)
        self.assertEqual(normalized[0].target_end_us, 1_160_000)
        self.assertTrue(visual["gate_passed"])
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 1)
        self.assertEqual(
            visual["large_intra_segment_gap_splits"][0]["dropped_boundary_filler_word_ids"],
            ["w001"],
        )

    def test_visual_pacing_does_not_split_single_char_content_side(self) -> None:
        rows = [
            ("w001", "我", 0, 800_000),
            ("w002", "继续说完整内容", 1_400_000, 2_800_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 2_800_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_800_000,
                target_start_us=0,
                target_end_us=2_800_000,
                word_ids=["w001", "w002"],
                text="我继续说完整内容",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].text, "我继续说完整内容")
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 0)
        self.assertEqual(visual["large_intra_segment_gap_candidates"][0]["reason"], "single_char_left_side_would_survive")

    def test_visual_pacing_keeps_normal_intra_segment_breath(self) -> None:
        rows = [
            ("w001", "前半句", 0, 1_300_000),
            ("w002", "后半句", 1_440_000, 2_740_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 2_740_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_740_000,
                target_start_us=0,
                target_end_us=2_740_000,
                word_ids=["w001", "w002"],
                text="前半句后半句",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].source_start_us, 0)
        self.assertEqual(normalized[0].source_end_us, 2_740_000)
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 0)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 0)

    def test_visual_pacing_reports_medium_intra_segment_gap_without_splitting(self) -> None:
        rows = [
            ("w001", "前半句", 0, 1_300_000),
            ("w002", "后半句", 1_480_000, 2_780_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 2_780_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_780_000,
                target_start_us=0,
                target_end_us=2_780_000,
                word_ids=["w001", "w002"],
                text="前半句后半句",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 0)
        self.assertEqual(visual["large_intra_segment_gap_candidates"][0]["reason"], "below_split_threshold")

    def test_visual_pacing_dense_gaps_still_split_when_segment_also_has_large_gap(self) -> None:
        rows = [
            ("w001", "第一句", 0, 600_000),
            ("w002", "第二句", 820_000, 1_420_000),
            ("w003", "第三句", 1_580_000, 2_180_000),
            ("w004", "第四句", 2_340_000, 2_940_000),
            ("w005", "第五句", 3_100_000, 3_700_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 3_700_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=3_700_000,
                target_start_us=0,
                target_end_us=3_700_000,
                word_ids=["w001", "w002", "w003", "w004", "w005"],
                text="第一句第二句第三句第四句第五句",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual([segment.text for segment in normalized], ["第一句", "第二句", "第三句", "第四句", "第五句"])
        self.assertGreaterEqual(visual["large_intra_segment_gap_split_count"], 4)
        reasons = {row["reason"] for row in visual["large_intra_segment_gap_candidates"]}
        self.assertIn("large_intra_segment_gap_split", reasons)
        self.assertIn("dense_intra_segment_gap_split", reasons)

    def test_visual_pacing_prunes_adjacent_splits_that_would_create_short_word_island(self) -> None:
        rows = [
            ("w001", "前段", 0, 600_000),
            ("w002", "短词", 820_000, 1_100_000),
            ("w003", "后段", 1_320_000, 1_920_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 1_920_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_920_000,
                target_start_us=0,
                target_end_us=1_920_000,
                word_ids=["w001", "w002", "w003"],
                text="前段短词后段",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 2)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 1)
        self.assertNotIn("短词", [segment.text for segment in normalized])
        self.assertIn("short_run_pruned", {row["reason"] for row in visual["large_intra_segment_gap_candidates"]})

    def test_visual_pacing_does_not_add_breath_cut_inside_full_cut_density_window(self) -> None:
        rows = [
            ("w001", "第一段", 0, 600_000),
            ("w002", "第二段", 800_000, 1_400_000),
            ("w003", "第三段", 1_600_000, 2_200_000),
            ("w004", "第四段", 2_400_000, 3_000_000),
            ("w005", "第五段", 3_200_000, 3_800_000),
            ("w006a", "前半", 4_000_000, 4_600_000),
            ("w006b", "后半", 4_820_000, 5_420_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(index, start, end),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=start,
                source_end_us=end,
                target_start_us=start,
                target_end_us=end,
                word_ids=[word_id],
                text=text,
            )
            for index, (word_id, text, start, end) in enumerate(rows[:5], start=1)
        ]
        initial.append(
            replace(
                _segment(6, 4_000_000, 5_420_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=4_000_000,
                source_end_us=5_420_000,
                target_start_us=4_000_000,
                target_end_us=5_420_000,
                word_ids=["w006a", "w006b"],
                text="前半后半",
            )
        )

        normalized, visual = split_large_intra_segment_gaps(initial, source_graph, [("primary_window", 0, 6_000_000)])

        self.assertEqual([segment.text for segment in normalized[-1:]], ["前半后半"])
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 0)
        self.assertEqual(visual["large_intra_segment_gap_candidates"][0]["reason"], "cut_density_window_protected")

    def test_visual_pacing_merges_safe_split_bridge_to_relieve_cut_density_burst(self) -> None:
        prefix_rows = [
            (f"wp{index:03d}", f"铺垫{index}", (index - 1) * 2_000_000, index * 2_000_000)
            for index in range(1, 10)
        ]
        dense_source_offset = 20_000_000
        dense_rows = [
            ("w001", "女同事", dense_source_offset + 0, dense_source_offset + 520_000),
            ("w002", "开始饥荒式的发情", dense_source_offset + 2_933_333, dense_source_offset + 4_566_667),
            ("w003", "第二级", dense_source_offset + 4_800_000, dense_source_offset + 5_360_000),
            ("w004", "恶化", dense_source_offset + 5_520_000, dense_source_offset + 6_033_333),
            ("w005", "被迫梭哈", dense_source_offset + 6_233_333, dense_source_offset + 7_266_667),
            ("w006", "因为你的世界只有这么一个女人", dense_source_offset + 7_766_667, dense_source_offset + 11_626_667),
        ]
        rows = [*prefix_rows, *dense_rows]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(index, (index - 1) * 2_000_000, index * 2_000_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=(index - 1) * 2_000_000,
                source_end_us=index * 2_000_000,
                target_start_us=(index - 1) * 2_000_000,
                target_end_us=index * 2_000_000,
                word_ids=[f"wp{index:03d}"],
                text=f"铺垫{index}",
            )
            for index in range(1, 10)
        ]
        initial.append(
            replace(
                _segment(10, 18_000_000, 29_626_667),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=dense_source_offset,
                source_end_us=dense_source_offset + 11_626_667,
                target_start_us=18_000_000,
                target_end_us=29_626_667,
                word_ids=["w001", "w002", "w003", "w004", "w005", "w006"],
                text="女同事开始饥荒式的发情第二级恶化被迫梭哈因为你的世界只有这么一个女人",
            ),
        )

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertTrue(visual["cut_density_gate_passed"], visual)
        self.assertLessEqual(visual["max_cuts_in_5s"], 5)
        self.assertNotIn("V21_VISUAL_CUT_DENSITY_FAILED", visual["blocker_codes"])
        self.assertGreaterEqual(visual["visual_pacing_merged_count"], 1)
        self.assertIn("第二级恶化", [segment.text for segment in normalized])

    def test_visual_pacing_does_not_split_large_gap_when_side_too_short(self) -> None:
        rows = [
            ("w001", "短句", 0, 240_000),
            ("w002", "后面完整表达", 800_000, 2_200_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        initial = [
            replace(
                _segment(1, 0, 2_200_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=2_200_000,
                target_start_us=0,
                target_end_us=2_200_000,
                word_ids=["w001", "w002"],
                text="短句后面完整表达",
            )
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(visual["large_intra_segment_gap_candidate_count"], 1)
        self.assertEqual(visual["large_intra_segment_gap_split_count"], 0)
        self.assertEqual(visual["large_intra_segment_gap_candidates"][0]["reason"], "left_side_too_short")

    def test_residual_visual_short_segments_merge_when_safe(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "啊", 0, 300_000),
                ("w002", "继续", 380_000, 780_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="啊", word_ids=["w001"]),
            replace(_segment(2, 380_000, 780_000), text="继续", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 1)
        self.assertTrue(visual["gate_passed"])
        self.assertEqual(visual["visual_short_segment_count_lt_1200ms_after_blocking"], 0)

    def test_residual_visual_short_segments_semantic_bridge_exception_not_blocking(self) -> None:
        source_graph = _graph_for_visual_merge_rows([("w001", "语义桥", 0, 700_000)])
        initial = [replace(_segment(1, 0, 700_000), text="语义桥", word_ids=["w001"])]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 1)
        self.assertTrue(visual["gate_passed"])
        self.assertEqual(visual["semantic_bridge_short_segment_count"], 1)
        self.assertEqual(visual["visual_short_segment_count_lt_1200ms_after_blocking"], 0)
        self.assertEqual(visual["residual_visual_short_segments"][0]["short_segment_status"], "semantic_bridge_exception")

    def test_residual_weak_filler_short_segment_merges_or_drops(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "就", 0, 240_000),
                ("w002", "继续表达", 300_000, 900_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 240_000), text="就", word_ids=["w001"]),
            replace(_segment(2, 300_000, 900_000), text="继续表达", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertLess(len(normalized), len(initial))
        self.assertTrue(visual["gate_passed"])
        self.assertEqual(visual["visual_short_segment_count_lt_1200ms_after_blocking"], 0)

    def test_one_char_weak_filler_micro_segment_cannot_survive(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "前面内容", 0, 1_200_000),
                ("w002", "就", 2_000_000, 2_240_000),
                ("w003", "后面内容继续", 3_000_000, 4_300_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 1_200_000), source_end_us=1_200_000, text="前面内容", word_ids=["w001"]),
            replace(_segment(2, 2_000_000, 2_240_000), source_start_us=2_000_000, source_end_us=2_240_000, text="就", word_ids=["w002"]),
            replace(_segment(3, 3_000_000, 4_300_000), source_start_us=3_000_000, source_end_us=4_300_000, text="后面内容继续", word_ids=["w003"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)
        captions = SubtitleRenderer().render(normalized, source_graph)
        alignment = build_caption_alignment_report(final_timeline=normalized, captions=captions)

        self.assertTrue(all(segment.text != "就" for segment in normalized))
        self.assertEqual(visual["visual_short_segment_count_lt_1200ms_after_blocking"], 0)
        self.assertEqual(alignment["one_char_caption_count"], 0)
        self.assertEqual(alignment["caption_too_short_count"], 0)

    def test_one_char_weak_filler_micro_segment_merges_when_safe(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "就", 0, 240_000),
                ("w002", "继续表达", 300_000, 1_200_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 240_000), text="就", word_ids=["w001"]),
            replace(_segment(2, 300_000, 1_200_000), source_start_us=300_000, source_end_us=1_200_000, text="继续表达", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)
        captions = SubtitleRenderer().render(normalized, source_graph)
        alignment = build_caption_alignment_report(final_timeline=normalized, captions=captions)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].text, "就继续表达")
        self.assertEqual(visual["merged_weak_filler_micro_segment_count"], 1)
        self.assertEqual(visual["dropped_weak_filler_micro_segment_count"], 0)
        self.assertEqual(alignment["one_char_caption_count"], 0)
        self.assertEqual(alignment["caption_too_short_count"], 0)

    def test_one_char_weak_filler_micro_segment_drops_when_merge_unsafe(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "就", 0, 240_000),
                ("w002", "继续表达很好", 900_000, 2_200_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 240_000), text="就", word_ids=["w001"]),
            replace(_segment(2, 900_000, 2_200_000), source_start_us=900_000, source_end_us=2_200_000, text="继续表达很好", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)
        captions = SubtitleRenderer().render(normalized, source_graph)
        alignment = build_caption_alignment_report(final_timeline=normalized, captions=captions)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].text, "继续表达很好")
        self.assertEqual(visual["merged_weak_filler_micro_segment_count"], 0)
        self.assertEqual(visual["dropped_weak_filler_micro_segment_count"], 1)
        self.assertEqual(visual["visual_short_segment_count_lt_1200ms_after_blocking"], 0)
        self.assertEqual(alignment["one_char_caption_count"], 0)
        self.assertEqual(alignment["caption_too_short_count"], 0)

    def test_visual_pacing_gate_blocks_when_short_segments_remain(self) -> None:
        class NoopVisualPacing:
            def normalize(self, final_timeline, source_graph):
                return final_timeline, {
                    "visual_pacing_executed": True,
                    "visual_pacing_merge_attempted_count": 0,
                    "visual_pacing_merged_count": 0,
                    "visual_short_segment_count_lt_1200ms_before": 1,
                    "visual_short_segment_count_lt_1200ms_after": 1,
                    "semantic_bridge_short_segment_count": 0,
                    "visual_pacing_blocker_codes": ["V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN"],
                }

        result = fake_real_draft_result()
        report = ArollEngine(visual_pacing=NoopVisualPacing()).run(
            ArollRunInput(
                mode="write",
                draft_data=result.draft_data,
                word_timeline=result.word_timeline,
                subtitles=result.subtitles,
                source_segments=result.source_segments,
                source_materials=result.source_materials,
                text_materials=result.text_materials,
                text_segments=result.text_segments,
                postwrite_mode="simulated",
            )
        )

        self.assertEqual(report.status, "blocked")
        visual = report.validator_report["visual_pacing_gate"]
        self.assertFalse(visual["gate_passed"])
        self.assertIn("V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN", visual["blocker_codes"])

    def test_visual_merge_does_not_cross_dropped_repeat_segment(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "啊", 0, 300_000),
                ("w_drop", "删", 400_000, 700_000),
                ("w002", "呃", 800_000, 1_100_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="啊", word_ids=["w001"]),
            replace(_segment(2, 800_000, 1_100_000), text="呃", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 2)
        self.assertFalse(visual["gate_passed"])
        self.assertEqual(visual["dropped_content_reintroduced_count"], 0)
        self.assertGreater(visual["visual_merge_safety_report"]["visual_pacing_blocked_unsafe_merge_attempt_count"], 0)

    def test_visual_merge_does_not_cross_hidden_repeat_cleanup_span(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "啊", 0, 300_000),
                ("w_hidden", "重复", 360_000, 660_000),
                ("w002", "呃", 720_000, 1_020_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="啊", word_ids=["w001"]),
            replace(_segment(2, 720_000, 1_020_000), text="呃", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 2)
        self.assertFalse(visual["gate_passed"])
        self.assertGreater(visual["visual_merge_safety_report"]["visual_pacing_blocked_unsafe_merge_attempt_count"], 0)

    def test_visual_merge_does_not_cross_boundary_overlap_cleanup_span(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "啊", 0, 300_000),
                ("w_boundary", "边界", 320_000, 620_000),
                ("w002", "呃", 680_000, 980_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="啊", word_ids=["w001"]),
            replace(_segment(2, 680_000, 980_000), text="呃", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 2)
        self.assertFalse(visual["gate_passed"])
        self.assertGreater(visual["visual_merge_safety_report"]["visual_pacing_blocked_unsafe_merge_attempt_count"], 0)

    def test_visual_merge_does_not_bridge_large_unspoken_gap(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "啊", 0, 300_000),
                ("w002", "呃", 800_000, 1_100_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="啊", word_ids=["w001"]),
            replace(_segment(2, 800_000, 1_100_000), text="呃", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 2)
        self.assertFalse(visual["gate_passed"])
        self.assertGreater(visual["visual_merge_safety_report"]["visual_pacing_blocked_unsafe_merge_attempt_count"], 0)

    def test_visual_merge_allows_small_safe_gap_same_source_window(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "啊", 0, 300_000),
                ("w002", "呃", 380_000, 680_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="啊", word_ids=["w001"]),
            replace(_segment(2, 380_000, 680_000), text="呃", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 1)
        self.assertTrue(visual["visual_merge_safety_gate_passed"])
        self.assertEqual(visual["unsafe_merge_group_count"], 0)
        self.assertEqual(visual["dropped_content_reintroduced_count"], 0)
        self.assertEqual(visual["max_bridged_gap_us"], 80_000)

    def test_visual_merge_safety_report_records_child_segments_and_gaps(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "啊", 0, 300_000),
                ("w002", "呃", 380_000, 680_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="啊", word_ids=["w001"]),
            replace(_segment(2, 380_000, 680_000), text="呃", word_ids=["w002"]),
        ]

        _normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)
        groups = visual["visual_merge_groups"]

        self.assertEqual(groups[0]["child_segment_ids"], ["v21_seg_000001", "v21_seg_000002"])
        self.assertEqual(groups[0]["source_window_id"], "primary_window")
        self.assertEqual(groups[0]["bridged_gaps"][0]["duration_us"], 80_000)
        self.assertTrue(groups[0]["merge_safe"])

    def test_visual_pacing_blocks_overmerged_four_segments_without_safety_proof(self) -> None:
        report = build_visual_pacing_report(
            final_timeline=[_segment(index, index * 1_000_000, (index + 1) * 1_000_000) for index in range(4)],
            captions=[_caption(index, "v21_seg_000001", index * 100_000, index * 100_000 + 80_000) for index in range(1, 109)],
            executed=True,
            merge_report={"visual_pacing_merged_count": 126},
        )

        self.assertFalse(report["gate_passed"])
        self.assertFalse(report["visual_merge_safety_gate_passed"])
        self.assertIn("V21_VISUAL_PACING_MISSING_MERGE_SAFETY_PROOF", report["blocker_codes"])

    def test_quality_gate_requires_visual_merge_safety(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": []},
            visual_pacing_gate={"gate_passed": True, "visual_pacing_executed": True, "blocker_codes": []},
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
        )

        self.assertFalse(quality["gate_passed"])
        self.assertIn("V21_VISUAL_PACING_UNSAFE_MERGE", quality["blocker_codes"])

    def test_caption_per_video_segment_ratio_high_requires_safe_bridge_proof(self) -> None:
        report = build_visual_pacing_report(
            final_timeline=[_segment(1, 0, 2_000_000)],
            captions=[_caption(index, "v21_seg_000001", index * 100_000, index * 100_000 + 80_000) for index in range(1, 13)],
            executed=True,
            merge_report={"visual_pacing_merged_count": 0},
        )

        self.assertFalse(report["gate_passed"])
        self.assertFalse(report["visual_merge_safety_gate_passed"])

    def test_final_video_segment_count_can_drop_only_when_merge_safety_passes(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "啊", 0, 300_000),
                ("w002", "呃", 380_000, 680_000),
            ]
        )
        initial = [
            replace(_segment(1, 0, 300_000), text="啊", word_ids=["w001"]),
            replace(_segment(2, 380_000, 680_000), text="呃", word_ids=["w002"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)

        self.assertEqual(len(normalized), 1)
        self.assertTrue(visual["gate_passed"])
        self.assertTrue(visual["visual_merge_safety_gate_passed"])

    def test_hidden_repeated_suffix_island_drops_whole_word_span(self) -> None:
        materials, text_segments = _template_rows()
        words = []
        cursor = 0
        for index, token in enumerate(["重复", "短语", "中间", "重复", "短语"], start=1):
            words.append(
                {
                    "word_id": f"w_{index:06d}",
                    "word_text": token,
                    "start_us": cursor,
                    "end_us": cursor + 200_000,
                    "subtitle_index": 1,
                    "subtitle_uid": "s001",
                }
            )
            cursor += 200_000

        report = ArollEngine().run(
            ArollRunInput(
                mode="write",
                source_segments=[{"id": "clip", "material_id": "main", "source_start_us": 0, "source_end_us": cursor + 500_000}],
                word_timeline=words,
                subtitles=[
                    {
                        "subtitle_uid": "s001",
                        "subtitle_index": 1,
                        "text": "重复短语中间重复短语",
                        "word_ids": [row["word_id"] for row in words],
                    }
                ],
                text_materials=materials,
                text_segments=text_segments,
                postwrite_mode="simulated",
            )
        )

        self.assertEqual(report.status, "ok", [blocker.code for blocker in report.blocker_report.blockers])
        self.assertEqual([caption.text for caption in report.captions], ["重复短语中间"])
        self.assertEqual(report.final_timeline[0].word_ids, ["w_000001", "w_000002", "w_000003"])
        self.assertTrue(report.validator_report["hidden_audio_repeat_validator"]["hidden_audio_repeat_gate_passed"])
        trace = [row for row in report.decision_trace if row.get("decision") == "drop_repeated_suffix_island"]
        self.assertEqual(trace[0]["dropped_word_ids"], ["w_000004", "w_000005"])

    def test_visual_pacing_cleanup_drops_hidden_repeat_created_by_merge(self) -> None:
        materials, text_segments = _template_rows()
        rows = [
            ("w_000001", "然后", 0, 300_000),
            ("w_000002", "亲手", 300_000, 600_000),
            ("w_000003", "摧毁", 600_000, 900_000),
            ("w_000004", "然后", 1_100_000, 1_400_000),
        ]
        words = [
            {
                "word_id": word_id,
                "word_text": text,
                "start_us": start,
                "end_us": end,
                "subtitle_index": 1,
                "subtitle_uid": "s001",
            }
            for word_id, text, start, end in rows
        ]

        engine = ArollEngine()
        source_graph = engine.ingest.build_source_graph(
            word_timeline=words,
            subtitles=[
                {
                    "subtitle_uid": "s001",
                    "subtitle_index": 1,
                    "text": "然后亲手摧毁然后",
                    "word_ids": [row["word_id"] for row in words],
                }
            ],
            source_segments=[{"id": "clip", "material_id": "main", "source_start_us": 0, "source_end_us": 2_000_000}],
            text_materials=materials,
            text_segments=text_segments,
        )
        initial = [
            replace(_segment(1, 0, 900_000), text="然后亲手摧毁", word_ids=["w_000001", "w_000002", "w_000003"]),
            replace(_segment(2, 1_100_000, 1_400_000), text="然后", word_ids=["w_000004"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)
        captions = engine.renderer.render(normalized, source_graph)

        self.assertEqual([segment.text for segment in normalized], ["然后亲手摧毁"])
        self.assertEqual(normalized[0].word_ids, ["w_000001", "w_000002", "w_000003"])
        self.assertEqual([caption.text for caption in captions], ["然后亲手摧毁"])
        self.assertTrue(visual["visual_pacing_executed"])
        self.assertGreater(visual["visual_pacing_merged_count"], 0)
        self.assertEqual(visual["visual_pacing_hidden_repeat_dropped_word_count"], 1)
        self.assertEqual(
            build_caption_alignment_report(final_timeline=normalized, captions=captions)["caption_overlap_count"],
            0,
        )

    def test_visual_pacing_cleanup_drops_caption_boundary_suffix_prefix_overlap(self) -> None:
        materials, text_segments = _template_rows()
        rows = [
            ("w_000001", "就是", 0, 300_000, 1, "s001"),
            ("w_000002", "在", 300_000, 500_000, 1, "s001"),
            ("w_000003", "亲手", 500_000, 800_000, 1, "s001"),
            ("w_000004", "摧毁", 800_000, 1_100_000, 1, "s001"),
            ("w_000005", "亲手", 1_300_000, 1_600_000, 2, "s002"),
            ("w_000006", "摧毁", 1_600_000, 1_900_000, 2, "s002"),
            ("w_000007", "男性", 1_900_000, 2_200_000, 2, "s002"),
        ]
        words = [
            {
                "word_id": word_id,
                "word_text": text,
                "start_us": start,
                "end_us": end,
                "subtitle_index": subtitle_index,
                "subtitle_uid": subtitle_uid,
            }
            for word_id, text, start, end, subtitle_index, subtitle_uid in rows
        ]
        engine = ArollEngine()
        source_graph = engine.ingest.build_source_graph(
            word_timeline=words,
            subtitles=[
                {
                    "subtitle_uid": "s001",
                    "subtitle_index": 1,
                    "text": "就是在亲手摧毁",
                    "word_ids": ["w_000001", "w_000002", "w_000003", "w_000004"],
                },
                {
                    "subtitle_uid": "s002",
                    "subtitle_index": 2,
                    "text": "亲手摧毁男性",
                    "word_ids": ["w_000005", "w_000006", "w_000007"],
                },
            ],
            source_segments=[{"id": "clip", "material_id": "main", "source_start_us": 0, "source_end_us": 2_500_000}],
            text_materials=materials,
            text_segments=text_segments,
        )
        initial = [
            replace(_segment(1, 0, 1_100_000), text="就是在亲手摧毁", word_ids=["w_000001", "w_000002", "w_000003", "w_000004"]),
            replace(_segment(2, 1_300_000, 2_200_000), text="亲手摧毁男性", word_ids=["w_000005", "w_000006", "w_000007"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)
        captions = engine.renderer.render(normalized, source_graph)

        self.assertEqual([segment.text for segment in normalized], ["就是在", "亲手摧毁男性"])
        self.assertEqual(normalized[0].word_ids, ["w_000001", "w_000002"])
        self.assertEqual([caption.text for caption in captions], ["就是在", "亲手摧毁男性"])
        self.assertEqual(visual["visual_pacing_boundary_overlap_dropped_word_count"], 2)

    def test_visual_pacing_merges_hard_too_short_residual_created_by_boundary_cleanup(self) -> None:
        materials, text_segments = _template_rows()
        rows = [
            ("w_000001", "反而", 0, 280_000, 1, "s001"),
            ("w_000002", "亲手", 280_000, 580_000, 1, "s001"),
            ("w_000003", "摧毁", 580_000, 880_000, 1, "s001"),
            ("w_000004", "亲手", 280_000, 580_000, 2, "s002"),
            ("w_000005", "摧毁", 580_000, 880_000, 2, "s002"),
            ("w_000006", "后续", 880_000, 1_180_000, 2, "s002"),
        ]
        words = [
            {
                "word_id": word_id,
                "word_text": text,
                "start_us": start,
                "end_us": end,
                "subtitle_index": subtitle_index,
                "subtitle_uid": subtitle_uid,
            }
            for word_id, text, start, end, subtitle_index, subtitle_uid in rows
        ]
        engine = ArollEngine()
        source_graph = engine.ingest.build_source_graph(
            word_timeline=words,
            subtitles=[
                {
                    "subtitle_uid": "s001",
                    "subtitle_index": 1,
                    "text": "反而亲手摧毁",
                    "word_ids": ["w_000001", "w_000002", "w_000003"],
                },
                {
                    "subtitle_uid": "s002",
                    "subtitle_index": 2,
                    "text": "亲手摧毁后续",
                    "word_ids": ["w_000004", "w_000005", "w_000006"],
                },
            ],
            source_segments=[{"id": "clip", "material_id": "main", "source_start_us": 0, "source_end_us": 1_200_000}],
            text_materials=materials,
            text_segments=text_segments,
        )
        initial = [
            replace(_segment(1, 0, 880_000), text="反而亲手摧毁", word_ids=["w_000001", "w_000002", "w_000003"]),
            replace(_segment(2, 280_000, 1_180_000), text="亲手摧毁后续", word_ids=["w_000004", "w_000005", "w_000006"]),
        ]

        normalized, visual = VisualPacingNormalizer().normalize(initial, source_graph)
        captions = engine.renderer.render(normalized, source_graph)
        alignment = build_caption_alignment_report(final_timeline=normalized, captions=captions)

        self.assertEqual([segment.text for segment in normalized], ["反而亲手摧毁后续"])
        self.assertEqual(visual["visual_pacing_boundary_overlap_dropped_word_count"], 2)
        self.assertEqual(visual["visual_short_segment_count_lt_1200ms_after_blocking"], 0)
        self.assertEqual(alignment["caption_too_short_count"], 0)

    def test_caption_uses_spoken_target_span(self) -> None:
        report = build_caption_alignment_report(
            final_timeline=[_segment(1, 0, 1_000_000)],
            captions=[_caption(1, "v21_seg_000001", 100_000, 700_000)],
        )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["caption_outside_video_count"], 0)

    def test_caption_alignment_blocks_ready_when_invalid(self) -> None:
        caption_report = build_caption_alignment_report(
            final_timeline=[_segment(1, 0, 500_000)],
            captions=[_caption(1, "v21_seg_000001", 400_000, 800_000, text="字")],
        )
        quality_report = build_quality_gate_report(caption_alignment_gate=caption_report)

        self.assertFalse(caption_report["gate_passed"])
        self.assertFalse(quality_report["gate_passed"])
        self.assertIn("V21_CAPTION_OUTSIDE_VIDEO_SEGMENT", quality_report["blocker_codes"])
        self.assertIn("V21_ONE_CHAR_CAPTION", quality_report["blocker_codes"])

    def test_caption_alignment_blocks_uncaptioned_final_timeline_words(self) -> None:
        segment = replace(
            _segment(1, 0, 1_200_000),
            word_ids=["w1", "w2"],
            text="前段后段",
        )
        caption = replace(
            _caption(1, "v21_seg_000001", 0, 600_000, text="前段"),
            word_ids=["w1"],
        )

        report = build_caption_alignment_report(final_timeline=[segment], captions=[caption])
        quality = build_quality_gate_report(caption_alignment_gate=report)

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["prewrite_uncaptioned_spoken_word_count"], 1)
        self.assertEqual(report["prewrite_uncaptioned_spoken_segment_count"], 1)
        self.assertEqual(report["prewrite_uncaptioned_spoken_word_rows"][0]["missing_word_ids"], ["w2"])
        self.assertIn("V21_PREWRITE_UNCAPTIONED_SPOKEN_WORDS", report["blocker_codes"])
        self.assertIn("V21_PREWRITE_UNCAPTIONED_SPOKEN_WORDS", quality["blocker_codes"])

    def test_caption_alignment_uses_containing_video_segment_not_self_final_segment(self) -> None:
        caption = CaptionRenderUnit(
            caption_id="v21_cap_000001",
            timeline_segment_ids=["v21_seg_000001"],
            word_ids=["w2"],
            text="字幕",
            target_start_us=100_000,
            target_end_us=500_000,
            source_subtitle_uids=["s1"],
            style_template_id="canonical_caption_template",
            spoken_source_start_us=1_300_000,
            spoken_source_end_us=1_600_000,
            containing_video_segment_id="v21_seg_000001",
        )
        report = build_caption_alignment_report(
            final_timeline=[_segment(1, 0, 1_000_000)],
            captions=[caption],
        )

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["caption_cross_primary_window_count"], 1)
        self.assertIn("V21_CAPTION_CROSSES_PRIMARY_SOURCE_WINDOW", report["blocker_codes"])

    def test_caption_too_short_separate_from_outside_video_count(self) -> None:
        report = build_caption_alignment_report(
            final_timeline=[_segment(1, 0, 1_000_000)],
            captions=[_caption(1, "v21_seg_000001", 100_000, 250_000, text="字幕")],
        )

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["caption_too_short_count"], 1)
        self.assertEqual(report["caption_outside_video_count"], 0)
        self.assertIn("V21_CAPTION_TOO_SHORT", report["blocker_codes"])

    def test_prewrite_validators_use_projected_safe_handle_view_for_single_caption_connector(self) -> None:
        graph = _graph_for_single_subtitle_words([("w001", "所以", 80_000, 280_000)])
        segment = FinalTimelineSegment(
            segment_id="v21_seg_000001",
            source_material_id="main",
            source_segment_id="primary_window",
            source_start_us=80_000,
            source_end_us=280_000,
            target_start_us=0,
            target_end_us=200_000,
            word_ids=["w001"],
            text="所以",
            decision_ids=[],
            spoken_source_start_us=80_000,
            spoken_source_end_us=280_000,
            clip_source_start_us=80_000,
            clip_source_end_us=380_000,
            lead_handle_us=0,
            tail_handle_us=100_000,
            debug_hints={
                "safe_handle_policy_enabled": True,
                "safe_handle_source_window_start_us": 80_000,
                "safe_handle_source_window_end_us": 380_000,
                "safe_handle_previous_target_end_us": 0,
                "safe_handle_next_target_start_us": 300_000,
            },
        )
        caption = CaptionRenderUnit(
            caption_id="v21_cap_000001",
            timeline_segment_ids=["v21_seg_000001"],
            word_ids=["w001"],
            text="所以",
            target_start_us=0,
            target_end_us=200_000,
            source_subtitle_uids=["s001"],
            style_template_id="canonical_caption_template",
            spoken_source_start_us=80_000,
            spoken_source_end_us=280_000,
            containing_video_segment_id="v21_seg_000001",
        )

        report = ReadOnlyValidators().run(
            source_graph=graph,
            decision_plan=DecisionPlan(decisions=[]),
            final_timeline=[segment],
            captions=[caption],
            material_write_plan={
                "materials": [{"id": "caption_mat_1"}],
                "segments": [
                    {"id": "caption_seg_1", "material_id": "caption_mat_1", "target_timerange": {"start": 0, "duration": 200_000}}
                ],
                "no_writer_fallback": True,
            },
            postwrite_mode="simulated",
        )

        self.assertTrue(report["prewrite_projected_write_view"]["prewrite_projected_write_view_applied"])
        self.assertEqual(report["prewrite_projected_write_view"]["prewrite_projection_short_video_segment_count"], 0)
        self.assertEqual(report["prewrite_projected_write_view"]["prewrite_projection_short_caption_count"], 0)
        self.assertEqual(report["rough_cut_quality_validator"]["segments_lt_300ms"], 0)
        self.assertEqual(report["caption_alignment_gate"]["caption_too_short_count"], 0)
        self.assertNotIn("V21_CAPTION_TOO_SHORT", report["caption_alignment_gate"]["blocker_codes"])

    def test_prewrite_projection_failure_blocks_validator_report(self) -> None:
        engine = ArollEngine()
        graph = _graph_for_single_subtitle_words([("w001", "正常内容", 0, 1_500_000)])
        segment = FinalTimelineSegment(
            segment_id="v21_seg_000001",
            source_material_id="main",
            source_segment_id="primary_window",
            source_start_us=0,
            source_end_us=1_500_000,
            target_start_us=0,
            target_end_us=1_500_000,
            word_ids=["w001"],
            text="正常内容",
            decision_ids=[],
        )
        caption = CaptionRenderUnit(
            caption_id="v21_cap_000001",
            timeline_segment_ids=["v21_seg_000001"],
            word_ids=["w001"],
            text="正常内容",
            target_start_us=0,
            target_end_us=1_500_000,
            source_subtitle_uids=["s001"],
            style_template_id="canonical_caption_template",
            containing_video_segment_id="v21_seg_000001",
        )
        material_write_plan, writer_blockers = engine.writer.build_write_plan(graph, [caption])
        self.assertFalse(writer_blockers)

        visual_pacing_report = {
            "visual_pacing_executed": True,
            "visual_pacing_merged_count": 0,
            "visual_pacing_blocker_codes": [],
            "visual_merge_safety_report": {
                "gate_passed": True,
                "blocker_codes": [],
                "unsafe_merge_group_count": 0,
                "dropped_content_reintroduced_count": 0,
            },
        }
        with patch(
            "aroll_v21.validate.validators.build_projected_write_view",
            side_effect=ProjectedWriteViewError("TEST_PROJECTION_FAILED", "projection failed"),
        ):
            report = ReadOnlyValidators().run(
                source_graph=graph,
                decision_plan=DecisionPlan(decisions=[]),
                final_timeline=[segment],
                captions=[caption],
                material_write_plan=material_write_plan,
                visual_pacing_report=visual_pacing_report,
                postwrite_mode="simulated",
            )

        projection = report["prewrite_projected_write_view"]
        self.assertFalse(report["validator_report_ok"])
        self.assertFalse(projection["prewrite_projected_write_view_gate_passed"])
        self.assertEqual(projection["prewrite_projection_error_code"], "TEST_PROJECTION_FAILED")
        self.assertIn("V21_PREWRITE_PROJECTED_WRITE_VIEW_FAILED", projection["prewrite_projected_write_view_blocker_codes"])
        self.assertFalse(report["quality_gate_report"]["gate_passed"])
        self.assertIn("V21_PREWRITE_PROJECTED_WRITE_VIEW_FAILED", report["quality_gate_report"]["blocker_codes"])

    def test_caption_gui_gate_blocks_orphan_or_floating_captions(self) -> None:
        orphan = CaptionRenderUnit(
            caption_id="v21_cap_orphan",
            timeline_segment_ids=[],
            word_ids=["w1"],
            text="字幕内容",
            target_start_us=100_000,
            target_end_us=700_000,
            source_subtitle_uids=["s1"],
            style_template_id="canonical_caption_template",
        )

        report = build_caption_alignment_report(final_timeline=[_segment(1, 0, 1_000_000)], captions=[orphan])

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["orphan_caption_count"], 1)
        self.assertEqual(report["floating_caption_count"], 1)
        self.assertIn("V21_CAPTION_GUI_TRACK_GATE_FAILED", report["blocker_codes"])

    def test_caption_gui_gate_requires_single_visible_caption_track(self) -> None:
        report = build_caption_alignment_report(
            final_timeline=[_segment(1, 0, 1_000_000)],
            captions=[_caption(1, "v21_seg_000001", 100_000, 700_000, text="字幕内容")],
            visible_caption_track_count=2,
            caption_lane_count=1,
        )
        quality = build_quality_gate_report(caption_alignment_gate=report)

        self.assertFalse(report["caption_gui_track_gate_passed"])
        self.assertFalse(quality["gate_passed"])
        self.assertEqual(report["visible_caption_track_count"], 2)
        self.assertIn("V21_CAPTION_GUI_TRACK_GATE_FAILED", quality["blocker_codes"])

    def test_subtitle_readability_allows_many_valid_tiny_captions(self) -> None:
        captions = [
            _caption(index, "v21_seg_000001", (index - 1) * 600_000, index * 600_000, text=f"短{index}")
            for index in range(1, 8)
        ]

        report = build_caption_alignment_report(
            final_timeline=[replace(_segment(1, 0, 5_000_000), word_ids=[f"w{index}" for index in range(1, 8)])],
            captions=captions,
        )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["captions_le_3_chars"], 7)
        self.assertEqual(report["captions_le_3_chars_cap"], 3)
        self.assertEqual(report["tiny_caption_fatal_count"], 0)
        self.assertEqual(report["tiny_caption_residual_density_window_count"], 0)
        self.assertNotIn("V21_SUBTITLE_READABILITY_GATE_FAILED", report["blocker_codes"])

    def test_subtitle_readability_blocks_hard_max_chars(self) -> None:
        report = build_caption_alignment_report(
            final_timeline=[_segment(1, 0, 2_000_000)],
            captions=[_caption(1, "v21_seg_000001", 100_000, 1_100_000, text="这是一个明显超过二十个字符限制的超长字幕块内容")],
        )

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["subtitle_hard_max_char_count"], 1)
        self.assertIn("V21_SUBTITLE_HARD_MAX_CHARS", report["blocker_codes"])
        self.assertIn("V21_SUBTITLE_READABILITY_GATE_FAILED", report["blocker_codes"])

    def test_caption_density_gate_blocks_bursty_caption_timeline(self) -> None:
        captions = [
            _caption(index, "v21_seg_000001", (index - 1) * 500_000, index * 500_000, text=f"字幕内容{index}")
            for index in range(1, 10)
        ]

        report = build_caption_alignment_report(
            final_timeline=[_segment(1, 0, 6_000_000)],
            captions=captions,
        )

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["max_captions_in_5s"], 9)
        self.assertGreater(report["caption_burst_density_count"], 0)
        self.assertIn("V21_SUBTITLE_READABILITY_GATE_FAILED", report["blocker_codes"])

    def test_v20_subtitle_interval_thresholds_are_gate_enforced(self) -> None:
        report = build_caption_alignment_report(
            final_timeline=[_segment(1, 0, 5_000_000)],
            captions=[
                _caption(1, "v21_seg_000001", 100_000, 4_000_000, text="字幕内容"),
            ],
        )
        quality = build_quality_gate_report(caption_alignment_gate=report)

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["subtitle_interval_too_long_count"], 1)
        self.assertIn("V21_SUBTITLE_TOO_LONG", report["blocker_codes"])
        self.assertIn("V21_SUBTITLE_READABILITY_GATE_FAILED", quality["blocker_codes"])

    def test_quality_gate_missing_required_gate_fail_closed(self) -> None:
        quality = build_quality_gate_report()

        self.assertFalse(quality["gate_passed"])
        self.assertFalse(quality["effective_speed_gate_present"])
        self.assertFalse(quality["final_caption_visible_repeat_gate_present"])
        self.assertFalse(quality["final_timeline_quality_guard_gate_present"])
        self.assertFalse(quality["semantic_adjudication_gate_present"])
        self.assertFalse(quality["visual_pacing_gate_present"])
        self.assertIn("final_caption_visible_repeat_gate", quality["missing_required_gates"])
        self.assertIn("final_timeline_quality_guard_gate", quality["missing_required_gates"])
        self.assertIn("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_MISSING", quality["blocker_codes"])
        self.assertIn("V21_FINAL_TIMELINE_QUALITY_GUARD_MISSING", quality["blocker_codes"])
        self.assertIn("V21_QUALITY_GATE_MISSING_REQUIRED_GATE", quality["blocker_codes"])

    def test_quality_gate_missing_final_visible_repeat_gate_fail_closed(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "final_repeat_high_count_after": 0},
            semantic_adjudication_gate=_semantic_gate_ok(),
            visual_pacing_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
            },
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertFalse(quality["final_caption_visible_repeat_gate_present"])
        self.assertIn("final_caption_visible_repeat_gate", quality["missing_required_gates"])
        self.assertIn("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_MISSING", quality["blocker_codes"])

    def test_quality_gate_missing_final_timeline_guard_fail_closed(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "final_repeat_high_count_after": 0},
            final_caption_visible_repeat_gate={"gate_passed": True, "blocker_codes": []},
            semantic_adjudication_gate=_semantic_gate_ok(),
            visual_pacing_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
            },
            caption_alignment_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "caption_gui_track_gate_passed": True,
                "subtitle_readability_gate_passed": True,
            },
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertFalse(quality["ready_for_user_manual_qc_preconditions_passed"])
        self.assertFalse(quality["final_timeline_quality_guard_gate_present"])
        self.assertFalse(quality["final_timeline_quality_guard_gate_passed"])
        self.assertIn("final_timeline_quality_guard_gate", quality["missing_required_gates"])
        self.assertIn("V21_FINAL_TIMELINE_QUALITY_GUARD_MISSING", quality["blocker_codes"])

    def test_final_caption_visible_gate_reports_policy_and_repair_signal_layers(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_000_000, text="然后亲手摧毁"),
                _caption(2, "v21_seg_000002", 1_100_000, 2_000_000, text="然后亲手摧毁"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertIn("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED", gate["blocker_codes"])
        self.assertIn("REPAIRABLE_FATAL", gate["final_visible_policy_decision_counts"])
        self.assertGreater(gate["final_visible_repair_signal_candidate_count"], 0)
        self.assertIn("visible_repeat", gate["final_visible_repair_signal_issue_types"])
        self.assertTrue(
            all(
                row["verdict"] in {"REPAIRABLE_FATAL", "BLOCKER_FATAL", "WARNING", "ALLOW", "HUMAN_REVIEW"}
                for row in gate["final_visible_policy_decisions"]
            )
        )

    def test_quality_gate_missing_semantic_adjudication_gate_fail_closed(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "final_repeat_high_count_after": 0},
            final_caption_visible_repeat_gate=_final_visible_gate_ok(),
            final_timeline_quality_guard_gate=_final_timeline_guard_ok(),
            visual_pacing_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
            },
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertFalse(quality["semantic_adjudication_gate_present"])
        self.assertFalse(quality["semantic_adjudication_gate_passed"])
        self.assertIn("V21_QUALITY_GATE_MISSING_REQUIRED_GATE", quality["blocker_codes"])

    def test_quality_gate_subreport_without_gate_passed_fails_closed(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": []},
            visual_pacing_gate={
                "gate_passed": True,
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
                "blocker_codes": [],
            },
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertFalse(quality["effective_speed_gate"]["gate_passed"])

    def test_prewrite_effective_speed_not_applicable_is_not_summary_passed(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={
                "gate_passed": False,
                "blocker_codes": [],
                "prewrite_pending": True,
                "not_applicable": True,
                "not_applicable_reason": "prewrite_source_binding_pending",
            },
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "final_repeat_high_count_after": 0},
            final_caption_visible_repeat_gate=_final_visible_gate_ok(),
            final_timeline_quality_guard_gate=_final_timeline_guard_ok(),
            visual_pacing_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
            },
            semantic_adjudication_gate=_semantic_gate_ok(),
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )
        report = RunReport(
            status="ok",
            source_graph=None,
            repeat_clusters=[],
            decision_plan=None,
            final_timeline=[],
            captions=[],
            material_write_plan={},
            validator_report={"validator_report_ok": True, "quality_gate_report": quality},
            postwrite_report={},
            blocker_report=BlockerReport(blocked=False, blockers=[]),
        )

        summary = build_run_summary(report)

        self.assertTrue(quality["gate_passed"])
        self.assertFalse(summary["effective_speed_gate_passed"])
        self.assertTrue(summary["effective_speed_not_applicable"])
        self.assertIsNone(summary["effective_speed_min"])
        self.assertIsNone(summary["effective_speed_max"])

    def test_quality_gate_normalizes_prewrite_effective_speed_placeholder(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": [], "prewrite_pending": True},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "final_repeat_high_count_after": 0},
            final_caption_visible_repeat_gate=_final_visible_gate_ok(),
            final_timeline_quality_guard_gate=_final_timeline_guard_ok(),
            visual_pacing_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
            },
            semantic_adjudication_gate=_semantic_gate_ok(),
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertTrue(quality["gate_passed"])
        self.assertFalse(quality["effective_speed_gate"]["gate_passed"])
        self.assertTrue(quality["effective_speed_gate"]["not_applicable"])
        self.assertEqual(quality["effective_speed_gate"]["not_applicable_reason"], "prewrite_source_binding_pending")

    def test_quality_gate_rejects_unscoped_effective_speed_not_applicable(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": False, "blocker_codes": [], "not_applicable": True},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "final_repeat_high_count_after": 0},
            visual_pacing_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
            },
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertIn("V21_EFFECTIVE_SPEED_GATE_FAILED", quality["blocker_codes"])

    def test_validator_blockers_include_quality_subgate_codes(self) -> None:
        blockers = ArollEngine()._validator_blockers(
            {
                "validators_read_only": True,
                "final_repeat_validator": {"final_repeat_gate_passed": True},
                "hidden_audio_repeat_validator": {"hidden_audio_repeat_gate_passed": True},
                "safe_cut_validator": {"safe_cut_boundary_gate_passed": True},
                "subtitle_coverage_validator": {"subtitle_coverage_gate_passed": True},
                "caption_alignment_gate": {"gate_passed": True, "blocker_codes": []},
                "final_caption_visible_repeat_gate": {"gate_passed": True, "blocker_codes": []},
                "subtitle_style_validator": {"prewrite_style_gate_ok": True},
                "rough_cut_quality_validator": {"rough_cut_quality_gate_passed": True},
                "postwrite_material_validator": {"postwrite_material_gate_ok": True},
                "semantic_final_review_validator": {"semantic_final_review_validator_passed": True},
                "quality_gate_report": {
                    "gate_passed": False,
                    "blocker_codes": ["V21_VISUAL_CUT_DENSITY_FAILED", "V21_SUBTITLE_READABILITY_GATE_FAILED"],
                },
            }
        )

        codes = [blocker.code for blocker in blockers]
        self.assertIn("V21_QUALITY_GATE_FAILED", codes)
        self.assertIn("V21_VISUAL_CUT_DENSITY_FAILED", codes)
        self.assertIn("V21_SUBTITLE_READABILITY_GATE_FAILED", codes)

    def test_quality_gate_preserves_subgate_diagnostic_fields(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "effective_speed_projected_row_missing_count": 0,
            },
            final_repeat_convergence_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "detector_report_present": True,
                "dropped_cluster_count": 2,
                "dropped_segment_count": 1,
                "clusters_per_dropped_segment": {"4": ["a", "b"]},
            },
            final_caption_visible_repeat_gate=_final_visible_gate_ok(),
            final_timeline_quality_guard_gate=_final_timeline_guard_ok(),
            visual_pacing_gate={
                "gate_passed": True,
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
                "blocker_codes": [],
            },
            semantic_adjudication_gate=_semantic_gate_ok(),
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertTrue(quality["gate_passed"])
        self.assertEqual(quality["effective_speed_gate"]["effective_speed_projected_row_missing_count"], 0)
        self.assertTrue(quality["final_repeat_convergence_gate"]["detector_report_present"])
        self.assertEqual(quality["final_repeat_convergence_gate"]["dropped_segment_count"], 1)

    def test_ready_requires_final_repeat_convergence_gate(self) -> None:
        quality = build_quality_gate_report(
            final_repeat_convergence_gate={
                "gate_passed": False,
                "blocker_codes": ["V21_FINAL_REPEAT_UNRESOLVED_AFTER_CONVERGENCE"],
            }
        )

        self.assertFalse(quality["gate_passed"])
        self.assertIn("V21_FINAL_REPEAT_UNRESOLVED_AFTER_CONVERGENCE", quality["blocker_codes"])

    def test_quality_gate_blocks_final_repeat_detector_not_executed(self) -> None:
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={
                "gate_passed": True,
                "blocker_codes": [],
                "detector_report_present": False,
            },
            visual_pacing_gate={
                "gate_passed": True,
                "visual_pacing_executed": True,
                "visual_merge_safety_gate_passed": True,
                "blocker_codes": [],
            },
            semantic_adjudication_gate=_semantic_gate_ok(),
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertIn("V21_FINAL_REPEAT_DETECTOR_REPORT_MISSING", quality["blocker_codes"])

    def test_ready_requires_visual_pacing_gate(self) -> None:
        quality = build_quality_gate_report(visual_pacing_gate={"gate_passed": False, "blocker_codes": ["V21_VISUAL_PACING_BLOCKED"]})

        self.assertFalse(quality["gate_passed"])
        self.assertIn("V21_VISUAL_PACING_BLOCKED", quality["blocker_codes"])

    def test_ready_requires_caption_alignment_gate(self) -> None:
        quality = build_quality_gate_report(caption_alignment_gate={"gate_passed": False, "blocker_codes": ["V21_CAPTION_OVERLAP"]})

        self.assertFalse(quality["gate_passed"])
        self.assertIn("V21_CAPTION_OVERLAP", quality["blocker_codes"])

    def test_quality_gate_report_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: FakeAdapter(result=fake_real_draft_result(root=root)),
            ):
                summary = run_operator(ArollV21OperatorConfig(mode="dry-run", run_dir=root / "run", draft_dir=draft_dir))

            quality_path = root / "run" / "quality_gate_report.json"
            quality = json.loads(quality_path.read_text("utf-8"))
            self.assertTrue(quality_path.exists())
            self.assertEqual(summary["READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT"], quality["gate_passed"])
            self.assertIn("effective_speed_gate", quality)
            self.assertIn("final_repeat_convergence_gate", quality)
            self.assertIn("visual_pacing_gate", quality)
            self.assertIn("caption_alignment_gate", quality)
            self.assertTrue(quality["post_write_actual_draft_audit_required_on_commit"])
            semantic_junk_path = root / "run" / "quality" / "pre_visible_semantic_junk_report.json"
            semantic_junk_candidates_path = root / "run" / "quality" / "semantic_junk_candidates.json"
            quality_ledger_path = root / "run" / "quality" / "quality_defect_ledger.json"
            manifest = json.loads((root / "run" / "artifact_manifest.json").read_text("utf-8"))
            self.assertTrue(semantic_junk_path.exists())
            self.assertTrue(semantic_junk_candidates_path.exists())
            self.assertTrue(quality_ledger_path.exists())
            self.assertIn("quality/pre_visible_semantic_junk_report.json", manifest["artifact_files"])
            semantic_junk = json.loads(semantic_junk_path.read_text("utf-8"))
            self.assertEqual(semantic_junk["detector_name"], "pre_visible_semantic_junk_candidate_detector")
            self.assertFalse(semantic_junk["pre_visible_semantic_junk_audit_only"])
            self.assertTrue(semantic_junk["pre_visible_semantic_junk_deterministic_apply_enabled"])

    def test_commit_requires_post_write_actual_draft_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            self.assertTrue(writeback_result.report["post_write_actual_draft_audit_required_on_commit"])
            self.assertTrue(writeback_result.report["post_write_actual_draft_audit_executed"])
            self.assertTrue(writeback_result.report["post_write_actual_draft_audit_gate_passed"])

    def test_post_write_audit_blocks_when_actual_draft_missing(self) -> None:
        def unreadable_encrypt(_jy_draftc: Path, _plain: Path, encrypted_out: Path) -> None:
            encrypted_out.write_text("not json and not decryptable", "utf-8")

        def failing_decrypt(_jy_draftc: Path, _encrypted: Path, _decrypted: Path) -> None:
            raise RuntimeError("decrypt unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(unreadable_encrypt, failing_decrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertFalse(writeback_result.success)
            self.assertEqual(writeback_result.blockers[0].code, "V21_POST_WRITE_ACTUAL_DRAFT_AUDIT_FAILED")
            self.assertFalse(writeback_result.report["post_write_actual_draft_loaded"])
            self.assertFalse(writeback_result.report["ready_for_user_manual_qc"])

    def test_post_write_audit_blocks_when_actual_quality_differs_from_plan(self) -> None:
        def drift_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            video_track = next(track for track in data["tracks"] if track["id"] == "video_track")
            video_track["segments"][0]["source_timerange"]["duration"] += 250_000
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(drift_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertFalse(writeback_result.success)
            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(audit["actual_video_rows_match_plan"])
            self.assertFalse(audit["actual_effective_speed_gate_passed"])
            self.assertIn("V21_POST_WRITE_ACTUAL_DRAFT_AUDIT_FAILED", audit["blocker_codes"])

    def test_post_write_audit_passes_when_actual_draft_matches_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertTrue(audit["gate_passed"])
            self.assertTrue(audit["actual_video_rows_match_plan"])
            self.assertTrue(audit["actual_caption_rows_match_plan"])
            self.assertTrue(audit["actual_effective_speed_gate_passed"])
            self.assertTrue(writeback_result.report["ready_for_user_manual_qc"])

    def test_actual_text_residue_gate_blocks_old_subtitle_residue(self) -> None:
        def residue_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            _append_actual_text_segment(
                data,
                segment_id="old_caption_residue_segment",
                material_id="old_caption_residue_material",
                text="旧字幕残留",
                start_us=350_000,
                duration_us=100_000,
            )
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(residue_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertFalse(audit["actual_text_residue_gate_passed"])
            self.assertEqual(audit["old_subtitle_residue_count"], 1)
            self.assertIn("actual_text_residue_gate_failed", [row["reason"] for row in audit["failure_reasons"]])

    def test_actual_text_residue_gate_blocks_text_after_final_video_end(self) -> None:
        def trailing_text_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            _append_actual_text_segment(
                data,
                segment_id="late_caption_residue_segment",
                material_id="late_caption_residue_material",
                text="拖尾旧字幕",
                start_us=900_000,
                duration_us=100_000,
            )
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(trailing_text_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertFalse(audit["actual_text_residue_gate_passed"])
            self.assertEqual(audit["text_after_final_video_end_count"], 1)
            self.assertEqual(audit["orphan_text_segment_count"], 1)

    def test_post_write_audit_fails_when_extra_caption_like_text_segments_exist(self) -> None:
        def extra_caption_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            _append_actual_text_segment(
                data,
                segment_id="extra_caption_like_segment",
                material_id="extra_caption_like_material",
                text="额外字幕",
                start_us=420_000,
                duration_us=80_000,
            )
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(extra_caption_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertFalse(audit["actual_has_no_extra_caption_like_text_segments"])
            self.assertFalse(writeback_result.report["writeback_success"])
            self.assertFalse(writeback_result.report["ready_for_user_manual_qc"])
            self.assertEqual(writeback_result.blockers[0].code, "V21_POST_WRITE_ACTUAL_DRAFT_AUDIT_FAILED")

    def test_actual_audio_coverage_gate_blocks_heard_but_uncaptioned_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)
            report = replace(report, captions=[replace(report.captions[0], word_ids=[])])

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertFalse(audit["actual_audio_coverage_gate_passed"])
            self.assertEqual(audit["audio_coverage_failure_count"], 1)
            self.assertEqual(audit["heard_but_uncaptioned_word_count"], 1)

    def test_audio_coverage_trim_or_blocks_heard_but_uncaptioned_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _caption_trim_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertTrue(audit["actual_audio_coverage_gate_passed"])
            self.assertEqual(audit["heard_but_uncaptioned_word_count"], 0)
            written = json.loads(draft_content.read_text("utf-8"))
            video_segment = next(track for track in written["tracks"] if track["id"] == "video_track")["segments"][0]
            self.assertEqual(video_segment["_v21_audio_coverage"]["source_start_us"], 500_000)
            self.assertEqual(video_segment["_v21_audio_coverage"]["source_end_us"], 800_000)

    def test_audio_coverage_blocks_dropped_but_reintroduced_words(self) -> None:
        def reintroducing_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            video_segment = next(track for track in data["tracks"] if track["id"] == "video_track")["segments"][0]
            video_segment["source_timerange"] = {"start": 100_000, "duration": 700_000}
            video_segment["_v21_audio_coverage"].update(
                {
                    "source_start_us": 100_000,
                    "source_end_us": 800_000,
                    "spoken_source_start_us": 100_000,
                    "spoken_source_end_us": 800_000,
                }
            )
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            before = draft_content.read_text("utf-8")
            result = fake_real_draft_result(root=root)
            report = _dropped_reintroduced_report(result)

            writeback_result = _audit_writeback(reintroducing_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertFalse(audit["actual_audio_coverage_gate_passed"])
            self.assertEqual(audit["dropped_but_reintroduced_word_count"], 1)
            self.assertEqual(draft_content.read_text("utf-8"), before)

    def test_video_source_interval_cannot_include_uncaptioned_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _caption_trim_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            written = json.loads(draft_content.read_text("utf-8"))
            video_segments = next(track for track in written["tracks"] if track["id"] == "video_track")["segments"]
            self.assertEqual(len(video_segments), 1)
            coverage = video_segments[0]["_v21_audio_coverage"]
            self.assertFalse(coverage["source_start_us"] < 400_000 and coverage["source_end_us"] > 100_000)
            self.assertEqual(coverage["word_ids"], ["w_keep"])

    def test_single_dropped_word_at_boundary_is_trimmed_from_video_source_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _caption_trim_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            written = json.loads(draft_content.read_text("utf-8"))
            video_segment = next(track for track in written["tracks"] if track["id"] == "video_track")["segments"][0]
            coverage = video_segment["_v21_audio_coverage"]
            self.assertEqual(coverage["source_start_us"], 500_000)
            self.assertEqual(coverage["source_end_us"], 800_000)
            self.assertEqual(coverage["word_ids"], ["w_keep"])

    def test_single_dropped_word_inside_segment_splits_video_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _inside_dropped_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            written = json.loads(draft_content.read_text("utf-8"))
            video_segments = next(track for track in written["tracks"] if track["id"] == "video_track")["segments"]
            self.assertEqual(len(video_segments), 2)
            coverages = [segment["_v21_audio_coverage"] for segment in video_segments]
            self.assertEqual([(row["source_start_us"], row["source_end_us"]) for row in coverages], [(100_000, 300_000), (600_000, 800_000)])
            self.assertEqual([row["word_ids"] for row in coverages], [["w_before"], ["w_after"]])

    def test_video_write_plan_is_gapless_after_split_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _inside_dropped_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            written = json.loads(draft_content.read_text("utf-8"))
            video_segments = next(track for track in written["tracks"] if track["id"] == "video_track")["segments"]
            self.assertEqual(
                [
                    (segment["target_timerange"]["start"], segment["target_timerange"]["duration"])
                    for segment in video_segments
                ],
                [(0, 200_000), (200_000, 200_000)],
            )
            self.assertTrue(writeback_result.report["gapless_video_write_plan_enabled"])
            self.assertEqual(writeback_result.report["gapless_final_video_end_us"], 400_000)

    def test_caption_times_repacked_after_video_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _inside_dropped_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            written = json.loads(draft_content.read_text("utf-8"))
            text_segment = next(track for track in written["tracks"] if track["id"] == "text_track")["segments"][0]
            self.assertEqual(text_segment["target_timerange"], {"start": 0, "duration": 400_000})
            self.assertLessEqual(
                text_segment["target_timerange"]["start"] + text_segment["target_timerange"]["duration"],
                writeback_result.report["gapless_final_video_end_us"],
            )

    def test_caption_does_not_cross_split_video_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _inside_dropped_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertTrue(writeback_result.success)
            self.assertEqual(audit["post_write_video_target_gap_count_gt_300ms"], 0)
            self.assertEqual(audit["caption_crosses_video_split_gap_count"], 0)
            self.assertEqual(audit["split_caption_container_mismatch_count"], 0)

    def test_jianying_canonical_save_collapse_keeps_caption_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _inside_dropped_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertTrue(writeback_result.success)
            self.assertTrue(audit["jianying_canonical_timeline_sync_gate_passed"])
            self.assertEqual(audit["final_video_end_us"], 400_000)
            self.assertEqual(audit["max_caption_end_us"], 400_000)
            self.assertEqual(audit["captions_after_final_video_end_count"], 0)
            self.assertEqual(audit["caption_video_drift_count"], 0)
            self.assertEqual(audit["max_caption_video_drift_us"], 0)
            self.assertEqual(audit["caption_words_not_covered_by_actual_video_count"], 0)

    def test_captions_after_final_video_end_blocked(self) -> None:
        def late_caption_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            text_track = next(track for track in data["tracks"] if track["id"] == "text_track")
            text_track["segments"][0]["target_timerange"] = {"start": 900_000, "duration": 100_000}
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _inside_dropped_report(result)

            writeback_result = _audit_writeback(late_caption_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertGreater(audit["captions_after_final_video_end_count"], 0)
            self.assertFalse(audit["jianying_canonical_timeline_sync_gate_passed"])
            self.assertIn("V21_JIANYING_CANONICAL_TIMELINE_SYNC_FAILED", audit["blocker_codes"])

    def test_canonical_timeline_sync_gate_blocks_old_gap_based_plan(self) -> None:
        def old_gap_plan_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            video_track = next(track for track in data["tracks"] if track["id"] == "video_track")
            video_track["segments"][1]["target_timerange"]["start"] = 600_000
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _inside_dropped_report(result)

            writeback_result = _audit_writeback(old_gap_plan_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertFalse(audit["jianying_canonical_timeline_sync_gate_passed"])
            self.assertEqual(audit["post_write_video_target_gap_count_gt_300ms"], 1)
            self.assertEqual(audit["caption_crosses_video_split_gap_count"], 1)
            self.assertIn("V21_JIANYING_CANONICAL_TIMELINE_SYNC_FAILED", audit["blocker_codes"])

    def test_dropped_word_is_not_solved_by_caption_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = _inside_dropped_report(result)

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertTrue(writeback_result.success)
            written = json.loads(draft_content.read_text("utf-8"))
            video_segments = next(track for track in written["tracks"] if track["id"] == "video_track")["segments"]
            self.assertNotIn("w_drop_inside", [word_id for segment in video_segments for word_id in segment["_v21_audio_coverage"]["word_ids"]])
            caption_material = written["materials"]["texts"][-1]
            self.assertNotIn("删掉", json.dumps(caption_material, ensure_ascii=False))

    def test_audio_coverage_zero_after_dropped_word_trim_or_split(self) -> None:
        for report_factory in (_caption_trim_report, _inside_dropped_report):
            with self.subTest(report_factory=report_factory.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                draft_dir, _draft_content, _template = create_disposable_draft(root)
                result = fake_real_draft_result(root=root)
                report = report_factory(result)

                writeback_result = _audit_writeback().commit(
                    draft_dir=draft_dir,
                    run_dir=root / "run",
                    real_draft_result=result,
                    run_report=report,
                    sacrificial_write_override_used=True,
                )

                audit = writeback_result.report["post_write_actual_draft_audit"]
                self.assertTrue(writeback_result.success)
                self.assertTrue(audit["actual_audio_coverage_gate_passed"])
                self.assertEqual(audit["audio_coverage_failure_count"], 0)
                self.assertEqual(audit["heard_but_uncaptioned_word_count"], 0)
                self.assertEqual(audit["dropped_but_reintroduced_word_count"], 0)

    def test_audio_coverage_allows_explicit_silence_or_handle_gap_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)
            tail_word = replace(
                report.source_graph.words[0],
                word_id="w_tail_handle",
                text="尾声",
                normalized_text="尾声",
                source_start_us=450_000,
                source_end_us=500_000,
            )
            report = replace(report, source_graph=replace(report.source_graph, words=[*report.source_graph.words, tail_word]))

            writeback_result = _audit_writeback().commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertTrue(writeback_result.success)
            self.assertTrue(audit["actual_audio_coverage_gate_passed"])
            self.assertEqual(audit["heard_but_uncaptioned_word_count"], 0)
            self.assertEqual(audit["actual_audio_coverage_report"]["allowed_handle_gap_word_count"], 0)

    def test_actual_visible_text_repeat_gate_scans_residue_text_segments(self) -> None:
        def repeat_residue_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            _append_actual_text_segment(
                data,
                segment_id="repeat_residue_segment",
                material_id="repeat_residue_material",
                text="测试",
                start_us=350_000,
                duration_us=100_000,
            )
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(repeat_residue_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertFalse(audit["actual_visible_text_repeat_gate_passed"])
            self.assertGreater(audit["actual_visible_repeat_candidate_count"], 0)

    def test_actual_caption_rows_exact_match_requires_no_extras(self) -> None:
        def exact_match_extra_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            _append_actual_text_segment(
                data,
                segment_id="exact_match_extra_segment",
                material_id="exact_match_extra_material",
                text="多余字幕",
                start_us=360_000,
                duration_us=100_000,
            )
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(exact_match_extra_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            audit = writeback_result.report["post_write_actual_draft_audit"]
            self.assertFalse(writeback_result.success)
            self.assertTrue(audit["expected_caption_rows_present"])
            self.assertFalse(audit["actual_caption_rows_exact_match_plan"])
            self.assertFalse(audit["actual_caption_rows_match_plan"])

    def test_writeback_audit_failure_rolls_back_target_files(self) -> None:
        original_copyfile = real_writeback_module.shutil.copyfile

        def tampering_copyfile(src: Path, dst: Path) -> str:
            src_path = Path(src)
            dst_path = Path(dst)
            if src_path.name == "draft_content.v21.modified.enc.json" and dst_path.name == "draft_content.json":
                data = json.loads(src_path.read_text("utf-8"))
                _append_actual_text_segment(
                    data,
                    segment_id="post_copy_residue_segment",
                    material_id="post_copy_residue_material",
                    text="复制后污染字幕",
                    start_us=350_000,
                    duration_us=100_000,
                )
                dst_path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
                return str(dst_path)
            return original_copyfile(src_path, dst_path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            before_draft = draft_content.read_text("utf-8")
            before_template = template.read_text("utf-8")
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            with patch("aroll_v21.writeback.real_draft_writeback.shutil.copyfile", side_effect=tampering_copyfile):
                writeback_result = _audit_writeback().commit(
                    draft_dir=draft_dir,
                    run_dir=root / "run",
                    real_draft_result=result,
                    run_report=report,
                    sacrificial_write_override_used=True,
                )

            self.assertFalse(writeback_result.success)
            self.assertFalse(writeback_result.report["commit_performed"])
            self.assertFalse(writeback_result.report["WRITE_SUCCESS"])
            self.assertTrue(writeback_result.report["rollback_performed"])
            self.assertTrue(writeback_result.report["rollback_success"])
            self.assertEqual(draft_content.read_text("utf-8"), before_draft)
            self.assertEqual(template.read_text("utf-8"), before_template)

    def test_commit_performed_false_means_no_persistent_target_write(self) -> None:
        def bad_staged_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            _append_actual_text_segment(
                data,
                segment_id="staged_residue_segment",
                material_id="staged_residue_material",
                text="预审污染字幕",
                start_us=350_000,
                duration_us=100_000,
            )
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            before_draft = draft_content.read_text("utf-8")
            before_template = template.read_text("utf-8")
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(bad_staged_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertFalse(writeback_result.success)
            self.assertFalse(writeback_result.report["commit_performed"])
            self.assertFalse(writeback_result.report["WRITE_SUCCESS"])
            self.assertTrue(all(value is False for value in writeback_result.report["target_writes"].values()))
            self.assertEqual(draft_content.read_text("utf-8"), before_draft)
            self.assertEqual(template.read_text("utf-8"), before_template)

    def test_ready_for_user_manual_qc_false_when_post_write_audit_failed(self) -> None:
        def drift_encrypt(_jy_draftc: Path, plain: Path, encrypted_out: Path) -> None:
            data = json.loads(plain.read_text("utf-8"))
            video_track = next(track for track in data["tracks"] if track["id"] == "video_track")
            video_track["segments"][0]["target_timerange"]["duration"] += 250_000
            encrypted_out.write_text(json.dumps(data, ensure_ascii=False), "utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            report = run_report_from_result(result)

            writeback_result = _audit_writeback(drift_encrypt).commit(
                draft_dir=draft_dir,
                run_dir=root / "run",
                real_draft_result=result,
                run_report=report,
                sacrificial_write_override_used=True,
            )

            self.assertFalse(writeback_result.success)
            self.assertFalse(writeback_result.report["ready_for_user_manual_qc"])
            self.assertFalse(writeback_result.report["writeback_success"])

    def test_caption_no_overlap_after_video_merge(self) -> None:
        segment = replace(_segment(1, 0, 1_200_000), word_ids=["w1", "w2"])
        report = build_caption_alignment_report(
            final_timeline=[segment],
            captions=[
                _caption(1, "v21_seg_000001", 0, 500_000),
                _caption(2, "v21_seg_000001", 500_000, 1_000_000),
            ],
        )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["caption_overlap_count"], 0)

    def test_caption_count_can_exceed_video_segment_count(self) -> None:
        segment = replace(_segment(1, 0, 1_200_000), word_ids=["w1", "w2"])
        visual = build_visual_pacing_report(
            final_timeline=[segment],
            captions=[
                _caption(1, "v21_seg_000001", 0, 500_000),
                _caption(2, "v21_seg_000001", 500_000, 1_000_000),
            ],
            executed=True,
            merge_report={
                "visual_pacing_executed": True,
                "visual_pacing_merge_attempted_count": 0,
                "visual_pacing_merged_count": 0,
                "visual_short_segment_count_lt_1200ms_before": 0,
                "visual_short_segment_count_lt_1200ms_after": 0,
                "semantic_bridge_short_segment_count": 0,
                "visual_pacing_blocker_codes": [],
            },
        )
        alignment = build_caption_alignment_report(
            final_timeline=[segment],
            captions=[
                _caption(1, "v21_seg_000001", 0, 500_000),
                _caption(2, "v21_seg_000001", 500_000, 1_000_000),
            ],
        )

        self.assertEqual(visual["final_video_segment_count"], 1)
        self.assertEqual(visual["caption_count"], 2)
        self.assertTrue(alignment["gate_passed"])

    def test_too_short_caption_merges_with_neighbor(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "就", 0, 240_000),
                ("w002", "继续", 300_000, 700_000),
            ]
        )
        segment = replace(_segment(1, 0, 900_000), source_end_us=900_000, word_ids=["w001", "w002"], text="就继续")

        captions = SubtitleRenderer().render([segment], source_graph)
        alignment = build_caption_alignment_report(final_timeline=[segment], captions=captions)

        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0].text, "就继续")
        self.assertEqual(alignment["caption_too_short_count"], 0)
        self.assertEqual(alignment["one_char_caption_count"], 0)

    def test_too_short_caption_extends_inside_containing_video_when_safe(self) -> None:
        segment = replace(_segment(1, 0, 600_000), word_ids=["w1"], text="短语")
        captions = [
            replace(
                _caption(1, "v21_seg_000001", 100_000, 250_000, text="短语"),
                containing_video_segment_id="v21_seg_000001",
            )
        ]

        cleaned = _cleanup_caption_units(captions, {"v21_seg_000001": segment})

        self.assertEqual(cleaned[0].target_start_us, 100_000)
        self.assertEqual(cleaned[0].target_end_us, 600_000)

    def test_one_char_caption_merges_with_neighbor(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "就", 0, 240_000),
                ("w002", "继续", 300_000, 700_000),
            ]
        )
        segment = replace(_segment(1, 0, 900_000), source_end_us=900_000, word_ids=["w001", "w002"], text="就继续")

        captions = SubtitleRenderer().render([segment], source_graph)

        self.assertEqual(len(captions), 1)
        self.assertGreater(len(captions[0].text), 1)

    def test_one_char_caption_count_zero_after_cleanup(self) -> None:
        source_graph = _graph_for_visual_merge_rows(
            [
                ("w001", "就", 0, 240_000),
                ("w002", "继续", 300_000, 700_000),
            ]
        )
        segment = replace(_segment(1, 0, 900_000), source_end_us=900_000, word_ids=["w001", "w002"], text="就继续")
        captions = SubtitleRenderer().render([segment], source_graph)
        report = build_caption_alignment_report(final_timeline=[segment], captions=captions)

        self.assertEqual(report["one_char_caption_count"], 0)
        self.assertEqual(report["residual_one_char_captions"], [])

    def test_caption_cleanup_does_not_merge_tiny_caption_into_overlong_interval(self) -> None:
        source_graph = _graph_for_single_subtitle_words(
            [
                ("w001", "咳", 0, 800_000),
                ("w002", "立刻", 2_066_666, 2_426_666),
                ("w003", "给", 2_426_666, 2_666_666),
                ("w004", "老子", 2_666_666, 2_866_666),
                ("w005", "关", 2_866_666, 3_066_666),
                ("w006", "了", 3_066_666, 3_226_666),
                ("w007", "弱智", 3_706_666, 4_266_666),
            ]
        )
        segment = replace(
            _segment(1, 0, 4_266_666),
            source_material_id="main",
            source_segment_id="primary_window",
            source_start_us=0,
            source_end_us=4_266_666,
            target_start_us=0,
            target_end_us=4_266_666,
            word_ids=["w001", "w002", "w003", "w004", "w005", "w006", "w007"],
            text="咳立刻给老子关了弱智",
        )

        captions = SubtitleRenderer().render([segment], source_graph)
        alignment = build_caption_alignment_report(final_timeline=[segment], captions=captions)

        self.assertEqual([caption.text for caption in captions], ["咳立刻给老子关了", "弱智"])
        self.assertLessEqual(max(caption.target_end_us - caption.target_start_us for caption in captions), 3_500_000)
        self.assertEqual(alignment["subtitle_interval_too_long_count"], 0)
        self.assertEqual(alignment["one_char_caption_count"], 0)
        self.assertTrue(alignment["gate_passed"], alignment)

    def test_final_visible_keeps_opening_command_and_following_expressive_tail(self) -> None:
        rows = [
            ("w001", "咳", 1_500_000, 2_300_000),
            ("w002", "立刻", 3_566_666, 3_926_666),
            ("w003", "给", 3_926_666, 4_166_666),
            ("w004", "老子", 4_166_666, 4_366_666),
            ("w005", "关", 4_366_666, 4_566_666),
            ("w006", "了", 4_566_666, 4_726_666),
            ("w007", "弱智", 5_206_666, 5_766_666),
            ("w008", "老子", 6_066_666, 6_346_666),
            ("w009", "最烦", 6_346_666, 6_746_666),
            ("w010", "这种", 6_746_666, 6_946_666),
            ("w011", "深夜", 6_946_666, 7_226_666),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = [
            replace(
                _segment(1, 0, 3_226_666),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=1_500_000,
                source_end_us=4_726_666,
                target_start_us=0,
                target_end_us=3_226_666,
                word_ids=[f"w{index:03d}" for index in range(1, 7)],
                text="咳立刻给老子关了",
            ),
            replace(
                _segment(2, 3_226_666, 3_786_666),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=5_206_666,
                source_end_us=5_766_666,
                target_start_us=3_226_666,
                target_end_us=3_786_666,
                word_ids=["w007"],
                text="弱智",
            ),
            replace(
                _segment(3, 3_786_666, 4_946_666),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=6_066_666,
                source_end_us=7_226_666,
                target_start_us=3_786_666,
                target_end_us=4_946_666,
                word_ids=[f"w{index:03d}" for index in range(8, 12)],
                text="老子最烦这种深夜",
            ),
        ]
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        visible = "".join(caption.text for caption in result.captions)
        self.assertNotIn("咳", visible)
        self.assertIn("立刻给老子关了弱智老子最烦这种深夜", visible)
        actions = result.report["final_visible_repair_actions"]
        self.assertNotIn(
            "弱智",
            [action.get("junk_text") for action in actions if action.get("decision") == "drop_isolated_junk_segment"],
        )
        self.assertNotIn(
            "老子关了",
            [action.get("target_text") for action in actions if action.get("issue_type") == "boundary_restart"],
        )
        self.assertNotIn(
            "立刻给",
            [action.get("dropped_text") for action in actions if action.get("reason") == "short_abandoned_open_clause"],
        )

    def test_visual_pacing_gate_allows_only_explicit_semantic_bridge_exceptions(self) -> None:
        allowed = build_visual_pacing_report(
            final_timeline=[replace(_segment(1, 0, 700_000), text="语义桥")],
            captions=[],
            executed=True,
            merge_report={"visual_pacing_executed": True},
        )
        blocked = build_visual_pacing_report(
            final_timeline=[
                replace(_segment(1, 0, 500_000), text="交配权"),
                replace(_segment(2, 500_000, 1_000_000), text="交配权"),
                replace(_segment(3, 1_000_000, 1_500_000), text="交配权"),
                replace(_segment(4, 1_500_000, 2_000_000), text="交配权"),
            ],
            captions=[],
            executed=True,
            merge_report={"visual_pacing_executed": True},
        )

        self.assertTrue(allowed["gate_passed"])
        self.assertEqual(allowed["visual_short_segment_count_lt_1200ms_after_blocking"], 0)
        self.assertEqual(allowed["semantic_bridge_reason_counts"], {"semantic_bridge_exception": 1})
        self.assertEqual(allowed["semantic_bridge_short_segment_details"][0]["text"], "语义桥")
        self.assertFalse(blocked["gate_passed"])
        self.assertEqual(blocked["visual_short_segment_count_lt_1200ms_after_blocking"], 4)

    def test_semantic_bridge_count_cap_blocks_abuse(self) -> None:
        timeline = [
            replace(_segment(index, (index - 1) * 700_000, index * 700_000), text=f"语义桥{index}")
            for index in range(1, 10)
        ]

        report = build_visual_pacing_report(
            final_timeline=timeline,
            captions=[],
            executed=True,
            merge_report={"visual_pacing_executed": True},
        )

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["semantic_bridge_short_segment_count"], 9)
        self.assertEqual(report["semantic_bridge_cap"], 8)
        self.assertIn("V21_VISUAL_SEMANTIC_BRIDGE_ABUSE", report["blocker_codes"])

    def test_semantic_bridge_cap_scales_with_long_timeline(self) -> None:
        timeline = []
        for index in range(1, 61):
            start = (index - 1) * 3_000_000
            duration = 700_000 if index <= 9 else 1_500_000
            text = f"语义桥{index}" if index <= 9 else f"长段内容{index}"
            timeline.append(replace(_segment(index, start, start + duration), text=text))

        report = build_visual_pacing_report(
            final_timeline=timeline,
            captions=[],
            executed=True,
            merge_report={"visual_pacing_executed": True},
        )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["semantic_bridge_short_segment_count"], 9)
        self.assertEqual(report["semantic_bridge_cap"], 9)
        self.assertNotIn("V21_VISUAL_SEMANTIC_BRIDGE_ABUSE", report["blocker_codes"])

    def test_visual_pacing_recomputes_counts_after_final_visible_repair(self) -> None:
        report = build_visual_pacing_report(
            final_timeline=[replace(_segment(1, 0, 700_000), text="语义桥")],
            captions=[],
            executed=True,
            merge_report={
                "visual_pacing_executed": True,
                "visual_short_segment_count_lt_1200ms_after": 2,
                "visual_short_segment_count_lt_1200ms_after_blocking": 1,
                "semantic_bridge_short_segment_count": 1,
                "visual_pacing_blocker_codes": ["V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN"],
                "final_visible_repair_action_count": 1,
            },
        )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["visual_short_segment_count_lt_1200ms_after"], 1)
        self.assertEqual(report["visual_short_segment_count_lt_1200ms_after_blocking"], 0)
        self.assertNotIn("V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN", report["blocker_codes"])

    def test_cut_density_gate_blocks_bursty_timeline(self) -> None:
        timeline = [
            replace(_segment(index, (index - 1) * 1_200_000, index * 1_200_000), text=f"长段{index}")
            for index in range(1, 13)
        ]

        report = build_visual_pacing_report(
            final_timeline=timeline,
            captions=[],
            executed=True,
            merge_report={"visual_pacing_executed": True},
        )

        self.assertFalse(report["gate_passed"])
        self.assertGreater(report["cuts_per_minute"], report["cut_density_thresholds"]["max_cuts_per_minute"])
        self.assertIn("V21_VISUAL_CUT_DENSITY_FAILED", report["blocker_codes"])

    def test_semantic_bridge_requires_reason_details(self) -> None:
        report = build_visual_pacing_report(
            final_timeline=[replace(_segment(1, 0, 700_000), text="语义桥")],
            captions=[],
            executed=True,
            merge_report={"visual_pacing_executed": True},
        )

        detail = report["semantic_bridge_short_segment_details"][0]
        self.assertEqual(detail["semantic_bridge_reason"], "semantic_bridge_exception")
        self.assertEqual(detail["text"], "语义桥")
        self.assertEqual(detail["duration_us"], 700_000)
        self.assertEqual(detail["why_not_merge"], ["semantic_bridge_exception_preserved"])

    def test_weak_filler_cannot_be_semantic_bridge(self) -> None:
        report = build_visual_pacing_report(
            final_timeline=[replace(_segment(1, 0, 700_000), text="然后")],
            captions=[],
            executed=True,
            merge_report={"visual_pacing_executed": True},
        )

        self.assertEqual(report["semantic_bridge_short_segment_count"], 0)
        self.assertEqual(report["semantic_bridge_short_segment_details"], [])
        self.assertFalse(report["residual_visual_short_segments"][0]["semantic_bridge"])
        self.assertTrue(report["residual_visual_short_segments"][0]["weak_filler"])

    def test_safe_merge_candidate_is_merged_not_reported_as_semantic_bridge(self) -> None:
        materials, text_segments = _template_rows()
        source_graph = ArollEngine().ingest.build_source_graph(
            word_timeline=[
                {
                    "word_id": "w001",
                    "word_text": "语义桥",
                    "start_us": 0,
                    "end_us": 700_000,
                    "subtitle_index": 1,
                    "subtitle_uid": "s001",
                },
                {
                    "word_id": "w002",
                    "word_text": "继续内容",
                    "start_us": 700_000,
                    "end_us": 2_000_000,
                    "subtitle_index": 1,
                    "subtitle_uid": "s001",
                },
            ],
            subtitles=[
                {
                    "subtitle_uid": "s001",
                    "subtitle_index": 1,
                    "text": "语义桥继续内容",
                    "word_ids": ["w001", "w002"],
                }
            ],
            source_segments=[
                {
                    "id": "primary_window",
                    "material_id": "main",
                    "type": "video",
                    "source_start_us": 0,
                    "source_end_us": 2_000_000,
                }
            ],
            text_materials=materials,
            text_segments=text_segments,
        )
        timeline = [
            replace(_segment(1, 0, 700_000), source_end_us=700_000, word_ids=["w001"], text="语义桥"),
            replace(
                _segment(2, 700_000, 2_000_000),
                source_start_us=700_000,
                source_end_us=2_000_000,
                word_ids=["w002"],
                text="继续内容",
            ),
        ]

        normalized, report = VisualPacingNormalizer().normalize(timeline, source_graph)

        self.assertTrue(report["gate_passed"])
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].text, "语义桥继续内容")
        self.assertEqual(report["semantic_bridge_safe_merge_candidate_count"], 0)
        self.assertEqual(report["semantic_bridge_short_segment_details"], [])

    def test_video_segments_can_be_fewer_than_captions_in_main_chain(self) -> None:
        result = _multi_caption_fake_result()
        report = ArollEngine().run(
            ArollRunInput(
                mode="write",
                draft_data=result.draft_data,
                word_timeline=result.word_timeline,
                subtitles=result.subtitles,
                source_segments=result.source_segments,
                source_materials=result.source_materials,
                text_materials=result.text_materials,
                text_segments=result.text_segments,
                postwrite_mode="simulated",
            )
        )

        self.assertEqual(report.status, "ok")
        self.assertLess(len(report.final_timeline), len(report.captions))
        self.assertGreater(report.validator_report["visual_pacing_gate"]["caption_per_video_segment_ratio"], 1.0)

    def test_caption_timeline_can_contain_multiple_captions_inside_one_video_segment(self) -> None:
        result = _multi_caption_fake_result()
        report = ArollEngine().run(
            ArollRunInput(
                mode="write",
                draft_data=result.draft_data,
                word_timeline=result.word_timeline,
                subtitles=result.subtitles,
                source_segments=result.source_segments,
                source_materials=result.source_materials,
                text_materials=result.text_materials,
                text_segments=result.text_segments,
                postwrite_mode="simulated",
            )
        )

        self.assertEqual(len(report.final_timeline), 1)
        self.assertGreater(len(report.captions), 1)
        self.assertTrue(all(caption.containing_video_segment_id == report.final_timeline[0].segment_id for caption in report.captions))
        self.assertTrue(all(report.final_timeline[0].target_start_us <= caption.target_start_us <= caption.target_end_us <= report.final_timeline[0].target_end_us for caption in report.captions))

    def test_resolved_template_map_count_matches_video_segment_count_not_caption_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, _draft_content, _template = create_disposable_draft(root)
            result = _multi_caption_fake_result(root=root)
            report = ArollEngine().run(
                ArollRunInput(
                    mode="write",
                    draft_data=result.draft_data,
                    word_timeline=result.word_timeline,
                    subtitles=result.subtitles,
                    source_segments=result.source_segments,
                    source_materials=result.source_materials,
                    text_materials=result.text_materials,
                    text_segments=result.text_segments,
                    postwrite_mode="simulated",
                )
            )

            preflight = DynamicSourceBindingPreflight(root_mirror_func=fake_root_mirror_not_required).preflight(
                draft_dir=draft_dir,
                real_draft_result=result,
                run_report=report,
                run_dir=root / "run",
            )

            self.assertTrue(preflight.success)
            self.assertEqual(preflight.report["resolved_template_map_count"], len(report.final_timeline))
            self.assertNotEqual(preflight.report["resolved_template_map_count"], len(report.captions))

    def test_no_sample_text_hardcoded_in_src(self) -> None:
        sample_tokens = [
            "交配权",
            "Steam上乱花",
            "笑人笑人家",
            "就这几就这几个",
            "provider的流水账",
            "后台私信全是",
            "6月19日",
            "集美",
        ]
        checked = 0
        for root in (Path("src"), Path("scripts")):
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".py", ".ps1", ".md", ".json", ".yaml", ".yml"}:
                    continue
                text = path.read_text("utf-8", errors="ignore")
                checked += 1
                for token in sample_tokens:
                    self.assertNotIn(token, text, msg=f"{token!r} leaked into {path}")
        self.assertGreater(checked, 0)

    def test_isolated_short_junk_uses_structural_rule_not_text_literal(self) -> None:
        rows = [
            ("w001", "前面长句保持上下文", 0, 1_200_000),
            ("w002", "怪词", 1_700_000, 2_000_000),
            ("w003", "后面长句继续表达", 2_500_000, 3_700_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = _timeline_from_word_rows(rows)
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([segment.text for segment in result.final_timeline], ["前面长句保持上下文", "后面长句继续表达"])
        self.assertIn("drop_isolated_junk_segment", [action["decision"] for action in result.report["final_visible_repair_actions"]])

    def test_final_visible_repair_allows_complex_case_more_than_10_passes(self) -> None:
        labels = ["甲一", "乙二", "丙三", "丁四", "戊五", "己六", "庚七", "辛八", "壬九", "癸十", "子十一"]
        rows: list[tuple[str, str, int, int]] = []
        cursor = 0
        for index, label in enumerate(labels, start=1):
            rows.append((f"w{index:03d}a", f"前段{label}", cursor, cursor + 500_000))
            cursor += 500_000
            rows.append((f"w{index:03d}b", f"的是后续{label}", cursor, cursor + 500_000))
            cursor += 500_000
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = _timeline_from_word_rows(rows)
        captions = renderer.render(timeline, source_graph)

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
            max_passes=20,
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertGreater(result.report["final_visible_repair_action_count"], 10)
        self.assertEqual(result.report["final_visible_repair_stop_reason"], "converged")
        self.assertEqual(result.report["final_visible_repair_final_counts"]["dangling_prefix_suffix_count"], 0)

    def test_final_visible_repair_stops_on_no_progress(self) -> None:
        rows = [("w001", "的是问题", 0, 800_000)]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = _timeline_from_word_rows(rows)
        captions = renderer.render(timeline, source_graph)
        no_op_step = final_visible_repair_module._RepairStep(
            final_timeline=timeline,
            captions=captions,
            action={"decision": "test_noop"},
            timeline_changed=False,
        )

        with patch.object(final_visible_repair_module, "_repair_next_issue", return_value=no_op_step):
            result = repair_final_visible_caption_issues(
                final_timeline=timeline,
                captions=captions,
                source_graph=source_graph,
                render_captions=lambda repaired: renderer.render(repaired, source_graph),
                max_passes=4,
            )

        self.assertFalse(result.report["final_visible_repair_success"])
        self.assertTrue(result.report["final_visible_repair_no_progress_detected"])
        self.assertEqual(result.report["final_visible_repair_stop_reason"], "no_progress_detected")
        self.assertEqual(result.report["final_visible_repair_unresolved"][0]["reason"], "no_progress_detected")

    def test_final_visible_repair_reports_max_pass_exhausted(self) -> None:
        rows = [("w001", "的是问题", 0, 800_000)]
        source_graph = _graph_for_single_subtitle_words(rows)
        renderer = SubtitleRenderer()
        timeline = _timeline_from_word_rows(rows)
        captions = renderer.render(timeline, source_graph)

        def progress_step(**kwargs):
            pass_index = int(kwargs["pass_index"])
            shifted = [
                replace(
                    caption,
                    target_end_us=int(caption.target_end_us) + pass_index,
                )
                for caption in kwargs["captions"]
            ]
            return final_visible_repair_module._RepairStep(
                final_timeline=kwargs["final_timeline"],
                captions=shifted,
                action={"decision": f"test_progress_{pass_index}"},
                timeline_changed=False,
            )

        with patch.object(final_visible_repair_module, "_repair_next_issue", side_effect=progress_step):
            result = repair_final_visible_caption_issues(
                final_timeline=timeline,
                captions=captions,
                source_graph=source_graph,
                render_captions=lambda repaired: renderer.render(repaired, source_graph),
                max_passes=3,
            )

        self.assertFalse(result.report["final_visible_repair_success"])
        self.assertTrue(result.report["final_visible_repair_max_pass_exhausted"])
        self.assertEqual(result.report["final_visible_repair_stop_reason"], "max_repair_passes_exhausted")
        self.assertEqual(result.report["final_visible_repair_unresolved"][0]["reason"], "max_repair_passes_exhausted")

    def test_caption_only_merge_materializes_same_segment_multi_caption(self) -> None:
        rows = [
            ("w001", "我们反对", 0, 800_000),
            ("w002", "的", 800_000, 900_000),
            ("w003", "是公共问题", 900_000, 1_600_000),
        ]
        source_graph = _graph_for_single_subtitle_words(rows)
        timeline = [
            replace(
                _segment(1, 0, 1_600_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=1_600_000,
                target_start_us=0,
                target_end_us=1_600_000,
                word_ids=["w001", "w002", "w003"],
                text="我们反对的是公共问题",
            )
        ]
        captions = [
            CaptionRenderUnit(
                caption_id="v21_cap_000001",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w001"],
                text="我们反对",
                target_start_us=0,
                target_end_us=800_000,
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
            CaptionRenderUnit(
                caption_id="v21_cap_000002",
                timeline_segment_ids=["v21_seg_000001"],
                word_ids=["w002", "w003"],
                text="的是公共问题",
                target_start_us=800_000,
                target_end_us=1_600_000,
                source_subtitle_uids=["s001"],
                style_template_id="canonical_caption_template",
                containing_video_segment_id="v21_seg_000001",
            ),
        ]
        renderer = SubtitleRenderer()

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda repaired: renderer.render(repaired, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"])
        self.assertEqual([caption.text for caption in result.captions], ["我们反对的是公共问题"])
        self.assertEqual(len(result.final_timeline), 1)
        self.assertEqual(result.report["final_visible_repair_final_timeline_counts"]["dangling_prefix_suffix_count"], 0)

    def test_caption_only_merge_materializes_partial_previous_segment_tail(self) -> None:
        self.test_caption_only_materialization_supports_partial_previous_segment_tail()

    def test_consumed_caption_not_counted_by_final_timeline_gate(self) -> None:
        self.test_caption_only_merge_removes_dangling_from_effective_visible_captions()

    def test_final_visible_drop_entire_caption_segment_reports_valid_action(self) -> None:
        self.test_final_visible_drops_isolated_semantic_junk_caption()

    def test_final_visible_trim_shared_segment_suffix_reports_valid_action(self) -> None:
        self.test_final_visible_trims_isolated_junk_suffix_from_shared_segment()

    def test_visual_pacing_recomputed_after_final_visible_timeline_repair(self) -> None:
        self.test_visual_pacing_recomputes_counts_after_final_visible_repair()

    def test_quality_gate_report_uses_final_timeline_after_repair(self) -> None:
        visual = build_visual_pacing_report(
            final_timeline=[replace(_segment(1, 0, 700_000), text="语义桥")],
            captions=[],
            executed=True,
            merge_report={
                "visual_pacing_executed": True,
                "visual_short_segment_count_lt_1200ms_after": 4,
                "visual_pacing_blocker_codes": ["V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN"],
                "final_visible_repair_action_count": 1,
                "semantic_bridge_safe_merge_candidates": [
                    {"segment_id": "stale_segment", "reason": "old_timeline_candidate"}
                ],
            },
        )
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": [], "final_repeat_high_count_after": 0},
            final_caption_visible_repeat_gate={"gate_passed": True, "blocker_codes": []},
            visual_pacing_gate=visual,
            caption_alignment_gate={"gate_passed": True, "blocker_codes": []},
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertTrue(visual["gate_passed"])
        self.assertEqual(visual["semantic_bridge_safe_merge_candidate_count"], 0)
        self.assertEqual(visual["semantic_bridge_safe_merge_candidates"], [])
        self.assertTrue(quality["visual_pacing_gate"]["gate_passed"])
        self.assertNotIn("V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN", quality["blocker_codes"])

    def test_engine_final_visible_cycle_reintroduced_by_visual_pacing_fails_closed(self) -> None:
        from aroll_v21.quality.final_visible_repair.result import FinalVisibleCaptionRepairResult

        class ReintroducingVisualPacing:
            def __init__(self) -> None:
                self.calls = 0
                self.initial_timeline = None

            def normalize(self, final_timeline, _source_graph):
                self.calls += 1
                if self.initial_timeline is None:
                    self.initial_timeline = list(final_timeline)
                    return list(final_timeline), self._report()
                return list(self.initial_timeline), self._report()

            def _report(self) -> dict[str, object]:
                return {
                    "visual_pacing_executed": True,
                    "visual_pacing_merge_attempted_count": 0,
                    "visual_pacing_merged_count": 0,
                    "visual_short_segment_count_lt_1200ms_after": 0,
                    "visual_short_segment_count_lt_1200ms_after_blocking": 0,
                    "visual_pacing_allowed_short_segment_threshold": 999,
                    "semantic_bridge_short_segment_count": 0,
                    "visual_pacing_blocker_codes": [],
                    "blocker_codes": [],
                }

        def fake_repair(*, final_timeline, captions, source_graph, render_captions):
            repaired = list(final_timeline)
            repaired[-1] = replace(repaired[-1], target_end_us=int(repaired[-1].target_end_us) + 1_000)
            return FinalVisibleCaptionRepairResult(
                final_timeline=repaired,
                captions=render_captions(repaired),
                report={
                    "final_visible_repair_attempted": True,
                    "final_visible_repair_success": True,
                    "final_visible_repair_action_count": 1,
                    "final_visible_repair_actions": [{"action": "test_repair"}],
                    "final_visible_repair_unresolved": [],
                    "final_visible_repair_passes_executed": 1,
                    "final_visible_repair_final_timeline_counts": {},
                },
            )

        materials, text_segments = _template_rows()
        visual_pacing = ReintroducingVisualPacing()
        with patch("aroll_v21.engine.repair_final_visible_caption_issues", side_effect=fake_repair):
            report = ArollEngine(visual_pacing=visual_pacing).run(
                ArollRunInput(
                    mode="write",
                    word_timeline=[
                        {
                            "word_id": "w001",
                            "word_text": "第一段",
                            "start_us": 0,
                            "end_us": 1_500_000,
                            "subtitle_index": 1,
                            "subtitle_uid": "s001",
                        },
                        {
                            "word_id": "w002",
                            "word_text": "第二段",
                            "start_us": 1_500_000,
                            "end_us": 3_000_000,
                            "subtitle_index": 2,
                            "subtitle_uid": "s002",
                        },
                    ],
                    subtitles=[
                        {"subtitle_uid": "s001", "subtitle_index": 1, "text": "第一段", "word_ids": ["w001"]},
                        {"subtitle_uid": "s002", "subtitle_index": 2, "text": "第二段", "word_ids": ["w002"]},
                    ],
                    source_segments=[
                        {
                            "id": "primary_window",
                            "material_id": "main",
                            "type": "video",
                            "source_start_us": 0,
                            "source_end_us": 3_000_000,
                        }
                    ],
                    source_materials=[{"source_material_id": "main", "type": "video", "duration_us": 3_000_000}],
                    text_materials=materials,
                    text_segments=text_segments,
                    postwrite_mode="simulated",
                )
            )

        repair_report = report.validator_report["final_visible_caption_repair_report"]
        visible_gate = report.validator_report["final_caption_visible_repeat_gate"]

        self.assertEqual(visual_pacing.calls, 2)
        self.assertFalse(repair_report["final_visible_repair_success"])
        self.assertTrue(repair_report["final_visible_repair_max_cycle_exhausted"])
        self.assertEqual(
            repair_report["final_visible_repair_stop_reason"],
            "visual_pacing_reintroduced_seen_repair_state",
        )
        phases = [row["phase"] for row in repair_report["quality_mutations"]]
        self.assertIn("final_visible_repair.cycle", phases)
        self.assertIn("visual_pacing.after_final_visible_repair", phases)
        self.assertGreaterEqual(repair_report["quality_mutation_count"], 2)
        self.assertEqual(repair_report["quality_mutation_rejected_count"], 0)
        self.assertTrue(
            all(row["before"]["timeline_signature_hash"] and row["after"]["timeline_signature_hash"] for row in repair_report["quality_mutations"])
        )
        self.assertFalse(visible_gate["gate_passed"])
        self.assertIn("V21_FINAL_VISIBLE_REPAIR_UNRESOLVED", visible_gate["blocker_codes"])

    def test_engine_quality_mutation_regression_is_reported_and_blocks_repair_success(self) -> None:
        from aroll_v21.quality.final_visible_repair.result import FinalVisibleCaptionRepairResult

        class NoOpVisualPacing:
            def normalize(self, final_timeline, _source_graph):
                return list(final_timeline), {
                    "visual_pacing_executed": True,
                    "visual_pacing_merge_attempted_count": 0,
                    "visual_pacing_merged_count": 0,
                    "visual_short_segment_count_lt_1200ms_after": 0,
                    "visual_short_segment_count_lt_1200ms_after_blocking": 0,
                    "visual_pacing_allowed_short_segment_threshold": 999,
                    "semantic_bridge_short_segment_count": 0,
                    "visual_pacing_blocker_codes": [],
                    "blocker_codes": [],
                }

        def fake_repair(*, final_timeline, captions, source_graph, render_captions):
            return FinalVisibleCaptionRepairResult(
                final_timeline=list(final_timeline),
                captions=[],
                report={
                    "final_visible_repair_attempted": True,
                    "final_visible_repair_success": True,
                    "final_visible_repair_action_count": 1,
                    "final_visible_repair_actions": [{"action": "drop_all_captions_for_regression_test"}],
                    "final_visible_repair_unresolved": [],
                    "final_visible_repair_passes_executed": 1,
                    "final_visible_repair_final_timeline_counts": {},
                },
            )

        materials, text_segments = _template_rows()
        with patch("aroll_v21.engine.repair_final_visible_caption_issues", side_effect=fake_repair):
            report = ArollEngine(visual_pacing=NoOpVisualPacing()).run(
                ArollRunInput(
                    mode="write",
                    word_timeline=[
                        {
                            "word_id": "w001",
                            "word_text": "完整字幕",
                            "start_us": 0,
                            "end_us": 1_500_000,
                            "subtitle_index": 1,
                            "subtitle_uid": "s001",
                        },
                    ],
                    subtitles=[
                        {"subtitle_uid": "s001", "subtitle_index": 1, "text": "完整字幕", "word_ids": ["w001"]},
                    ],
                    source_segments=[
                        {
                            "id": "primary_window",
                            "material_id": "main",
                            "type": "video",
                            "source_start_us": 0,
                            "source_end_us": 1_500_000,
                        }
                    ],
                    source_materials=[{"source_material_id": "main", "type": "video", "duration_us": 1_500_000}],
                    text_materials=materials,
                    text_segments=text_segments,
                    postwrite_mode="simulated",
                )
            )

        repair_report = report.validator_report["final_visible_caption_repair_report"]

        self.assertFalse(repair_report["final_visible_repair_success"])
        self.assertEqual(repair_report["final_visible_repair_stop_reason"], "quality_mutation_regression_detected")
        self.assertEqual(repair_report["quality_mutation_rejected_count"], 1)
        self.assertEqual(repair_report["quality_mutation_rejection_reasons"], ["caption_alignment_blocker_introduced"])
        self.assertIn("V21_PREWRITE_UNCAPTIONED_SPOKEN_WORDS", repair_report["quality_mutation_introduced_blocker_codes"])
        rejected = [row for row in repair_report["quality_mutations"] if not row["accepted"]]
        self.assertEqual(rejected[0]["phase"], "final_visible_repair.cycle")
        self.assertEqual(rejected[0]["rule_name"], "repair_final_visible_caption_issues")

    def test_engine_accepts_caption_too_short_as_pending_visual_pacing_recheck(self) -> None:
        mutation = {
            "accepted": False,
            "rejection_reason": "blocking_short_segment_count_increased",
            "before": {
                "final_visible_fatal_count": 3,
                "blocking_short_segment_count": 0,
                "caption_alignment_blocker_codes": [],
                "visual_blocker_codes": [],
            },
            "after": {
                "final_visible_fatal_count": 0,
                "blocking_short_segment_count": 2,
                "caption_alignment_blocker_codes": ["V21_CAPTION_TOO_SHORT"],
                "visual_blocker_codes": ["V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN"],
            },
        }

        ArollEngine()._accept_pending_visual_pacing_recheck(mutation)

        self.assertTrue(mutation["accepted"])
        self.assertTrue(mutation["audit_only"])
        self.assertTrue(mutation["pending_visual_pacing_recheck"])
        self.assertEqual(mutation["audit_only_rejection_reason"], "blocking_short_segment_count_increased")
        self.assertEqual(mutation["rejection_reason"], "")

    def test_engine_rolls_back_visual_pacing_regression_after_final_visible_repair(self) -> None:
        class RegressingVisualPacing:
            def __init__(self) -> None:
                self.calls = 0
                self.initial_timeline = None

            def normalize(self, _final_timeline, _source_graph):
                self.calls += 1
                if self.initial_timeline is None:
                    self.initial_timeline = [
                        replace(
                            _segment(1, 0, 800_000),
                            source_material_id="main",
                            source_segment_id="primary_window",
                            word_ids=["w001"],
                            text="我们反对",
                        ),
                        replace(
                            _segment(2, 800_000, 1_600_000),
                            source_material_id="main",
                            source_segment_id="primary_window",
                            word_ids=["w002"],
                            text="的是公共问题",
                        ),
                    ]
                return list(self.initial_timeline), {
                    "visual_pacing_executed": True,
                    "visual_pacing_merge_attempted_count": 0,
                    "visual_pacing_merged_count": 0,
                    "visual_short_segment_count_lt_1200ms_after": 0,
                    "visual_short_segment_count_lt_1200ms_after_blocking": 0,
                    "visual_pacing_allowed_short_segment_threshold": 999,
                    "semantic_bridge_short_segment_count": 0,
                    "visual_pacing_blocker_codes": [],
                    "blocker_codes": [],
                }

        materials, text_segments = _template_rows()
        visual_pacing = RegressingVisualPacing()
        report = ArollEngine(visual_pacing=visual_pacing).run(
            ArollRunInput(
                mode="write",
                word_timeline=[
                    {"word_id": "w001", "word_text": "我们反对", "start_us": 0, "end_us": 800_000, "subtitle_index": 1, "subtitle_uid": "s001"},
                    {"word_id": "w002", "word_text": "的是公共问题", "start_us": 800_000, "end_us": 1_600_000, "subtitle_index": 2, "subtitle_uid": "s002"},
                ],
                subtitles=[
                    {"subtitle_uid": "s001", "subtitle_index": 1, "text": "我们反对", "word_ids": ["w001"]},
                    {"subtitle_uid": "s002", "subtitle_index": 2, "text": "的是公共问题", "word_ids": ["w002"]},
                ],
                source_segments=[
                    {"id": "primary_window", "material_id": "main", "type": "video", "source_start_us": 0, "source_end_us": 1_600_000}
                ],
                source_materials=[{"source_material_id": "main", "type": "video", "duration_us": 1_600_000}],
                text_materials=materials,
                text_segments=text_segments,
                postwrite_mode="simulated",
            )
        )

        repair_report = report.validator_report["final_visible_caption_repair_report"]
        visible_gate = report.validator_report["final_caption_visible_repeat_gate"]

        self.assertEqual(visual_pacing.calls, 2)
        self.assertEqual([segment.text for segment in report.final_timeline], ["我们反对的是公共问题"])
        self.assertTrue(repair_report["final_visible_repair_success"])
        self.assertEqual(
            repair_report["final_visible_repair_stop_reason"],
            "visual_pacing_regression_reverted_after_final_visible_repair",
        )
        self.assertEqual(repair_report["quality_mutation_rejected_count"], 1)
        rejected = [row for row in repair_report["quality_mutations"] if not row["accepted"]]
        self.assertEqual(rejected[0]["phase"], "visual_pacing.after_final_visible_repair")
        self.assertTrue(visible_gate["gate_passed"])

    def test_final_visible_repair_drops_caption_level_aborted_final_repeat_containment(self) -> None:
        words = [
            ("w001", "她", 0, 100_000, 1),
            ("w002", "对", 100_000, 200_000, 1),
            ("w003", "你", 200_000, 300_000, 1),
            ("w004", "有", 300_000, 400_000, 1),
            ("w005", "金融", 400_000, 500_000, 1),
            ("w006", "性", 500_000, 600_000, 1),
            ("w007", "的", 600_000, 700_000, 1),
            ("w008", "喜欢", 700_000, 800_000, 1),
            ("w009", "扫描", 900_000, 1_000_000, 2),
            ("w010", "你", 1_000_000, 1_100_000, 2),
            ("w011", "第一", 1_100_000, 1_200_000, 2),
            ("w012", "金融", 1_200_000, 1_300_000, 3),
            ("w013", "性", 1_300_000, 1_400_000, 3),
            ("w014", "喜欢", 1_400_000, 1_500_000, 3),
        ]
        source_graph = ArollEngine().ingest.build_source_graph(
            word_timeline=[
                {
                    "word_id": word_id,
                    "word_text": text,
                    "start_us": start_us,
                    "end_us": end_us,
                    "subtitle_index": subtitle_index,
                    "subtitle_uid": f"s{subtitle_index:03d}",
                }
                for word_id, text, start_us, end_us, subtitle_index in words
            ],
            subtitles=[
                {
                    "subtitle_uid": "s001",
                    "subtitle_index": 1,
                    "text": "她对你有金融性的喜欢",
                    "word_ids": [f"w{index:03d}" for index in range(1, 9)],
                },
                {
                    "subtitle_uid": "s002",
                    "subtitle_index": 2,
                    "text": "扫描你第一",
                    "word_ids": ["w009", "w010", "w011"],
                },
                {
                    "subtitle_uid": "s003",
                    "subtitle_index": 3,
                    "text": "金融性喜欢",
                    "word_ids": ["w012", "w013", "w014"],
                },
            ],
            source_segments=[{"id": "primary_window", "material_id": "main", "type": "video", "source_start_us": 0, "source_end_us": 1_600_000}],
            source_materials=[{"source_material_id": "main", "type": "video", "duration_us": 1_600_000}],
        )
        final_timeline = [
            replace(
                _segment(1, 0, 800_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=0,
                source_end_us=800_000,
                word_ids=[f"w{index:03d}" for index in range(1, 9)],
                text="她对你有金融性的喜欢",
            ),
            replace(
                _segment(2, 800_000, 1_400_000),
                source_material_id="main",
                source_segment_id="primary_window",
                source_start_us=900_000,
                source_end_us=1_500_000,
                word_ids=[f"w{index:03d}" for index in range(9, 15)],
                text="扫描你第一金融性喜欢",
            ),
        ]
        captions = [
            replace(
                _caption(1, "v21_seg_000001", 0, 800_000, text="她对你有金融性的喜欢"),
                word_ids=[f"w{index:03d}" for index in range(1, 9)],
            ),
            replace(_caption(2, "v21_seg_000002", 800_000, 1_100_000, text="扫描你第一"), word_ids=["w009", "w010", "w011"]),
            replace(_caption(3, "v21_seg_000002", 1_100_000, 1_400_000, text="金融性喜欢"), word_ids=["w012", "w013", "w014"]),
        ]

        result = repair_final_visible_caption_issues(
            final_timeline=final_timeline,
            captions=captions,
            source_graph=source_graph,
            render_captions=lambda timeline: SubtitleRenderer().render(timeline, source_graph),
        )

        self.assertTrue(result.report["final_visible_repair_success"], result.report["final_visible_repair_unresolved"])
        self.assertTrue(
            any(
                action.get("issue_type") == "final_target_repeat_caption_containment"
                and action.get("decision") == "trim_shorter_repeated_words"
                for action in result.report["final_visible_repair_actions"]
            )
        )
        self.assertEqual("".join(segment.text for segment in result.final_timeline), "她对你有金融性的喜欢扫描你第一")
        kept_word_ids = [word_id for segment in result.final_timeline for word_id in segment.word_ids]
        self.assertFalse({"w012", "w013", "w014"} & set(kept_word_ids))

    def test_engine_quality_gate_report_uses_final_timeline_after_repair(self) -> None:
        class StaleVisualPacing:
            def __init__(self) -> None:
                self.calls = 0

            def normalize(self, _final_timeline, _source_graph):
                self.calls += 1
                if self.calls > 1:
                    return list(_final_timeline), build_visual_pacing_report(
                        final_timeline=list(_final_timeline),
                        captions=[],
                        executed=True,
                        source_graph=_source_graph,
                        merge_report={
                            "visual_pacing_executed": True,
                            "visual_pacing_rerun_after_final_visible_repair": True,
                        },
                    )
                timeline = [
                    replace(
                        _segment(1, 0, 800_000),
                        source_material_id="main",
                        source_segment_id="primary_window",
                        source_start_us=0,
                        source_end_us=800_000,
                        target_start_us=0,
                        target_end_us=800_000,
                        word_ids=["w001"],
                        text="我们反对",
                    ),
                    replace(
                        _segment(2, 800_000, 1_600_000),
                        source_material_id="main",
                        source_segment_id="primary_window",
                        source_start_us=800_000,
                        source_end_us=1_600_000,
                        target_start_us=800_000,
                        target_end_us=1_600_000,
                        word_ids=["w002"],
                        text="的是公共问题",
                    ),
                ]
                return timeline, {
                    "visual_pacing_executed": True,
                    "visual_pacing_merge_attempted_count": 0,
                    "visual_pacing_merged_count": 0,
                    "visual_short_segment_count_lt_1200ms_after": 2,
                    "visual_short_segment_count_lt_1200ms_after_blocking": 2,
                    "semantic_bridge_short_segment_count": 0,
                    "semantic_bridge_safe_merge_candidates": [
                        {"segment_id": "stale_segment", "reason": "old_timeline_candidate"}
                    ],
                    "visual_pacing_blocker_codes": ["V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN"],
                    "blocker_codes": ["V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN"],
                }

        materials, text_segments = _template_rows()
        visual_pacing = StaleVisualPacing()
        report = ArollEngine(visual_pacing=visual_pacing).run(
            ArollRunInput(
                mode="write",
                word_timeline=[
                    {"word_id": "w001", "word_text": "我们反对", "start_us": 0, "end_us": 800_000, "subtitle_index": 1, "subtitle_uid": "s001"},
                    {"word_id": "w002", "word_text": "的是公共问题", "start_us": 800_000, "end_us": 1_600_000, "subtitle_index": 2, "subtitle_uid": "s002"},
                ],
                subtitles=[
                    {"subtitle_uid": "s001", "subtitle_index": 1, "text": "我们反对", "word_ids": ["w001"]},
                    {"subtitle_uid": "s002", "subtitle_index": 2, "text": "的是公共问题", "word_ids": ["w002"]},
                ],
                source_segments=[
                    {"id": "primary_window", "material_id": "main", "type": "video", "source_start_us": 0, "source_end_us": 1_600_000}
                ],
                source_materials=[{"source_material_id": "main", "type": "video", "duration_us": 1_600_000}],
                text_materials=materials,
                text_segments=text_segments,
                postwrite_mode="simulated",
            )
        )
        visual = report.validator_report["visual_pacing_gate"]
        quality = report.validator_report["quality_gate_report"]
        summary = build_run_summary(report)

        self.assertEqual(report.status, "ok")
        self.assertEqual([segment.text for segment in report.final_timeline], ["我们反对的是公共问题"])
        self.assertEqual(visual_pacing.calls, 2)
        self.assertGreater(report.validator_report["final_visible_caption_repair_report"]["final_visible_repair_action_count"], 0)
        self.assertEqual(
            report.validator_report["final_visible_caption_repair_report"]["final_visible_repair_visual_pacing_rerun_count"],
            1,
        )
        self.assertTrue(visual["gate_passed"])
        self.assertEqual(visual["final_video_segment_count"], 1)
        self.assertEqual(visual["caption_count"], 1)
        self.assertEqual(visual["visual_short_segment_count_lt_1200ms_after"], 0)
        self.assertEqual(visual["semantic_bridge_safe_merge_candidate_count"], 0)
        self.assertEqual(visual["semantic_bridge_safe_merge_candidates"], [])
        self.assertNotIn("V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN", visual["blocker_codes"])
        self.assertTrue(quality["visual_pacing_gate"]["gate_passed"])
        self.assertNotIn("V21_VISUAL_PACING_SHORT_SEGMENTS_REMAIN", quality["blocker_codes"])
        self.assertTrue(summary["visual_pacing_gate_passed"])
        self.assertEqual(summary["final_video_segment_count"], 1)

    def test_final_visible_ngram_does_not_block_common_middle_phrase(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_000_000, text="他的后台私信全是消息"),
                _caption(2, "v21_seg_000002", 1_100_000, 2_300_000, text="你找那种漂亮女生的后台私信去看看"),
            ]
        )

        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["ngram_repeat_count"], 0)

    def test_final_visible_ngram_still_blocks_true_visible_repeat(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_000_000, text="甲后台私信"),
                _caption(2, "v21_seg_000002", 1_100_000, 2_100_000, text="后台私信丙丁"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertGreaterEqual(gate["visible_repeat_candidate_count"], 1)
        self.assertIn("V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED", gate["blocker_codes"])

    def test_final_visible_ngram_latest_seen_warns_after_earlier_common_middle_phrase(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "v21_seg_000001", 0, 1_000_000, text="他的后台私信全是消息"),
                _caption(2, "v21_seg_000002", 1_100_000, 2_100_000, text="甲后台私信乙"),
                _caption(3, "v21_seg_000003", 2_200_000, 3_200_000, text="后台私信丙丁"),
            ]
        )

        self.assertTrue(gate["gate_passed"], gate)
        self.assertEqual(gate["ngram_repeat_count"], 0)
        self.assertGreaterEqual(gate["ngram_repeat_raw_count"], 1)
        candidate = gate["visible_repeat_warning_candidates"][0]
        self.assertEqual(candidate["caption_id"], "v21_cap_000002")
        self.assertEqual(candidate["related_caption_id"], "v21_cap_000003")


if __name__ == "__main__":
    unittest.main()
