from __future__ import annotations

import unittest

from aroll_v21.ir.models import (
    CanonicalSourceGraph,
    CanonicalWord,
    CaptionRenderUnit,
    FinalTimelineSegment,
    SourceGraphInvariantReport,
)
from aroll_v21.quality.final_visible_caption_repair import repair_final_visible_caption_issues
from aroll_v21.quality.final_timeline_quality_guard import build_final_timeline_quality_guard_report
from aroll_v21.quality.quality_audit import build_quality_snapshot, build_timeline_mutation
from aroll_v21.quality.quality_gate import build_quality_gate_report
from aroll_v21.render.subtitle_renderer import SubtitleRenderer


def _word(word_id: str, text: str, start_us: int, end_us: int, subtitle_index: int) -> CanonicalWord:
    return CanonicalWord(
        word_id=word_id,
        text=text,
        normalized_text=text,
        source_start_us=start_us,
        source_end_us=end_us,
        source_material_id="main",
        source_segment_id="primary",
        subtitle_uid=f"s{subtitle_index:03d}",
        subtitle_index=subtitle_index,
        char_start=None,
        char_end=None,
        confidence=0.99,
        is_cuttable_left=True,
        is_cuttable_right=True,
    )


def _graph(words: list[CanonicalWord]) -> CanonicalSourceGraph:
    return CanonicalSourceGraph(
        words=words,
        edit_units=[],
        subtitle_rows=[],
        source_materials=[],
        source_segments=[{"id": "primary", "material_id": "main", "source_start_us": 0, "source_end_us": 100_000_000}],
        text_materials=[],
        text_segments=[],
        invariant_report=SourceGraphInvariantReport(
            single_source_graph_ok=True,
            all_words_have_source_time=True,
            all_edit_units_have_word_ids=True,
            unbound_word_count=0,
            unbound_subtitle_count=0,
            blocker_count=0,
        ),
    )


def _segment(
    segment_id: str,
    word_ids: list[str],
    text: str,
    start_us: int,
    end_us: int,
    *,
    debug_hints: dict[str, object] | None = None,
) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=segment_id,
        source_material_id="main",
        source_segment_id="primary",
        source_start_us=start_us,
        source_end_us=end_us,
        target_start_us=start_us,
        target_end_us=end_us,
        word_ids=word_ids,
        text=text,
        decision_ids=[],
        lead_handle_us=0,
        tail_handle_us=0,
        debug_hints=debug_hints or {},
    )


def _caption(caption_id: str, segment_id: str, word_ids: list[str], text: str, start_us: int, end_us: int) -> CaptionRenderUnit:
    return CaptionRenderUnit(
        caption_id=caption_id,
        timeline_segment_ids=[segment_id],
        word_ids=word_ids,
        text=text,
        target_start_us=start_us,
        target_end_us=end_us,
        source_subtitle_uids=[],
        style_template_id="canonical_caption_template",
        spoken_source_start_us=start_us,
        spoken_source_end_us=end_us,
        containing_video_segment_id=segment_id,
    )


class FinalTimelineQualityGuardTest(unittest.TestCase):
    def test_reports_short_content_island_before_completed_restart(self) -> None:
        words = [
            _word("w065", "心形", 21_900_000, 22_700_000, 12),
            _word("w066", "用", 22_966_666, 23_120_000, 13),
            _word("w067", "个", 23_120_000, 23_240_000, 13),
            _word("w068", "玫瑰花", 23_240_000, 23_680_000, 13),
            _word("w069", "摆", 23_680_000, 23_840_000, 13),
            _word("w070", "个", 23_840_000, 23_960_000, 13),
            _word("w071", "土", 23_960_000, 24_120_000, 13),
            _word("w072", "到", 24_120_000, 24_280_000, 13),
            _word("w073", "爆", 24_280_000, 24_520_000, 13),
            _word("w074", "的", 24_520_000, 24_700_000, 13),
            _word("w075", "心", 24_700_000, 25_200_000, 13),
        ]
        timeline = [
            _segment(
                "v21_seg_000006",
                ["w065"],
                "心形",
                21_900_000,
                22_700_000,
                debug_hints={
                    "visual_pacing_large_intra_segment_gap_split": True,
                    "safe_handle_policy_enabled": True,
                    "safe_handle_requested_lead_us": 320_000,
                },
            ),
            _segment(
                "v21_seg_000007",
                ["w066", "w067", "w068", "w069", "w070", "w071", "w072", "w073", "w074", "w075"],
                "用个玫瑰花摆个土到爆的心",
                22_966_666,
                25_200_000,
            ),
        ]

        report = build_final_timeline_quality_guard_report(
            source_graph=_graph(words),
            final_timeline=timeline,
            captions=[_caption("cap006", "v21_seg_000006", ["w065"], "心形", 21_900_000, 22_700_000)],
        )

        candidates = [row for row in report["candidates"] if row["type"] == "short_restart_residue_island"]
        self.assertEqual(len(candidates), 1, report)
        self.assertEqual(candidates[0]["segment_id"], "v21_seg_000006")
        self.assertEqual(candidates[0]["related_segment_id"], "v21_seg_000007")
        self.assertEqual(candidates[0]["overlap_text"], "心")
        intents = report["repair_intent_report"]["repair_intents"]
        drop_intents = [row for row in intents if row["intent_type"] == "drop_restart_residue_segment"]
        self.assertEqual(len(drop_intents), 1, report)
        self.assertEqual(drop_intents[0]["segment_id"], "v21_seg_000006")
        self.assertEqual(drop_intents[0]["drop_word_ids"], ["w065"])
        self.assertTrue(drop_intents[0]["safe_cut_recompute_required"])
        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["blocking_candidate_type_counts"]["short_restart_residue_island"], 1)
        self.assertIn("V21_FINAL_TIMELINE_SHORT_RESTART_RESIDUE", report["blocker_codes"])

    def test_semantic_bridge_single_overlap_expected_removed_text_still_visible_blocks_write(self) -> None:
        words = [
            _word("w190", "四分", 62_560_000, 63_360_000, 31),
            _word("w191", "但凡", 63_666_666, 64_026_666, 32),
            _word("w192", "任何", 64_026_666, 64_360_000, 32),
            _word("w193", "一个", 64_360_000, 64_620_000, 32),
            _word("w194", "四分女", 64_620_000, 65_100_000, 32),
        ]
        timeline = [
            _segment(
                "v21_seg_000020",
                ["w190"],
                "四分",
                62_560_000,
                63_360_000,
                debug_hints={"safe_handle_requested_lead_us": 320_000},
            ),
            _segment(
                "v21_seg_000021",
                ["w191", "w192", "w193", "w194"],
                "但凡任何一个四分女",
                63_666_666,
                65_100_000,
            ),
        ]

        report = build_final_timeline_quality_guard_report(
            source_graph=_graph(words),
            final_timeline=timeline,
            captions=[],
        )

        candidates = [row for row in report["candidates"] if row["type"] == "short_restart_residue_island"]
        self.assertEqual(len(candidates), 1, report)
        self.assertEqual(candidates[0]["severity"], "warning")
        self.assertTrue(candidates[0]["is_semantic_bridge"])
        self.assertFalse(candidates[0]["is_visual_gap_split"])
        self.assertFalse(candidates[0]["semantic_restart_residue_safe_drop"])
        residuals = [row for row in report["candidates"] if row["type"] == "expected_removed_text_still_visible"]
        self.assertEqual(len(residuals), 1, report)
        self.assertEqual(residuals[0]["expected_removed_text"], "四分")
        self.assertEqual(residuals[0]["source_intent_type"], "drop_restart_residue_segment")
        self.assertIn("V21_FINAL_TIMELINE_EXPECTED_REMOVED_TEXT_STILL_VISIBLE", report["blocker_codes"])
        self.assertFalse(report["gate_passed"])

    def test_repair_pipeline_drops_multi_overlap_semantic_restart_residue(self) -> None:
        words = [
            _word("w189", "给", 62_560_000, 62_720_000, 31),
            _word("w190", "你", 62_720_000, 62_900_000, 31),
            _word("w191", "四分", 62_900_000, 63_360_000, 31),
            _word("w192", "但凡", 63_666_666, 64_026_666, 32),
            _word("w193", "任何", 64_026_666, 64_360_000, 32),
            _word("w194", "一个", 64_360_000, 64_620_000, 32),
            _word("w195", "四分女", 64_620_000, 65_100_000, 32),
            _word("w196", "但凡", 65_180_000, 65_460_000, 32),
            _word("w197", "给", 65_460_000, 65_620_000, 32),
            _word("w198", "你", 65_620_000, 65_780_000, 32),
            _word("w199", "释放", 65_780_000, 66_060_000, 32),
            _word("w200", "好感", 66_060_000, 66_360_000, 32),
        ]
        graph = _graph(words)
        timeline = [
            _segment(
                "v21_seg_000020",
                ["w189", "w190", "w191"],
                "给你四分",
                62_560_000,
                63_360_000,
                debug_hints={"safe_handle_requested_lead_us": 320_000},
            ),
            _segment(
                "v21_seg_000021",
                ["w192", "w193", "w194", "w195", "w196", "w197", "w198", "w199", "w200"],
                "但凡任何一个四分女但凡给你释放好感",
                63_666_666,
                66_360_000,
            ),
        ]
        renderer = SubtitleRenderer()

        initial_report = build_final_timeline_quality_guard_report(
            source_graph=graph,
            final_timeline=timeline,
            captions=renderer.render(timeline, graph),
        )
        candidate = [row for row in initial_report["candidates"] if row["type"] == "short_restart_residue_island"][0]
        self.assertTrue(candidate["semantic_restart_residue_safe_drop"], initial_report)
        self.assertEqual(candidate["semantic_restart_overlap_units"], ["四分", "给你"])
        intent = initial_report["repair_intent_report"]["repair_intents"][0]
        self.assertEqual(intent["safety_level"], "deterministic_candidate")

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=renderer.render(timeline, graph),
            source_graph=graph,
            render_captions=lambda rows: renderer.render(rows, graph),
            max_passes=8,
        )

        visible = "".join(caption.text for caption in result.captions)
        self.assertNotIn("给你四分", visible)
        self.assertEqual([segment.text for segment in result.final_timeline], ["但凡任何一个四分女但凡给你释放好感"])
        final_report = build_final_timeline_quality_guard_report(
            source_graph=graph,
            final_timeline=result.final_timeline,
            captions=result.captions,
        )
        self.assertTrue(final_report["gate_passed"], final_report)

    def test_complete_short_question_is_not_restart_residue(self) -> None:
        words = [
            _word("w213", "你", 70_533_333, 70_773_333, 38),
            _word("w214", "信", 70_773_333, 70_893_333, 38),
            _word("w215", "不信", 70_893_333, 71_453_333, 38),
            _word("w216", "你", 71_800_000, 72_080_000, 39),
            _word("w217", "属于", 72_080_000, 72_360_000, 39),
            _word("w218", "是", 72_360_000, 72_560_000, 39),
            _word("w219", "被迫", 72_560_000, 72_840_000, 39),
            _word("w220", "中仓", 73_000_000, 73_360_000, 39),
        ]
        timeline = [
            _segment(
                "v21_seg_000022",
                ["w213", "w214", "w215"],
                "你信不信",
                70_533_333,
                71_453_333,
                debug_hints={"visual_pacing_large_intra_segment_gap_split": True},
            ),
            _segment(
                "v21_seg_000023",
                ["w216", "w217", "w218", "w219", "w220"],
                "你属于是被迫中仓",
                71_800_000,
                73_360_000,
            ),
        ]

        report = build_final_timeline_quality_guard_report(
            source_graph=_graph(words),
            final_timeline=timeline,
            captions=[],
        )

        self.assertNotIn("short_restart_residue_island", report["candidate_type_counts"], report)
        self.assertEqual(report["repair_intent_count"], 0, report)
        self.assertTrue(report["gate_passed"], report)

    def test_enumeration_slot_is_not_restart_residue(self) -> None:
        words = [
            _word("w870", "以下", 10_000_000, 10_200_000, 1),
            _word("w871", "几个", 10_200_000, 10_500_000, 1),
            _word("w872", "基调", 10_500_000, 10_800_000, 1),
            _word("w873", "的", 10_800_000, 10_900_000, 1),
            _word("w874", "香水", 10_900_000, 11_200_000, 1),
            _word("w875", "雪松", 11_600_000, 11_900_000, 2),
            _word("w876", "檀木", 12_300_000, 12_600_000, 2),
            _word("w877", "香根草", 12_900_000, 13_300_000, 2),
            _word("w878", "那么", 13_300_000, 13_600_000, 3),
            _word("w879", "雪松", 13_600_000, 13_900_000, 3),
            _word("w880", "的", 13_900_000, 14_000_000, 3),
            _word("w881", "冷", 14_000_000, 14_300_000, 3),
        ]
        timeline = [
            _segment("v21_seg_000001", ["w870", "w871", "w872", "w873", "w874"], "以下几个基调的香水", 10_000_000, 11_200_000),
            _segment(
                "v21_seg_000002",
                ["w875", "w876"],
                "雪松檀木",
                11_600_000,
                12_600_000,
                debug_hints={"safe_handle_requested_lead_us": 320_000},
            ),
            _segment("v21_seg_000003", ["w877", "w878", "w879", "w880", "w881"], "香根草那么雪松的冷", 12_900_000, 14_300_000),
        ]

        report = build_final_timeline_quality_guard_report(
            source_graph=_graph(words),
            final_timeline=timeline,
            captions=[],
        )

        self.assertNotIn("short_restart_residue_island", report["candidate_type_counts"], report)
        self.assertTrue(report["gate_passed"], report)

    def test_particle_only_overlap_is_not_restart_residue(self) -> None:
        words = [
            _word("w886", "檀木", 20_000_000, 20_280_000, 1),
            _word("w887", "的", 20_280_000, 20_440_000, 1),
            _word("w888", "沉", 20_520_000, 20_800_000, 1),
            _word("w889", "香根草", 21_000_000, 21_480_000, 2),
            _word("w890", "的", 21_480_000, 21_620_000, 2),
            _word("w891", "泥土味", 21_620_000, 22_100_000, 2),
        ]
        timeline = [
            _segment(
                "v21_seg_000001",
                ["w886", "w887", "w888"],
                "檀木的沉",
                20_000_000,
                20_800_000,
                debug_hints={"safe_handle_requested_lead_us": 320_000},
            ),
            _segment("v21_seg_000002", ["w889", "w890", "w891"], "香根草的泥土味", 21_000_000, 22_100_000),
        ]

        report = build_final_timeline_quality_guard_report(
            source_graph=_graph(words),
            final_timeline=timeline,
            captions=[],
        )

        self.assertNotIn("short_restart_residue_island", report["candidate_type_counts"], report)
        self.assertTrue(report["gate_passed"], report)

    def test_reports_dangling_connector_caption_mismatch_and_missing_lead_handle(self) -> None:
        words = [
            _word("w137", "手", 47_186_666, 47_533_333, 24),
            _word("w138", "所以", 48_066_666, 48_266_666, 25),
            _word("w139", "你", 48_300_000, 48_460_000, 25),
            _word("w140", "手里", 48_460_000, 48_780_000, 25),
            _word("w141", "你", 48_900_000, 49_060_000, 25),
            _word("w142", "手里", 49_060_000, 49_420_000, 25),
            _word("w143", "呢", 49_420_000, 49_620_000, 25),
            _word("w144", "你", 49_786_666, 49_946_666, 25),
            _word("w145", "手里", 49_946_666, 50_366_666, 25),
            _word("w146", "呢", 50_366_666, 50_566_666, 25),
            _word("w147", "仅剩", 50_566_666, 50_926_666, 26),
            _word("w148", "的", 50_926_666, 51_046_666, 26),
            _word("w149", "两个", 51_046_666, 51_386_666, 26),
            _word("w150", "钢镚", 51_386_666, 51_726_666, 26),
        ]
        timeline = [
            _segment(
                "v21_seg_000016",
                ["w137", "w138"],
                "手所以",
                47_186_666,
                48_266_666,
                debug_hints={
                    "safe_handle_policy_enabled": True,
                    "safe_handle_requested_lead_us": 320_000,
                },
            ),
            _segment(
                "v21_seg_000017",
                ["w144", "w145", "w146", "w147", "w148", "w149", "w150"],
                "你手里呢仅剩的两个钢镚",
                49_786_666,
                51_726_666,
            ),
        ]
        captions = [
            _caption("cap020", "v21_seg_000016", ["w137", "w138"], "手里就捏着", 47_186_666, 48_266_666),
            _caption("cap021", "v21_seg_000017", ["w144", "w145", "w146"], "仅有的一枚", 49_786_666, 50_566_666),
        ]

        report = build_final_timeline_quality_guard_report(
            source_graph=_graph(words),
            final_timeline=timeline,
            captions=captions,
        )

        by_type = report["candidate_type_counts"]
        self.assertGreaterEqual(by_type["dangling_word_before_connector"], 1, report)
        self.assertGreaterEqual(by_type["caption_video_word_text_mismatch"], 2, report)
        self.assertGreaterEqual(by_type["missing_requested_lead_handle"], 1, report)
        connector = [row for row in report["candidates"] if row["type"] == "dangling_word_before_connector"][0]
        self.assertEqual(connector["dangling_word_ids"], ["w137"])
        self.assertEqual(connector["connector_text"], "所以")
        self.assertEqual(connector["following_unselected_word_ids"], ["w139", "w140", "w141", "w142", "w143"])
        intents = report["repair_intent_report"]["repair_intents"]
        intent_types = report["repair_intent_type_counts"]
        self.assertGreaterEqual(intent_types["trim_dangling_words_before_connector"], 1, report)
        self.assertGreaterEqual(intent_types["rerender_caption_from_source_words"], 2, report)
        self.assertGreaterEqual(intent_types["recompute_missing_lead_handle"], 1, report)
        trim_intent = [row for row in intents if row["intent_type"] == "trim_dangling_words_before_connector"][0]
        self.assertEqual(trim_intent["segment_id"], "v21_seg_000016")
        self.assertEqual(trim_intent["drop_word_ids"], ["w137"])
        self.assertEqual(trim_intent["keep_anchor_word_ids"], ["w138"])
        self.assertEqual(trim_intent["requested_lead_handle_us"], 320_000)
        caption_intent = [
            row
            for row in intents
            if row["intent_type"] == "rerender_caption_from_source_words" and row["caption_id"] == "cap021"
        ][0]
        self.assertEqual(caption_intent["expected_caption_text"], "你手里呢")
        self.assertIn(
            "caption changes cannot mask a physical timeline source-word mismatch",
            caption_intent["safety_checks"],
        )

    def test_source_words_are_authoritative_without_segment_text_fallback(self) -> None:
        words = [
            _word("w001", "甲", 1_000_000, 1_200_000, 1),
            _word("w002", "乙丙丁戊", 1_300_000, 2_000_000, 2),
        ]
        timeline = [
            _segment(
                "v21_seg_000001",
                ["missing"],
                "乙",
                1_000_000,
                1_200_000,
                debug_hints={"visual_pacing_large_intra_segment_gap_split": True},
            ),
            _segment("v21_seg_000002", ["w002"], "乙丙丁戊", 1_300_000, 2_000_000),
        ]

        report = build_final_timeline_quality_guard_report(
            source_graph=_graph(words),
            final_timeline=timeline,
            captions=[],
        )

        self.assertNotIn("short_restart_residue_island", report["candidate_type_counts"])
        self.assertEqual(report["repair_intent_count"], 0, report)
        self.assertTrue(report["gate_passed"])

    def test_caption_text_mismatch_blocks_without_physical_residue(self) -> None:
        words = [_word("w001", "甲", 1_000_000, 1_300_000, 1)]
        timeline = [_segment("v21_seg_000001", ["w001"], "甲", 1_000_000, 1_300_000)]
        captions = [_caption("cap001", "v21_seg_000001", ["w001"], "乙", 1_000_000, 1_300_000)]

        report = build_final_timeline_quality_guard_report(
            source_graph=_graph(words),
            final_timeline=timeline,
            captions=captions,
        )

        self.assertEqual(report["candidate_type_counts"]["caption_video_word_text_mismatch"], 1, report)
        self.assertEqual(report["blocking_candidate_count"], 1, report)
        self.assertEqual(report["caption_source_text_mismatch_blocking_count"], 1, report)
        self.assertIn("V21_FINAL_TIMELINE_CAPTION_SOURCE_TEXT_MISMATCH", report["blocker_codes"])
        self.assertFalse(report["gate_passed"], report)

    def test_quality_mutation_rejects_introduced_final_timeline_guard_blocker(self) -> None:
        words = [_word("w001", "source", 1_000_000, 2_500_000, 1)]
        graph = _graph(words)
        timeline = [_segment("v21_seg_000001", ["w001"], "source", 1_000_000, 2_500_000)]
        before_captions = [_caption("cap001", "v21_seg_000001", ["w001"], "source", 1_000_000, 2_500_000)]
        after_captions = [_caption("cap001", "v21_seg_000001", ["w001"], "target", 1_000_000, 2_500_000)]

        before = build_quality_snapshot(
            source_graph=graph,
            final_timeline=timeline,
            captions=before_captions,
        )
        after = build_quality_snapshot(
            source_graph=graph,
            final_timeline=timeline,
            captions=after_captions,
        )
        mutation = build_timeline_mutation(
            phase="regression_test",
            rule_name="caption_text_rewrite",
            before=before,
            after=after,
        ).to_report()

        self.assertFalse(mutation["accepted"], mutation)
        self.assertEqual(mutation["rejection_reason"], "final_timeline_quality_guard_blocker_introduced")
        self.assertIn("V21_FINAL_TIMELINE_CAPTION_SOURCE_TEXT_MISMATCH", mutation["introduced_blocker_codes"])
        self.assertIn("V21_FINAL_TIMELINE_CAPTION_SOURCE_TEXT_MISMATCH", mutation["after"]["final_timeline_quality_guard_blocker_codes"])

    def test_quality_gate_blocks_final_timeline_guard_failures(self) -> None:
        guard = {
            "gate_passed": False,
            "blocker_codes": ["V21_FINAL_TIMELINE_SHORT_RESTART_RESIDUE"],
            "blocking_candidate_count": 1,
        }

        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": []},
            final_caption_visible_repeat_gate={"gate_passed": True, "blocker_codes": []},
            semantic_adjudication_gate={"semantic_adjudication_gate_passed": True, "blocker_codes": []},
            visual_pacing_gate={"gate_passed": True, "visual_pacing_executed": True, "visual_merge_safety_gate_passed": True, "blocker_codes": []},
            caption_alignment_gate={"gate_passed": True, "caption_gui_track_gate_passed": True, "subtitle_readability_gate_passed": True, "blocker_codes": []},
            final_timeline_quality_guard_gate=guard,
            ready_for_user_manual_qc_preconditions_passed=True,
        )

        self.assertFalse(quality["gate_passed"])
        self.assertFalse(quality["ready_for_user_manual_qc_preconditions_passed"])
        self.assertIn("V21_FINAL_TIMELINE_QUALITY_GUARD_FAILED", quality["blocker_codes"])
        self.assertIn("V21_FINAL_TIMELINE_SHORT_RESTART_RESIDUE", quality["blocker_codes"])

    def test_repair_pipeline_applies_short_restart_residue_intent(self) -> None:
        words = [
            _word("w065", "心形", 21_900_000, 22_700_000, 12),
            _word("w066", "用", 22_966_666, 23_120_000, 13),
            _word("w067", "个", 23_120_000, 23_240_000, 13),
            _word("w068", "玫瑰花", 23_240_000, 23_680_000, 13),
            _word("w069", "摆", 23_680_000, 23_840_000, 13),
            _word("w070", "个", 23_840_000, 23_960_000, 13),
            _word("w071", "土", 23_960_000, 24_120_000, 13),
            _word("w072", "到", 24_120_000, 24_280_000, 13),
            _word("w073", "爆", 24_280_000, 24_520_000, 13),
            _word("w074", "的", 24_520_000, 24_700_000, 13),
            _word("w075", "心", 24_700_000, 25_200_000, 13),
        ]
        graph = _graph(words)
        timeline = [
            _segment(
                "v21_seg_000006",
                ["w065"],
                "心形",
                21_900_000,
                22_700_000,
                debug_hints={"visual_pacing_large_intra_segment_gap_split": True},
            ),
            _segment(
                "v21_seg_000007",
                ["w066", "w067", "w068", "w069", "w070", "w071", "w072", "w073", "w074", "w075"],
                "用个玫瑰花摆个土到爆的心",
                22_966_666,
                25_200_000,
            ),
        ]
        renderer = SubtitleRenderer()

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=renderer.render(timeline, graph),
            source_graph=graph,
            render_captions=lambda rows: renderer.render(rows, graph),
            max_passes=8,
        )

        self.assertEqual(len(result.final_timeline), 1)
        self.assertEqual(result.final_timeline[0].word_ids, ["w066", "w067", "w068", "w069", "w070", "w071", "w072", "w073", "w074", "w075"])
        self.assertNotIn("心形", "".join(caption.text for caption in result.captions))
        actions = result.report["final_timeline_repair_intent_actions"]
        self.assertEqual(actions[0]["intent_type"], "drop_restart_residue_segment")
        self.assertEqual(actions[0]["decision"], "drop_restart_residue_segment")

    def test_repair_pipeline_trims_connector_residue_and_recomputes_handle(self) -> None:
        words = [
            _word("w137", "手", 47_186_666, 47_533_333, 24),
            _word("w138", "所以", 48_066_666, 48_266_666, 25),
            _word("w139", "你", 48_300_000, 48_460_000, 25),
            _word("w140", "手里", 48_460_000, 48_780_000, 25),
            _word("w141", "你", 48_900_000, 49_060_000, 25),
            _word("w142", "手里", 49_060_000, 49_420_000, 25),
            _word("w143", "呢", 49_420_000, 49_620_000, 25),
            _word("w144", "你", 49_786_666, 49_946_666, 25),
            _word("w145", "手里", 49_946_666, 50_366_666, 25),
            _word("w146", "呢", 50_366_666, 50_566_666, 25),
            _word("w147", "仅剩", 50_566_666, 50_926_666, 26),
            _word("w148", "的", 50_926_666, 51_046_666, 26),
            _word("w149", "两个", 51_046_666, 51_386_666, 26),
            _word("w150", "钢镚", 51_386_666, 51_726_666, 26),
        ]
        graph = _graph(words)
        timeline = [
            _segment(
                "v21_seg_000016",
                ["w137", "w138"],
                "手所以",
                47_186_666,
                48_266_666,
                debug_hints={
                    "safe_handle_policy_enabled": True,
                    "safe_handle_requested_lead_us": 320_000,
                },
            ),
            _segment(
                "v21_seg_000017",
                ["w144", "w145", "w146", "w147", "w148", "w149", "w150"],
                "你手里呢仅剩的两个钢镚",
                49_786_666,
                51_726_666,
            ),
        ]
        captions = [
            _caption("cap020", "v21_seg_000016", ["w137", "w138"], "手里就捏着", 47_186_666, 48_266_666),
            _caption("cap021", "v21_seg_000017", ["w144", "w145", "w146"], "仅有的一枚", 49_786_666, 50_566_666),
        ]
        renderer = SubtitleRenderer()

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=captions,
            source_graph=graph,
            render_captions=lambda rows: renderer.render(rows, graph),
            max_passes=8,
        )

        first = result.final_timeline[0]
        self.assertEqual(first.word_ids, ["w138"])
        self.assertEqual(first.text, "所以")
        self.assertGreater(first.lead_handle_us, 0)
        self.assertLess(first.clip_source_start_us, first.source_start_us)
        visible_text = "".join(caption.text for caption in result.captions)
        self.assertNotIn("手里就捏着", visible_text)
        self.assertNotIn("仅有的一枚", visible_text)
        self.assertIn("所以", visible_text)
        self.assertIn("你手里呢", visible_text)
        guard = build_final_timeline_quality_guard_report(
            source_graph=graph,
            final_timeline=result.final_timeline,
            captions=result.captions,
        )
        self.assertNotIn("V21_FINAL_TIMELINE_SAFE_CUT_HANDLE_MISSING", guard["blocker_codes"])
        actions = result.report["final_timeline_repair_intent_actions"]
        self.assertEqual(actions[0]["intent_type"], "trim_dangling_words_before_connector")
        self.assertEqual(actions[0]["decision"], "trim_dangling_words_before_connector")

    def test_safe_handle_recompute_keeps_partial_available_lead(self) -> None:
        words = [
            _word("w001", "上一句", 1_000_000, 1_900_000, 1),
            _word("w002", "所以", 2_100_000, 2_300_000, 2),
            _word("w003", "继续", 2_300_000, 2_700_000, 2),
        ]
        graph = _graph(words)
        timeline = [
            _segment(
                "v21_seg_000001",
                ["w001"],
                "上一句",
                1_000_000,
                1_900_000,
                debug_hints={
                    "safe_handle_policy_enabled": True,
                    "safe_handle_requested_lead_us": 320_000,
                    "safe_handle_requested_tail_us": 220_000,
                },
            ),
            _segment(
                "v21_seg_000002",
                ["w002", "w003"],
                "所以继续",
                2_100_000,
                2_700_000,
                debug_hints={
                    "safe_handle_policy_enabled": True,
                    "safe_handle_requested_lead_us": 320_000,
                    "safe_handle_requested_tail_us": 220_000,
                },
            ),
        ]
        renderer = SubtitleRenderer()

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=renderer.render(timeline, graph),
            source_graph=graph,
            render_captions=lambda rows: renderer.render(rows, graph),
            max_passes=4,
        )

        second = result.final_timeline[1]
        self.assertEqual(second.lead_handle_us, 200_000)
        self.assertEqual(second.clip_source_start_us, 1_900_000)
        guard = build_final_timeline_quality_guard_report(
            source_graph=graph,
            final_timeline=result.final_timeline,
            captions=result.captions,
        )
        self.assertTrue(guard["gate_passed"], guard)

    def test_intra_sentence_pause_gap_blocks_and_emits_safe_split_intent(self) -> None:
        words = [
            _word("w001", "前半句", 0, 700_000, 1),
            _word("w002", "后半句", 1_080_000, 1_780_000, 1),
        ]
        graph = _graph(words)
        timeline = [_segment("v21_seg_000001", ["w001", "w002"], "前半句后半句", 0, 1_780_000)]

        report = build_final_timeline_quality_guard_report(
            source_graph=graph,
            final_timeline=timeline,
            captions=[_caption("cap001", "v21_seg_000001", ["w001", "w002"], "前半句后半句", 0, 1_780_000)],
        )

        candidates = [row for row in report["candidates"] if row["type"] == "intra_sentence_pause_gap_exceeds_limit"]
        self.assertEqual(len(candidates), 1, report)
        self.assertEqual(candidates[0]["gap_us"], 380_000)
        self.assertTrue(candidates[0]["safe_split_available"])
        self.assertFalse(report["gate_passed"])
        self.assertIn("V21_FINAL_TIMELINE_INTRA_SENTENCE_PAUSE_GAP_EXCEEDS_LIMIT", report["blocker_codes"])
        intents = report["repair_intent_report"]["repair_intents"]
        split_intents = [row for row in intents if row["intent_type"] == "split_intra_sentence_pause_gap"]
        self.assertEqual(len(split_intents), 1, report)
        self.assertEqual(split_intents[0]["safety_level"], "deterministic_candidate")
        self.assertTrue(split_intents[0]["safe_cut_recompute_required"])
        self.assertEqual(split_intents[0]["left_word_ids"], ["w001"])
        self.assertEqual(split_intents[0]["right_word_ids"], ["w002"])

    def test_repair_pipeline_splits_intra_sentence_pause_gap_and_recomputes_safe_cut(self) -> None:
        words = [
            _word("w001", "前半句", 0, 700_000, 1),
            _word("w002", "后半句", 1_080_000, 1_780_000, 1),
        ]
        graph = _graph(words)
        timeline = [_segment("v21_seg_000001", ["w001", "w002"], "前半句后半句", 0, 1_780_000)]
        renderer = SubtitleRenderer()

        result = repair_final_visible_caption_issues(
            final_timeline=timeline,
            captions=renderer.render(timeline, graph),
            source_graph=graph,
            render_captions=lambda rows: renderer.render(rows, graph),
            max_passes=8,
        )

        self.assertEqual([segment.text for segment in result.final_timeline], ["前半句", "后半句"])
        self.assertEqual([segment.target_start_us for segment in result.final_timeline], [0, 700_000])
        self.assertTrue(
            all(segment.debug_hints.get("safe_handle_recomputed_by_final_timeline_intent") for segment in result.final_timeline)
        )
        actions = result.report["final_timeline_repair_intent_actions"]
        self.assertTrue(any(action["intent_type"] == "split_intra_sentence_pause_gap" for action in actions), actions)
        guard = build_final_timeline_quality_guard_report(
            source_graph=graph,
            final_timeline=result.final_timeline,
            captions=result.captions,
        )
        self.assertTrue(guard["gate_passed"], guard)


if __name__ == "__main__":
    unittest.main()
