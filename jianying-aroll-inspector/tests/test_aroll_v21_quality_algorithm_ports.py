from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.engine import build_run_summary
from aroll_v21.ir.models import FinalTimelineSegment
from aroll_v21.quality.repeat_span_repair import contained_repeat_drop_side, dropped_span_report, longest_suffix_prefix_overlap
from aroll_v21.quality.safe_boundary import source_range_aligned_to_word_boundaries, trailing_word_ids_for_suffix_overlap
from aroll_v21.quality.subtitle_readability import (
    HARD_MAX_CHARS,
    HARD_MAX_DURATION_US,
    MIN_DURATION_US,
    TARGET_MAX_CHARS,
    subtitle_interval_report,
)
from aroll_v21.quality.tiny_segment_classifier import (
    MIN_VIDEO_SEGMENT_US,
    PREFERRED_MIN_VIDEO_SEGMENT_US,
    WEAK_TINY_TEXT,
    classify_tiny_segment,
)
from tests.test_aroll_v21_sacrificial_write_override import fake_real_draft_result


ROOT = Path(__file__).resolve().parents[1]


class ArollV21QualityAlgorithmPortsTests(unittest.TestCase):
    def test_word_timeline_count_reports_accepted_native_words(self) -> None:
        result = fake_real_draft_result()
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

        summary = build_run_summary(report)

        self.assertEqual(summary["word_timeline_count"], len(result.word_timeline))
        self.assertEqual(summary["word_timeline_count_source"], "source_graph")

    def test_v20_display_subtitle_thresholds_ported_to_v21(self) -> None:
        self.assertEqual(TARGET_MAX_CHARS, 18)
        self.assertEqual(HARD_MAX_CHARS, 20)
        self.assertEqual(MIN_DURATION_US, 500_000)
        self.assertEqual(HARD_MAX_DURATION_US, 3_500_000)

        renderer_source = (ROOT / "src" / "aroll_v21" / "render" / "subtitle_renderer.py").read_text("utf-8")
        self.assertIn("split_words_for_display", renderer_source)
        self.assertIn("merge_tiny_display_fragments", renderer_source)
        self.assertNotIn("aroll_display_subtitle_planner", renderer_source)
        self.assertNotIn("phase4e", renderer_source)
        self.assertNotIn("downstream_repair", renderer_source)

    def test_v20_subtitle_interval_guard_overlap_and_duration_ported(self) -> None:
        captions = [
            SimpleNamespace(caption_id="c1", text="正常字幕", target_start_us=0, target_end_us=400_000),
            SimpleNamespace(caption_id="c2", text="超长字幕" * 8, target_start_us=390_000, target_end_us=4_100_000),
        ]

        report = subtitle_interval_report(captions)

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["subtitle_interval_overlap_count"], 1)
        self.assertEqual(report["subtitle_interval_too_short_count"], 1)
        self.assertEqual(report["subtitle_interval_too_long_count"], 1)
        self.assertEqual(report["subtitle_hard_max_char_count"], 1)

    def test_v20_tiny_segment_classifier_ported_without_legacy_ids(self) -> None:
        self.assertEqual(MIN_VIDEO_SEGMENT_US, 500_000)
        self.assertEqual(PREFERRED_MIN_VIDEO_SEGMENT_US, 700_000)
        self.assertIn("然后", WEAK_TINY_TEXT)
        weak = classify_tiny_segment(_final_segment("seg1", "然后", 0, 300_000, ["w1"]))
        bridge = classify_tiny_segment(_final_segment("seg2", "关键转折", 0, 600_000, ["w2"]))

        self.assertTrue(weak.hard_tiny_artifact)
        self.assertEqual(weak.merge_candidate_reason, "hard_tiny_artifact")
        self.assertTrue(bridge.semantic_bridge)
        self.assertEqual(bridge.merge_candidate_reason, "semantic_bridge_exception")

        source = "\n".join(
            [
                (ROOT / "src" / "aroll_v21" / "quality" / "tiny_segment_classifier.py").read_text("utf-8"),
                (ROOT / "src" / "aroll_v21" / "quality" / "visual_pacing.py").read_text("utf-8"),
            ]
        )
        for forbidden in ("6月15日", "6月16日", "嘉豪", "随意", "肆意", "这说明", "safe_cut_boundary_resolver"):
            self.assertNotIn(forbidden, source)

    def test_v20_safe_boundary_whole_word_guard_ported(self) -> None:
        words = {
            "w1": SimpleNamespace(word_id="w1", text="完整", source_start_us=100_000, source_end_us=300_000),
            "w2": SimpleNamespace(word_id="w2", text="边界", source_start_us=300_000, source_end_us=600_000),
        }
        aligned = _final_segment("seg1", "完整边界", 100_000, 600_000, ["w1", "w2"])
        cut_inside = _final_segment("seg2", "完整边界", 150_000, 600_000, ["w1", "w2"])

        self.assertTrue(source_range_aligned_to_word_boundaries(segment=aligned, word_lookup=words))
        self.assertFalse(source_range_aligned_to_word_boundaries(segment=cut_inside, word_lookup=words))
        self.assertEqual(
            trailing_word_ids_for_suffix_overlap(segment=aligned, word_lookup=words, overlap="边界"),
            ["w2"],
        )

    def test_v20_repeat_span_repair_ported_without_downstream_repair(self) -> None:
        self.assertEqual(contained_repeat_drop_side("短句", "短句扩展"), "drop_left")
        self.assertEqual(longest_suffix_prefix_overlap(["甲", "乙", "丙"], ["乙", "丙", "丁"]), 2)
        report = dropped_span_report(
            [
                {
                    "route": "final_target_repeat",
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "drop_recommended",
                    "dropped_segment_indices": [4],
                    "applied": True,
                },
                {
                    "route": "final_target_repeat",
                    "cluster_id": "final_target_repeat_tc_0001",
                    "decision": "drop_recommended",
                    "dropped_segment_indices": [4],
                    "applied": True,
                },
            ]
        )

        self.assertEqual(report["dropped_cluster_count"], 1)
        self.assertEqual(report["dropped_segment_count"], 1)
        self.assertEqual(report["clusters_per_dropped_segment"], {"4": ["final_target_repeat_tc_0001"]})
        source = (ROOT / "src" / "aroll_v21" / "quality" / "repeat_span_repair.py").read_text("utf-8")
        self.assertNotIn("downstream_repair", source)
        self.assertNotIn("repair_applier", source)

    def test_v20_quality_migration_manifest_has_real_targets(self) -> None:
        manifest = (ROOT / "docs" / "v21_v20_quality_algorithm_migration.md").read_text("utf-8")
        for target in (
            "src/aroll_v21/quality/subtitle_readability.py",
            "src/aroll_v21/quality/tiny_segment_classifier.py",
            "src/aroll_v21/quality/safe_boundary.py",
            "src/aroll_v21/quality/repeat_span_repair.py",
        ):
            self.assertIn(target, manifest)
        self.assertIn("Migration status: migrated", manifest)
        self.assertIn("Migration status: audited_not_migrated", manifest)
        self.assertNotIn("嘉豪", manifest)
        self.assertNotIn("phase4e_full_aroll", manifest)


def _final_segment(segment_id: str, text: str, start_us: int, end_us: int, word_ids: list[str]) -> FinalTimelineSegment:
    return FinalTimelineSegment(
        segment_id=segment_id,
        source_material_id="",
        source_segment_id=None,
        source_start_us=start_us,
        source_end_us=end_us,
        target_start_us=start_us,
        target_end_us=end_us,
        word_ids=word_ids,
        text=text,
        decision_ids=[],
    )


if __name__ == "__main__":
    unittest.main()
