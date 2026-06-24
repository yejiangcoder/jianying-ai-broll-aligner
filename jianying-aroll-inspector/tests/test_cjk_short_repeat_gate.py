from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aroll_cjk_short_repeat_gate import detect_cjk_short_repeats
from aroll_final_repeat_gate import build_final_repeat_gate_report
from aroll_hidden_audio_repeat_gate import build_hidden_audio_repeat_report


class CjkShortRepeatGateTest(unittest.TestCase):
    def test_detects_adjacent_clause_prefix_overlap(self) -> None:
        candidates = detect_cjk_short_repeats(
            [{"fragment_id": "f1", "fragment_text": "就像螃蟹效应一样，就看，就看到有人想爬出粪坑"}]
        )

        self.assertTrue(
            any(
                row["type"] == "boundary_prefix_containment"
                and row["issue_type"] == "cjk_adjacent_clause_overlap"
                and row["overlap"] == "就看"
                and row["severity"] == "fatal"
                for row in candidates
            )
        )

    def test_detects_pronoun_connector_restart(self) -> None:
        candidates = detect_cjk_short_repeats(
            [{"fragment_id": "f1", "fragment_text": "我就我发现有这么一群精分的数字游民"}]
        )

        self.assertTrue(
            any(
                row["type"] == "restart_disfluency"
                and row["issue_type"] == "cjk_pronoun_connector_restart"
                and row["overlap"] == "我就我"
                and row["severity"] == "fatal"
                for row in candidates
            )
        )

    def test_normal_pronoun_connector_sentence_passes(self) -> None:
        candidates = detect_cjk_short_repeats(
            [{"fragment_id": "f1", "fragment_text": "我就发现有这么一群精分的数字游民"}]
        )

        self.assertEqual(candidates, [])

    def test_normal_single_clause_sentence_passes(self) -> None:
        candidates = detect_cjk_short_repeats(
            [{"fragment_id": "f1", "fragment_text": "就看到有人想爬出粪坑"}]
        )

        self.assertEqual(candidates, [])

    def test_detects_adjacent_subtitle_boundary_overlap(self) -> None:
        candidates = detect_cjk_short_repeats(
            [
                {"fragment_id": "f1", "fragment_text": "就看"},
                {"fragment_id": "f2", "fragment_text": "就看到有人想爬出粪坑"},
            ]
        )

        self.assertTrue(
            any(
                row["type"] == "boundary_prefix_containment"
                and row["issue_type"] == "cjk_adjacent_subtitle_boundary_overlap"
                and row["overlap"] == "就看"
                and row["severity"] == "fatal"
                for row in candidates
            )
        )

    def test_detects_restart_disfluency_across_adjacent_subtitle_boundary(self) -> None:
        candidates = detect_cjk_short_repeats(
            [
                {"fragment_id": "f1", "fragment_text": "我就"},
                {"fragment_id": "f2", "fragment_text": "我发现有这么一群人"},
            ]
        )

        self.assertTrue(
            any(
                row["type"] == "restart_disfluency"
                and row["scope"] == "subtitle_boundary"
                and row["overlap"] == "我就我"
                and row["severity"] == "fatal"
                for row in candidates
            )
        )

    def test_normal_adjacent_subtitles_pass(self) -> None:
        candidates = detect_cjk_short_repeats(
            [
                {"fragment_id": "f1", "fragment_text": "这就是螃蟹效应"},
                {"fragment_id": "f2", "fragment_text": "有人想爬出粪坑"},
            ]
        )

        self.assertEqual(candidates, [])

    def test_label_reuse_before_definition_passes_within_caption(self) -> None:
        candidates = detect_cjk_short_repeats(
            [{"fragment_id": "f1", "fragment_text": "这种行为都叫表白，表白等于释放信号"}]
        )

        self.assertFalse(
            any(
                row["type"] == "boundary_prefix_containment"
                and row["overlap"] == "表白"
                for row in candidates
            ),
            candidates,
        )

    def test_label_reuse_before_definition_passes_across_subtitle_boundary(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [
                {"fragment_id": "f1", "fragment_text": "这种行为都叫表白"},
                {"fragment_id": "f2", "fragment_text": "表白等于释放信号"},
            ],
        )

        self.assertTrue(report["final_repeat_gate_passed"], report)
        self.assertEqual(report["final_cjk_short_repeat_fatal_count"], 0)
        self.assertFalse(
            any(
                row["type"] == "boundary_prefix_containment"
                and row["overlap"] == "表白"
                for row in report["blocking_issues"]
            ),
            report,
        )

    def test_detects_exact_short_repeat(self) -> None:
        candidates = detect_cjk_short_repeats(
            [{"fragment_id": "f1", "fragment_text": "然后然后"}]
        )

        self.assertTrue(
            any(
                row["type"] == "intra_subtitle_ngram_repeat"
                and row["overlap"] == "然后"
                and row["severity"] == "fatal"
                for row in candidates
            )
        )

    def test_a_not_a_question_structure_is_not_blocking(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "就国南能不能不要规训自己人呐"}],
        )

        self.assertTrue(report["final_repeat_gate_passed"])
        self.assertEqual(report["final_cjk_short_repeat_fatal_count"], 0)
        self.assertEqual(report["blocking_issues"], [])

    def test_numeral_headed_reduplication_is_not_blocking(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "一寸一寸往前挪"}],
        )

        self.assertTrue(report["final_repeat_gate_passed"])
        self.assertEqual(report["final_cjk_short_repeat_fatal_count"], 0)
        self.assertEqual(report["blocking_issues"], [])
        self.assertGreaterEqual(report["final_cjk_short_repeat_warning_count"], 1)

    def test_amount_modifier_reduplication_is_not_blocking(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "你甚至会为了省下区区几百块"}],
        )

        self.assertTrue(report["final_repeat_gate_passed"])
        self.assertEqual(report["final_cjk_short_repeat_fatal_count"], 0)
        self.assertEqual(report["blocking_issues"], [])

    def test_numeral_headed_boundary_reduplication_is_not_blocking(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [
                {"fragment_id": "f1", "fragment_text": "我发现一寸"},
                {"fragment_id": "f2", "fragment_text": "一寸地往前挪"},
            ],
        )

        self.assertTrue(report["final_repeat_gate_passed"])
        self.assertEqual(report["final_cjk_short_repeat_fatal_count"], 0)
        self.assertEqual(report["blocking_issues"], [])
        self.assertGreaterEqual(report["final_cjk_short_repeat_warning_count"], 1)

    def test_high_confidence_short_repeats_remain_blocking(self) -> None:
        for text, overlap in (("然后然后", "然后"), ("就是就是", "就是"), ("我我", "我")):
            with self.subTest(text=text):
                report = build_final_repeat_gate_report(
                    {"issues": []},
                    [{"fragment_id": "f1", "fragment_text": text}],
                )

                self.assertFalse(report["final_repeat_gate_passed"])
                self.assertEqual(report["final_cjk_short_repeat_fatal_count"], 1)
                self.assertTrue(any(row["overlap"] == overlap for row in report["blocking_issues"]))

    def test_prefix_containment_examples_remain_blocking(self) -> None:
        for left, right, overlap in (
            ("最后只", "最后只能像一个小丑一样被迫接盘38万彩礼", "最后只"),
            ("敢张", "敢张口管你要38万彩礼的底气", "敢张"),
        ):
            with self.subTest(left=left):
                report = build_final_repeat_gate_report(
                    {"issues": []},
                    [
                        {"fragment_id": "f1", "fragment_text": left},
                        {"fragment_id": "f2", "fragment_text": right},
                    ],
                )

                self.assertFalse(report["final_repeat_gate_passed"])
                self.assertTrue(
                    any(
                        row["type"] == "boundary_prefix_containment"
                        and row["overlap"] == overlap
                        and row["severity"] == "fatal"
                        for row in report["blocking_issues"]
                    )
                )

    def test_adjacent_modifier_semantic_redundancy_blocks_for_self_review(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "随意的肆意的踩踏"}],
        )

        self.assertFalse(report["final_repeat_gate_passed"])
        self.assertEqual(report["adjacent_modifier_semantic_redundancy_fatal_count"], 1)
        issue = report["blocking_issues"][0]
        self.assertEqual(issue["type"], "adjacent_modifier_semantic_redundancy")
        self.assertTrue(issue["requires_self_review"])
        self.assertIn("block_write", issue["review_options"])

    def test_adjacent_modifier_semantic_redundancy_with_punctuation_blocks(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "随意的、肆意的踩踏"}],
        )

        self.assertFalse(report["final_repeat_gate_passed"])
        self.assertEqual(report["adjacent_modifier_semantic_redundancy_fatal_count"], 1)
        self.assertEqual(report["blocking_issues"][0]["type"], "adjacent_modifier_semantic_redundancy")

    def test_adjacent_modifier_semantic_redundancy_across_subtitle_boundary_is_warning(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [
                {"fragment_id": "f1", "fragment_text": "随意的"},
                {"fragment_id": "f2", "fragment_text": "肆意的踩踏"},
            ],
        )

        self.assertTrue(report["final_repeat_gate_passed"])
        self.assertEqual(report["adjacent_modifier_semantic_redundancy_fatal_count"], 0)
        self.assertEqual(report["blocking_issues"], [])
        issue = report["adjacent_modifier_semantic_redundancy_candidates"][0]
        self.assertEqual(issue["scope"], "subtitle_boundary")
        self.assertEqual(issue["severity"], "warning")
        self.assertEqual(issue["prev_text"], "随意的")
        self.assertEqual(issue["next_text"], "肆意的踩踏")

    def test_adjacent_modifier_semantic_redundancy_also_blocks_hidden_gate(self) -> None:
        report = build_hidden_audio_repeat_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "随意的肆意的踩踏", "word_ids": []}],
            [],
        )

        self.assertFalse(report["hidden_audio_repeat_gate_passed"])
        self.assertEqual(report["adjacent_modifier_semantic_redundancy_fatal_count"], 1)
        self.assertTrue(report["blocking_issues"])
        self.assertEqual(report["blocking_issues"][0]["type"], "adjacent_modifier_semantic_redundancy")

    def test_single_modifier_phrase_does_not_trigger_modifier_redundancy_gate(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "随意地踩踏"}],
        )

        self.assertTrue(report["final_repeat_gate_passed"])
        self.assertEqual(report["adjacent_modifier_semantic_redundancy_count"], 0)

    def test_long_normal_de_phrases_do_not_trigger_modifier_redundancy_gate(self) -> None:
        for text in (
            "是你杀死的年少时候的自己就",
            "抢回属于你自己的资源你的关注度你",
            "所有人的视线里的交配权",
        ):
            with self.subTest(text=text):
                report = build_final_repeat_gate_report(
                    {"issues": []},
                    [{"fragment_id": "f1", "fragment_text": text}],
                )
                self.assertTrue(report["final_repeat_gate_passed"], report)
                self.assertEqual(report["adjacent_modifier_semantic_redundancy_fatal_count"], 0)

    def test_normal_then_sentence_passes(self) -> None:
        candidates = detect_cjk_short_repeats(
            [{"fragment_id": "f1", "fragment_text": "然后他开始解释这个问题"}]
        )

        self.assertEqual(candidates, [])

    def test_candidate_report_contains_standard_fields(self) -> None:
        candidates = detect_cjk_short_repeats(
            [
                {"fragment_id": "f1", "fragment_text": "就看"},
                {"fragment_id": "f2", "fragment_text": "就看到有人想爬出粪坑"},
            ]
        )
        candidate = next(row for row in candidates if row["severity"] == "fatal")

        for key in ("type", "severity", "prev_text", "next_text", "text", "overlap", "span", "reason"):
            self.assertIn(key, candidate)
        self.assertEqual(candidate["type"], "boundary_prefix_containment")
        self.assertEqual(candidate["prev_text"], "就看")
        self.assertEqual(candidate["next_text"], "就看到有人想爬出粪坑")

    def test_single_char_boundary_overlap_is_warning_not_blocking(self) -> None:
        candidates = detect_cjk_short_repeats(
            [
                {"fragment_id": "f1", "fragment_text": "他"},
                {"fragment_id": "f2", "fragment_text": "他说这个问题"},
            ]
        )
        report = build_final_repeat_gate_report(
            {"issues": []},
            [
                {"fragment_id": "f1", "fragment_text": "他"},
                {"fragment_id": "f2", "fragment_text": "他说这个问题"},
            ],
        )

        self.assertTrue(any(row["severity"] == "warning" and row["overlap"] == "他" for row in candidates))
        self.assertTrue(report["final_repeat_gate_passed"])
        self.assertEqual(report["final_cjk_short_repeat_fatal_count"], 0)

    def test_hidden_gate_passes_when_only_nonfatal_or_empty_findings_exist(self) -> None:
        clean_report = build_hidden_audio_repeat_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "正常的一句话", "word_ids": []}],
            [],
        )
        warning_report = build_hidden_audio_repeat_report(
            {"issues": []},
            [
                {"fragment_id": "f1", "fragment_text": "他", "word_ids": []},
                {"fragment_id": "f2", "fragment_text": "他说这个问题", "word_ids": []},
            ],
            [],
        )

        self.assertTrue(clean_report["hidden_audio_repeat_gate_passed"])
        self.assertEqual(clean_report["word_timeline_repeated_island_count"], 0)
        self.assertEqual(clean_report["issues"], [])
        self.assertEqual(clean_report["blocking_issues"], [])
        self.assertTrue(warning_report["hidden_audio_repeat_gate_passed"])
        self.assertEqual(warning_report["final_spoken_text_short_repeat_fatal_count"], 0)
        self.assertEqual(warning_report["blocking_issues"], [])

    def test_hidden_gate_failure_has_blocking_issue(self) -> None:
        report = build_hidden_audio_repeat_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "我就我发现有这么一群人", "word_ids": []}],
            [],
        )

        self.assertFalse(report["hidden_audio_repeat_gate_passed"])
        self.assertGreater(len(report["blocking_issues"]), 0)
        self.assertGreater(len(report["issues"]), 0)
        self.assertTrue(any(row["type"] == "restart_disfluency" for row in report["blocking_issues"]))

    def test_final_repeat_gate_blocks_final_subtitle_short_overlap(self) -> None:
        report = build_final_repeat_gate_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "我就我发现有这么一群精分的数字游民"}],
        )

        self.assertFalse(report["final_repeat_gate_passed"])
        self.assertEqual(report["final_cjk_short_repeat_count"], 1)
        self.assertEqual(report["final_cjk_short_repeat_fatal_count"], 1)
        self.assertGreater(len(report["blocking_issues"]), 0)

    def test_hidden_gate_checks_final_spoken_text_when_word_islands_are_clean(self) -> None:
        report = build_hidden_audio_repeat_report(
            {"issues": []},
            [{"fragment_id": "f1", "fragment_text": "就像螃蟹效应一样，就看，就看到有人想爬出粪坑", "word_ids": []}],
            [],
        )

        self.assertFalse(report["hidden_audio_repeat_gate_passed"])
        self.assertGreater(report["final_spoken_text_short_repeat_count"], 0)

    def test_production_core_has_no_known_qc_phrase_patch(self) -> None:
        removed_legacy_files = (
            "aroll_downstream_repair_pipeline.py",
            "aroll_final_target_repeat_repair.py",
            "aroll_repair_applier.py",
            "aroll_hidden_repeat_repair.py",
        )
        for name in removed_legacy_files:
            self.assertFalse((SRC / name).exists(), name)

        existing_core_files = (
            SRC / "aroll_adjacent_modifier_semantic_redundancy_gate.py",
            SRC / "aroll_cjk_short_repeat_gate.py",
            SRC / "aroll_hidden_audio_repeat_gate.py",
            SRC / "aroll_final_repeat_gate.py",
            *(SRC / "aroll_v21").rglob("*.py"),
        )
        source = "\n".join(
            path.read_text("utf-8")
            for path in existing_core_files
        )
        for forbidden in (
            "嘉豪",
            "螃蟹效应",
            "数字游民",
            "能不能",
            "一寸一寸",
            "最后只",
            "最后只能",
            "敢张",
            "敢张口",
            "你跪在地上",
            "你们是在",
            "你们是极度恐慌",
            "我就我发现",
            "随意的肆意的踩踏",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
