from __future__ import annotations

import json
import unittest
from pathlib import Path

from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.ir import CaptionRenderUnit
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate


ROOT = Path(__file__).resolve().parents[1]


def _material_rows() -> tuple[list[dict], list[dict]]:
    payload = json.loads((ROOT / "fixtures" / "real_materials" / "normal_caption_template.json").read_text("utf-8"))
    return [payload["material"]], [payload["segment"]]


def _run_input(parts: list[str]) -> ArollRunInput:
    text_materials, text_segments = _material_rows()
    words = []
    subtitles = []
    cursor = 0
    for subtitle_index, text in enumerate(parts, start=1):
        word_ids = []
        for char in text:
            word_id = f"w_{len(words) + 1:06d}"
            word_ids.append(word_id)
            words.append(
                {
                    "word_id": word_id,
                    "word_text": char,
                    "source_start_us": cursor,
                    "source_end_us": cursor + 90_000,
                    "source_material_id": "main_video",
                    "source_segment_id": "clip_1",
                    "subtitle_uid": f"s{subtitle_index}",
                    "subtitle_index": subtitle_index,
                }
            )
            cursor += 90_000
        subtitles.append(
            {
                "subtitle_uid": f"s{subtitle_index}",
                "subtitle_index": subtitle_index,
                "text": text,
                "word_ids": word_ids,
            }
        )
        cursor += 120_000
    return ArollRunInput(
        source_segments=[
            {
                "id": "clip_1",
                "material_id": "main_video",
                "source_start_us": 0,
                "source_end_us": cursor + 500_000,
            }
        ],
        word_timeline=words,
        subtitles=subtitles,
        text_materials=text_materials,
        text_segments=text_segments,
        postwrite_mode="simulated",
    )


def _caption(index: int, text: str) -> CaptionRenderUnit:
    return CaptionRenderUnit(
        caption_id=f"cap_{index:06d}",
        timeline_segment_ids=[f"seg_{index:06d}"],
        word_ids=[f"w_{index:06d}"],
        text=text,
        target_start_us=(index - 1) * 500_000,
        target_end_us=index * 500_000,
        source_subtitle_uids=[f"s{index}"],
        style_template_id="canonical_caption_template",
    )


class ArollV21GuiQualityResidualTests(unittest.TestCase):
    def test_modifier_redundancy_repairs_redundant_modifier_without_hardcoding_sample(self) -> None:
        report = ArollEngine().run(_run_input(["快乐的开心的孩子"]))

        self.assertEqual([segment.text for segment in report.final_timeline], ["开心的孩子"])
        self.assertEqual([caption.text for caption in report.captions], ["开心的孩子"])
        self.assertFalse(report.decision_plan.semantic_request_payloads)
        self.assertTrue(report.decision_plan.split_decisions)
        self.assertEqual(report.validator_report["final_caption_visible_repeat_gate"]["modifier_redundancy_residual_count"], 0)

    def test_self_repair_aborted_phrase_drops_incomplete_restart(self) -> None:
        left = "但是这个市场的方"
        right = "但是这个的方向上"
        report = ArollEngine().run(_run_input([left, right]))

        self.assertEqual([segment.text for segment in report.final_timeline], [right])
        self.assertEqual([caption.text for caption in report.captions], [right])
        trace = [row for row in report.decision_trace if row.get("route") == "self_repair_aborted_phrase"]
        self.assertTrue(any(row.get("decision") == "drop_left_keep_right" for row in trace))

    def test_self_repair_keeps_completed_rephrase(self) -> None:
        completed = "但是这个的方向上"
        report = ArollEngine().run(_run_input(["但是这个市场的方", completed]))

        self.assertEqual("".join(caption.text for caption in report.captions), completed)

    def test_parallel_enumeration_is_not_treated_as_aborted_restart(self) -> None:
        report = ArollEngine().run(
            _run_input(
                [
                    "那么第二生理性的喜欢",
                    "他要观察你的结构",
                    "观察你的状态",
                    "观察你的表达",
                    "观察你的言行举止",
                    "看看第一眼能不能激发兴趣",
                ]
            )
        )

        final_text = "".join(segment.text for segment in report.final_timeline)
        caption_text = "".join(caption.text for caption in report.captions)
        for expected in ("他要观察你的结构", "观察你的状态", "观察你的表达", "观察你的言行举止"):
            self.assertIn(expected, final_text)
            self.assertIn(expected, caption_text)
        scan_requests = [
            row
            for row in report.decision_plan.semantic_request_payloads
            if "观察你的" in str(row.get("left_text") or row.get("right_text") or row.get("candidate_text") or "")
        ]
        self.assertEqual(scan_requests, [])

    def test_self_repair_requires_semantic_adjudication_when_ambiguous(self) -> None:
        gate = build_final_caption_visible_repeat_gate(
            [
                _caption(1, "我们讨论项目方向"),
                _caption(2, "我们讨论项目结果"),
            ]
        )

        self.assertFalse(gate["gate_passed"])
        self.assertIn("V21_SELF_REPAIR_ABORTED_PHRASE_UNRESOLVED", gate["blocker_codes"])
        candidate = gate["self_repair_aborted_phrase_candidates"][0]
        self.assertTrue(candidate["requires_semantic_adjudication"])
        self.assertFalse(candidate["deterministic_drop_left"])

    def test_no_hardcoded_finance_angle_phrase(self) -> None:
        source = "\n".join(path.read_text("utf-8") for path in (ROOT / "src" / "aroll_v21").rglob("*.py"))

        for forbidden in ("金融市场的角", "金融的视角下", "随意的肆意的踩踏"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
