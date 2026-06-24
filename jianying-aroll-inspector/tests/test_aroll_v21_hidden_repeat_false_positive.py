from __future__ import annotations

import unittest

from aroll_v21 import ArollEngine, ArollRunInput
from tests.test_aroll_v21_cjk_a_not_a_not_hidden_repeat import _run_text
from tests.test_aroll_v21_cjk_a_not_a_not_hidden_repeat import _material_rows


def _run_word_tokens(tokens: list[str]) -> object:
    text_materials, text_segments = _material_rows()
    rows = []
    cursor = 0
    for index, token in enumerate(tokens, start=1):
        end = cursor + max(80_000, len(token) * 80_000)
        rows.append(
            {
                "word_id": f"w{index}",
                "word_text": token,
                "start_us": cursor,
                "end_us": end,
                "subtitle_uid": "s1",
                "subtitle_index": 1,
            }
        )
        cursor = end
    return ArollEngine().run(
        ArollRunInput(
            source_segments=[{"id": "clip_1", "material_id": "main_video", "source_start_us": 0, "source_end_us": cursor + 200_000}],
            word_timeline=rows,
            subtitles=[{"subtitle_uid": "s1", "subtitle_index": 1, "text": "".join(tokens), "word_ids": [row["word_id"] for row in rows]}],
            text_materials=text_materials,
            text_segments=text_segments,
        )
    )


class ArollV21HiddenRepeatFalsePositiveTests(unittest.TestCase):
    def test_plain_negative_duplicate_still_detected(self) -> None:
        report = _run_text("不要不要继续")

        self.assertTrue(report.repeat_clusters)
        self.assertIn("hidden_audio_repeat", {cluster.repeat_type for cluster in report.repeat_clusters})

    def test_a_not_a_false_positive_does_not_create_split_human_review(self) -> None:
        report = _run_text("就国南能不能不要规训自己人呐")

        self.assertNotIn("UNIT_SPLIT_REQUIRES_HUMAN_REVIEW", [blocker.code for blocker in report.blocker_report.blockers])
        self.assertNotIn("hidden_audio_repeat", {cluster.repeat_type for cluster in report.repeat_clusters})

    def test_quantity_reduplication_modifier_does_not_create_hidden_repeat(self) -> None:
        report = _run_text("尊严一寸一寸地给老子抢回来")

        self.assertNotIn("UNIT_SPLIT_REQUIRES_HUMAN_REVIEW", [blocker.code for blocker in report.blocker_report.blockers])
        self.assertNotIn("hidden_audio_repeat", {cluster.repeat_type for cluster in report.repeat_clusters})

    def test_amount_reduplication_modifier_does_not_drop_first_word_token(self) -> None:
        report = _run_word_tokens(["你", "甚至", "会", "为了", "省下", "区", "区", "几", "毛钱"])

        self.assertEqual(report.status, "ok", [blocker.code for blocker in report.blocker_report.blockers])
        self.assertEqual("".join(caption.text for caption in report.captions), "你甚至会为了省下区区几毛钱")
        self.assertNotIn("hidden_audio_repeat", {cluster.repeat_type for cluster in report.repeat_clusters})


if __name__ == "__main__":
    unittest.main()
