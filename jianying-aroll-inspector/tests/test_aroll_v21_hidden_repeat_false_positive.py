from __future__ import annotations

import unittest

from tests.test_aroll_v21_cjk_a_not_a_not_hidden_repeat import _run_text


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


if __name__ == "__main__":
    unittest.main()
