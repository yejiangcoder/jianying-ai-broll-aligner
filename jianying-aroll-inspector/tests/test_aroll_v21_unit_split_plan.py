from __future__ import annotations

import json
import unittest
from pathlib import Path

from aroll_text_normalize import normalize_text
from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.compiler import FinalTimelineCompiler
from aroll_v21.decision import SemanticDecisionPlanner
from aroll_v21.evidence import CandidateEvidenceBuilder
from aroll_v21.ingest import DraftIngest
from aroll_v21.ir import CandidateEvidence, DecisionPlan, EditUnit, RepeatCluster, UnitSplitPlan


ROOT = Path(__file__).resolve().parents[1]


def _material_rows() -> tuple[list[dict], list[dict]]:
    payload = json.loads((ROOT / "fixtures" / "real_materials" / "normal_caption_template.json").read_text("utf-8"))
    return [payload["material"]], [payload["segment"]]


def _split_run_input(*, cut_policy: str = "word_boundary") -> ArollRunInput:
    text_materials, text_segments = _material_rows()
    return ArollRunInput(
        source_segments=[{"id": "clip_1", "material_id": "main_video", "source_start_us": 0, "source_end_us": 1200000}],
        word_timeline=[
            {"word_id": "w1", "word_text": "然后", "start_us": 0, "end_us": 200000, "subtitle_uid": "s1", "subtitle_index": 1},
            {"word_id": "w2", "word_text": "然后", "start_us": 200000, "end_us": 400000, "subtitle_uid": "s1", "subtitle_index": 1},
            {"word_id": "w3", "word_text": "他开始解释", "start_us": 400000, "end_us": 1000000, "subtitle_uid": "s1", "subtitle_index": 1},
        ],
        subtitles=[
            {
                "subtitle_uid": "s1",
                "subtitle_index": 1,
                "text": "然后然后他开始解释",
                "word_ids": ["w1", "w2", "w3"],
                "cut_policy": cut_policy,
            }
        ],
        text_materials=text_materials,
        text_segments=text_segments,
    )


def _manual_unit_split_cluster(
    cluster_id: str,
    words: list[str],
    drop_text: str,
    *,
    include_word_tokens: bool = True,
) -> RepeatCluster:
    word_ids = [f"w{index}" for index in range(1, len(words) + 1)]
    text = "".join(words)
    unit = EditUnit(
        unit_id="s1",
        word_ids=word_ids,
        text=text,
        normalized_text=normalize_text(text),
        source_start_us=0,
        source_end_us=len(words) * 100_000,
        subtitle_uids=["s1"],
        source_material_ids=["main_video"],
        kind="repeat",
        cut_policy="word_boundary",
    )
    metadata = {
        "spans": [{"phrase": drop_text}],
        "candidate": {"phrase": drop_text},
    }
    if include_word_tokens:
        metadata["word_tokens"] = [
            {"word_id": word_id, "text": word}
            for word_id, word in zip(word_ids, words)
        ]
    evidence = CandidateEvidence(
        evidence_id=f"evidence_{cluster_id}",
        evidence_type="hidden_audio_repeat",
        unit_ids=[unit.unit_id],
        word_ids=word_ids,
        text=text,
        normalized_text=normalize_text(text),
        reason="manual unit split binding test",
        confidence=0.9,
        requires_semantic_decision=False,
        metadata=metadata,
    )
    return RepeatCluster(
        cluster_id=cluster_id,
        variants=[unit],
        repeat_type="hidden_audio_repeat",
        evidence=[evidence],
        local_recommendation="requires_unit_split",
    )


def _plan_for_words(words: list[str], *, text: str | None = None):
    text_materials, text_segments = _material_rows()
    word_timeline = []
    cursor = 0
    for index, word in enumerate(words, start=1):
        word_timeline.append(
            {
                "word_id": f"w{index}",
                "word_text": word,
                "start_us": cursor,
                "end_us": cursor + 100_000,
                "subtitle_uid": "s1",
                "subtitle_index": 1,
            }
        )
        cursor += 100_000
    source_text = text if text is not None else "".join(words)
    graph = DraftIngest().build_source_graph(
        word_timeline=word_timeline,
        subtitles=[
            {
                "subtitle_uid": "s1",
                "subtitle_index": 1,
                "text": source_text,
                "word_ids": [row["word_id"] for row in word_timeline],
            }
        ],
        source_segments=[{"id": "clip_1", "material_id": "main_video", "source_start_us": 0, "source_end_us": cursor}],
        text_materials=text_materials,
        text_segments=text_segments,
    )
    clusters = CandidateEvidenceBuilder().build(graph)
    return SemanticDecisionPlanner().plan(clusters), clusters


class ArollV21UnitSplitPlanTests(unittest.TestCase):
    def test_intra_edit_unit_repeat_generates_split_decision(self) -> None:
        run_input = _split_run_input()
        graph = DraftIngest().build_source_graph(
            word_timeline=run_input.word_timeline,
            subtitles=run_input.subtitles,
            source_segments=run_input.source_segments,
            text_materials=run_input.text_materials,
            text_segments=run_input.text_segments,
        )
        clusters = CandidateEvidenceBuilder().build(graph)
        plan = SemanticDecisionPlanner().plan(clusters)

        self.assertFalse(plan.blocked, [blocker.code for blocker in plan.blockers])
        self.assertTrue(plan.split_decisions)
        self.assertEqual(plan.split_decisions[0].unit_id, "s1")
        self.assertTrue(set(plan.split_decisions[0].drop_word_ids) < {"w1", "w2", "w3"})
        self.assertNotIn("REPEAT_CLUSTER_REQUIRES_UNIT_SPLIT", [blocker.code for blocker in plan.blockers])

    def test_compiler_removes_drop_word_ids_from_split_decision(self) -> None:
        run_input = _split_run_input()
        graph = DraftIngest().build_source_graph(
            word_timeline=run_input.word_timeline,
            subtitles=run_input.subtitles,
            source_segments=run_input.source_segments,
            text_materials=run_input.text_materials,
            text_segments=run_input.text_segments,
        )
        plan = DecisionPlan(
            decisions=[],
            split_decisions=[
                UnitSplitPlan(
                    split_id="split_test",
                    cluster_id="repeat_test",
                    unit_id="s1",
                    drop_word_ids=["w1"],
                    keep_word_ids=["w2", "w3"],
                    reason="drop first duplicate phrase",
                )
            ],
        )

        timeline, blockers = FinalTimelineCompiler().compile(graph, plan)

        self.assertEqual(blockers, [])
        self.assertEqual("".join(segment.text for segment in timeline), "然后他开始解释")
        self.assertNotIn("w1", [word_id for segment in timeline for word_id in segment.word_ids])

    def test_engine_uses_split_decision_and_final_gate_passes(self) -> None:
        report = ArollEngine().run(_split_run_input())

        self.assertEqual(report.status, "ok", [blocker.code for blocker in report.blocker_report.blockers])
        self.assertEqual("".join(caption.text for caption in report.captions), "然后他开始解释")
        self.assertTrue(report.decision_plan.split_decisions)

    def test_unsafe_edit_unit_split_blocks_with_specific_reason(self) -> None:
        report = ArollEngine().run(_split_run_input(cut_policy="unsafe"))

        self.assertEqual(report.status, "blocked")
        codes = [blocker.code for blocker in report.blocker_report.blockers]
        self.assertIn("UNIT_SPLIT_UNSAFE_BOUNDARY", codes)
        self.assertNotIn("REPEAT_CLUSTER_REQUIRES_UNIT_SPLIT", codes)

    def test_requires_unit_split_reuses_existing_split_decision_same_unit_drop_text(self) -> None:
        first = _manual_unit_split_cluster("repeat_001000", ["笑人", "笑人家", "拼什么"], "笑人")
        second = _manual_unit_split_cluster("repeat_003000", ["笑人", "笑人家", "拼什么"], "笑人", include_word_tokens=False)

        plan = SemanticDecisionPlanner().plan([first, second])

        self.assertEqual([blocker.code for blocker in plan.blockers], [])
        self.assertEqual(len(plan.split_decisions), 2)
        self.assertEqual(plan.split_decisions[0].drop_word_ids, ["w1"])
        self.assertEqual(plan.split_decisions[1].drop_word_ids, ["w1"])
        self.assertEqual(plan.split_decisions[1].metadata["binding_source"], "reuse_existing_split_decision")
        self.assertEqual(plan.split_decisions[1].metadata["reused_split_id"], plan.split_decisions[0].split_id)

    def test_requires_unit_split_binds_exact_repeated_ngram(self) -> None:
        cluster = _manual_unit_split_cluster("repeat_003000", ["然后", "然后", "继续说明"], "然后")

        plan = SemanticDecisionPlanner().plan([cluster])

        self.assertEqual([blocker.code for blocker in plan.blockers], [])
        self.assertEqual(plan.split_decisions[0].drop_word_ids, ["w1"])
        self.assertEqual(plan.split_decisions[0].keep_word_ids, ["w2", "w3"])
        self.assertEqual(plan.split_decisions[0].metadata["binding_source"], "exact_repeated_ngram")

    def test_requires_unit_split_binds_short_phrase_before_longer_prefix_word(self) -> None:
        cluster = _manual_unit_split_cluster("repeat_003000", ["笑人", "笑人家", "拼什么"], "笑人")

        plan = SemanticDecisionPlanner().plan([cluster])

        self.assertEqual([blocker.code for blocker in plan.blockers], [])
        self.assertEqual(plan.split_decisions[0].drop_word_ids, ["w1"])
        self.assertEqual(plan.split_decisions[0].keep_word_ids, ["w2", "w3"])
        self.assertEqual(plan.split_decisions[0].metadata["binding_source"], "short_phrase_before_longer_prefix_word")

    def test_requires_unit_split_luan_hua_luan_huaqian_whole_word_safe(self) -> None:
        plan, _clusters = _plan_for_words(
            ["你在是", "你在Steam上", "乱", "花", "乱", "花钱"],
            text="你在是你在Steam上乱花乱花钱",
        )

        self.assertEqual([blocker.code for blocker in plan.blockers], [])
        self.assertTrue(plan.split_decisions)
        self.assertEqual(plan.split_decisions[0].drop_word_ids, ["w3", "w4"])
        self.assertEqual(plan.split_decisions[0].keep_word_ids, ["w1", "w2", "w5", "w6"])
        self.assertNotIn("w6", plan.split_decisions[0].drop_word_ids)

    def test_requires_unit_split_xiaoren_xiaorenjia_reuses_or_binds_safely(self) -> None:
        plan, _clusters = _plan_for_words(["呃", "笑人", "笑人家", "拼什么"])

        self.assertEqual([blocker.code for blocker in plan.blockers], [])
        self.assertTrue(plan.split_decisions)
        self.assertEqual(plan.split_decisions[0].drop_word_ids, ["w2"])
        self.assertEqual(plan.split_decisions[0].keep_word_ids, ["w1", "w3", "w4"])

    def test_requires_unit_split_jiuzheji_jiuzhejige_reuses_or_binds_safely(self) -> None:
        plan, _clusters = _plan_for_words(["就这几", "就这几个", "廉价的词"])

        self.assertEqual([blocker.code for blocker in plan.blockers], [])
        self.assertTrue(plan.split_decisions)
        self.assertEqual(plan.split_decisions[0].drop_word_ids, ["w1"])
        self.assertEqual(plan.split_decisions[0].keep_word_ids, ["w2", "w3"])

    def test_requires_unit_split_fail_closed_when_no_whole_word_safe_binding(self) -> None:
        cluster = _manual_unit_split_cluster("repeat_003000", ["然后然后", "继续说明"], "然后")

        plan = SemanticDecisionPlanner().plan([cluster])

        self.assertEqual(plan.split_decisions, [])
        self.assertIn("UNIT_SPLIT_REQUIRES_HUMAN_REVIEW", [blocker.code for blocker in plan.blockers])
        self.assertEqual(plan.blockers[0].context["failed_reason"], "word_token_binding_no_safe_whole_word_binding")
        payload = plan.semantic_request_payloads[0]
        self.assertEqual(payload["split_summary"]["binding"], "missing")
        self.assertEqual(payload["split_summary"]["failed_reason"], "word_token_binding_no_safe_whole_word_binding")

    def test_no_hardcoded_jimei_unit_split_sample_phrases_in_src(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = "\n".join(path.read_text("utf-8") for path in (root / "src").rglob("*.py"))

        for token in ("Steam", "乱花", "笑人", "就这几", "集美", "6月19日"):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
