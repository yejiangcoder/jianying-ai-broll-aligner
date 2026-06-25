from __future__ import annotations

import unittest

from aroll_adjacent_modifier_semantic_redundancy_gate import detect_adjacent_modifier_semantic_redundancy
from aroll_v21.decision import SemanticDecisionPlanner
from aroll_v21.evidence import CandidateEvidenceBuilder
from aroll_v21.ingest import DraftIngest


def _graph_for_text(text: str):
    word_rows = []
    word_ids = []
    cursor = 0
    for index, char in enumerate(text, start=1):
        word_id = f"w_{index:06d}"
        word_ids.append(word_id)
        word_rows.append(
            {
                "word_id": word_id,
                "word_text": char,
                "source_start_us": cursor,
                "source_end_us": cursor + 80_000,
                "source_material_id": "main",
                "source_segment_id": "clip",
                "subtitle_uid": "s_001",
                "subtitle_index": 1,
            }
        )
        cursor += 80_000
    return DraftIngest().build_source_graph(
        word_timeline=word_rows,
        subtitles=[{"subtitle_uid": "s_001", "subtitle_index": 1, "text": text, "word_ids": word_ids}],
        source_segments=[{"id": "clip", "material_id": "main", "source_start_us": 0, "source_end_us": cursor + 1_000_000}],
    )


def _semantic_plan_for_text(text: str):
    clusters = CandidateEvidenceBuilder().build(_graph_for_text(text))
    return SemanticDecisionPlanner().plan(clusters)


class ArollV21ModifierRedundancyScopeTests(unittest.TestCase):
    def test_same_head_adjacent_modifiers_trigger_modifier_redundancy(self) -> None:
        candidates = detect_adjacent_modifier_semantic_redundancy([{"fragment_id": "f1", "fragment_text": "快乐的开心的孩子"}])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["type"], "adjacent_modifier_semantic_redundancy")
        self.assertEqual(candidates[0]["scope"], "intra_subtitle")
        self.assertEqual(candidates[0]["left_modifier"], "快乐")
        self.assertEqual(candidates[0]["right_modifier"], "开心")
        self.assertEqual(candidates[0]["head_text"], "孩子")

    def test_relational_determiner_phrase_does_not_trigger_modifier_redundancy(self) -> None:
        candidates = detect_adjacent_modifier_semantic_redundancy([{"fragment_id": "f1", "fragment_text": "把你本该有的骨骼的张力直接"}])

        self.assertEqual(candidates, [])

    def test_verb_object_fragment_does_not_trigger_modifier_redundancy(self) -> None:
        candidates = detect_adjacent_modifier_semantic_redundancy([{"fragment_id": "f1", "fragment_text": "能完美咬合你的真实的肩宽"}])

        self.assertEqual(candidates, [])

    def test_existential_property_phrase_does_not_trigger_modifier_redundancy(self) -> None:
        candidates = detect_adjacent_modifier_semantic_redundancy([{"fragment_id": "f1", "fragment_text": "必须服务你本来就有的自律的身材"}])

        self.assertEqual(candidates, [])

    def test_action_intent_phrase_does_not_trigger_modifier_redundancy(self) -> None:
        candidates = detect_adjacent_modifier_semantic_redundancy([{"fragment_id": "f1", "fragment_text": "你要做的唯一的动作就是"}])

        self.assertEqual(candidates, [])

    def test_noun_phrase_plus_emphasis_adverb_plus_predicate_does_not_trigger(self) -> None:
        candidates = detect_adjacent_modifier_semantic_redundancy([{"fragment_id": "f1", "fragment_text": "勇敢的人真的能成功"}])

        self.assertEqual(candidates, [])

    def test_round_fixture_same_head_modifier_generates_split_without_semantic_payload(self) -> None:
        plan = _semantic_plan_for_text("随意的肆意的踩踏")

        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(len(plan.split_decisions), 1)
        self.assertEqual(plan.split_decisions[0].cluster_id, "repeat_002000")

    def test_round_fixture_noun_scope_emphasis_does_not_enter_semantic_payload(self) -> None:
        plan = _semantic_plan_for_text("自信的人真的能拿到结果")

        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.split_decisions, [])

    def test_round_fixture_existential_property_does_not_generate_split(self) -> None:
        plan = _semantic_plan_for_text("本来就有的自律的身材")

        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.split_decisions, [])

    def test_round_fixture_action_intent_does_not_generate_split(self) -> None:
        plan = _semantic_plan_for_text("你要做的唯一的动作就是")

        self.assertEqual(plan.semantic_request_payloads, [])
        self.assertEqual(plan.split_decisions, [])


if __name__ == "__main__":
    unittest.main()
