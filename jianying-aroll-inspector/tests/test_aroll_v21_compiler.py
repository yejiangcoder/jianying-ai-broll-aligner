from __future__ import annotations

import json
import unittest
from pathlib import Path

from aroll_v21 import ArollEngine, ArollRunInput
from aroll_v21.decision import DeepSeekSemanticPlanner
from aroll_v21.ingest import DraftIngest
from aroll_v21.ir import dataclass_to_dict
from aroll_v21.render import SubtitleRenderer
from aroll_v21.writer import CaptionMaterialWriter


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str):
    return json.loads((ROOT / path).read_text("utf-8"))


def material_fixture(name: str) -> tuple[dict, dict]:
    payload = load_json(f"fixtures/real_materials/{name}.json")
    return payload["material"], payload["segment"]


def base_material_rows() -> tuple[list[dict], list[dict]]:
    normal_material, normal_segment = material_fixture("normal_caption_template")
    giant_material, giant_segment = material_fixture("giant_title_material")
    callout_material, callout_segment = material_fixture("callout_text_material")
    return [normal_material, giant_material, callout_material], [normal_segment, giant_segment, callout_segment]


def repeated_run_input() -> ArollRunInput:
    text_materials, text_segments = base_material_rows()
    return ArollRunInput(
        source_segments=[
            {"id": "clip_a", "material_id": "main_video_a", "source_start_us": 0, "source_end_us": 2_000_000}
        ],
        word_timeline=[
            {"word_id": "w001", "word_text": "你跪在地上", "start_us": 100_000, "end_us": 500_000, "subtitle_index": 1, "subtitle_uid": "s001"},
            {"word_id": "w002", "word_text": "你跪在地上", "start_us": 520_000, "end_us": 900_000, "subtitle_index": 2, "subtitle_uid": "s002"},
            {"word_id": "w003", "word_text": "叫大佬", "start_us": 920_000, "end_us": 1_200_000, "subtitle_index": 3, "subtitle_uid": "s003"},
        ],
        subtitles=[
            {"subtitle_uid": "s001", "subtitle_index": 1, "text": "你跪在地上", "word_ids": ["w001"]},
            {"subtitle_uid": "s002", "subtitle_index": 2, "text": "你跪在地上", "word_ids": ["w002"]},
            {"subtitle_uid": "s003", "subtitle_index": 3, "text": "叫大佬", "word_ids": ["w003"]},
        ],
        text_materials=text_materials,
        text_segments=text_segments,
    )


class PhysicalFieldPlanner:
    def decide(self, clusters):
        return [
            {
                "cluster_id": clusters[0].cluster_id,
                "keep_unit_id": clusters[0].variants[0].unit_id,
                "drop_unit_ids": [],
                "reason": "invalid physical control attempt",
                "confidence": 0.9,
                "requires_human_review": False,
                "source_start_us": 123,
            }
        ]


class ArollV21CompilerTests(unittest.TestCase):
    def test_engine_compiles_from_word_truth_and_clears_adjacent_exact_repeat(self) -> None:
        report = ArollEngine().run(repeated_run_input())

        self.assertEqual(report.status, "ok", dataclass_to_dict(report.blocker_report))
        self.assertTrue(report.source_graph.invariant_report.single_source_graph_ok)
        self.assertTrue(all(segment.word_ids for segment in report.final_timeline))
        self.assertTrue(all(caption.timeline_segment_ids for caption in report.captions))
        final_text = "".join(caption.text for caption in report.captions)
        self.assertEqual(final_text, "你跪在地上叫大佬")
        self.assertNotIn("你跪在地上你跪在地上", final_text)
        self.assertTrue(report.material_write_plan["no_writer_fallback"])
        self.assertEqual(report.material_write_plan["canonical_caption_template_id"], "caption_template_001")
        self.assertTrue(report.validator_report["validators_read_only"])
        self.assertTrue(report.validator_report["safe_cut_validator"]["safe_cut_boundary_gate_passed"])

    def test_caption_writer_preserves_content_json_styles_and_updates_text(self) -> None:
        run_input = repeated_run_input()
        graph = DraftIngest().build_source_graph(
            word_timeline=run_input.word_timeline,
            subtitles=run_input.subtitles,
            source_segments=run_input.source_segments,
            text_materials=run_input.text_materials,
            text_segments=run_input.text_segments,
        )
        captions = SubtitleRenderer().render([], graph)
        captions = [
            captions[0] if captions else None
        ]
        # Build a direct caption to avoid depending on compiler behavior in this writer unit test.
        from aroll_v21.ir import CaptionRenderUnit

        caption = CaptionRenderUnit(
            caption_id="cap",
            timeline_segment_ids=["seg"],
            word_ids=["w002"],
            text="更新后的字幕",
            target_start_us=0,
            target_end_us=1_000_000,
            source_subtitle_uids=["s002"],
            style_template_id="canonical_caption_template",
        )
        plan, blockers = CaptionMaterialWriter().build_write_plan(graph, [caption])
        self.assertFalse(blockers)
        material = plan["materials"][0]
        content = json.loads(material["content"])
        base_content = json.loads(material["base_content"])
        self.assertEqual(content["text"], "更新后的字幕")
        self.assertEqual(base_content["text"], "更新后的字幕")
        self.assertTrue(content["styles"])
        self.assertEqual(content["styles"][0]["range"]["end"], len("更新后的字幕"))

    def test_deepseek_decision_cannot_contain_physical_fields(self) -> None:
        text_materials, text_segments = base_material_rows()
        run_input = ArollRunInput(
            source_segments=[{"id": "clip_a", "material_id": "main_video_a", "source_start_us": 0, "source_end_us": 2_000_000}],
            word_timeline=[
                {"word_id": "w001", "word_text": "随意的肆意的踩踏", "start_us": 100_000, "end_us": 900_000, "subtitle_index": 1, "subtitle_uid": "s001"}
            ],
            subtitles=[{"subtitle_uid": "s001", "subtitle_index": 1, "text": "随意的肆意的踩踏", "word_ids": ["w001"]}],
            text_materials=text_materials,
            text_segments=text_segments,
        )
        report = ArollEngine(deepseek_planner=PhysicalFieldPlanner()).run(run_input)

        self.assertEqual(report.status, "blocked")
        codes = [blocker.code for blocker in report.blocker_report.blockers]
        self.assertIn("DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS", codes)

    def test_compiler_refuses_to_drop_unsafe_edit_unit(self) -> None:
        run_input = repeated_run_input()
        subtitles = [
            dict(run_input.subtitles[0], cut_policy="unsafe"),
            *run_input.subtitles[1:],
        ]
        report = ArollEngine().run(
            ArollRunInput(
                source_segments=run_input.source_segments,
                word_timeline=run_input.word_timeline,
                subtitles=subtitles,
                text_materials=run_input.text_materials,
                text_segments=run_input.text_segments,
            )
        )
        self.assertEqual(report.status, "blocked")
        self.assertIn("UNSAFE_EDIT_UNIT_DROP_BLOCKED", [blocker.code for blocker in report.blocker_report.blockers])

    def test_no_hardcoded_uat_terms_in_v21_src(self) -> None:
        terms = [
            "嘉豪",
            "数字游民",
            "螃蟹效应",
            "敢张",
            "敢张口",
            "最后只",
            "最后只能",
            "你跪在地上",
            "你们是在",
            "你们是极度恐慌",
            "我就我发现",
            "能不能",
            "一寸一寸",
        ]
        src_text = "\n".join(path.read_text("utf-8") for path in (ROOT / "src" / "aroll_v21").rglob("*.py"))
        for term in terms:
            self.assertNotIn(term, src_text)


if __name__ == "__main__":
    unittest.main()
