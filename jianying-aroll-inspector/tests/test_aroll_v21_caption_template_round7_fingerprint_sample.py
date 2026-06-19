from __future__ import annotations

import copy
import json
import unittest

from aroll_v21.ingest import DraftIngest
from aroll_v21.ir import CaptionRenderUnit
from aroll_v21.writer import CaptionMaterialWriter


FONT_PATH = "D:/JianyingPro/10.7.0.14095/Resources/Font/SystemFont/zh-hans.ttf"


def _base_content(text: str) -> dict:
    return {
        "styles": [
            {
                "fill": {"alpha": 1.0, "content": {"render_type": "solid", "solid": {"alpha": 1.0, "color": [1.0, 1.0, 1.0]}}},
                "font": {"id": "", "path": FONT_PATH},
                "range": [0, len(text)],
                "size": 5.0,
            }
        ],
        "text": text,
    }


def _round7_material(material_id: str, text: str, *, keyword: bool) -> dict:
    content = copy.deepcopy(_base_content(text))
    if keyword and len(text) >= 6:
        content["styles"] = [
            {**copy.deepcopy(_base_content(text)["styles"][0]), "range": [0, 2]},
            {
                "fill": {"content": {"render_type": "solid", "solid": {"color": [1, 0.87058824300766, 0]}}},
                "font": {"id": "", "path": FONT_PATH},
                "range": [2, len(text) - 1],
                "size": 6.5,
                "strokes": [{"content": {"render_type": "solid", "solid": {"color": [0, 0, 0]}}, "width": 0.06, "mode": 1}],
                "useLetterColor": True,
            },
            {**copy.deepcopy(_base_content(text)["styles"][0]), "range": [len(text) - 1, len(text)]},
        ]
    return {
        "id": material_id,
        "type": "subtitle",
        "font_size": 5.0,
        "font_path": FONT_PATH,
        "text_color": "#FFFFFF",
        "initial_scale": 1.0,
        "content": json.dumps(content, ensure_ascii=False),
        "base_content": json.dumps(_base_content(text), ensure_ascii=False),
        "subtitle_keywords": {"range": [{"location": 2, "length": max(1, len(text) - 3), "source_type": "server"}]} if keyword else None,
        "words": {"start_time": [0], "end_time": [200], "text": [text[:1]]},
        "current_words": {},
        "recognize_text": text,
        "extra_material_refs": [f"volatile_ref_{material_id}"],
    }


def _round7_segment(material_id: str, index: int) -> dict:
    return {
        "id": f"seg_{index:03d}",
        "material_id": material_id,
        "track_id": "round7_text_track",
        "track_type": "text",
        "target_timerange": {"start": index * 200_000, "duration": 180_000},
        "clip": {"scale": {"x": 1.0, "y": 1.0}, "transform": {"x": 0.0, "y": -0.73}, "flip": {}},
        "render_index": 14_000 + index,
        "track_render_index": 1,
        "source": "segmentsourcenormal",
    }


class ArollV21CaptionTemplateRound7FingerprintSampleTests(unittest.TestCase):
    def test_round7_110_safe_candidates_collapse_to_one_stable_group(self) -> None:
        materials: list[dict] = []
        segments: list[dict] = []
        subtitles: list[dict] = []
        words: list[dict] = []
        captions: list[CaptionRenderUnit] = []
        for index in range(1, 111):
            material_id = f"round7_text_{index:03d}"
            subtitle_uid = f"round7_subtitle_{index:03d}"
            word_id = f"round7_word_{index:03d}"
            text = "普通字幕" * (index % 5 + 1)
            materials.append(_round7_material(material_id, text, keyword=(index % 2 == 0)))
            segments.append(_round7_segment(material_id, index))
            subtitles.append({"subtitle_uid": subtitle_uid, "subtitle_index": index, "text": text, "word_ids": [word_id], "text_material_id": material_id})
            words.append({"word_id": word_id, "word_text": text, "start_us": index * 200_000, "end_us": index * 200_000 + 100_000, "subtitle_uid": subtitle_uid, "subtitle_index": index})
            captions.append(
                CaptionRenderUnit(
                    caption_id=f"cap_{index:03d}",
                    timeline_segment_ids=[f"timeline_{index:03d}"],
                    word_ids=[word_id],
                    text=text,
                    target_start_us=index * 200_000,
                    target_end_us=index * 200_000 + 100_000,
                    source_subtitle_uids=[subtitle_uid],
                    style_template_id="canonical_caption_template",
                )
            )
        graph = DraftIngest().build_source_graph(
            word_timeline=words,
            subtitles=subtitles,
            source_segments=[{"id": "clip", "material_id": "main_video", "source_start_us": 0, "source_end_us": 30_000_000}],
            text_materials=materials,
            text_segments=segments,
        )

        plan, blockers = CaptionMaterialWriter().build_write_plan(graph, captions)

        self.assertEqual(blockers, [])
        report = plan["template_report"]
        self.assertEqual(report["candidate_count"], 110)
        self.assertEqual(report["fingerprint_group_count"], 1)
        self.assertEqual(report["stable_fingerprint_group_count"], 1)
        self.assertEqual(report["selection_reason"], "single_safe_fingerprint_group")
        self.assertEqual(plan["writer_fallback_count"], 0)
        self.assertIn("content.text", report["volatile_fields_ignored"])
        self.assertIn("subtitle_keywords", report["volatile_fields_ignored"])
        self.assertTrue(report["fingerprint_debug_samples"])


if __name__ == "__main__":
    unittest.main()
