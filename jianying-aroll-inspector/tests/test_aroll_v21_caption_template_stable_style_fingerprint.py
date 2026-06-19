from __future__ import annotations

import copy
import json
import unittest

from aroll_v21.writer.caption_material_writer import caption_template_fingerprint
from tests.test_aroll_v21_caption_template_round7_fingerprint_sample import _round7_material, _round7_segment


class ArollV21CaptionTemplateStableStyleFingerprintTests(unittest.TestCase):
    def test_round7_volatile_text_keyword_word_and_id_fields_do_not_change_fingerprint(self) -> None:
        material_a = _round7_material("material_a", "就国南能不能不要规训自己人呐", keyword=True)
        material_b = _round7_material("material_b", "你嘲笑嘉豪是对自己人", keyword=False)
        segment_a = _round7_segment("material_a", 1)
        segment_b = _round7_segment("material_b", 99)

        self.assertEqual(caption_template_fingerprint(material_a, segment_a), caption_template_fingerprint(material_b, segment_b))

    def test_real_font_size_color_stroke_transform_and_scale_differences_still_change_fingerprint(self) -> None:
        material_a = _round7_material("material_a", "普通字幕", keyword=False)
        material_b = _round7_material("material_b", "普通字幕", keyword=False)
        segment_a = _round7_segment("material_a", 1)
        segment_b = _round7_segment("material_b", 2)

        material_b["font_size"] = 6.0
        material_b["text_color"] = "#FFFF00"
        base_content = json.loads(material_b["base_content"])
        base_content["styles"][0]["size"] = 6.0
        base_content["styles"][0]["fill"]["content"]["solid"]["color"] = [1.0, 1.0, 0.0]
        base_content["styles"][0]["strokes"] = [{"width": 0.08, "content": {"render_type": "solid", "solid": {"color": [0, 0, 0]}}}]
        material_b["base_content"] = json.dumps(base_content, ensure_ascii=False)
        segment_b = copy.deepcopy(segment_b)
        segment_b["clip"]["scale"] = {"x": 1.15, "y": 1.15}
        segment_b["clip"]["transform"] = {"x": 0.0, "y": -0.55}

        self.assertNotEqual(caption_template_fingerprint(material_a, segment_a), caption_template_fingerprint(material_b, segment_b))


if __name__ == "__main__":
    unittest.main()
