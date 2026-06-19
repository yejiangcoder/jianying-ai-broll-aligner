from __future__ import annotations

import unittest
from pathlib import Path

from aroll_v21.decision.semantic_decision_planner import FORBIDDEN_DEEPSEEK_FIELDS


ROOT = Path(__file__).resolve().parents[1]
V21_SRC = ROOT / "src" / "aroll_v21"


def v21_source_text() -> str:
    return "\n".join(path.read_text("utf-8") for path in V21_SRC.rglob("*.py"))


class ArollV21NoDriftTests(unittest.TestCase):
    def test_no_legacy_patch_pipeline_imports_in_v21(self) -> None:
        text = v21_source_text()
        forbidden = [
            "aroll_phase4e_full_aroll",
            "aroll_uat_full",
            "aroll_downstream_repair_pipeline",
            "aroll_repair_applier",
            "aroll_safe_cut_boundary_resolver",
            "material_text_rows",
            "run_downstream_repair_pipeline",
            "apply_repair_proposals",
            "resolve_safe_cut_boundaries",
        ]
        for token in forbidden:
            self.assertNotIn(token, text)

    def test_no_repair_fixup_resolver_auto_fix_logic_in_v21_main_path(self) -> None:
        text = v21_source_text().lower()
        for token in ("auto_fix", "fixup", "downstream_repair", "safe_cut_boundary_resolver", "safe_cut_expand"):
            self.assertNotIn(token, text)
        self.assertNotIn("or segment.text", text)
        self.assertNotIn("aroll_shared_edit_utils", text)

    def test_deepseek_forbidden_physical_fields_are_locked(self) -> None:
        for field in (
            "source_start_us",
            "source_end_us",
            "target_start_us",
            "target_end_us",
            "edl",
            "final_edl",
            "material_id",
            "segment_id",
            "draft_content",
        ):
            self.assertIn(field, FORBIDDEN_DEEPSEEK_FIELDS)

    def test_real_uat_phrases_are_not_hardcoded_in_v21_src(self) -> None:
        text = v21_source_text()
        for term in (
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
        ):
            self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
