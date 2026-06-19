from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [ROOT / "src" / "aroll_v21", ROOT / "tools"]


def _python_files() -> list[Path]:
    return [path for root in SCAN_ROOTS for path in root.rglob("*.py")]


def _source_text() -> str:
    return "\n".join(path.read_text("utf-8") for path in _python_files())


def _all_src_text() -> str:
    return "\n".join(path.read_text("utf-8") for path in (ROOT / "src").rglob("*.py"))


class ArollV21FinalStaticHiddenBugScanTests(unittest.TestCase):
    def test_no_forbidden_v20_or_real_draft_hardcoded_paths_or_phrases(self) -> None:
        text = _source_text()
        for token in (
            "material_text_rows",
            "D:\\JianyingPro Drafts",
            "6月15日",
            "嘉豪",
            "随意的",
            "肆意的",
            "踩踏",
            "这说明",
        ):
            self.assertNotIn(token, text)

    def test_no_hardcoded_qc_sample_phrases_in_src(self) -> None:
        text = _all_src_text()
        for token in (
            "嘉豪",
            "粪坑",
            "舞台中央",
            "6月18日",
        ):
            self.assertNotIn(token, text)

    def test_no_forbidden_v20_imports_or_repair_calls(self) -> None:
        forbidden_modules = (
            "aroll_phase4e",
            "aroll_uat_full",
            "aroll_downstream_repair_pipeline",
            "aroll_repair_applier",
            "aroll_safe_cut_boundary_resolver",
        )
        forbidden_calls = (
            "run_downstream_repair_pipeline",
            "apply_repair_proposals",
            "resolve_safe_cut_boundaries",
        )
        for path in _python_files():
            tree = ast.parse(path.read_text("utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(any(alias.name.startswith(module) for module in forbidden_modules), str(path))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(any(module.startswith(item) for item in forbidden_modules), str(path))
                elif isinstance(node, ast.Call):
                    name = ""
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    self.assertNotIn(name, forbidden_calls, str(path))

    def test_no_bare_except_pass_or_broad_except_returning_success(self) -> None:
        for path in _python_files():
            tree = ast.parse(path.read_text("utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                is_bare = node.type is None
                body_is_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                self.assertFalse(is_bare and body_is_pass, str(path))
                if isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return):
                            payload = ast.dump(child.value) if child.value is not None else ""
                            self.assertNotRegex(payload, r"success.*True|WritebackResult.*True", str(path))

    def test_no_naive_first_track_selection_or_fake_success_literals(self) -> None:
        text = _source_text()
        self.assertNotRegex(text, r"tracks\s*\[\s*0\s*\].*(text|subtitle|video)")
        self.assertNotRegex(text, r"first\s+(text|video)\s+track")
        self.assertNotRegex(text, r"selected\s+first\s+(text|video)\s+track")
        success_sites = [
            (path, lineno, line.strip())
            for path in _python_files()
            for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1)
            if re.search(r'"WRITE_SUCCESS"\s*:\s*True|"commit_performed"\s*:\s*True|commit_performed=True', line)
        ]
        allowed_paths = {ROOT / "src" / "aroll_v21" / "operator.py", ROOT / "src" / "aroll_v21" / "writeback" / "real_draft_writeback.py"}
        for path, _lineno, line in success_sites:
            self.assertIn(path, allowed_paths, line)
        writeback_source = (ROOT / "src" / "aroll_v21" / "writeback" / "real_draft_writeback.py").read_text("utf-8")
        self.assertIn('"writeback_success": True', writeback_source)
        self.assertIn('"WRITE_SUCCESS": True', writeback_source)


if __name__ == "__main__":
    unittest.main()
