from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aroll_v21.operator import ArollV21OperatorConfig, run_operator

from tests.aroll_v21_contract_assertions import (
    assert_material_caption_timeline_contract,
    assert_run_summary_contract,
    assert_writeback_report_contract,
)
from tests.test_aroll_v21_full_chain_internal_self_check import (
    ExternalWordTimelineAdapter,
    add_non_aroll_tracks,
    root_mirror_required,
)
from tests.test_aroll_v21_sacrificial_write_override import (
    create_disposable_draft,
    fake_real_draft_result,
    fake_real_writeback,
)


def writeback_factory(*args, **kwargs):
    return fake_real_writeback(jy_draftc=kwargs.get("jy_draftc"), root_mirror_func=root_mirror_required)


def read_json(path: Path):
    return json.loads(path.read_text("utf-8"))


class ArollV21BackendContractFullChainTests(unittest.TestCase):
    def test_operator_engine_validator_writeback_artifact_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir, draft_content, template = create_disposable_draft(root)
            result = fake_real_draft_result(root=root)
            add_non_aroll_tracks(result)
            word_timeline_json = root / "external_word_timeline.json"
            word_timeline_json.write_text(json.dumps(result.word_timeline, ensure_ascii=False), "utf-8")
            semantic_decisions_json = root / "semantic_decisions.json"
            semantic_decisions_json.write_text("[]", "utf-8")
            before_draft = draft_content.read_text("utf-8")
            before_template = template.read_text("utf-8")

            with patch(
                "aroll_v21.operator.RealDraftIngestAdapter",
                lambda *a, **k: ExternalWordTimelineAdapter(result=result),
            ), patch("aroll_v21.operator.RealDraftWriteback", writeback_factory):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="write",
                        run_dir=root / "run",
                        draft_dir=draft_dir,
                        jy_draftc=root / "jy draftc with spaces.exe",
                        word_timeline_json=word_timeline_json,
                        semantic_decisions_json=semantic_decisions_json,
                        commit=True,
                        allow_sacrificial_write_without_postwrite_decrypt=True,
                        report_profile="debug",
                    )
                )

            run_dir = root / "run"
            artifacts = {
                name: run_dir / name
                for name in (
                    "source_graph.json",
                    "decision_plan.json",
                    "final_timeline.json",
                    "captions.json",
                    "material_write_plan.json",
                    "validator_report.json",
                    "writeback_report.json",
                    "run_summary.json",
                )
            }
            for name, path in artifacts.items():
                self.assertTrue(path.exists(), name)

            source_graph = read_json(artifacts["source_graph.json"])
            decision_plan = read_json(artifacts["decision_plan.json"])
            final_timeline = read_json(artifacts["final_timeline.json"])
            captions = read_json(artifacts["captions.json"])
            material_write_plan = read_json(artifacts["material_write_plan.json"])
            validator_report = read_json(artifacts["validator_report.json"])
            writeback_report = read_json(artifacts["writeback_report.json"])
            persisted_summary = read_json(artifacts["run_summary.json"])

            self.assertTrue(source_graph.get("words"))
            self.assertIn("decisions", decision_plan)
            self.assertTrue(validator_report["validator_report_ok"])
            assert_material_caption_timeline_contract(self, final_timeline, captions, material_write_plan)
            assert_writeback_report_contract(self, writeback_report)
            assert_run_summary_contract(self, summary, writeback_report=writeback_report)
            self.assertEqual(summary, persisted_summary)

            self.assertNotEqual(draft_content.read_text("utf-8"), before_draft)
            self.assertNotEqual(template.read_text("utf-8"), before_template)
            self.assertTrue((draft_dir / "draft_content.json").exists())
            self.assertTrue((draft_dir / "template-2.tmp").exists())
            self.assertTrue(writeback_report["target_writes"][str(draft_dir / "draft_content.json")])
            self.assertTrue(writeback_report["target_writes"][str(draft_dir / "template-2.tmp")])


if __name__ == "__main__":
    unittest.main()
