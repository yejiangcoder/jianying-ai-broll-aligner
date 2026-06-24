from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from aroll_v21.ir.models import (
    CanonicalSourceGraph,
    CanonicalWord,
    CaptionRenderUnit,
    FinalTimelineSegment,
    SourceGraphInvariantReport,
    dataclass_to_dict,
)
from aroll_v21.quality.caption_alignment import build_caption_alignment_report
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate
from aroll_v21.quality.quality_gate import build_quality_gate_report
from aroll_v21.quality.visual_pacing import build_visual_pacing_report
from aroll_v21.render.subtitle_renderer import SubtitleRenderer


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "replay_final_visible_repair_from_run.py"

spec = importlib.util.spec_from_file_location("replay_final_visible_repair_from_run", TOOL_PATH)
assert spec is not None and spec.loader is not None
replay_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_tool)


class ReplayFinalVisibleRepairToolTests(unittest.TestCase):
    def test_replay_repairs_existing_run_without_mutating_input_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            output_dir = root / "replay"
            run_dir.mkdir()
            graph = self._source_graph()
            timeline = [
                FinalTimelineSegment(
                    segment_id="seg_001",
                    source_material_id="main",
                    source_segment_id="primary",
                    source_start_us=0,
                    source_end_us=2_220_000,
                    target_start_us=0,
                    target_end_us=2_220_000,
                    word_ids=["w001", "w002", "w003", "w004", "w005"],
                    text="保留你连错的",
                    decision_ids=[],
                    spoken_source_start_us=0,
                    spoken_source_end_us=2_220_000,
                    clip_source_start_us=0,
                    clip_source_end_us=2_220_000,
                )
            ]
            renderer = SubtitleRenderer()
            captions = renderer.render(timeline, graph)
            self._write_run_artifacts(run_dir, graph, timeline, captions)

            report = replay_tool.replay_final_visible_repair_from_run(run_dir, output_dir=output_dir)

            self.assertTrue(report["read_only_input"])
            self.assertTrue(report["does_not_write_draft"])
            self.assertTrue(report["changed"])
            self.assertGreaterEqual(report["final_visible_repair_action_count"], 1)
            input_timeline = self._read_json(run_dir / "final_timeline.json")
            output_timeline = self._read_json(output_dir / "final_timeline.json")
            output_captions = self._read_json(output_dir / "captions.json")
            self.assertEqual(input_timeline[0]["text"], "保留你连错的")
            self.assertEqual(output_timeline[0]["text"], "保留")
            self.assertEqual(output_captions[0]["text"], "保留")
            replay_report = self._read_json(output_dir / "replay_final_visible_repair_report.json")
            self.assertEqual(replay_report["report_kind"], replay_tool.REPORT_KIND)

    def _source_graph(self) -> CanonicalSourceGraph:
        rows = [
            ("w001", "保留", 0, 1_500_000),
            ("w002", "你", 1_600_000, 1_760_000),
            ("w003", "连", 1_760_000, 1_900_000),
            ("w004", "错", 1_900_000, 2_060_000),
            ("w005", "的", 2_060_000, 2_220_000),
            ("w006", "你", 3_000_000, 3_160_000),
            ("w007", "连", 3_160_000, 3_300_000),
            ("w008", "怎么", 3_300_000, 3_540_000),
            ("w009", "错", 3_540_000, 3_700_000),
            ("w010", "都", 3_700_000, 3_920_000),
        ]
        words = [
            CanonicalWord(
                word_id=word_id,
                text=text,
                normalized_text=text,
                source_start_us=start,
                source_end_us=end,
                source_material_id="main",
                source_segment_id="primary",
                subtitle_uid="s001",
                subtitle_index=1,
                char_start=None,
                char_end=None,
                confidence=None,
                is_cuttable_left=True,
                is_cuttable_right=True,
            )
            for word_id, text, start, end in rows
        ]
        return CanonicalSourceGraph(
            words=words,
            edit_units=[],
            subtitle_rows=[
                {
                    "subtitle_uid": "s001",
                    "subtitle_index": 1,
                    "text": "".join(text for _word_id, text, _start, _end in rows),
                    "word_ids": [word_id for word_id, _text, _start, _end in rows],
                }
            ],
            source_materials=[{"source_material_id": "main", "type": "video", "duration_us": 4_500_000}],
            source_segments=[
                {
                    "id": "primary",
                    "material_id": "main",
                    "type": "video",
                    "source_start_us": 0,
                    "source_end_us": 4_500_000,
                }
            ],
            text_materials=[],
            text_segments=[],
            invariant_report=SourceGraphInvariantReport(
                single_source_graph_ok=True,
                all_words_have_source_time=True,
                all_edit_units_have_word_ids=True,
                unbound_word_count=0,
                unbound_subtitle_count=0,
                blocker_count=0,
                blockers=[],
            ),
        )

    def _write_run_artifacts(
        self,
        run_dir: Path,
        graph: CanonicalSourceGraph,
        timeline: list[FinalTimelineSegment],
        captions: list[CaptionRenderUnit],
    ) -> None:
        caption_alignment = build_caption_alignment_report(final_timeline=timeline, captions=captions)
        visible_repeat = build_final_caption_visible_repeat_gate(captions)
        visual = build_visual_pacing_report(final_timeline=timeline, captions=captions, executed=True, source_graph=graph)
        quality = build_quality_gate_report(
            effective_speed_gate={"gate_passed": True, "blocker_codes": []},
            final_repeat_convergence_gate={"gate_passed": True, "blocker_codes": []},
            final_caption_visible_repeat_gate=visible_repeat,
            semantic_adjudication_gate={
                "semantic_adjudication_gate_passed": True,
                "semantic_request_count": 0,
                "semantic_request_unresolved_count": 0,
                "fatal_semantic_issue_count": 0,
                "blocker_codes": [],
            },
            visual_pacing_gate=visual,
            caption_alignment_gate=caption_alignment,
            ready_for_user_manual_qc_preconditions_passed=True,
        )
        self._write_json(run_dir / "source_graph.json", graph)
        self._write_json(run_dir / "final_timeline.json", timeline)
        self._write_json(run_dir / "captions.json", captions)
        self._write_json(run_dir / "quality_gate_report.json", quality)
        self._write_json(run_dir / "semantic_adjudication_report.json", {"blocker_codes": []})
        self._write_json(run_dir / "run_summary.json", {"status": "ok", "ready_for_disposable_write_pre_audit": True})

    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(dataclass_to_dict(payload), ensure_ascii=False, indent=2), "utf-8")

    def _read_json(self, path: Path):
        return json.loads(path.read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
