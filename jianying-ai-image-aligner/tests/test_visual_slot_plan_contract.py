from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "direct_draft_broll_writer.py"
SRC = SCRIPT.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location("direct_draft_broll_writer", SCRIPT)
writer = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = writer
SPEC.loader.exec_module(writer)


def test_visual_slot_rows_blocks_not_ready_plan() -> None:
    raw = {
        "schema_version": "visual_slot_plan.v1",
        "ready_for_image_alignment": False,
        "slots": [
            {
                "slot_id": "B01",
                "image_id": "01",
                "target_start_us": 0,
                "target_end_us": 1_000_000,
                "container_video_segment_ids": ["video_1"],
            }
        ],
    }

    try:
        writer.visual_slot_rows(raw)
    except RuntimeError as exc:
        assert "ready_for_image_alignment=false" in str(exc)
    else:
        raise AssertionError("not-ready visual_slot_plan should block")


def test_visual_slot_rows_accepts_legacy_plan_without_ready_flag() -> None:
    rows = writer.visual_slot_rows(
        {
            "slots": [
                {
                    "slot_id": "B01",
                    "image_id": "01",
                    "target_start_us": 0,
                    "target_end_us": 1_000_000,
                    "container_video_segment_ids": ["video_1"],
                }
            ]
        }
    )

    assert rows[0]["slot_id"] == "B01"
