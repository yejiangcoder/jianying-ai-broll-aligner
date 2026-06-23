from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from draft_runtime_binding import DraftRuntimeBinding


DEFAULT_STATE_PATH = Path(r"D:\auto_clip_runtime\video_pipeline\current_draft.json")

STAGE_ALIASES = {
    "broll_design_qc_passed": "broll_design_ready",
    "ai_image_qc_passed": "ai_images_ready",
}
AROLL_QC_PASSED_STAGES = {
    "aroll_qc_passed",
    "broll_design_ready",
    "ai_images_ready",
    "image_alignment_written",
}
VALID_STAGES = {"rolled_back_baseline", "aroll_written", *AROLL_QC_PASSED_STAGES}


def normalize_stage(stage: str) -> str:
    normalized = STAGE_ALIASES.get(stage, stage)
    if normalized not in VALID_STAGES:
        allowed = ", ".join(sorted(VALID_STAGES | set(STAGE_ALIASES)))
        raise ValueError(f"未知 current_draft stage：{stage}；允许值：{allowed}")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Bind the current draft state for downstream video pipeline stages.")
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--jy-draftc", type=Path, default=None)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--stage", default="aroll_written")
    parser.add_argument("--source", default="manual_qc")
    args = parser.parse_args()

    if not args.draft_dir.exists():
        raise FileNotFoundError(f"draft_dir 不存在：{args.draft_dir}")

    out_dir = args.state_path.parent / "binding_checks" / time.strftime("bind_current_draft_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    binding = DraftRuntimeBinding.bind(args.draft_dir, args.jy_draftc, out_dir)
    stage = normalize_stage(args.stage)
    state: dict[str, Any] = {
        "version": "video_pipeline_current_draft_v1",
        "draft_dir": str(args.draft_dir),
        "draft_name": args.draft_dir.name,
        "stage": stage,
        "aroll_qc_passed": stage in AROLL_QC_PASSED_STAGES,
        "bound_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": args.source,
        "timeline_id": binding.timeline_id,
        "timeline_name": binding.timeline_name,
        "jy_draftc": str(binding.jy_draftc),
    }
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    args.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")

    print(f"CURRENT_DRAFT_STATE={args.state_path}")
    print(f"BOUND_DRAFT_DIR={args.draft_dir}")
    print(f"AROLL_QC_PASSED={state['aroll_qc_passed']}")
    print(f"TIMELINE_ID={binding.timeline_id}")
    print(f"TIMELINE_NAME={binding.timeline_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
