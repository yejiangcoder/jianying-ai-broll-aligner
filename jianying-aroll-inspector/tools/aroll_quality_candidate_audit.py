from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


TOOL_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TOOL_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aroll_runtime_paths import get_aroll_audits_dir  # noqa: E402
from aroll_v21.quality.quality_candidate_audit import (  # noqa: E402
    build_run_quality_candidate_audit,
    render_quality_candidate_audit_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a non-blocking A-Roll quality candidate audit from run artifacts.")
    parser.add_argument("--run-dir", action="append", required=True, help="A V21 run directory. Repeat for multiple runs.")
    parser.add_argument("--out-root", default="", help="Output root. Defaults to the external A-Roll audits runtime directory.")
    parser.add_argument("--name", default="", help="Optional audit folder name.")
    args = parser.parse_args(argv)

    out_root = Path(args.out_root) if str(args.out_root or "").strip() else get_aroll_audits_dir()
    audit_name = str(args.name or "").strip() or f"quality_candidate_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = out_root / audit_name
    out_dir.mkdir(parents=True, exist_ok=True)

    audits = [build_run_quality_candidate_audit(Path(value)) for value in args.run_dir]
    payload: dict[str, Any] = {
        "audit_name": "aroll_quality_candidate_audit_batch",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "run_count": len(audits),
        "candidate_count": sum(int(row.get("candidate_count") or 0) for row in audits),
        "runs": audits,
    }
    (out_dir / "quality_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    (out_dir / "quality_candidates.md").write_text(render_quality_candidate_audit_markdown(audits), "utf-8")
    print(json.dumps({"output_dir": str(out_dir), "candidate_count": payload["candidate_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
