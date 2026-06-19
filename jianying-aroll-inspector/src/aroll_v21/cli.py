from __future__ import annotations

import argparse
from pathlib import Path

from aroll_v21.operator import ArollV21OperatorConfig, run_operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline V21 A-Roll compiler runner.")
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--draft-dir", type=Path)
    parser.add_argument("--jy-draftc", type=Path)
    parser.add_argument("--word-timeline-json", type=Path)
    parser.add_argument("--semantic-decisions-json", type=Path)
    parser.add_argument("--output-dir", "--run-dir", dest="run_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "write", "verify-only"], default="dry-run")
    parser.add_argument("--postwrite-materials-json", type=Path)
    parser.add_argument("--ready-run-dir", type=Path)
    parser.add_argument("--report-profile", choices=["minimal", "standard", "debug"], default="standard")
    parser.add_argument(
        "--semantic-mode",
        choices=["auto", "deterministic-baseline", "semantic-requests-only", "deepseek", "fail-closed", "default"],
        default="auto",
    )
    parser.add_argument("--simulate-write", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--allow-sacrificial-write-without-postwrite-decrypt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_operator(
        ArollV21OperatorConfig(
            mode=args.mode,
            run_dir=args.run_dir,
            input_json=args.input_json,
            draft_dir=args.draft_dir,
            jy_draftc=args.jy_draftc,
            word_timeline_json=args.word_timeline_json,
            semantic_decisions_json=args.semantic_decisions_json,
            postwrite_materials_json=args.postwrite_materials_json,
            simulate_write=bool(args.simulate_write),
            commit=bool(args.commit),
            allow_sacrificial_write_without_postwrite_decrypt=bool(
                args.allow_sacrificial_write_without_postwrite_decrypt
            ),
            semantic_mode=str(args.semantic_mode or "default"),
            ready_run_dir=args.ready_run_dir,
            report_profile=args.report_profile,
        )
    )
    print(f"status={summary.get('status')}")
    print(f"write_status={summary.get('write_status')}")
    print(f"run_dir={args.run_dir}")
    return 0 if summary.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
