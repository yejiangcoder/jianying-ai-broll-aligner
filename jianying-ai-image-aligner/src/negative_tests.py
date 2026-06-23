from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from direct_draft_broll_writer import (
    assert_slots_inside_video_segments,
    load_visual_slot_plan,
    normalized_image_files,
    validate_broll_image_contract,
    validate_no_slot_overlaps,
    validate_slot_confidence,
    validate_slot_plan_ids,
)
from draft_runtime_binding import (
    DraftRuntimeBinding,
    changed_hash_paths,
    collect_draft_write_hashes,
    read_json,
    write_json,
)
from runtime_paths import get_runtime_root


REPO_DIR = Path(__file__).resolve().parents[1]


def copy_case_inputs(case_dir: Path, broll: Path, image_dir: Path, visual_slot_plan: Path) -> tuple[Path, Path, Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    broll_copy = case_dir / broll.name
    plan_copy = case_dir / visual_slot_plan.name
    image_copy = case_dir / "images"
    shutil.copy2(broll, broll_copy)
    shutil.copy2(visual_slot_plan, plan_copy)
    shutil.copytree(image_dir, image_copy)
    image_files = normalized_image_files(image_copy)
    plan_data = load_plan(plan_copy)
    for slot in plan_data["slots"]:
        image_id = str(slot.get("image_id") or "")
        if image_id in image_files:
            slot["image_path"] = str(image_files[image_id])
    write_plan(plan_copy, plan_data)
    return broll_copy, image_copy, plan_copy


def run_preflight_validators(broll: Path, image_dir: Path, visual_slot_plan: Path, draft_data: dict[str, Any]) -> None:
    image_ids, image_files = validate_broll_image_contract(broll, image_dir)
    slots = load_visual_slot_plan(visual_slot_plan, image_dir, image_files)
    validate_slot_plan_ids(slots, image_ids)
    validate_no_slot_overlaps(slots)
    validate_slot_confidence(slots)
    assert_slots_inside_video_segments(slots, draft_data)


def replace_first_table_id(broll: Path, new_id: str) -> None:
    lines = broll.read_text("utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("|") and "AI静态图" in line:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and cells[0].isdigit():
                cells[0] = new_id
                lines[index] = "| " + " | ".join(cells) + " |"
                broll.write_text("\n".join(lines) + "\n", "utf-8")
                return
    raise RuntimeError("没有找到可修改的 B-roll AI静态图表格行")


def load_plan(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("slots"), list):
        raise RuntimeError(f"visual_slot_plan 格式不符合测试预期：{path}")
    return data


def write_plan(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def mutate_slot_plan_id(plan_path: Path) -> None:
    data = load_plan(plan_path)
    data["slots"][0]["image_id"] = "99"
    write_plan(plan_path, data)


def mutate_slot_overlap(plan_path: Path) -> None:
    data = load_plan(plan_path)
    first = data["slots"][0]
    second = data["slots"][1]
    second["target_start_us"] = int(first["target_end_us"]) - 100_000
    second["target_end_us"] = max(int(second["target_start_us"]) + 100_000, int(second["target_end_us"]))
    second["duration_us"] = int(second["target_end_us"]) - int(second["target_start_us"])
    write_plan(plan_path, data)


def mutate_slot_exceeds_container(plan_path: Path, draft_data: dict[str, Any]) -> None:
    data = load_plan(plan_path)
    slot = data["slots"][0]
    container_id = str(slot["container_video_segment_ids"][0])
    video_start = None
    for track in draft_data.get("tracks", []):
        if track.get("type") != "video":
            continue
        for segment in track.get("segments", []):
            if str(segment.get("id") or "") == container_id:
                timerange = segment.get("target_timerange") or {}
                video_start = int(timerange.get("start") or 0)
                break
        if video_start is not None:
            break
    if video_start is None:
        raise RuntimeError(f"找不到测试 container video segment：{container_id}")
    # Keep this mutation focused on the video-container validator. Moving the
    # first slot just before its container avoids triggering the overlap gate
    # first when the next real slot starts close to this container's end.
    slot["target_start_us"] = video_start - 200_000
    slot["target_end_us"] = video_start - 100_000
    slot["duration_us"] = int(slot["target_end_us"]) - int(slot["target_start_us"])
    write_plan(plan_path, data)


def synthetic_gap_data() -> dict[str, Any]:
    return {
        "materials": {"videos": [{"id": "m1", "material_name": "v1"}, {"id": "m2", "material_name": "v2"}]},
        "tracks": [
            {
                "type": "video",
                "name": "main",
                "segments": [
                    {
                        "id": "v1",
                        "material_id": "m1",
                        "target_timerange": {"start": 0, "duration": 1_000_000},
                    },
                    {
                        "id": "v2",
                        "material_id": "m2",
                        "target_timerange": {"start": 2_000_000, "duration": 1_000_000},
                    },
                ],
            }
        ],
    }


def case_slot_crosses_gap() -> None:
    slots = [
        {
            "slot_id": "gap_case",
            "start_us": 500_000,
            "end_us": 2_500_000,
            "container_video_segment_ids": ["v1", "v2"],
        }
    ]
    assert_slots_inside_video_segments(slots, synthetic_gap_data())


def powershell() -> str:
    return "powershell"


def run_wrapper_expect_failure(args: list[str], env_state_path: Path) -> str:
    env_command = (
        f"$env:VIDEO_PIPELINE_CURRENT_DRAFT_STATE = '{str(env_state_path)}'; "
        + " ".join(args)
    )
    result = subprocess.run(
        [powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", env_command],
        cwd=str(REPO_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode == 0:
        raise RuntimeError("wrapper unexpectedly succeeded")
    return result.stdout


def case_unbound_current_draft(case_dir: Path, broll: Path, image_dir: Path, plan: Path, jy_draftc: Path) -> None:
    missing_state = case_dir / "missing_current_draft.json"
    output = run_wrapper_expect_failure(
        [
            ".\\run_pipeline_contract_check.ps1",
            "-BrollMd",
            f"'{broll}'",
            "-ImageDir",
            f"'{image_dir}'",
            "-VisualSlotPlan",
            f"'{plan}'",
            "-JyDraftc",
            f"'{jy_draftc}'",
        ],
        missing_state,
    )
    if "current draft state does not exist" not in output:
        raise RuntimeError(output)


def case_aroll_qc_not_passed(
    case_dir: Path,
    draft_dir: Path,
    broll: Path,
    image_dir: Path,
    plan: Path,
    jy_draftc: Path,
) -> None:
    state = {
        "version": "video_pipeline_current_draft_v1",
        "draft_dir": str(draft_dir),
        "aroll_qc_passed": False,
    }
    state_path = case_dir / "current_draft_qc_false.json"
    state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), "utf-8")
    output = run_wrapper_expect_failure(
        [
            ".\\run_pipeline_contract_check.ps1",
            "-BrollMd",
            f"'{broll}'",
            "-ImageDir",
            f"'{image_dir}'",
            "-VisualSlotPlan",
            f"'{plan}'",
            "-JyDraftc",
            f"'{jy_draftc}'",
        ],
        state_path,
    )
    if "A-Roll QC is not marked passed" not in output:
        raise RuntimeError(output)


def copy_draft_template(draft_dir: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(draft_dir, destination)
    return destination


def case_post_write_rollback(case_dir: Path, draft_dir: Path, jy_draftc: Path) -> dict[str, Any]:
    clone = copy_draft_template(draft_dir, case_dir / "disposable_draft")
    bind_dir = case_dir / "bind"
    bind_dir.mkdir(parents=True, exist_ok=True)
    binding = DraftRuntimeBinding.bind(clone, jy_draftc, bind_dir)
    before = collect_draft_write_hashes(clone)

    def key_value_writer() -> None:
        key_value = clone / "key_value.json"
        data = read_json(key_value) if key_value.exists() else {}
        data["NEGATIVE_TEST_MARKER"] = {"materialName": "negative-test"}
        write_json(key_value, data)

    try:
        binding.write_encrypted_transaction(
            "invalid encrypted payload from negative test",
            key_value_writer,
            case_dir / "transaction",
            post_write_validator=lambda: (_ for _ in ()).throw(RuntimeError("FORCED_POST_WRITE_AUDIT_FAILURE")),
        )
    except RuntimeError as exc:
        if "FORCED_POST_WRITE_AUDIT_FAILURE" not in str(exc):
            raise
    else:
        raise RuntimeError("transaction unexpectedly succeeded")

    after = collect_draft_write_hashes(clone)
    unrestored = sorted(changed_hash_paths(before, after))
    if unrestored:
        raise RuntimeError(f"rollback hash mismatch: {unrestored}")
    return {"rollback_restored": True, "disposable_draft": str(clone)}


def expect_block(name: str, expected: str, func: Callable[[], Any]) -> dict[str, Any]:
    try:
        detail = func()
    except Exception as exc:
        message = str(exc)
        passed = expected in message if expected else True
        return {
            "name": name,
            "passed": passed,
            "blocked": True,
            "message": message,
            "expected_substring": expected,
        }
    return {
        "name": name,
        "passed": False,
        "blocked": False,
        "message": f"case did not block; detail={detail}",
        "expected_substring": expected,
    }


def expect_success(name: str, func: Callable[[], Any]) -> dict[str, Any]:
    try:
        detail = func()
    except Exception as exc:
        return {"name": name, "passed": False, "blocked": True, "message": str(exc)}
    return {"name": name, "passed": True, "blocked": False, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run negative tests for the image aligner contract.")
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--broll", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--visual-slot-plan", type=Path, required=True)
    parser.add_argument("--jy-draftc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (
        get_runtime_root() / "negative_tests" / f"negative_tests_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    binding = DraftRuntimeBinding.bind(args.draft_dir, args.jy_draftc, out_dir / "base_bind")
    draft_data = binding.decrypt_timeline(out_dir / "base_draft_content.dec.json")

    report: dict[str, Any] = {
        "draft_dir": str(args.draft_dir),
        "broll": str(args.broll),
        "image_dir": str(args.image_dir),
        "visual_slot_plan": str(args.visual_slot_plan),
        "jy_draftc": str(args.jy_draftc),
        "out_dir": str(out_dir),
        "cases": [],
    }

    def package_case(case_name: str) -> tuple[Path, Path, Path, Path]:
        case_dir = out_dir / case_name
        broll, image_dir, plan = copy_case_inputs(case_dir, args.broll, args.image_dir, args.visual_slot_plan)
        return case_dir, broll, image_dir, plan

    case_dir, broll, image_dir, plan = package_case("missing_image")
    sorted(image_dir.glob("*.png"))[0].unlink()
    report["cases"].append(
        expect_block(
            "missing_image_must_block",
            "B-roll 表格 AI 编号与图片目录不一致",
            lambda: run_preflight_validators(broll, image_dir, plan, draft_data),
        )
    )

    case_dir, broll, image_dir, plan = package_case("extra_image")
    first_image = sorted(image_dir.glob("*.png"))[0]
    shutil.copy2(first_image, image_dir / "extra_AI_99_extra.png")
    report["cases"].append(
        expect_block(
            "extra_image_must_block",
            "B-roll 表格 AI 编号与图片目录不一致",
            lambda: run_preflight_validators(broll, image_dir, plan, draft_data),
        )
    )

    case_dir, broll, image_dir, plan = package_case("broll_table_static_mismatch")
    replace_first_table_id(broll, "99")
    report["cases"].append(
        expect_block(
            "broll_table_static_list_mismatch_must_block",
            "B-roll 表格 AI 编号与图片目录不一致",
            lambda: run_preflight_validators(broll, image_dir, plan, draft_data),
        )
    )

    case_dir, broll, image_dir, plan = package_case("slot_plan_image_id_mismatch")
    mutate_slot_plan_id(plan)
    report["cases"].append(
        expect_block(
            "visual_slot_plan_id_image_dir_mismatch_must_block",
            "slot image_id 与 image_path 文件名编号不一致",
            lambda: run_preflight_validators(broll, image_dir, plan, draft_data),
        )
    )

    case_dir, broll, image_dir, plan = package_case("slot_overlap")
    mutate_slot_overlap(plan)
    report["cases"].append(
        expect_block(
            "slot_overlap_must_block",
            "visual_slot_plan 存在同轨重叠 slot",
            lambda: run_preflight_validators(broll, image_dir, plan, draft_data),
        )
    )

    case_dir, broll, image_dir, plan = package_case("slot_exceeds_container")
    mutate_slot_exceeds_container(plan, draft_data)
    report["cases"].append(
        expect_block(
            "slot_exceeds_container_must_block",
            "SLOT_CROSSES_VIDEO_GAP_OR_EXCEEDS_CONTAINER",
            lambda: run_preflight_validators(broll, image_dir, plan, draft_data),
        )
    )

    report["cases"].append(
        expect_block(
            "slot_crosses_video_gap_must_block",
            "SLOT_CROSSES_VIDEO_GAP_OR_EXCEEDS_CONTAINER",
            case_slot_crosses_gap,
        )
    )

    case_dir, broll, image_dir, plan = package_case("unbound_current_draft")
    report["cases"].append(
        expect_success(
            "unbound_current_draft_must_block",
            lambda: case_unbound_current_draft(case_dir, broll, image_dir, plan, args.jy_draftc),
        )
    )

    case_dir, broll, image_dir, plan = package_case("aroll_qc_not_passed")
    report["cases"].append(
        expect_success(
            "aroll_qc_not_passed_must_block",
            lambda: case_aroll_qc_not_passed(case_dir, args.draft_dir, broll, image_dir, plan, args.jy_draftc),
        )
    )

    report["cases"].append(
        expect_success(
            "post_write_audit_failure_must_rollback",
            lambda: case_post_write_rollback(out_dir / "post_write_rollback", args.draft_dir, args.jy_draftc),
        )
    )

    failed = [case for case in report["cases"] if not case.get("passed")]
    report["passed"] = not failed
    report["failed_count"] = len(failed)
    report["case_count"] = len(report["cases"])
    report_path = out_dir / "negative_test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    print(f"NEGATIVE_TEST_REPORT={report_path}")
    print(f"NEGATIVE_TEST_PASSED={report['passed']}")
    print(f"NEGATIVE_TEST_CASE_COUNT={report['case_count']}")
    print(f"NEGATIVE_TEST_FAILED_COUNT={report['failed_count']}")
    for case in report["cases"]:
        print(f"{case['name']}={'PASS' if case.get('passed') else 'FAIL'}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
