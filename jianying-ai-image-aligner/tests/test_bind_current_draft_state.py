from __future__ import annotations

from bind_current_draft import AROLL_QC_PASSED_STAGES, normalize_stage


def test_aroll_written_does_not_mark_aroll_qc_passed() -> None:
    stage = normalize_stage("aroll_written")

    assert stage == "aroll_written"
    assert stage not in AROLL_QC_PASSED_STAGES


def test_aroll_qc_passed_requires_explicit_stage() -> None:
    stage = normalize_stage("aroll_qc_passed")

    assert stage == "aroll_qc_passed"
    assert stage in AROLL_QC_PASSED_STAGES
