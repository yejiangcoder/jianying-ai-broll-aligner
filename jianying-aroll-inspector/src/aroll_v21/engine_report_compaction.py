from __future__ import annotations

import json
from typing import Any

from aroll_v21.ir.models import dataclass_to_dict

COMPACT_RUNTIME_REPORT_DROP_KEYS = {
    "post_write_actual_draft_audit",
    "staged_post_write_actual_draft_audit",
    "postwrite_actual_draft_audit",
    "actual_draft_data",
    "draft_data",
}


def _compact_runtime_report_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    compact: dict[str, Any] = {}
    omitted: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if key in COMPACT_RUNTIME_REPORT_DROP_KEYS:
            omitted[key] = {
                "omitted": True,
                "reason": "debug_payload_available_only_in_debug_report_profile",
                "approx_json_bytes": len(json.dumps(dataclass_to_dict(value), ensure_ascii=False)),
            }
            continue
        compact[key] = value
    if omitted:
        compact["compact_report_omitted_debug_payloads"] = omitted
    return compact


def _resolved_semantic_decision_rows(decision_plan) -> list[dict[str, Any]]:
    if decision_plan is None:
        empty_rows: list[dict[str, Any]] = []
        return empty_rows
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in list(getattr(decision_plan, "semantic_decision_rows", []) or []):
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "")
        if not cluster_id or cluster_id in seen:
            continue
        if str(row.get("_blocker_code") or ""):
            continue
        rows.append(dict(row))
        seen.add(cluster_id)
    return rows

