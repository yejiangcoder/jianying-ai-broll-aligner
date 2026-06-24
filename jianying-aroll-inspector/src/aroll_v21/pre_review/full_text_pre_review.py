from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aroll_v21.decision.deepseek_semantic_planner import (
    DEFAULT_DEEPSEEK_MODEL,
    FORBIDDEN_PROVIDER_FIELDS,
    deepseek_provider_from_runtime_config,
)
from aroll_v21.operator_io import read_json


REPORT_KIND = "aroll_v21_full_text_pre_review"
REPORT_VERSION = 1
DEFAULT_MAX_ITEMS = 900
DEFAULT_MAX_PROMPT_CHARS = 120_000
ALLOWED_ISSUE_TYPES = {
    "untrimmed_non_speech",
    "large_breath_gap",
    "broken_word_or_truncated_syllable",
    "stutter_or_restart_not_removed",
    "bad_repeat",
    "rhetorical_repeat_false_positive",
    "semantic_jump",
    "caption_fragment",
    "visible_caption_mismatch",
    "other",
}
ALLOWED_SEVERITIES = {"info", "warning", "fatal_candidate"}
ALLOWED_QC_RECOMMENDATIONS = {"pass", "warning", "block_candidate"}


class FullTextPreReviewProvider(Protocol):
    provider_name: str

    def review(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PreReviewRunResult:
    report: dict[str, Any]
    markdown: str
    hotspots: dict[str, Any]
    triage: dict[str, Any]


class DeepSeekFullTextPreReviewProvider:
    provider_name = "deepseek_full_text_pre_review"

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.provider_called_count = 0

    @classmethod
    def from_runtime_config(cls) -> "DeepSeekFullTextPreReviewProvider | None":
        provider = deepseek_provider_from_runtime_config()
        if provider is None:
            missing: DeepSeekFullTextPreReviewProvider | None = None
            return missing
        return cls(provider)

    def review(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not getattr(self.provider, "api_key", ""):
            raise RuntimeError("DEEPSEEK_PRE_REVIEW_PROVIDER_MISSING")
        request_payload = {
            "schema": _provider_response_schema(),
            "review_task": {
                "goal": "Review the final Chinese A-Roll rough cut as a non-mutating QC auditor.",
                "strict_limits": [
                    "Return JSON only.",
                    "Do not return physical edit fields.",
                    "Only reference provided review_item_id values.",
                    "Do not invent exact audio defects when no local audio signal is provided.",
                ],
                "allowed_issue_types": sorted(ALLOWED_ISSUE_TYPES),
                "allowed_severities": sorted(ALLOWED_SEVERITIES),
                "allowed_qc_recommendations": sorted(ALLOWED_QC_RECOMMENDATIONS),
            },
            "payload": payload,
        }
        body_payload: dict[str, Any] = {
            "model": str(getattr(self.provider, "model", "") or DEFAULT_DEEPSEEK_MODEL),
            "thinking": {"type": str(getattr(self.provider, "thinking_type", "") or "enabled")},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict Chinese video rough-cut pre-review auditor. "
                        "You only produce non-mutating QC findings. Return JSON only. "
                        "Never output source_start_us, source_end_us, target_start_us, target_end_us, "
                        "material_id, source_material_id, source_segment_id, segment_id, final_timeline, "
                        "final_edl, edl, or draft_content."
                    ),
                },
                {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        if body_payload["thinking"]["type"] == "enabled":
            body_payload["reasoning_effort"] = str(getattr(self.provider, "reasoning_effort", "") or "high")
        else:
            body_payload["temperature"] = 0
        request = urllib.request.Request(
            str(getattr(self.provider, "base_url", "") or "https://api.deepseek.com/chat/completions"),
            data=json.dumps(body_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {getattr(self.provider, 'api_key', '')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.provider_called_count += 1
        try:
            with urllib.request.urlopen(request, timeout=int(getattr(self.provider, "timeout_s", 90) or 90)) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"DEEPSEEK_PRE_REVIEW_HTTP_{exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DEEPSEEK_PRE_REVIEW_NETWORK_ERROR: {exc}") from exc
        envelope = json.loads(raw)
        content = str(((envelope.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        if not content.strip():
            raise RuntimeError("DEEPSEEK_PRE_REVIEW_EMPTY_RESPONSE")
        decoded = json.loads(content)
        if not isinstance(decoded, dict):
            raise RuntimeError("DEEPSEEK_PRE_REVIEW_RESPONSE_NOT_OBJECT")
        return decoded


def run_full_text_pre_review(
    run_dir: Path,
    *,
    provider: FullTextPreReviewProvider | None = None,
    output_dir: Path | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> PreReviewRunResult:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir or run_dir)
    payload = build_review_payload(run_dir, max_items=max_items, max_prompt_chars=max_prompt_chars)
    provider_metadata = _provider_metadata(provider)
    started = time.time()
    if provider is None:
        report = _base_report(
            payload,
            provider_metadata=provider_metadata,
            status="provider_missing",
            qc_recommendation="warning",
            error="DeepSeek full-text pre-review provider is not configured",
            started=started,
        )
    else:
        try:
            raw_response = provider.review(payload)
            report = _normalize_provider_response(
                raw_response,
                payload,
                provider_metadata=provider_metadata,
                started=started,
            )
        except (RuntimeError, ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            report = _base_report(
                payload,
                provider_metadata=provider_metadata,
                status="provider_failed",
                qc_recommendation="warning",
                error=str(exc),
                started=started,
            )
    hotspots = _hotspots_from_report(report, payload)
    triage = _triage_from_report(report, hotspots)
    report["triage_summary"] = triage.get("summary") or {}
    report["triage_report_path"] = "quality/full_text_pre_review_triage.json"
    markdown = _markdown_from_report(report, hotspots, triage)
    write_pre_review_outputs(output_dir, report, markdown, hotspots, triage)
    return PreReviewRunResult(report=report, markdown=markdown, hotspots=hotspots, triage=triage)


def build_review_payload(
    run_dir: Path,
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    source_graph = _safe_read_json(run_dir / "source_graph.json", {})
    final_timeline = _safe_read_json(run_dir / "final_timeline.json", [])
    captions = _safe_read_json(run_dir / "captions.json", [])
    run_summary = _safe_read_json(run_dir / "run_summary.json", {})
    quality_gate = _safe_read_json(run_dir / "quality_gate_report.json", {})
    visible_repeat_gate = _safe_read_json(run_dir / "final_caption_visible_repeat_gate.json", {})
    visible_repair = _safe_read_json(run_dir / "final_visible_caption_repair_report.json", {})
    semantic_report = _safe_read_json(run_dir / "semantic_adjudication_report.json", {})
    decision_trace = _safe_read_json(run_dir / "decision_trace.json", [])

    items = _review_items(final_timeline, captions, max_items=max_items)
    payload: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "report_version": REPORT_VERSION,
        "run_dir_name": run_dir.name,
        "run_artifact_hash": _run_artifact_hash(run_dir),
        "sidecar_only": True,
        "non_blocking": True,
        "audio_limitations": {
            "raw_audio_not_provided_to_llm": True,
            "audio_defects_require_local_signal": True,
        },
        "summary": _compact_summary(run_summary),
        "quality_signals": {
            "quality_gate": _compact_quality_payload(quality_gate),
            "visible_repeat_gate": _compact_quality_payload(visible_repeat_gate),
            "visible_repair": _compact_quality_payload(visible_repair),
            "semantic_adjudication": _compact_quality_payload(semantic_report),
            "decision_trace_count": len(decision_trace) if isinstance(decision_trace, list) else 0,
        },
        "original_transcript_excerpt": _original_transcript_excerpt(source_graph),
        "final_output_text": _joined_text(items),
        "review_items": items,
        "issue_rubric": _issue_rubric(),
    }
    payload = _fit_payload_to_prompt_budget(payload, max_prompt_chars=max_prompt_chars)
    return payload


def write_pre_review_outputs(
    output_dir: Path,
    report: dict[str, Any],
    markdown: str,
    hotspots: dict[str, Any],
    triage: dict[str, Any],
) -> None:
    quality_dir = Path(output_dir) / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "full_text_pre_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        "utf-8",
    )
    (quality_dir / "full_text_pre_review.md").write_text(markdown, "utf-8")
    (quality_dir / "full_text_pre_review_hotspots.json").write_text(
        json.dumps(hotspots, ensure_ascii=False, indent=2),
        "utf-8",
    )
    (quality_dir / "full_text_pre_review_triage.json").write_text(
        json.dumps(triage, ensure_ascii=False, indent=2),
        "utf-8",
    )
    (quality_dir / "full_text_pre_review_triage.md").write_text(
        _markdown_from_triage(triage),
        "utf-8",
    )


def _provider_response_schema() -> dict[str, Any]:
    return {
        "qc_recommendation": "pass|warning|block_candidate",
        "summary": "short Chinese summary",
        "issues": [
            {
                "issue_type": "one allowed issue type",
                "severity": "info|warning|fatal_candidate",
                "confidence": 0.0,
                "review_item_ids": ["review_item_id from payload.review_items"],
                "evidence_text": "short quoted evidence, not a physical edit field",
                "reason": "why this needs human QC",
                "suggested_human_qc": "what the editor should listen/watch for",
            }
        ],
    }


def _review_items(final_timeline: Any, captions: Any, *, max_items: int) -> list[dict[str, Any]]:
    caption_texts_by_segment = _caption_texts_by_segment(captions)
    source_rows = final_timeline if isinstance(final_timeline, list) and final_timeline else captions
    source_kind = "final_timeline" if source_rows is final_timeline else "captions"
    items: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows if isinstance(source_rows, list) else []):
        if not isinstance(row, dict):
            continue
        text = _clean_text(row.get("text"))
        if not text:
            continue
        item: dict[str, Any] = {
            "review_item_id": f"review_{len(items):04d}",
            "source_kind": source_kind,
            "ordinal": len(items),
            "text": _clip_text(text, 360),
        }
        caption_texts = caption_texts_by_segment.get(str(row.get("segment_id") or ""))
        if caption_texts:
            item["visible_caption_count"] = len(caption_texts)
            item["visible_caption_texts"] = [_clip_text(text, 220) for text in caption_texts[:20]]
            item["visible_caption_texts_truncated"] = len(caption_texts) > 20
            item["visible_caption_text_sequence"] = _clip_text(" / ".join(caption_texts), 1600)
        if len(items) > 0:
            item["previous_text"] = _clip_text(items[-1].get("text", ""), 160)
        items.append(item)
        if len(items) >= max(1, int(max_items or DEFAULT_MAX_ITEMS)):
            break
    return items


def _caption_texts_by_segment(captions: Any) -> dict[str, list[str]]:
    by_segment: dict[str, list[str]] = {}
    for row in captions if isinstance(captions, list) else []:
        if not isinstance(row, dict):
            continue
        text = _clean_text(row.get("text"))
        if not text:
            continue
        for segment_id in row.get("timeline_segment_ids") or []:
            by_segment.setdefault(str(segment_id), []).append(text)
    return by_segment


def _fit_payload_to_prompt_budget(payload: dict[str, Any], *, max_prompt_chars: int) -> dict[str, Any]:
    budget = max(20_000, int(max_prompt_chars or DEFAULT_MAX_PROMPT_CHARS))
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    while len(json.dumps(payload, ensure_ascii=False)) > budget and len(payload.get("review_items") or []) > 20:
        items = list(payload.get("review_items") or [])
        payload["review_items"] = items[: max(20, int(len(items) * 0.85))]
        payload["final_output_text"] = _joined_text(payload["review_items"])
        payload["truncated_for_prompt_budget"] = True
    payload.setdefault("truncated_for_prompt_budget", False)
    payload["review_item_count"] = len(payload.get("review_items") or [])
    return payload


def _normalize_provider_response(
    response: dict[str, Any],
    payload: dict[str, Any],
    *,
    provider_metadata: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    sanitized, dropped = _drop_forbidden_fields(response)
    item_ids = {str(item.get("review_item_id") or "") for item in payload.get("review_items") or []}
    issues: list[dict[str, Any]] = []
    for raw_issue in sanitized.get("issues") or []:
        if not isinstance(raw_issue, dict):
            continue
        issue_type = str(raw_issue.get("issue_type") or "other")
        if issue_type not in ALLOWED_ISSUE_TYPES:
            issue_type = "other"
        severity = str(raw_issue.get("severity") or "warning")
        if severity not in ALLOWED_SEVERITIES:
            severity = "warning"
        review_item_ids = [
            str(item_id)
            for item_id in raw_issue.get("review_item_ids") or []
            if str(item_id) in item_ids
        ]
        issues.append(
            {
                "issue_id": f"FTPR-{len(issues) + 1:03d}",
                "issue_type": issue_type,
                "severity": severity,
                "confidence": _bounded_float(raw_issue.get("confidence"), default=0.5),
                "review_item_ids": review_item_ids,
                "evidence_text": _clip_text(_clean_text(raw_issue.get("evidence_text")), 240),
                "reason": _clip_text(_clean_text(raw_issue.get("reason")), 360),
                "suggested_human_qc": _clip_text(_clean_text(raw_issue.get("suggested_human_qc")), 360),
            }
        )
    qc_recommendation = str(sanitized.get("qc_recommendation") or "warning")
    if qc_recommendation not in ALLOWED_QC_RECOMMENDATIONS:
        qc_recommendation = "warning"
    report = _base_report(
        payload,
        provider_metadata=provider_metadata,
        status="ok",
        qc_recommendation=qc_recommendation,
        error="",
        started=started,
    )
    report["provider_summary"] = _clip_text(_clean_text(sanitized.get("summary")), 800)
    report["issues"] = issues
    report["issue_count"] = len(issues)
    report["issue_count_by_severity"] = _count_by_key(issues, "severity")
    report["issue_count_by_type"] = _count_by_key(issues, "issue_type")
    report["provider_forbidden_fields_dropped"] = sorted(set(dropped))
    return report


def _base_report(
    payload: dict[str, Any],
    *,
    provider_metadata: dict[str, Any],
    status: str,
    qc_recommendation: str,
    error: str,
    started: float,
) -> dict[str, Any]:
    return {
        "report_kind": REPORT_KIND,
        "report_version": REPORT_VERSION,
        "status": status,
        "sidecar_only": True,
        "non_blocking": True,
        "does_not_change_ready_gate": True,
        "does_not_write_draft": True,
        "qc_recommendation": qc_recommendation,
        "error": error,
        "provider": provider_metadata,
        "run_dir_name": payload.get("run_dir_name"),
        "run_artifact_hash": payload.get("run_artifact_hash"),
        "review_item_count": payload.get("review_item_count", len(payload.get("review_items") or [])),
        "truncated_for_prompt_budget": bool(payload.get("truncated_for_prompt_budget")),
        "audio_limitations": payload.get("audio_limitations") or {},
        "summary": payload.get("summary") or {},
        "issues": [],
        "issue_count": 0,
        "issue_count_by_severity": {},
        "issue_count_by_type": {},
        "elapsed_seconds": round(max(0.0, time.time() - started), 3),
    }


def _hotspots_from_report(report: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    by_id = {str(item.get("review_item_id") or ""): item for item in payload.get("review_items") or []}
    hotspots: list[dict[str, Any]] = []
    for issue in report.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        items = [
            {
                "review_item_id": item_id,
                "ordinal": by_id.get(item_id, {}).get("ordinal"),
                "text": by_id.get(item_id, {}).get("text", ""),
                "visible_caption_count": by_id.get(item_id, {}).get("visible_caption_count", 0),
                "visible_caption_texts": by_id.get(item_id, {}).get("visible_caption_texts", []),
                "visible_caption_texts_truncated": by_id.get(item_id, {}).get("visible_caption_texts_truncated", False),
                "visible_caption_text_sequence": by_id.get(item_id, {}).get("visible_caption_text_sequence", ""),
            }
            for item_id in issue.get("review_item_ids") or []
            if item_id in by_id
        ]
        hotspots.append(
            {
                "issue_id": issue.get("issue_id"),
                "issue_type": issue.get("issue_type"),
                "severity": issue.get("severity"),
                "confidence": issue.get("confidence"),
                "review_items": items,
                "suggested_human_qc": issue.get("suggested_human_qc"),
            }
        )
    return {
        "report_kind": REPORT_KIND + "_hotspots",
        "report_version": REPORT_VERSION,
        "sidecar_only": True,
        "non_blocking": True,
        "hotspot_count": len(hotspots),
        "hotspots": hotspots,
    }


TRIAGE_BUCKETS = {
    "deterministic_rule_backlog": "Candidate for future local deterministic repair rule; never mutate directly from provider output.",
    "human_audio_review": "Needs human listening or local audio evidence before any edit decision.",
    "asr_or_text_mismatch_review": "Likely ASR, wording, or visible-caption wording review; do not auto-rewrite.",
    "pre_review_false_positive_candidate": "Likely provider overreach or issue-type mismatch; lower priority unless confirmed by human QC.",
    "informational_only": "Keep as context only.",
}
DETERMINISTIC_RULE_BACKLOG_TYPES = {"bad_repeat", "stutter_or_restart_not_removed"}
HUMAN_AUDIO_REVIEW_TYPES = {
    "untrimmed_non_speech",
    "large_breath_gap",
    "broken_word_or_truncated_syllable",
    "semantic_jump",
    "caption_fragment",
}
ASR_OR_TEXT_MISMATCH_TYPES = {"visible_caption_mismatch", "other"}


def _triage_from_report(report: dict[str, Any], hotspots: dict[str, Any]) -> dict[str, Any]:
    hotspot_by_issue_id = {
        str(row.get("issue_id") or ""): row
        for row in hotspots.get("hotspots") or []
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for issue in report.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        issue_id = str(issue.get("issue_id") or "")
        hotspot = hotspot_by_issue_id.get(issue_id, {})
        bucket, tags = _triage_bucket_and_tags(issue, hotspot)
        rows.append(
            {
                "issue_id": issue_id,
                "issue_type": str(issue.get("issue_type") or ""),
                "severity": str(issue.get("severity") or ""),
                "confidence": _bounded_float(issue.get("confidence"), default=0.0),
                "triage_bucket": bucket,
                "triage_bucket_description": TRIAGE_BUCKETS.get(bucket, ""),
                "tags": tags,
                "automation_policy": _triage_automation_policy(bucket),
                "review_item_ids": list(issue.get("review_item_ids") or []),
                "evidence_text": str(issue.get("evidence_text") or ""),
                "reason": str(issue.get("reason") or ""),
                "suggested_human_qc": str(issue.get("suggested_human_qc") or ""),
                "review_items": _triage_review_items(hotspot),
            }
        )
    return {
        "report_kind": REPORT_KIND + "_triage",
        "report_version": REPORT_VERSION,
        "sidecar_only": True,
        "non_blocking": True,
        "does_not_change_ready_gate": True,
        "does_not_write_draft": True,
        "automated_timeline_mutation_allowed": False,
        "block_candidate_is_qc_only": True,
        "triage_policy": {
            "deterministic_rule_backlog_requires_code_change_and_regression_tests": True,
            "human_audio_review_requires_editor_or_local_audio_evidence": True,
            "asr_or_text_mismatch_review_does_not_auto_rewrite": True,
            "pre_review_false_positive_candidate_lowers_priority": True,
        },
        "source_report": {
            "status": report.get("status"),
            "qc_recommendation": report.get("qc_recommendation"),
            "issue_count": report.get("issue_count", len(rows)),
            "issue_count_by_type": report.get("issue_count_by_type") or {},
            "issue_count_by_severity": report.get("issue_count_by_severity") or {},
        },
        "summary": {
            "triage_item_count": len(rows),
            "triage_count_by_bucket": _count_by_key(rows, "triage_bucket"),
            "triage_count_by_severity": _count_by_key(rows, "severity"),
            "deterministic_rule_backlog_count": sum(1 for row in rows if row.get("triage_bucket") == "deterministic_rule_backlog"),
            "human_audio_review_count": sum(1 for row in rows if row.get("triage_bucket") == "human_audio_review"),
            "asr_or_text_mismatch_review_count": sum(1 for row in rows if row.get("triage_bucket") == "asr_or_text_mismatch_review"),
            "pre_review_false_positive_candidate_count": sum(
                1 for row in rows if row.get("triage_bucket") == "pre_review_false_positive_candidate"
            ),
        },
        "triage_items": rows,
    }


def _triage_bucket_and_tags(issue: dict[str, Any], hotspot: dict[str, Any]) -> tuple[str, list[str]]:
    issue_type = str(issue.get("issue_type") or "")
    severity = str(issue.get("severity") or "")
    confidence = _bounded_float(issue.get("confidence"), default=0.0)
    tags: list[str] = []
    if severity == "fatal_candidate":
        tags.append("provider_fatal_candidate")
    if confidence >= 0.85:
        tags.append("high_provider_confidence")
    if not issue.get("review_item_ids"):
        tags.append("no_valid_review_item_id")
        return "pre_review_false_positive_candidate", tags
    if issue_type == "rhetorical_repeat_false_positive":
        tags.append("rhetorical_repeat_should_not_auto_drop")
        return "pre_review_false_positive_candidate", tags
    if issue_type == "visible_caption_mismatch":
        if _all_review_items_match_visible_caption_sequence(hotspot):
            tags.append("visible_caption_sequence_matches_final_text")
            tags.append("issue_type_likely_overstated")
        return "asr_or_text_mismatch_review", tags
    if issue_type in DETERMINISTIC_RULE_BACKLOG_TYPES:
        tags.append("derive_local_rule_only")
        tags.append("regression_test_required")
        return "deterministic_rule_backlog", tags
    if issue_type in HUMAN_AUDIO_REVIEW_TYPES:
        tags.append("audio_confirmation_required")
        if issue_type in {"caption_fragment", "semantic_jump", "broken_word_or_truncated_syllable"}:
            tags.append("local_rule_backlog_possible_after_confirmation")
        return "human_audio_review", tags
    if issue_type in ASR_OR_TEXT_MISMATCH_TYPES:
        tags.append("wording_review_required")
        return "asr_or_text_mismatch_review", tags
    return "informational_only", tags


def _triage_automation_policy(bucket: str) -> str:
    if bucket == "deterministic_rule_backlog":
        return "collect_as_rule_backlog_only_no_direct_edit"
    if bucket == "human_audio_review":
        return "human_or_local_audio_confirmation_required_before_rule_design"
    if bucket == "asr_or_text_mismatch_review":
        return "manual_wording_review_no_auto_rewrite"
    if bucket == "pre_review_false_positive_candidate":
        return "lower_priority_verify_before_action"
    return "informational_no_action"


def _triage_review_items(hotspot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in hotspot.get("review_items") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "review_item_id": item.get("review_item_id"),
                "ordinal": item.get("ordinal"),
                "text": item.get("text", ""),
                "visible_caption_count": item.get("visible_caption_count", 0),
                "visible_caption_text_sequence": item.get("visible_caption_text_sequence", ""),
                "visible_caption_texts_truncated": bool(item.get("visible_caption_texts_truncated")),
            }
        )
    return rows


def _all_review_items_match_visible_caption_sequence(hotspot: dict[str, Any]) -> bool:
    items = [item for item in hotspot.get("review_items") or [] if isinstance(item, dict)]
    if not items:
        return False
    for item in items:
        text = _normalized_comparison_text(str(item.get("text") or ""))
        sequence = _normalized_comparison_text(
            str(item.get("visible_caption_text_sequence") or "".join(item.get("visible_caption_texts") or []))
        )
        if not text or not sequence or text != sequence:
            return False
    return True


def _normalized_comparison_text(text: str) -> str:
    return "".join(char for char in str(text or "") if char.isalnum() or "\u4e00" <= char <= "\u9fff").lower()


def _markdown_from_report(report: dict[str, Any], hotspots: dict[str, Any], triage: dict[str, Any]) -> str:
    lines = [
        "# A-Roll Full Text Pre Review",
        "",
        f"- status: {report.get('status')}",
        f"- qc_recommendation: {report.get('qc_recommendation')}",
        f"- sidecar_only: {str(report.get('sidecar_only')).lower()}",
        f"- non_blocking: {str(report.get('non_blocking')).lower()}",
        f"- review_item_count: {report.get('review_item_count')}",
        "",
    ]
    if report.get("provider_summary"):
        lines.extend(["## Summary", "", str(report["provider_summary"]), ""])
    if report.get("error"):
        lines.extend(["## Error", "", str(report["error"]), ""])
    lines.extend(["## Triage", ""])
    summary = triage.get("summary") or {}
    for bucket, count in (summary.get("triage_count_by_bucket") or {}).items():
        lines.append(f"- {bucket}: {count}")
    lines.append("")
    lines.extend(["## Issues", ""])
    if not report.get("issues"):
        lines.append("- No provider issues returned.")
    for issue in report.get("issues") or []:
        item_ids = ", ".join(str(item_id) for item_id in issue.get("review_item_ids") or [])
        lines.append(
            f"- {issue.get('issue_id')} [{issue.get('severity')}/{issue.get('issue_type')}] "
            f"{issue.get('evidence_text')} ({item_ids})"
        )
        if issue.get("reason"):
            lines.append(f"  reason: {issue.get('reason')}")
        if issue.get("suggested_human_qc"):
            lines.append(f"  qc: {issue.get('suggested_human_qc')}")
    lines.extend(["", "## Hotspots", ""])
    for hotspot in hotspots.get("hotspots") or []:
        lines.append(f"- {hotspot.get('issue_id')}: {len(hotspot.get('review_items') or [])} review item(s)")
    lines.append("")
    return "\n".join(lines)


def _markdown_from_triage(triage: dict[str, Any]) -> str:
    lines = [
        "# A-Roll Full Text Pre Review Triage",
        "",
        f"- sidecar_only: {str(triage.get('sidecar_only')).lower()}",
        f"- non_blocking: {str(triage.get('non_blocking')).lower()}",
        f"- automated_timeline_mutation_allowed: {str(triage.get('automated_timeline_mutation_allowed')).lower()}",
        f"- block_candidate_is_qc_only: {str(triage.get('block_candidate_is_qc_only')).lower()}",
        "",
        "## Summary",
        "",
    ]
    summary = triage.get("summary") or {}
    for key in (
        "triage_item_count",
        "deterministic_rule_backlog_count",
        "human_audio_review_count",
        "asr_or_text_mismatch_review_count",
        "pre_review_false_positive_candidate_count",
    ):
        lines.append(f"- {key}: {summary.get(key, 0)}")
    lines.extend(["", "## Items", ""])
    for item in triage.get("triage_items") or []:
        lines.append(
            f"- {item.get('issue_id')} [{item.get('severity')}/{item.get('issue_type')}] "
            f"{item.get('triage_bucket')}: {item.get('evidence_text')}"
        )
        if item.get("automation_policy"):
            lines.append(f"  automation_policy: {item.get('automation_policy')}")
        if item.get("tags"):
            lines.append(f"  tags: {', '.join(str(tag) for tag in item.get('tags') or [])}")
    lines.append("")
    return "\n".join(lines)


def _safe_read_json(path: Path, default: Any) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError, EOFError, json.JSONDecodeError):
        return default


def _compact_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        empty: dict[str, Any] = {}
        return empty
    keys = [
        "status",
        "READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT",
        "write_status",
        "writeback_success",
        "final_video_segment_count",
        "caption_count",
        "semantic_unresolved_count",
        "final_repeat_convergence_gate_passed",
        "final_caption_visible_repeat_gate_passed",
        "quality_gate_passed",
        "visual_pacing_gate_passed",
        "caption_alignment_gate_passed",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def _compact_quality_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        empty: dict[str, Any] = {}
        return empty
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if key.endswith("_count") or key.endswith("_passed") or key in {
            "status",
            "gate_passed",
            "blocked",
            "write_allowed",
            "semantic_unresolved_count",
        }:
            compact[key] = value
    return compact


def _original_transcript_excerpt(source_graph: Any) -> str:
    words = source_graph.get("words") if isinstance(source_graph, dict) else []
    text = "".join(_clean_text(row.get("text")) for row in words if isinstance(row, dict))
    return _clip_text(text, 6000)


def _joined_text(items: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item.get('review_item_id')}: {item.get('text')}" for item in items)


def _run_artifact_hash(run_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "run_summary.json",
        "final_timeline.json",
        "final_timeline.json.gz",
        "captions.json",
        "captions.json.gz",
        "quality_gate_report.json",
    ):
        path = run_dir / name
        if path.exists() and path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _issue_rubric() -> list[dict[str, str]]:
    return [
        {"issue_type": "untrimmed_non_speech", "meaning": "清嗓子、咳嗽等非表达内容疑似未剪掉；无音频证据时只能标记为人工复听"},
        {"issue_type": "large_breath_gap", "meaning": "一句话内部疑似留下过长气口或节奏断裂"},
        {"issue_type": "broken_word_or_truncated_syllable", "meaning": "词首或词尾疑似被切掉，导致听感像缺字"},
        {"issue_type": "stutter_or_restart_not_removed", "meaning": "口吃、重启、忘词残留，且不是有效表达"},
        {"issue_type": "bad_repeat", "meaning": "机械重复或识别重复，非修辞强化"},
        {"issue_type": "rhetorical_repeat_false_positive", "meaning": "正确的重复强调被系统误判为应删除"},
        {"issue_type": "semantic_jump", "meaning": "上下句连接突兀，像中间被误删"},
        {"issue_type": "caption_fragment", "meaning": "字幕或片段明显不是完整表达"},
        {"issue_type": "visible_caption_mismatch", "meaning": "最终片段文本与可见字幕疑似不一致"},
    ]


def _drop_forbidden_fields(value: Any) -> tuple[Any, list[str]]:
    dropped: list[str] = []

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, child in node.items():
                key_text = str(key)
                if key_text in FORBIDDEN_PROVIDER_FIELDS:
                    dropped.append(key_text)
                    continue
                out[key_text] = walk(child)
            return out
        if isinstance(node, list):
            return [walk(child) for child in node]
        return node

    return walk(value), dropped


def _provider_metadata(provider: FullTextPreReviewProvider | None) -> dict[str, Any]:
    if provider is None:
        return {"provider_name": "none", "configured": False}
    deepseek = getattr(provider, "provider", None)
    return {
        "provider_name": str(getattr(provider, "provider_name", "unknown")),
        "configured": True,
        "model": str(getattr(deepseek, "model", "") or ""),
        "thinking_type": str(getattr(deepseek, "thinking_type", "") or ""),
        "reasoning_effort": str(getattr(deepseek, "reasoning_effort", "") or ""),
        "config_source": str(getattr(deepseek, "config_source", "") or ""),
    }


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split())


def _clip_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only A-Roll full-text pre-review from a V21 run directory.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--max-prompt-chars", type=int, default=DEFAULT_MAX_PROMPT_CHARS)
    parser.add_argument("--allow-missing-provider", action="store_true")
    args = parser.parse_args(argv)
    provider = DeepSeekFullTextPreReviewProvider.from_runtime_config()
    result = run_full_text_pre_review(
        args.run_dir,
        provider=provider,
        output_dir=args.output_dir,
        max_items=args.max_items,
        max_prompt_chars=args.max_prompt_chars,
    )
    print(
        json.dumps(
            {
                "status": result.report.get("status"),
                "qc_recommendation": result.report.get("qc_recommendation"),
                "issue_count": result.report.get("issue_count"),
                "triage_summary": result.triage.get("summary") or {},
                "output_dir": str(Path(args.output_dir or args.run_dir) / "quality"),
            },
            ensure_ascii=False,
        )
    )
    if result.report.get("status") == "provider_missing" and not args.allow_missing_provider:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
