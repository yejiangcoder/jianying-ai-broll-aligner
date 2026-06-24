from __future__ import annotations

import json
import os
import re
import sys
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from aroll_runtime_paths import CONFIG_DIR, get_deepseek_config_path
from aroll_v21.decision.semantic_adjudication import (
    legacy_row_from_adjudication_decision,
    request_from_cluster,
)
from aroll_v21.decision.semantic_contracts import (
    SemanticAdjudicationDecision,
    SemanticAdjudicationDecisionType,
    SemanticAdjudicationProvider,
    SemanticAdjudicationRequest,
    semantic_contract_to_dict,
)
from aroll_v21.ir.models import RepeatCluster


FORBIDDEN_PROVIDER_FIELDS = {
    "source_start_us",
    "source_end_us",
    "target_start_us",
    "target_end_us",
    "material_id",
    "source_material_id",
    "source_segment_id",
    "segment_id",
    "final_timeline",
    "final_edl",
    "edl",
    "draft_content",
}

DEFAULT_DEEPSEEK_CONFIG_PATHS = (
    CONFIG_DIR / "deepseek.yaml",
)
LEGACY_REFERENCE_DEEPSEEK_CONFIG_ENV = "REFERENCE_VIDEO_DATA_CATCHER_DEEPSEEK_CONFIG_PATH"
DEFAULT_BATCH_MAX_ATTEMPTS = 3
DEFAULT_SINGLE_ATTEMPT_TIMEOUT_SECONDS = 180
DEFAULT_TOTAL_BATCH_TIMEOUT_SECONDS = 600
DEFAULT_BATCH_MAX_ISSUES = 6
DEFAULT_BATCH_MAX_PROMPT_CHARS = 120_000
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 180
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_THINKING_TYPE = "enabled"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "high"
SEMANTIC_BATCH_PROVIDER_FAILED_CODE = "V21_SEMANTIC_BATCH_PROVIDER_FAILED"
SEMANTIC_BATCH_PROVIDER_MISSING_CODE = "V21_SEMANTIC_BATCH_PROVIDER_MISSING"
SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE = "V21_SEMANTIC_BATCH_PARTIAL_RESPONSE"
SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE = "V21_SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW"
BATCH_METADATA_FIELDS = (
    "deepseek_batch_enabled",
    "deepseek_batch_request_count",
    "deepseek_batch_attempt_count",
    "deepseek_batch_retry_count",
    "deepseek_batch_issue_count",
    "deepseek_batch_resolved_count",
    "deepseek_batch_unresolved_count",
    "deepseek_batch_missing_issue_ids",
    "deepseek_batch_unknown_issue_ids",
    "deepseek_batch_error",
    "deepseek_batch_chunk_count",
    "deepseek_batch_chunk_sizes",
    "deepseek_batch_request",
    "deepseek_batch_response",
    "deepseek_batch_error_payload",
)


class _RetryableBatchError(RuntimeError):
    pass


class _NonRetryableBatchError(RuntimeError):
    pass


class DeepSeekSemanticProvider:
    provider_name = "deepseek_semantic_planner"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/chat/completions",
        model: str = DEFAULT_DEEPSEEK_MODEL,
        thinking_type: str = "",
        reasoning_effort: str = DEFAULT_DEEPSEEK_REASONING_EFFORT,
        timeout_s: int = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
        config_source: str = "",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        normalized = _normalize_deepseek_model_settings(model, thinking_type, reasoning_effort)
        self.model = normalized["model"]
        self.thinking_type = normalized["thinking_type"]
        self.reasoning_effort = normalized["reasoning_effort"]
        self.timeout_s = min(max(1, int(timeout_s or DEFAULT_DEEPSEEK_TIMEOUT_SECONDS)), DEFAULT_SINGLE_ATTEMPT_TIMEOUT_SECONDS)
        self.config_source = config_source
        self._reset_batch_state()

    def decide(self, requests: Sequence[SemanticAdjudicationRequest]) -> list[SemanticAdjudicationDecision]:
        self._reset_batch_state()
        if not requests:
            empty_decisions: list[SemanticAdjudicationDecision] = []
            return empty_decisions
        if not _is_configured_api_key(self.api_key):
            raise _NonRetryableBatchError("V21_SEMANTIC_BATCH_PROVIDER_MISSING: DeepSeek API key is not configured")
        issue_ids = [str(request.issue_id or "") for request in requests]
        duplicate_ids = sorted({issue_id for issue_id in issue_ids if issue_ids.count(issue_id) > 1})
        if not all(issue_ids) or duplicate_ids:
            raise _NonRetryableBatchError(f"V21_SEMANTIC_BATCH_SCHEMA_ERROR: issue_id must be non-empty and unique; duplicates={duplicate_ids}")
        chunks = self._chunk_requests(list(requests))
        self.deepseek_batch_enabled = True
        self.deepseek_batch_request_count = len(chunks)
        self.deepseek_batch_issue_count = len(requests)
        self.deepseek_batch_chunk_count = len(chunks)
        self.deepseek_batch_chunk_sizes = [len(chunk) for chunk in chunks]
        decisions: list[SemanticAdjudicationDecision] = []
        started = time.monotonic()
        for index, chunk in enumerate(chunks, start=1):
            if time.monotonic() - started > DEFAULT_TOTAL_BATCH_TIMEOUT_SECONDS:
                self.deepseek_batch_error = "V21_SEMANTIC_BATCH_PROVIDER_FAILED: total batch timeout exceeded"
                self.deepseek_batch_error_payload = {"error": self.deepseek_batch_error, "chunk_index": index}
                raise RuntimeError(self.deepseek_batch_error)
            decisions.extend(self._decide_chunk_with_retry(chunk, index=index))
        by_issue = {decision.issue_id: decision for decision in decisions if str(decision.issue_id or "")}
        missing = [issue_id for issue_id in issue_ids if issue_id not in by_issue]
        human_review = [
            decision.issue_id
            for decision in decisions
            if decision.decision == SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW or decision.requires_human_review
        ]
        self.deepseek_batch_missing_issue_ids = sorted(set([*self.deepseek_batch_missing_issue_ids, *missing]))
        self.deepseek_batch_resolved_count = len([issue_id for issue_id in issue_ids if issue_id in by_issue and issue_id not in set(human_review)])
        self.deepseek_batch_unresolved_count = len(set([*self.deepseek_batch_missing_issue_ids, *human_review]))
        return decisions

    def _reset_batch_state(self) -> None:
        self.provider_called_count = 0
        self.deepseek_batch_enabled = True
        self.deepseek_batch_request_count = 0
        self.deepseek_batch_attempt_count = 0
        self.deepseek_batch_retry_count = 0
        self.deepseek_batch_issue_count = 0
        self.deepseek_batch_resolved_count = 0
        self.deepseek_batch_unresolved_count = 0
        self.deepseek_batch_missing_issue_ids: list[str] = []
        self.deepseek_batch_unknown_issue_ids: list[str] = []
        self.deepseek_batch_error = ""
        self.deepseek_batch_chunk_count = 0
        self.deepseek_batch_chunk_sizes: list[int] = []
        self.deepseek_batch_request: dict[str, Any] = {}
        self.deepseek_batch_response: dict[str, Any] = {}
        self.deepseek_batch_error_payload: dict[str, Any] = {}

    def _chunk_requests(self, requests: list[SemanticAdjudicationRequest]) -> list[list[SemanticAdjudicationRequest]]:
        chunks: list[list[SemanticAdjudicationRequest]] = []
        current: list[SemanticAdjudicationRequest] = []
        current_chars = 0
        for request in requests:
            row_chars = len(json.dumps(self._issue_payload(request), ensure_ascii=False))
            if (
                current
                and (len(current) >= DEFAULT_BATCH_MAX_ISSUES or current_chars + row_chars > DEFAULT_BATCH_MAX_PROMPT_CHARS)
            ):
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(request)
            current_chars += row_chars
        if current:
            chunks.append(current)
        return chunks

    def _decide_chunk_with_retry(
        self,
        requests: list[SemanticAdjudicationRequest],
        *,
        index: int,
    ) -> list[SemanticAdjudicationDecision]:
        expected_issue_ids = [request.issue_id for request in requests]
        expected_batch_id = self._batch_id(requests, chunk_index=index)
        last_error = ""
        partial_decisions: list[SemanticAdjudicationDecision] = []
        for attempt in range(1, DEFAULT_BATCH_MAX_ATTEMPTS + 1):
            error_hint = last_error if attempt > 1 else ""
            self.deepseek_batch_attempt_count += 1
            if attempt > 1:
                self.deepseek_batch_retry_count += 1
            try:
                response = self._call_batch_once(requests, chunk_index=index, attempt=attempt, error_hint=error_hint)
                response_batch_id = str(response.get("batch_id") or "")
                if response_batch_id != expected_batch_id:
                    raise _RetryableBatchError(
                        f"schema validation failed: batch_id mismatch expected={expected_batch_id} actual={response_batch_id}"
                    )
                decisions = self._parse_batch_response(response)
                decisions = self._filter_and_validate_decisions(decisions, expected_issue_ids, requests)
                by_issue = {decision.issue_id: decision for decision in decisions}
                missing = [issue_id for issue_id in expected_issue_ids if issue_id not in by_issue]
                if missing:
                    partial_decisions = decisions
                    message = "response missing required issue_id: " + ",".join(missing)
                    if attempt < DEFAULT_BATCH_MAX_ATTEMPTS:
                        last_error = message
                        continue
                    self.deepseek_batch_missing_issue_ids = sorted(set([*self.deepseek_batch_missing_issue_ids, *missing]))
                    self.deepseek_batch_error = f"V21_SEMANTIC_BATCH_PARTIAL_RESPONSE: {message}"
                    self.deepseek_batch_error_payload = {
                        "error": self.deepseek_batch_error,
                        "missing_issue_ids": missing,
                        "chunk_index": index,
                        "attempt": attempt,
                    }
                    return partial_decisions
                return decisions
            except _RetryableBatchError as exc:
                last_error = str(exc)
                if attempt < DEFAULT_BATCH_MAX_ATTEMPTS:
                    continue
                self.deepseek_batch_error = f"V21_SEMANTIC_BATCH_PROVIDER_FAILED: {last_error}"
                self.deepseek_batch_error_payload = {
                    "error": self.deepseek_batch_error,
                    "chunk_index": index,
                    "attempt": attempt,
                }
                raise RuntimeError(self.deepseek_batch_error) from exc
            except _NonRetryableBatchError as exc:
                self.deepseek_batch_error = str(exc)
                self.deepseek_batch_error_payload = {
                    "error": self.deepseek_batch_error,
                    "chunk_index": index,
                    "attempt": attempt,
                }
                raise
        return partial_decisions

    def _call_batch_once(
        self,
        requests: list[SemanticAdjudicationRequest],
        *,
        chunk_index: int,
        attempt: int,
        error_hint: str = "",
    ) -> dict[str, Any]:
        batch_payload = self._batch_payload(requests, chunk_index=chunk_index, attempt=attempt, error_hint=error_hint)
        if not self.deepseek_batch_request:
            self.deepseek_batch_request = batch_payload
        else:
            existing_chunks = self.deepseek_batch_request.setdefault("chunks", [])
            if isinstance(existing_chunks, list):
                existing_chunks.append(batch_payload)
        payload = {
            "model": self.model,
            "thinking": {"type": self.thinking_type},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You adjudicate Chinese transcript semantic repeat issues in one batch. "
                        "Return JSON only using the requested batch_id and one decision per issue_id. "
                        "Do not output physical edit fields such as source_start_us, source_end_us, "
                        "target_start_us, target_end_us, material_id, segment_id, or draft content."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(batch_payload, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        if self.thinking_type == "enabled":
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = 0
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.provider_called_count += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise _NonRetryableBatchError(f"DEEPSEEK_SEMANTIC_AUTH_FAILED: HTTP {exc.code}") from exc
            if exc.code == 429 or 500 <= exc.code <= 599:
                raise _RetryableBatchError(f"DEEPSEEK_SEMANTIC_RETRYABLE_HTTP_{exc.code}") from exc
            raise _NonRetryableBatchError(f"DEEPSEEK_SEMANTIC_HTTP_{exc.code}") from exc
        except urllib.error.URLError as exc:
            raise _RetryableBatchError(f"DEEPSEEK_SEMANTIC_NETWORK_ERROR: {exc}") from exc
        if not raw.strip():
            raise _RetryableBatchError("empty response")
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _RetryableBatchError(f"JSON parse failed: {exc}") from exc
        content = str(((envelope.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        if not content.strip():
            raise _RetryableBatchError("empty response content")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise _RetryableBatchError(f"JSON parse failed: {exc}") from exc
        self.deepseek_batch_response = decoded if isinstance(decoded, dict) else {"raw": decoded}
        return self.deepseek_batch_response

    def _batch_payload(
        self,
        requests: list[SemanticAdjudicationRequest],
        *,
        chunk_index: int,
        attempt: int,
        error_hint: str = "",
    ) -> dict[str, Any]:
        batch_id = self._batch_id(requests, chunk_index=chunk_index)
        schema = {
            "batch_id": batch_id,
            "decisions": [
                {
                    "issue_id": "string",
                    "decision": "keep_all|drop_left|drop_right|keep_longest_drop_others|drop_recommended|drop_aborted|repair_text|requires_human_review|no_decision",
                    "decision_type": "one of allowed_actions for that issue",
                    "action": "same value as decision_type",
                    "confidence": 0.0,
                    "reason": "string",
                    "drop_side": None,
                    "drop_word_ids": [],
                    "keep_word_ids": [],
                    "requires_human_review": False,
                }
            ],
        }
        payload = {
            "batch_id": batch_id,
            "mode": "auto",
            "chunk_index": chunk_index,
            "attempt": attempt,
            "schema": schema,
            "issues": [self._issue_payload(request) for request in requests],
        }
        if error_hint:
            payload["previous_attempt_error"] = error_hint
        if attempt >= DEFAULT_BATCH_MAX_ATTEMPTS:
            payload["strict_schema_only"] = True
        return payload

    def _batch_id(self, requests: list[SemanticAdjudicationRequest], *, chunk_index: int) -> str:
        seed = "|".join(request.issue_id for request in requests)
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return f"semantic_batch_{digest}_{chunk_index:02d}"

    def _issue_payload(self, request: SemanticAdjudicationRequest) -> dict[str, Any]:
        candidate_text = str(request.local_context.get("candidate_text") or request.local_context.get("text") or "")
        if not candidate_text:
            candidate_text = str(request.text_before or request.text_after or "")
        return {
            "issue_id": request.issue_id,
            "issue_type": request.issue_type.value,
            "severity": request.severity.value,
            "candidate_text": candidate_text,
            "left_text": request.text_before,
            "right_text": request.text_after,
            "context_before": str(request.local_context.get("context_before") or ""),
            "context_after": str(request.local_context.get("context_after") or ""),
            "allowed_actions": list(request.allowed_decisions),
            "caption_ids": list(request.candidate_caption_ids),
            "segment_ids": list(request.candidate_segment_ids),
            "word_ids": list(request.word_ids),
            "recommended_action": request.recommended_action,
            "why_local_policy_cannot_decide": request.why_local_policy_cannot_decide,
            "local_context": dict(request.local_context),
        }

    def _parse_batch_response(self, response: dict[str, Any]) -> list[SemanticAdjudicationDecision]:
        rows = response.get("decisions")
        if not isinstance(rows, list):
            raise _RetryableBatchError("schema validation failed: decisions must be a list")
        decisions: list[SemanticAdjudicationDecision] = []
        for row in rows:
            if isinstance(row, dict):
                decisions.append(_decision_from_provider_row(row))
        return decisions

    def _filter_and_validate_decisions(
        self,
        decisions: list[SemanticAdjudicationDecision],
        expected_issue_ids: list[str],
        requests: list[SemanticAdjudicationRequest],
    ) -> list[SemanticAdjudicationDecision]:
        expected = set(expected_issue_ids)
        allowed_by_issue = {request.issue_id: set(request.allowed_decisions) for request in requests}
        filtered: list[SemanticAdjudicationDecision] = []
        unknown: list[str] = []
        seen: set[str] = set()
        for decision in decisions:
            issue_id = str(decision.issue_id or "")
            if issue_id not in expected:
                if issue_id:
                    unknown.append(issue_id)
                continue
            if issue_id in seen:
                raise _RetryableBatchError(f"schema validation failed: duplicate decision for issue_id={issue_id}")
            seen.add(issue_id)
            allowed = allowed_by_issue.get(issue_id) or set()
            if decision.decision.value not in allowed:
                raise _RetryableBatchError(
                    f"response action not in allowed_actions: issue_id={issue_id} action={decision.decision.value}"
                )
            filtered.append(decision)
        if unknown:
            self.deepseek_batch_unknown_issue_ids = sorted(set([*self.deepseek_batch_unknown_issue_ids, *unknown]))
        return filtered


class DeepSeekSemanticPlannerAdapter:
    """Adapter from the new request/decision provider contract to the legacy cluster planner shape."""

    provider_name = "deepseek_semantic_planner"

    def __init__(self, provider: SemanticAdjudicationProvider) -> None:
        self.provider = provider
        self.rows: list[dict[str, Any]] = []
        self.request_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.provider_called_count = 0
        self.deepseek_provider_configured = True
        self.deepseek_provider_config_source = str(getattr(provider, "config_source", "") or "")
        self.deepseek_provider_error = ""
        self.commit_reused_semantic_cache = bool(getattr(provider, "commit_reused_semantic_cache", False))
        self.semantic_cache_input_hash = str(getattr(provider, "semantic_cache_input_hash", "") or "")
        self.semantic_cache_issue_count = int(getattr(provider, "semantic_cache_issue_count", 0) or 0)
        self.semantic_cache_resolved_count = int(getattr(provider, "semantic_cache_resolved_count", 0) or 0)
        self.semantic_cache_unresolved_count = int(getattr(provider, "semantic_cache_unresolved_count", 0) or 0)
        for field_name in BATCH_METADATA_FIELDS:
            setattr(self, field_name, getattr(provider, field_name, _batch_metadata_default(field_name)))

    def decide(self, clusters: list[RepeatCluster]) -> list[dict[str, Any]]:
        requests = [request_from_cluster(cluster) for cluster in clusters]
        self.request_rows = [semantic_contract_to_dict(request) for request in requests]
        provider_called_before = int(getattr(self.provider, "provider_called_count", 0) or 0)
        try:
            decisions = self.provider.decide(requests)
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
            self.deepseek_provider_error = str(exc)
            self._sync_provider_batch_metadata(provider_called_before)
            self.decision_rows = []
            self.rows = [
                {
                    "cluster_id": cluster.cluster_id,
                    "_blocker_code": SEMANTIC_BATCH_PROVIDER_FAILED_CODE,
                    "_severity": "write_blocker",
                    "_message": "DeepSeek provider failed while adjudicating semantic request",
                    "_provider_error": self.deepseek_provider_error,
                }
                for cluster in clusters
            ]
            return list(self.rows)
        self._sync_provider_batch_metadata(provider_called_before)
        self.deepseek_provider_error = ""
        decisions_by_issue = {decision.issue_id: decision for decision in decisions}
        rows: list[dict[str, Any]] = []
        for cluster in clusters:
            decision = decisions_by_issue.get(cluster.cluster_id)
            if decision is None:
                rows.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "_blocker_code": SEMANTIC_BATCH_PARTIAL_RESPONSE_CODE,
                        "_severity": "write_blocker",
                        "_message": "DeepSeek provider did not return a decision for this semantic request",
                    }
                )
                continue
            forbidden = _forbidden_provider_fields(semantic_contract_to_dict(decision))
            if forbidden:
                rows.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "_blocker_code": "DEEPSEEK_DECISION_HAS_PHYSICAL_FIELDS",
                        "_severity": "write_blocker",
                        "_message": "DeepSeek provider returned forbidden physical timeline/material fields",
                        "_forbidden_fields": forbidden,
                    }
                )
                continue
            if decision.decision == SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW or decision.requires_human_review:
                rows.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "_blocker_code": SEMANTIC_BATCH_REQUIRES_HUMAN_REVIEW_CODE,
                        "_severity": "write_blocker",
                        "_message": decision.reason or "DeepSeek provider returned requires_human_review",
                        "_decision": decision.decision.value,
                    }
                )
                continue
            rows.append(legacy_row_from_adjudication_decision(decision, cluster))
        self.decision_rows = [semantic_contract_to_dict(decision) for decision in decisions]
        self.rows = rows
        return rows

    def _sync_provider_batch_metadata(self, provider_called_before: int) -> None:
        provider_called_after = int(getattr(self.provider, "provider_called_count", provider_called_before) or 0)
        delta = max(0, provider_called_after - provider_called_before)
        if delta == 0 and self.request_rows:
            delta = 1
        self.provider_called_count += delta
        self.commit_reused_semantic_cache = bool(getattr(self.provider, "commit_reused_semantic_cache", self.commit_reused_semantic_cache))
        self.semantic_cache_input_hash = str(getattr(self.provider, "semantic_cache_input_hash", self.semantic_cache_input_hash) or "")
        for field_name in BATCH_METADATA_FIELDS:
            setattr(self, field_name, getattr(self.provider, field_name, getattr(self, field_name, _batch_metadata_default(field_name))))
        if not self.commit_reused_semantic_cache:
            self.semantic_cache_issue_count = int(getattr(self, "deepseek_batch_issue_count", 0) or 0)
            self.semantic_cache_resolved_count = int(getattr(self, "deepseek_batch_resolved_count", 0) or 0)
            self.semantic_cache_unresolved_count = int(getattr(self, "deepseek_batch_unresolved_count", 0) or 0)


def deepseek_provider_from_env() -> DeepSeekSemanticProvider | None:
    return deepseek_provider_from_runtime_config()


def deepseek_provider_from_runtime_config() -> DeepSeekSemanticProvider | None:
    api_key = str(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_TOKEN") or "").strip()
    base_url = str(os.environ.get("DEEPSEEK_API_URL") or "").strip()
    model = str(os.environ.get("DEEPSEEK_MODEL") or "").strip()
    timeout_s = _configured_timeout_s(None)
    config = _load_deepseek_config_from_files()
    if config is not None and _is_configured_api_key(config.get("api_key", "")):
        return DeepSeekSemanticProvider(
            api_key=config["api_key"],
            base_url=_chat_completions_url(config.get("base_url", "")),
            model=config.get("model") or DEFAULT_DEEPSEEK_MODEL,
            thinking_type=config.get("thinking_type") or str(os.environ.get("DEEPSEEK_THINKING_TYPE") or ""),
            reasoning_effort=config.get("reasoning_effort") or str(os.environ.get("DEEPSEEK_REASONING_EFFORT") or ""),
            timeout_s=_configured_timeout_s(config),
            config_source=config.get("config_source", ""),
        )
    if not _is_configured_api_key(api_key):
        no_provider: DeepSeekSemanticProvider | None = None
        return no_provider
    config_source = "env:DEEPSEEK_API_KEY" if os.environ.get("DEEPSEEK_API_KEY") else "env:DEEPSEEK_API_TOKEN"
    return DeepSeekSemanticProvider(
        api_key=api_key,
        base_url=_chat_completions_url(base_url or "https://api.deepseek.com/chat/completions"),
        model=model or DEFAULT_DEEPSEEK_MODEL,
        thinking_type=str(os.environ.get("DEEPSEEK_THINKING_TYPE") or ""),
        reasoning_effort=str(os.environ.get("DEEPSEEK_REASONING_EFFORT") or ""),
        timeout_s=timeout_s,
        config_source=config_source,
    )


def deepseek_provider_from_config_file(path: Path) -> DeepSeekSemanticProvider | None:
    try:
        config = _parse_deepseek_yaml_config(path)
    except OSError:
        no_provider: DeepSeekSemanticProvider | None = None
        return no_provider
    if not _is_configured_api_key(config.get("api_key", "")) or not config.get("base_url"):
        no_provider: DeepSeekSemanticProvider | None = None
        return no_provider
    return DeepSeekSemanticProvider(
        api_key=config["api_key"],
        base_url=_chat_completions_url(config["base_url"]),
        model=config.get("model") or DEFAULT_DEEPSEEK_MODEL,
        thinking_type=config.get("thinking_type") or str(os.environ.get("DEEPSEEK_THINKING_TYPE") or ""),
        reasoning_effort=config.get("reasoning_effort") or str(os.environ.get("DEEPSEEK_REASONING_EFFORT") or ""),
        timeout_s=_configured_timeout_s(config),
        config_source=path.name,
    )


def _load_deepseek_config_from_files(paths: Sequence[Path] | None = None) -> dict[str, str] | None:
    candidate_paths = _deepseek_config_candidate_paths(paths)
    for path in candidate_paths:
        try:
            if not path.exists() or not path.is_file():
                continue
            config = _parse_deepseek_yaml_config(path)
        except OSError:
            continue
        if _is_configured_api_key(config.get("api_key", "")) and config.get("base_url"):
            config["config_source"] = path.name
            return config
    no_config: dict[str, str] | None = None
    return no_config


def _deepseek_config_candidate_paths(paths: Sequence[Path] | None = None) -> list[Path]:
    candidates: list[Path] = []
    explicit_env = str(os.environ.get("DEEPSEEK_CONFIG_PATH") or "").strip()
    if explicit_env:
        candidates.append(Path(explicit_env))
    reference_env = str(os.environ.get(LEGACY_REFERENCE_DEEPSEEK_CONFIG_ENV) or "").strip()
    if reference_env:
        candidates.append(Path(reference_env))
    if paths is None:
        default_config_path = get_deepseek_config_path()
        if not _skip_real_project_deepseek_config_during_pytest(default_config_path):
            candidates.append(default_config_path)
        path_rows = [
            path
            for path in DEFAULT_DEEPSEEK_CONFIG_PATHS
            if not _skip_real_project_deepseek_config_during_pytest(path)
        ]
    else:
        path_rows = list(paths)
    candidates.extend(path_rows)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _skip_real_project_deepseek_config_during_pytest(path: Path) -> bool:
    if "pytest" not in sys.modules:
        return False
    if os.environ.get("DEEPSEEK_CONFIG_PATH"):
        return False
    try:
        return path.resolve() == (CONFIG_DIR / "deepseek.yaml").resolve()
    except OSError:
        return False


def _batch_metadata_default(field_name: str) -> Any:
    if field_name in {
        "deepseek_batch_enabled",
    }:
        return True
    if field_name in {
        "deepseek_batch_missing_issue_ids",
        "deepseek_batch_unknown_issue_ids",
        "deepseek_batch_chunk_sizes",
    }:
        return list()
    if field_name in {
        "deepseek_batch_request",
        "deepseek_batch_response",
        "deepseek_batch_error_payload",
    }:
        return dict()
    if field_name == "deepseek_batch_error":
        return ""
    return 0


def _parse_deepseek_yaml_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    lines = path.read_text("utf-8").splitlines()
    in_deepseek = False
    deepseek_indent = -1
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip().lstrip("\ufeff")
        if stripped == "deepseek:":
            in_deepseek = True
            deepseek_indent = indent
            continue
        if in_deepseek and indent <= deepseek_indent and re.match(r"^[\w.-]+:", stripped):
            in_deepseek = False
        if in_deepseek and ":" in stripped:
            key, raw_value = stripped.split(":", 1)
            normalized_key = key.strip().replace("-", "_")
            if normalized_key in {"api_key", "base_url", "model", "thinking_type", "thinking", "reasoning_effort", "timeout_s", "timeout"}:
                if normalized_key == "thinking":
                    normalized_key = "thinking_type"
                if normalized_key == "timeout":
                    normalized_key = "timeout_s"
                values[normalized_key] = _strip_yaml_scalar(raw_value)
    if not values.get("model"):
        models = [
            _strip_yaml_scalar(match.group(1))
            for match in re.finditer(r"(?m)^\s*model\s*:\s*(.+?)\s*$", "\n".join(lines))
        ]
        if DEFAULT_DEEPSEEK_MODEL in models:
            values["model"] = DEFAULT_DEEPSEEK_MODEL
        elif "deepseek-v4-flash" in models:
            values["model"] = "deepseek-v4-flash"
        elif "deepseek-chat" in models:
            values["model"] = DEFAULT_DEEPSEEK_MODEL
            values.setdefault("thinking_type", "disabled")
        elif "deepseek-reasoner" in models:
            values["model"] = DEFAULT_DEEPSEEK_MODEL
            values.setdefault("thinking_type", "enabled")
        elif models:
            values["model"] = models[-1]
    return values


def _is_configured_api_key(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and "REPLACE_WITH" not in text


def _strip_yaml_scalar(value: str) -> str:
    text = value.strip()
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return text.strip()


def _configured_timeout_s(config: dict[str, str] | None) -> int:
    env_value = str(os.environ.get("DEEPSEEK_TIMEOUT_S") or "").strip()
    if env_value:
        return _safe_timeout_s(env_value, DEFAULT_DEEPSEEK_TIMEOUT_SECONDS)
    if config is not None:
        config_value = str(config.get("timeout_s") or "").strip()
        if config_value:
            return _safe_timeout_s(config_value, DEFAULT_DEEPSEEK_TIMEOUT_SECONDS)
    return DEFAULT_DEEPSEEK_TIMEOUT_SECONDS


def _safe_timeout_s(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chat_completions_url(base_url: str) -> str:
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return "https://api.deepseek.com/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def _normalize_deepseek_model_settings(model: str, thinking_type: str = "", reasoning_effort: str = "") -> dict[str, str]:
    raw_model = str(model or "").strip() or DEFAULT_DEEPSEEK_MODEL
    raw_thinking = str(thinking_type or "").strip().lower()
    raw_effort = str(reasoning_effort or "").strip().lower()

    if raw_model == "deepseek-chat":
        raw_model = DEFAULT_DEEPSEEK_MODEL
        raw_thinking = raw_thinking or "disabled"
    elif raw_model == "deepseek-reasoner":
        raw_model = DEFAULT_DEEPSEEK_MODEL
        raw_thinking = raw_thinking or "enabled"

    if raw_thinking not in {"enabled", "disabled"}:
        raw_thinking = DEFAULT_DEEPSEEK_THINKING_TYPE
    if raw_effort not in {"high", "max"}:
        raw_effort = DEFAULT_DEEPSEEK_REASONING_EFFORT

    return {
        "model": raw_model,
        "thinking_type": raw_thinking,
        "reasoning_effort": raw_effort,
    }


def _decision_from_provider_row(row: dict[str, Any]) -> SemanticAdjudicationDecision:
    decision = str(
        row.get("decision")
        or row.get("decision_type")
        or row.get("action")
        or SemanticAdjudicationDecisionType.NO_DECISION.value
    )
    if decision not in {item.value for item in SemanticAdjudicationDecisionType}:
        decision = SemanticAdjudicationDecisionType.NO_DECISION.value
    return SemanticAdjudicationDecision(
        issue_id=str(row.get("issue_id") or row.get("cluster_id") or ""),
        decision=SemanticAdjudicationDecisionType(decision),
        reason=str(row.get("reason") or ""),
        confidence=float(row.get("confidence") or 0.0),
        provider_name=str(row.get("provider_name") or "deepseek_semantic_planner"),
        keep_unit_id=str(row.get("keep_unit_id") or ""),
        drop_unit_ids=[str(item) for item in row.get("drop_unit_ids") or [] if str(item)],
        unit_id=str(row.get("unit_id") or ""),
        drop_word_ids=[str(item) for item in row.get("drop_word_ids") or [] if str(item)],
        keep_word_ids=[str(item) for item in row.get("keep_word_ids") or [] if str(item)],
        repair_text=str(row.get("repair_text") or ""),
        requires_human_review=bool(row.get("requires_human_review")),
        metadata={
            key: value
            for key, value in row.items()
            if key not in {"issue_id", "cluster_id", "decision", "decision_type", "action", "reason", "confidence"}
        },
    )


def _forbidden_provider_fields(value: Any) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                text = str(key)
                if text in FORBIDDEN_PROVIDER_FIELDS:
                    found.add(text)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)
