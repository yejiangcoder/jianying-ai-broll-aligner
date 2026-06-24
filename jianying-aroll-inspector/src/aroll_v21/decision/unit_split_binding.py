from __future__ import annotations

from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import RepeatCluster, UnitSplitPlan

def _unit_split_binding(
    cluster: RepeatCluster,
    *,
    existing_splits: list[UnitSplitPlan] | None = None,
) -> dict[str, Any]:
    if not cluster.variants:
        return _missing_split_binding("cluster_has_no_unit")
    unit = cluster.variants[0]
    drop_texts = _unit_split_drop_texts(cluster)
    reused = _reuse_existing_unit_split(unit.unit_id, drop_texts, existing_splits or [])
    if reused is not None:
        return reused
    metadata_binding = _metadata_unit_split_binding(cluster)
    if metadata_binding is not None:
        return metadata_binding
    token_binding = _word_token_unit_split_binding(cluster, drop_texts)
    if token_binding is not None:
        return token_binding
    failed_reason = _first_split_failed_reason(cluster) or (
        "word_token_binding_no_safe_whole_word_binding"
        if drop_texts
        else "drop_text_missing_for_unit_split_binding"
    )
    return _missing_split_binding(failed_reason, drop_texts=drop_texts)


def _missing_split_binding(failed_reason: str, *, drop_texts: list[str] | None = None) -> dict[str, Any]:
    return {
        "drop_word_ids": [],
        "keep_word_ids": [],
        "drop_text": (drop_texts or [""])[0] if drop_texts else "",
        "normalized_drop_text": normalize_text((drop_texts or [""])[0]) if drop_texts else "",
        "binding": "missing",
        "binding_source": "",
        "failed_reason": failed_reason,
    }


def _reuse_existing_unit_split(
    unit_id: str,
    drop_texts: list[str],
    existing_splits: list[UnitSplitPlan],
) -> dict[str, Any] | None:
    normalized_targets: list[str] = []
    for text in drop_texts:
        normalized = normalize_text(text)
        if normalized and normalized not in normalized_targets:
            normalized_targets.append(normalized)
    if not normalized_targets:
        return None
    for split in existing_splits:
        if split.unit_id != unit_id:
            continue
        split_texts = _split_plan_drop_texts(split)
        split_normalized = {normalize_text(text) for text in split_texts if normalize_text(text)}
        matched_drop_text = next((text for text in normalized_targets if text in split_normalized), "")
        if not matched_drop_text:
            continue
        return {
            "drop_word_ids": list(split.drop_word_ids),
            "keep_word_ids": list(split.keep_word_ids),
            "drop_text": matched_drop_text,
            "normalized_drop_text": matched_drop_text,
            "binding": "whole_word",
            "binding_source": "reuse_existing_split_decision",
            "reused_split_id": split.split_id,
            "failed_reason": "",
        }
    return None


def _split_plan_drop_texts(split: UnitSplitPlan) -> list[str]:
    metadata = split.metadata if isinstance(split.metadata, dict) else {}
    values: list[str] = []
    for key in ("drop_text", "normalized_drop_text"):
        value = str(metadata.get(key) or "")
        if value:
            values.append(value)
    for value in metadata.get("drop_texts") or []:
        if str(value):
            values.append(str(value))
    return values


def _metadata_unit_split_binding(cluster: RepeatCluster) -> dict[str, Any] | None:
    if not cluster.variants:
        return None
    unit = cluster.variants[0]
    for evidence in cluster.evidence:
        metadata = evidence.metadata or {}
        drop_word_ids = [str(word_id) for word_id in metadata.get("split_drop_word_ids") or [] if str(word_id)]
        keep_word_ids = [str(word_id) for word_id in metadata.get("split_keep_word_ids") or [] if str(word_id)]
        if not drop_word_ids:
            for span in metadata.get("spans") or []:
                if not isinstance(span, dict) or span.get("source") != "word_audio_sequence":
                    continue
                start = int(span.get("start_token") or 0)
                size = int(span.get("token_ngram_size") or 0)
                if size > 0:
                    drop_word_ids = unit.word_ids[start : start + size]
                    keep_word_ids = [word_id for word_id in unit.word_ids if word_id not in set(drop_word_ids)]
                    break
        if not _safe_unit_split_ids(unit, drop_word_ids, keep_word_ids):
            continue
        drop_text = str(metadata.get("split_drop_text") or metadata.get("drop_text") or "")
        normalized_drop_text = normalize_text(str(metadata.get("normalized_drop_text") or drop_text))
        return {
            "drop_word_ids": drop_word_ids,
            "keep_word_ids": keep_word_ids,
            "drop_text": drop_text,
            "normalized_drop_text": normalized_drop_text,
            "binding": "whole_word",
            "binding_source": str(metadata.get("split_binding_source") or "metadata_split_word_ids"),
            "failed_reason": "",
        }
    return None


def _word_token_unit_split_binding(cluster: RepeatCluster, drop_texts: list[str]) -> dict[str, Any] | None:
    if not cluster.variants:
        return None
    unit = cluster.variants[0]
    for evidence in cluster.evidence:
        tokens = _metadata_word_tokens(evidence.metadata or {}, unit)
        if not tokens:
            continue
        for drop_text in drop_texts:
            binding = _bind_drop_text_to_whole_word_tokens(unit, tokens, drop_text)
            if binding is not None:
                binding["binding_source"] = str(binding.get("binding_source") or "source_word_token_binding")
                return binding
    return None


def _metadata_word_tokens(metadata: dict[str, Any], unit: Any) -> list[dict[str, str]]:
    raw_tokens = metadata.get("word_tokens") or metadata.get("unit_word_tokens") or metadata.get("source_word_tokens") or []
    if not isinstance(raw_tokens, list):
        return []
    by_id: dict[str, str] = {}
    for row in raw_tokens:
        if not isinstance(row, dict):
            continue
        word_id = str(row.get("word_id") or "")
        text = normalize_text(str(row.get("text") or row.get("word_text") or ""))
        if word_id and text:
            by_id[word_id] = text
    ordered = [{"word_id": word_id, "text": by_id[word_id]} for word_id in unit.word_ids if word_id in by_id]
    if len(ordered) != len(unit.word_ids):
        return []
    return ordered


def _bind_drop_text_to_whole_word_tokens(
    unit: Any,
    tokens: list[dict[str, str]],
    drop_text: str,
) -> dict[str, Any] | None:
    target = normalize_text(drop_text)
    if not target:
        return None
    exact_spans = _exact_whole_word_spans(tokens, target)
    if not exact_spans:
        return None
    for span in exact_spans:
        if any(other["start"] >= span["end"] for other in exact_spans):
            return _split_binding_from_span(unit, span, drop_text, "exact_repeated_ngram")
        if _later_prefix_repeat_exists(tokens, span, target):
            return _split_binding_from_span(unit, span, drop_text, "short_phrase_before_longer_prefix_word")
    return None


def _exact_whole_word_spans(tokens: list[dict[str, str]], target: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for start in range(len(tokens)):
        joined = ""
        word_ids: list[str] = []
        for end in range(start, len(tokens)):
            joined += normalize_text(tokens[end]["text"])
            word_ids.append(str(tokens[end]["word_id"]))
            if joined == target:
                spans.append({"start": start, "end": end + 1, "word_ids": list(word_ids)})
                break
            if len(joined) >= len(target) or not target.startswith(joined):
                break
    return spans


def _later_prefix_repeat_exists(tokens: list[dict[str, str]], span: dict[str, Any], target: str) -> bool:
    for start in range(int(span["end"]), len(tokens)):
        joined = ""
        for end in range(start, len(tokens)):
            joined += normalize_text(tokens[end]["text"])
            if joined == target:
                return True
            if joined.startswith(target):
                return True
            if not target.startswith(joined):
                break
    return False


def _split_binding_from_span(unit: Any, span: dict[str, Any], drop_text: str, binding_source: str) -> dict[str, Any]:
    drop_word_ids = [str(word_id) for word_id in span.get("word_ids") or [] if str(word_id)]
    keep_word_ids = [word_id for word_id in unit.word_ids if word_id not in set(drop_word_ids)]
    if not _safe_unit_split_ids(unit, drop_word_ids, keep_word_ids):
        return _missing_split_binding("word_token_binding_produced_unsafe_word_ids", drop_texts=[drop_text])
    return {
        "drop_word_ids": drop_word_ids,
        "keep_word_ids": keep_word_ids,
        "drop_text": drop_text,
        "normalized_drop_text": normalize_text(drop_text),
        "binding": "whole_word",
        "binding_source": binding_source,
        "failed_reason": "",
    }


def _safe_unit_split_ids(unit: Any, drop_word_ids: list[str], keep_word_ids: list[str]) -> bool:
    unit_word_ids = set(unit.word_ids)
    drop_set = set(drop_word_ids)
    keep_set = set(keep_word_ids)
    return bool(drop_set and keep_set and drop_set < unit_word_ids and keep_set <= unit_word_ids and not (drop_set & keep_set))


def _unit_split_drop_texts(cluster: RepeatCluster) -> list[str]:
    values: list[str] = []
    for evidence in cluster.evidence:
        metadata = evidence.metadata or {}
        for key in ("split_drop_text", "drop_text", "normalized_drop_text"):
            value = str(metadata.get(key) or "")
            if value:
                values.append(value)
        candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else {}
        for key in ("drop_text", "normalized_drop_text", "phrase", "overlap"):
            value = str(candidate.get(key) or "")
            if value:
                values.append(value)
        for span in metadata.get("spans") or []:
            if isinstance(span, dict):
                value = str(span.get("phrase") or span.get("overlap") or "")
                if value:
                    values.append(value)
    result: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in {normalize_text(item) for item in result}:
            result.append(value)
    return result


def _first_split_failed_reason(cluster: RepeatCluster) -> str:
    for evidence in cluster.evidence:
        metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
        reason = str(metadata.get("split_failed_reason") or metadata.get("failed_reason") or "")
        if reason:
            return reason
    return ""

