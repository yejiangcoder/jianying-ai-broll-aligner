from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.quality.boundary_overlap import is_explanatory_term_reuse, is_semantic_label_reuse_boundary
from aroll_v21.quality.pre_visible_semantic_junk_candidate_detector import MIN_HIGH_CONFIDENCE


MAX_HIDDEN_AUDIO_PREFIX_CHARS = 3
MAX_HIDDEN_AUDIO_PREFIX_GAP_US = 900_000
MIN_RESTARTED_WORD_CHARS = 2
MAX_VISIBLE_PREFIX_CHARS = 3
MAX_VISIBLE_INTERNAL_RESTART_OFFSET = 5
MIN_VISIBLE_RESTART_REMAINDER_CHARS = 4
MAX_ADJACENT_RESTART_FRAGMENT_CHARS = 10
TRIVIAL_SINGLE_WORDS = {"的", "了", "呢", "啊", "吗", "吧", "个", "一", "是", "在"}
LEXICAL_REDUPLICATION_BOUNDARY_WORDS = ("上", "里", "中", "下", "的", "款", "版", "版型")
LEXICAL_REDUPLICATION_PREFIX_STOPWORDS = (
    "你",
    "我",
    "他",
    "她",
    "它",
    "这",
    "那",
    "就",
    "也",
    "还",
    "都",
    "不",
    "没",
    "很",
    "更",
    "再",
    "又",
)


def build_run_quality_candidate_audit(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    captions = _read_json_file(run_dir / "captions.json") or []
    final_timeline = _read_json_file(run_dir / "final_timeline.json") or []
    source_graph = _read_json_file(run_dir / "source_graph.json") or _read_gzip_json_file(run_dir / "source_graph.json.gz") or {}
    pre_report = _read_json_file(run_dir / "quality" / "pre_visible_semantic_junk_report.json") or {}
    return build_quality_candidate_audit_from_artifacts(
        run_name=run_dir.name,
        run_dir=str(run_dir),
        captions=captions,
        final_timeline=final_timeline,
        source_graph=source_graph,
        pre_visible_semantic_junk_report=pre_report,
    )


def build_quality_candidate_audit_from_artifacts(
    *,
    run_name: str,
    run_dir: str,
    captions: list[dict[str, Any]],
    final_timeline: list[dict[str, Any]],
    source_graph: dict[str, Any],
    pre_visible_semantic_junk_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    words_by_id = {str(row.get("word_id") or ""): row for row in list(source_graph.get("words") or [])}
    caption_index = _caption_index_by_word_id(captions)
    selected_words = _selected_timeline_words(final_timeline, words_by_id)
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        _pre_visible_semantic_junk_candidates(
            run_name=run_name,
            run_dir=run_dir,
            report=pre_visible_semantic_junk_report or {},
        )
    )
    candidates.extend(
        _hidden_audio_prefix_restart_candidates(
            run_name=run_name,
            run_dir=run_dir,
            selected_words=selected_words,
            caption_index=caption_index,
        )
    )
    candidates.extend(
        _intraword_audio_restart_normalization_candidates(
            run_name=run_name,
            run_dir=run_dir,
            selected_words=selected_words,
            caption_index=caption_index,
        )
    )
    candidates.extend(
        _visible_internal_prefix_restart_candidates(
            run_name=run_name,
            run_dir=run_dir,
            captions=captions,
        )
    )
    candidates.extend(
        _adjacent_visible_restart_candidates(
            run_name=run_name,
            run_dir=run_dir,
            captions=captions,
        )
    )
    deduped = _dedupe_candidates(candidates)
    by_type: dict[str, int] = {}
    for candidate in deduped:
        by_type[str(candidate.get("type") or "")] = by_type.get(str(candidate.get("type") or ""), 0) + 1
    return {
        "audit_name": "aroll_quality_candidate_audit",
        "run_name": run_name,
        "run_dir": run_dir,
        "candidate_count": len(deduped),
        "candidate_counts_by_type": by_type,
        "candidates": deduped,
    }


def render_quality_candidate_audit_markdown(audits: list[dict[str, Any]]) -> str:
    lines = [
        "# A-Roll Quality Candidate Audit",
        "",
        "This audit is non-blocking and review-only. It lists generic quality candidates detected from runtime artifacts.",
        "",
    ]
    total = sum(int(audit.get("candidate_count") or 0) for audit in audits)
    lines.append(f"Total candidates: {total}")
    lines.append("")
    for audit in audits:
        lines.append(f"## {audit.get('run_name')}")
        lines.append("")
        lines.append(f"Run dir: `{audit.get('run_dir')}`")
        lines.append("")
        counts = dict(audit.get("candidate_counts_by_type") or {})
        if counts:
            lines.append("Counts by type:")
            for key in sorted(counts):
                lines.append(f"- `{key}`: {counts[key]}")
            lines.append("")
        for index, candidate in enumerate(list(audit.get("candidates") or []), start=1):
            lines.append(f"### {index}. {candidate.get('type')}")
            lines.append("")
            lines.append(f"- confidence: `{candidate.get('confidence')}`")
            lines.append(f"- recommendation: `{candidate.get('review_recommendation')}`")
            lines.append(f"- visible_text: `{candidate.get('visible_text')}`")
            lines.append(f"- native_words_text: `{candidate.get('native_words_text')}`")
            lines.append(f"- target_caption_ids: `{candidate.get('target_caption_ids')}`")
            lines.append(f"- target_segment_ids: `{candidate.get('target_segment_ids')}`")
            lines.append(f"- target_word_ids: `{candidate.get('target_word_ids')}`")
            evidence = candidate.get("evidence") or {}
            if evidence:
                lines.append(f"- evidence: `{json.dumps(evidence, ensure_ascii=False)}`")
            context = candidate.get("context") or {}
            if context:
                lines.append(f"- context: `{json.dumps(context, ensure_ascii=False)}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _pre_visible_semantic_junk_candidates(
    *,
    run_name: str,
    run_dir: str,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(list(report.get("pre_visible_semantic_junk_candidates") or []), start=1):
        confidence = float(row.get("local_confidence") or 0.0)
        rows.append(
            _candidate(
                run_name=run_name,
                run_dir=run_dir,
                index=index,
                candidate_type="pre_visible_semantic_junk_candidate",
                confidence=confidence,
                visible_text=str(row.get("visible_text") or ""),
                native_words_text=str(row.get("native_words_text") or ""),
                target_caption_ids=list(row.get("target_caption_ids") or []),
                target_segment_ids=list(row.get("target_segment_ids") or []),
                target_word_ids=list(row.get("target_word_ids") or []),
                source_start_us=int(row.get("source_start_us") or 0),
                source_end_us=int(row.get("source_end_us") or 0),
                evidence={
                    "source_candidate_id": str(row.get("candidate_id") or ""),
                    "source_candidate_type": str(row.get("type") or ""),
                    "proposed_action": str(row.get("proposed_action") or ""),
                    "provider_required": bool(row.get("provider_required")),
                    "safe_to_deterministic_apply": (
                        str(row.get("proposed_action") or "") == "drop_fragment"
                        and confidence >= MIN_HIGH_CONFIDENCE
                        and not bool(row.get("provider_required"))
                    ),
                    **dict(row.get("evidence") or {}),
                },
                context={
                    "previous_context": list(row.get("previous_context") or []),
                    "next_context": list(row.get("next_context") or []),
                },
                review_recommendation="review_candidate_or_confirm_deterministic_drop",
            )
        )
    return rows


def _hidden_audio_prefix_restart_candidates(
    *,
    run_name: str,
    run_dir: str,
    selected_words: list[dict[str, Any]],
    caption_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(selected_words) - 1):
        left = selected_words[index]
        right = selected_words[index + 1]
        left_text = normalize_text(str(left.get("text") or ""))
        right_text = normalize_text(str(right.get("text") or ""))
        if not _is_hidden_audio_prefix_restart(left_text, right_text):
            continue
        gap_us = int(right.get("source_start_us") or 0) - int(left.get("source_end_us") or 0)
        if gap_us > MAX_HIDDEN_AUDIO_PREFIX_GAP_US:
            continue
        word_ids = [str(left.get("word_id") or ""), str(right.get("word_id") or "")]
        captions = _captions_for_word_ids(caption_index, word_ids)
        rows.append(
            _candidate(
                run_name=run_name,
                run_dir=run_dir,
                index=len(rows) + 1,
                candidate_type="hidden_audio_prefix_restart",
                confidence=_hidden_audio_prefix_confidence(left_text, right_text, gap_us),
                visible_text="".join(str(row.get("text") or "") for row in captions),
                native_words_text=f"{left_text}{right_text}",
                target_caption_ids=[str(row.get("caption_id") or "") for row in captions],
                target_segment_ids=_unique(
                    [str(left.get("timeline_segment_id") or ""), str(right.get("timeline_segment_id") or "")]
                ),
                target_word_ids=word_ids,
                source_start_us=int(left.get("source_start_us") or 0),
                source_end_us=int(right.get("source_end_us") or 0),
                evidence={
                    "prefix_word_text": left_text,
                    "restart_word_text": right_text,
                    "source_gap_us": gap_us,
                    "reason": "short selected word is immediately restarted by the next selected word",
                },
                context={
                    "previous_words": _word_context(selected_words, index, before=True),
                    "next_words": _word_context(selected_words, index + 1, before=False),
                },
                review_recommendation="review_audio_stutter_candidate",
            )
        )
    return rows


def _intraword_audio_restart_normalization_candidates(
    *,
    run_name: str,
    run_dir: str,
    selected_words: list[dict[str, Any]],
    caption_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, word in enumerate(selected_words):
        debug_hints = dict(word.get("debug_hints") or {})
        original_text = normalize_text(str(debug_hints.get("original_text") or ""))
        normalized_word_text = normalize_text(str(word.get("text") or ""))
        reason = str(debug_hints.get("normalization_reason") or "")
        if not _is_intraword_audio_restart_normalization(debug_hints, original_text, normalized_word_text):
            continue
        previous_text = normalize_text(str(selected_words[index - 1].get("text") or "")) if index > 0 else ""
        next_text = normalize_text(str(selected_words[index + 1].get("text") or "")) if index + 1 < len(selected_words) else ""
        if _looks_like_prefixed_reduplicated_lexeme(original_text, next_text, previous_text):
            continue
        word_id = str(word.get("word_id") or "")
        captions = _captions_for_word_ids(caption_index, [word_id])
        rows.append(
            _candidate(
                run_name=run_name,
                run_dir=run_dir,
                index=len(rows) + 1,
                candidate_type="intraword_audio_restart_normalized",
                confidence=0.94,
                visible_text="".join(str(row.get("text") or "") for row in captions),
                native_words_text=original_text or normalized_word_text,
                target_caption_ids=[str(row.get("caption_id") or "") for row in captions],
                target_segment_ids=[str(word.get("timeline_segment_id") or "")],
                target_word_ids=[word_id],
                source_start_us=int(word.get("source_start_us") or 0),
                source_end_us=int(word.get("source_end_us") or 0),
                evidence={
                    "original_text": original_text,
                    "normalized_word_text": normalized_word_text,
                    "normalization_reason": reason,
                    "word_boundary_safe_to_auto_apply": False,
                    "reason": "ASR word contains an internal restart that was normalized for text but may remain audible",
                },
                context={
                    "previous_words": _word_context(selected_words, index, before=True),
                    "next_words": _word_context(selected_words, index, before=False),
                },
                review_recommendation="review_audio_or_subword_restart_candidate",
            )
        )
    return rows


def _visible_internal_prefix_restart_candidates(
    *,
    run_name: str,
    run_dir: str,
    captions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = _ordered_captions(captions)
    for caption_index, caption in enumerate(ordered):
        text = normalize_text(str(caption.get("text") or ""))
        match = _visible_internal_prefix_restart(text)
        if match is None:
            continue
        prefix, second_index = match
        rows.append(
            _candidate(
                run_name=run_name,
                run_dir=run_dir,
                index=len(rows) + 1,
                candidate_type="visible_internal_prefix_restart",
                confidence=_visible_internal_prefix_confidence(prefix, second_index, text),
                visible_text=str(caption.get("text") or ""),
                native_words_text=str(caption.get("text") or ""),
                target_caption_ids=[str(caption.get("caption_id") or "")],
                target_segment_ids=[str(value) for value in list(caption.get("timeline_segment_ids") or [])],
                target_word_ids=[str(value) for value in list(caption.get("word_ids") or [])],
                source_start_us=int(caption.get("spoken_source_start_us") or 0),
                source_end_us=int(caption.get("spoken_source_end_us") or 0),
                evidence={
                    "prefix_text": prefix,
                    "second_prefix_index": second_index,
                    "reason": "caption begins with a short prefix that restarts inside the same visible caption",
                },
                context={
                    "previous_captions": [str(row.get("text") or "") for row in ordered[max(0, caption_index - 2) : caption_index]],
                    "next_captions": [str(row.get("text") or "") for row in ordered[caption_index + 1 : caption_index + 3]],
                },
                review_recommendation="review_visible_caption_restart_candidate",
            )
        )
    return rows


def _adjacent_visible_restart_candidates(
    *,
    run_name: str,
    run_dir: str,
    captions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = _ordered_captions(captions)
    for index in range(len(ordered) - 1):
        left = ordered[index]
        right = ordered[index + 1]
        left_text = normalize_text(str(left.get("text") or ""))
        right_text = normalize_text(str(right.get("text") or ""))
        if not _is_adjacent_visible_restart(left_text, right_text):
            continue
        rows.append(
            _candidate(
                run_name=run_name,
                run_dir=run_dir,
                index=len(rows) + 1,
                candidate_type="adjacent_visible_restart",
                confidence=0.82,
                visible_text=f"{left.get('text') or ''} / {right.get('text') or ''}",
                native_words_text=f"{left_text}{right_text}",
                target_caption_ids=[str(left.get("caption_id") or ""), str(right.get("caption_id") or "")],
                target_segment_ids=_unique(
                    [
                        str(value)
                        for caption in (left, right)
                        for value in list(caption.get("timeline_segment_ids") or [])
                    ]
                ),
                target_word_ids=[
                    str(value)
                    for caption in (left, right)
                    for value in list(caption.get("word_ids") or [])
                ],
                source_start_us=int(left.get("spoken_source_start_us") or 0),
                source_end_us=int(right.get("spoken_source_end_us") or 0),
                evidence={
                    "left_text": left_text,
                    "right_text": right_text,
                    "reason": "short adjacent caption is reopened by the next visible caption",
                },
                context={
                    "previous_captions": [str(row.get("text") or "") for row in ordered[max(0, index - 2) : index]],
                    "next_captions": [str(row.get("text") or "") for row in ordered[index + 2 : index + 4]],
                },
                review_recommendation="review_adjacent_visible_restart_candidate",
            )
        )
    return rows


def _is_hidden_audio_prefix_restart(left_text: str, right_text: str) -> bool:
    if not left_text or not right_text:
        return False
    if len(left_text) > MAX_HIDDEN_AUDIO_PREFIX_CHARS or len(right_text) < MIN_RESTARTED_WORD_CHARS:
        return False
    if left_text in TRIVIAL_SINGLE_WORDS:
        return False
    if not _contains_cjk(left_text + right_text):
        return False
    return right_text.startswith(left_text) and right_text != left_text


def _is_intraword_audio_restart_normalization(
    debug_hints: dict[str, Any],
    original_text: str,
    normalized_word_text: str,
) -> bool:
    if not bool(debug_hints.get("intraword_cjk_restart_normalized")):
        return False
    if not original_text or not normalized_word_text or original_text == normalized_word_text:
        return False
    if not _contains_cjk(original_text):
        return False
    return original_text.startswith(normalized_word_text) or _has_leading_repeated_cjk_char(original_text)


def _looks_like_prefixed_reduplicated_lexeme(original_text: str, next_text: str, previous_text: str) -> bool:
    if len(original_text) != 2 or original_text[0] != original_text[1]:
        return False
    if len(previous_text) != 1 or previous_text in LEXICAL_REDUPLICATION_PREFIX_STOPWORDS:
        return False
    if not _contains_cjk(previous_text):
        return False
    return bool(next_text) and next_text.startswith(LEXICAL_REDUPLICATION_BOUNDARY_WORDS)


def _has_leading_repeated_cjk_char(text: str) -> bool:
    return len(text) >= 2 and text[0] == text[1] and _contains_cjk(text[:1])


def _hidden_audio_prefix_confidence(left_text: str, right_text: str, gap_us: int) -> float:
    score = 0.72
    if len(left_text) == 1:
        score += 0.08
    if len(right_text) >= len(left_text) + 2:
        score += 0.08
    if gap_us <= 300_000:
        score += 0.08
    return round(min(0.96, score), 6)


def _visible_internal_prefix_restart(text: str) -> tuple[str, int] | None:
    if len(text) < MIN_VISIBLE_RESTART_REMAINDER_CHARS + 2 or not _contains_cjk(text):
        no_match: tuple[str, int] | None = None
        return no_match
    for prefix_len in range(1, min(MAX_VISIBLE_PREFIX_CHARS, len(text) - 1) + 1):
        prefix = text[:prefix_len]
        if prefix in TRIVIAL_SINGLE_WORDS or not _contains_cjk(prefix):
            continue
        second_index = text.find(prefix, prefix_len)
        if second_index < 0 or second_index > MAX_VISIBLE_INTERNAL_RESTART_OFFSET:
            continue
        if len(text) - (second_index + prefix_len) < MIN_VISIBLE_RESTART_REMAINDER_CHARS:
            continue
        return prefix, second_index
    no_match: tuple[str, int] | None = None
    return no_match


def _visible_internal_prefix_confidence(prefix: str, second_index: int, text: str) -> float:
    score = 0.72
    if len(prefix) == 1 and second_index <= 3:
        score += 0.1
    if len(text) >= second_index + len(prefix) + 6:
        score += 0.08
    return round(min(0.95, score), 6)


def _is_adjacent_visible_restart(left_text: str, right_text: str) -> bool:
    if not left_text or not right_text:
        return False
    if len(left_text) > MAX_ADJACENT_RESTART_FRAGMENT_CHARS:
        return False
    if len(right_text) < len(left_text) + 2:
        return False
    if not right_text.startswith(left_text):
        return False
    if is_semantic_label_reuse_boundary(left_text, right_text, left_text):
        return False
    if is_explanatory_term_reuse(left_text, right_text) or is_explanatory_term_reuse(right_text, left_text):
        return False
    return True


def _candidate(
    *,
    run_name: str,
    run_dir: str,
    index: int,
    candidate_type: str,
    confidence: float,
    visible_text: str,
    native_words_text: str,
    target_caption_ids: list[str],
    target_segment_ids: list[str],
    target_word_ids: list[str],
    source_start_us: int,
    source_end_us: int,
    evidence: dict[str, Any],
    context: dict[str, Any],
    review_recommendation: str,
) -> dict[str, Any]:
    return {
        "candidate_id": f"{run_name}_{candidate_type}_{index:06d}",
        "run_name": run_name,
        "run_dir": run_dir,
        "type": candidate_type,
        "confidence": round(float(confidence), 6),
        "visible_text": visible_text,
        "native_words_text": native_words_text,
        "target_caption_ids": _unique([str(value) for value in target_caption_ids]),
        "target_segment_ids": _unique([str(value) for value in target_segment_ids]),
        "target_word_ids": _unique([str(value) for value in target_word_ids]),
        "source_start_us": int(source_start_us),
        "source_end_us": int(source_end_us),
        "evidence": evidence,
        "context": context,
        "review_recommendation": review_recommendation,
        "audit_only": True,
    }


def _selected_timeline_words(
    final_timeline: list[dict[str, Any]],
    words_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in sorted(
        final_timeline,
        key=lambda row: (int(row.get("target_start_us") or 0), int(row.get("target_end_us") or 0), str(row.get("segment_id") or "")),
    ):
        for word_id in list(segment.get("word_ids") or []):
            word = dict(words_by_id.get(str(word_id)) or {})
            if not word:
                continue
            word["timeline_segment_id"] = str(segment.get("segment_id") or "")
            word["timeline_target_start_us"] = int(segment.get("target_start_us") or 0)
            word["timeline_target_end_us"] = int(segment.get("target_end_us") or 0)
            rows.append(word)
    return rows


def _caption_index_by_word_id(captions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for caption in captions:
        for word_id in list(caption.get("word_ids") or []):
            index.setdefault(str(word_id), []).append(caption)
    return index


def _captions_for_word_ids(
    caption_index: dict[str, list[dict[str, Any]]],
    word_ids: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for word_id in word_ids:
        for caption in caption_index.get(str(word_id), []):
            caption_id = str(caption.get("caption_id") or "")
            if caption_id in seen:
                continue
            seen.add(caption_id)
            rows.append(caption)
    return _ordered_captions(rows)


def _word_context(words: list[dict[str, Any]], index: int, *, before: bool) -> list[str]:
    if before:
        start = max(0, index - 3)
        rows = words[start:index]
    else:
        rows = words[index + 1 : index + 4]
    return [str(row.get("text") or "") for row in rows]


def _ordered_captions(captions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        captions,
        key=lambda row: (int(row.get("target_start_us") or 0), int(row.get("target_end_us") or 0), str(row.get("caption_id") or "")),
    )


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        key = (
            str(candidate.get("type") or ""),
            tuple(candidate.get("target_word_ids") or []),
            tuple(candidate.get("target_caption_ids") or []),
            normalize_text(str(candidate.get("visible_text") or "")),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(candidate)
    rows.sort(
        key=lambda row: (
            int(row.get("source_start_us") or 0),
            str(row.get("type") or ""),
            str(row.get("candidate_id") or ""),
        )
    )
    return rows


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        missing: Any = None
        return missing
    return json.loads(path.read_text("utf-8"))


def _read_gzip_json_file(path: Path) -> Any:
    if not path.exists():
        missing: Any = None
        return missing
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _unique(values: list[str]) -> list[str]:
    rows: list[str] = []
    for value in values:
        if value and value not in rows:
            rows.append(value)
    return rows
