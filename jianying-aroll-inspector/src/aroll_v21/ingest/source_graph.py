from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from aroll_text_normalize import normalize_text
from aroll_v21.ir.models import (
    Blocker,
    CanonicalSourceGraph,
    CanonicalWord,
    EditUnit,
    SourceGraphInvariantReport,
)

_CJK_TEXT_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")
_RESULT_COMPLEMENT_STARTS = (
    "出",
    "到",
    "成",
    "回",
    "来",
    "去",
    "起",
    "下",
    "上",
    "完",
    "掉",
    "走",
    "开",
    "住",
    "好",
    "进",
)


def _text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("fragment_text") or row.get("subtitle_text") or row.get("word_text") or "")


def _start(row: dict[str, Any]) -> int:
    for key in ("source_start_us", "start_us", "source_timeline_start_us"):
        if row.get(key) is not None:
            return int(row.get(key) or 0)
    for key in ("source_timerange", "target_timerange", "timerange"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("start") is not None:
            return int(value.get("start") or 0)
    return 0


def _end(row: dict[str, Any]) -> int:
    if row.get("source_end_us") is not None:
        return int(row.get("source_end_us") or 0)
    if row.get("end_us") is not None:
        return int(row.get("end_us") or 0)
    if row.get("source_timeline_end_us") is not None:
        return int(row.get("source_timeline_end_us") or 0)
    if row.get("duration_us") is not None:
        return _start(row) + int(row.get("duration_us") or 0)
    for key in ("source_timerange", "target_timerange", "timerange"):
        value = row.get(key)
        if isinstance(value, dict):
            if value.get("end") is not None:
                return int(value.get("end") or 0)
            if value.get("duration") is not None:
                return _start(row) + int(value.get("duration") or 0)
    return 0


class TrackDetector:
    """Extract minimal track/material rows from already-loaded draft-like data."""

    def text_materials(self, draft_data: dict[str, Any]) -> list[dict[str, Any]]:
        materials = draft_data.get("materials") or {}
        rows = materials.get("texts") if isinstance(materials, dict) else []
        return [dict(row) for row in (rows or []) if isinstance(row, dict)]

    def text_segments(self, draft_data: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for track in draft_data.get("tracks") or []:
            if not isinstance(track, dict):
                continue
            track_type = str(track.get("type") or track.get("track_type") or "")
            if "text" not in track_type.lower() and "subtitle" not in track_type.lower():
                continue
            for segment in track.get("segments") or []:
                if isinstance(segment, dict):
                    rows.append(dict(segment))
        return rows

    def source_segments(self, draft_data: dict[str, Any], explicit_segments: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        if explicit_segments is not None:
            return [dict(row) for row in explicit_segments]
        rows: list[dict[str, Any]] = []
        for track in draft_data.get("tracks") or []:
            if not isinstance(track, dict):
                continue
            track_type = str(track.get("type") or track.get("track_type") or "")
            if "video" not in track_type.lower():
                continue
            for segment in track.get("segments") or []:
                if isinstance(segment, dict):
                    rows.append(dict(segment))
        return rows

    def source_materials(
        self,
        draft_data: dict[str, Any],
        source_segments: list[dict[str, Any]],
        explicit_materials: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if explicit_materials is not None:
            return [dict(row) for row in explicit_materials]
        materials = draft_data.get("materials") if isinstance(draft_data.get("materials"), dict) else {}
        rows: list[dict[str, Any]] = []
        for key, material_type in (("videos", "video"), ("audios", "audio")):
            for material in materials.get(key) or []:
                if not isinstance(material, dict):
                    continue
                material_id = str(material.get("id") or material.get("material_id") or "")
                if not material_id:
                    continue
                rows.append(
                    {
                        "source_material_id": material_id,
                        "path": str(material.get("path") or material.get("local_material_id") or material.get("name") or ""),
                        "duration_us": int(material.get("duration_us") or material.get("duration") or 0),
                        "type": material_type,
                        "metadata": {k: v for k, v in material.items() if k not in {"id", "path", "duration_us", "duration"}},
                    }
                )
        if rows:
            return rows
        seen: set[str] = set()
        for segment in source_segments:
            material_id = str(segment.get("material_id") or segment.get("source_material_id") or "")
            if not material_id or material_id in seen:
                continue
            seen.add(material_id)
            rows.append(
                {
                    "source_material_id": material_id,
                    "path": str(segment.get("path") or ""),
                    "duration_us": int(segment.get("duration_us") or max(0, _end(segment) - _start(segment))),
                    "type": "video" if "video" in str(segment.get("track_type") or "").lower() else "unknown",
                    "metadata": {"inventory_source": "source_segment"},
                }
            )
        return rows


class CanonicalSourceGraphBuilder:
    def __init__(self, track_detector: TrackDetector | None = None) -> None:
        self.track_detector = track_detector or TrackDetector()

    def build(
        self,
        *,
        draft_data: dict[str, Any] | None = None,
        word_timeline: list[dict[str, Any]],
        subtitles: list[dict[str, Any]],
        source_segments: list[dict[str, Any]] | None = None,
        source_materials: list[dict[str, Any]] | None = None,
        text_materials: list[dict[str, Any]] | None = None,
        text_segments: list[dict[str, Any]] | None = None,
    ) -> CanonicalSourceGraph:
        draft_data = draft_data or {}
        explicit_source_materials_provided = source_materials is not None
        source_segments = self.track_detector.source_segments(draft_data, source_segments)
        source_materials = self.track_detector.source_materials(draft_data, source_segments, source_materials)
        text_materials = [dict(row) for row in (text_materials if text_materials is not None else self.track_detector.text_materials(draft_data))]
        text_segments = [dict(row) for row in (text_segments if text_segments is not None else self.track_detector.text_segments(draft_data))]

        blockers: list[Blocker] = []
        source_material_ids = {str(row.get("source_material_id") or row.get("id") or "") for row in source_materials}
        if explicit_source_materials_provided and source_segments and not source_materials:
            blockers.append(
                Blocker(
                    "SOURCE_MATERIAL_INVENTORY_EMPTY",
                    "source material inventory was explicitly provided but empty",
                    "ingest",
                )
            )
        for segment in source_segments:
            material_id = str(segment.get("material_id") or segment.get("source_material_id") or "")
            if not material_id:
                blockers.append(
                    Blocker(
                        "SOURCE_SEGMENT_MATERIAL_UNBOUND",
                        "source segment has no source material id",
                        "ingest",
                        context={"segment_id": str(segment.get("id") or "")},
                    )
                )
            elif source_material_ids and material_id not in source_material_ids:
                blockers.append(
                    Blocker(
                        "SOURCE_SEGMENT_MATERIAL_NOT_IN_INVENTORY",
                        "source segment material id is absent from source material inventory",
                        "ingest",
                        context={"segment_id": str(segment.get("id") or ""), "source_material_id": material_id},
                    )
                )
        words = [
            self._canonical_word(index=index, row=row, source_segments=source_segments, blockers=blockers)
            for index, row in enumerate(word_timeline, start=1)
        ]
        words = _normalize_intraword_cjk_restarts(words)
        edit_units = self._edit_units(subtitles, words, blockers, source_segments)
        if not words:
            blockers.append(Blocker("SOURCE_WORD_TIMELINE_EMPTY", "source graph has no canonical words", "ingest"))
        if not edit_units:
            blockers.append(Blocker("SOURCE_EDIT_UNITS_EMPTY", "source graph has no edit units", "ingest"))
        unbound_word_count = sum(1 for word in words if word.source_start_us >= word.source_end_us)
        unbound_subtitle_count = sum(1 for unit in edit_units if not unit.word_ids)
        invariant = SourceGraphInvariantReport(
            single_source_graph_ok=len([b for b in blockers if b.severity == "fatal"]) == 0,
            all_words_have_source_time=all(word.source_end_us > word.source_start_us for word in words),
            all_edit_units_have_word_ids=all(bool(unit.word_ids) for unit in edit_units),
            unbound_word_count=unbound_word_count,
            unbound_subtitle_count=unbound_subtitle_count,
            blocker_count=len([b for b in blockers if b.severity == "fatal"]),
            blockers=blockers,
        )
        return CanonicalSourceGraph(
            words=words,
            edit_units=edit_units,
            subtitle_rows=[dict(row) for row in subtitles],
            source_materials=source_materials,
            source_segments=source_segments,
            text_materials=text_materials,
            text_segments=text_segments,
            invariant_report=invariant,
        )

    def _canonical_word(
        self,
        *,
        index: int,
        row: dict[str, Any],
        source_segments: list[dict[str, Any]],
        blockers: list[Blocker],
    ) -> CanonicalWord:
        start = _start(row)
        end = _end(row)
        legacy_source_material_id = str(row.get("source_material_id") or row.get("material_id") or "")
        legacy_source_segment_id = str(row.get("source_segment_id") or "")
        debug_hints = dict(row.get("debug_hints") or {})
        if legacy_source_material_id:
            debug_hints.setdefault("legacy_source_material_id", legacy_source_material_id)
        if legacy_source_segment_id:
            debug_hints.setdefault("legacy_source_segment_id", legacy_source_segment_id)
        word_id = str(row.get("word_id") or f"word_{index:06d}")
        text = str(row.get("word_text") or row.get("text") or "")
        if end <= start:
            blockers.append(Blocker("SOURCE_WORD_TIME_UNBOUND", "word has no valid source time", "ingest", context={"word_id": word_id}))
        return CanonicalWord(
            word_id=word_id,
            text=text,
            normalized_text=normalize_text(text),
            source_start_us=start,
            source_end_us=end,
            source_material_id="",
            source_segment_id=None,
            subtitle_uid=str(row.get("subtitle_uid") or "") or None,
            subtitle_index=int(row.get("subtitle_index")) if row.get("subtitle_index") is not None else None,
            char_start=int(row.get("char_start")) if row.get("char_start") is not None else None,
            char_end=int(row.get("char_end")) if row.get("char_end") is not None else None,
            confidence=float(row.get("confidence")) if row.get("confidence") is not None else None,
            is_cuttable_left=bool(row.get("is_cuttable_left", True)),
            is_cuttable_right=bool(row.get("is_cuttable_right", True)),
            debug_hints=debug_hints,
        )

    def _segment_for_time(self, segments: list[dict[str, Any]], start: int, end: int) -> dict[str, Any] | None:
        for segment in segments:
            seg_start = _start(segment)
            seg_end = _end(segment)
            if seg_start <= start and end <= seg_end:
                return segment
        return None

    def _edit_units(
        self,
        subtitles: list[dict[str, Any]],
        words: list[CanonicalWord],
        blockers: list[Blocker],
        source_segments: list[dict[str, Any]],
    ) -> list[EditUnit]:
        words_by_id = {word.word_id: word for word in words}
        units: list[EditUnit] = []
        for index, row in enumerate(subtitles, start=1):
            explicit_word_ids = [str(word_id) for word_id in row.get("word_ids") or [] if str(word_id)]
            word_ids = [word_id for word_id in explicit_word_ids if word_id in words_by_id]
            subtitle_uid = str(row.get("subtitle_uid") or row.get("fragment_id") or "")
            subtitle_index = int(row.get("subtitle_index") or index)
            if not word_ids:
                word_ids = [
                    word.word_id
                    for word in words
                    if (subtitle_uid and word.subtitle_uid == subtitle_uid)
                    or (word.subtitle_index is not None and word.subtitle_index == subtitle_index)
                ]
            if not word_ids:
                start = _start(row)
                end = _end(row)
                if end > start:
                    word_ids = [word.word_id for word in words if start <= word.source_start_us and word.source_end_us <= end]
            if not word_ids:
                diagnostic = self._diagnostic_binding(row, source_segments)
                blockers.append(
                    Blocker(
                        "EDIT_UNIT_WORD_BINDING_MISSING",
                        "subtitle/edit unit could not bind to canonical words",
                        "ingest",
                        context={
                            "subtitle_uid": subtitle_uid,
                            "subtitle_index": subtitle_index,
                            "text": _text(row),
                            **diagnostic,
                        },
                    )
                )
            unit_words = [words_by_id[word_id] for word_id in word_ids if word_id in words_by_id]
            source_start = min((word.source_start_us for word in unit_words), default=_start(row))
            source_end = max((word.source_end_us for word in unit_words), default=_end(row))
            word_text = "".join(word.text for word in unit_words)
            has_normalized_word = any(
                bool(word.debug_hints.get("intraword_cjk_restart_normalized"))
                for word in unit_words
            )
            text = word_text if has_normalized_word and word_text else (_text(row) or word_text)
            units.append(
                EditUnit(
                    unit_id=str(row.get("unit_id") or row.get("fragment_id") or subtitle_uid or f"unit_{index:06d}"),
                    word_ids=word_ids,
                    text=text,
                    normalized_text=normalize_text(text),
                    source_start_us=source_start,
                    source_end_us=source_end,
                    subtitle_uids=[subtitle_uid] if subtitle_uid else [],
                    source_material_ids=[],
                    kind=str(row.get("kind") or "sentence"),  # type: ignore[arg-type]
                    cut_policy=str(row.get("cut_policy") or "word_boundary"),  # type: ignore[arg-type]
                )
            )
        return units

    def _diagnostic_binding(self, row: dict[str, Any], source_segments: list[dict[str, Any]]) -> dict[str, Any]:
        start = _start(row)
        end = _end(row)
        candidates: list[dict[str, Any]] = []
        if end <= start:
            return {
                "binding_status": "diagnostic_only",
                "subtitle_target_time": {"start_us": start, "end_us": end},
                "diagnostic_source_segment_candidates": candidates,
            }
        for segment in source_segments:
            seg_start = _start(segment)
            seg_end = _end(segment)
            overlap = max(0, min(end, seg_end) - max(start, seg_start))
            if overlap <= 0:
                continue
            duration = max(1, end - start)
            candidates.append(
                {
                    "candidate_source_segment_id": str(segment.get("id") or ""),
                    "candidate_source_material_id": str(segment.get("material_id") or segment.get("source_material_id") or ""),
                    "binding_confidence": round(overlap / duration, 4),
                }
            )
        candidates.sort(key=lambda item: float(item.get("binding_confidence") or 0.0), reverse=True)
        best = candidates[0] if candidates else {}
        return {
            "binding_status": "diagnostic_only",
            "subtitle_target_time": {"start_us": start, "end_us": end},
            "candidate_source_segment_id": best.get("candidate_source_segment_id", ""),
            "candidate_source_material_id": best.get("candidate_source_material_id", ""),
            "binding_confidence": best.get("binding_confidence", 0.0),
            "diagnostic_source_segment_candidates": candidates[:3],
        }


class DraftIngest:
    def __init__(self, builder: CanonicalSourceGraphBuilder | None = None) -> None:
        self.builder = builder or CanonicalSourceGraphBuilder()

    def build_source_graph(self, **kwargs: Any) -> CanonicalSourceGraph:
        return self.builder.build(**kwargs)


def _normalize_intraword_cjk_restarts(words: list[CanonicalWord]) -> list[CanonicalWord]:
    normalized: list[CanonicalWord] = []
    for index, word in enumerate(words):
        next_text = words[index + 1].text if index + 1 < len(words) else ""
        replacement = _intraword_cjk_restart_replacement(word.text, next_text)
        if replacement is None or replacement == word.text:
            normalized.append(word)
            continue
        normalized.append(
            replace(
                word,
                text=replacement,
                normalized_text=normalize_text(replacement),
                debug_hints={
                    **dict(word.debug_hints or {}),
                    "intraword_cjk_restart_normalized": True,
                    "original_text": word.text,
                    "normalization_reason": "leading_single_char_restart_before_result_complement",
                },
            )
        )
    return normalized


def _intraword_cjk_restart_replacement(text: str, next_text: str) -> str | None:
    raw = str(text or "")
    if not raw or not _CJK_TEXT_RE.fullmatch(raw):
        no_replacement: str | None = None
        return no_replacement
    if len(raw) >= 3 and raw[0] == raw[1] and raw[2:].startswith(_RESULT_COMPLEMENT_STARTS):
        return raw[1:]
    next_normalized = normalize_text(next_text)
    if (
        len(raw) == 2
        and raw[0] == raw[1]
        and next_normalized.startswith(_RESULT_COMPLEMENT_STARTS)
    ):
        return raw[0]
    no_replacement: str | None = None
    return no_replacement
