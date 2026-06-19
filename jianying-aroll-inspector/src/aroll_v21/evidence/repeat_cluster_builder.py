from __future__ import annotations

from typing import Any

from aroll_adjacent_modifier_semantic_redundancy_gate import detect_adjacent_modifier_semantic_redundancy
from aroll_cjk_short_repeat_gate import detect_cjk_short_repeats
from aroll_text_normalize import normalize_text, repeated_phrase_spans
from aroll_v21.ir.models import CandidateEvidence, CanonicalSourceGraph, EditUnit, RepeatCluster
from aroll_v21.quality.repeat_span_repair import self_repair_aborted_phrase_candidate


CJK_NUMERAL_PREFIXES = tuple("一二两三四五六七八九十百千万半几多")
REDUPLICATION_MODIFIER_SUFFIXES = ("的", "地")


def _unit_display_rows(units: list[EditUnit]) -> list[dict[str, Any]]:
    return [
        {
            "fragment_id": unit.unit_id,
            "fragment_text": unit.text,
            "text": unit.text,
            "word_ids": unit.word_ids,
            "source_start_us": unit.source_start_us,
            "source_end_us": unit.source_end_us,
        }
        for unit in units
    ]


class CandidateEvidenceBuilder:
    """Build repeat/take evidence from the canonical source graph without editing it."""

    def build(self, source_graph: CanonicalSourceGraph) -> list[RepeatCluster]:
        units = source_graph.edit_units
        words_by_id = {word.word_id: word for word in source_graph.words}
        clusters: list[RepeatCluster] = []
        clusters.extend(self._adjacent_exact_repeats(units))
        clusters.extend(self._boundary_prefix_containment(units, words_by_id))
        clusters.extend(self._cjk_short_repeats(units, words_by_id))
        clusters.extend(self._modifier_redundancy(units, words_by_id))
        clusters.extend(self._self_repair_aborted_phrases(units, words_by_id))
        clusters.extend(self._intra_unit_repeats(units, words_by_id))
        return self._dedupe(clusters)

    def _evidence(
        self,
        *,
        seq: int,
        evidence_type: str,
        units: list[EditUnit],
        reason: str,
        confidence: float,
        requires_semantic_decision: bool,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateEvidence:
        return CandidateEvidence(
            evidence_id=f"evidence_{seq:06d}",
            evidence_type=evidence_type,  # type: ignore[arg-type]
            unit_ids=[unit.unit_id for unit in units],
            word_ids=[word_id for unit in units for word_id in unit.word_ids],
            text=" ".join(unit.text for unit in units).strip(),
            normalized_text=normalize_text("".join(unit.text for unit in units)),
            reason=reason,
            confidence=confidence,
            requires_semantic_decision=requires_semantic_decision,
            metadata=metadata or {},
        )

    def _cluster(
        self,
        *,
        seq: int,
        repeat_type: str,
        units: list[EditUnit],
        evidence: CandidateEvidence,
        local_recommendation: str | None,
    ) -> RepeatCluster:
        return RepeatCluster(
            cluster_id=f"repeat_{seq:06d}",
            variants=units,
            repeat_type=repeat_type,  # type: ignore[arg-type]
            evidence=[evidence],
            local_recommendation=local_recommendation,
        )

    def _adjacent_exact_repeats(self, units: list[EditUnit]) -> list[RepeatCluster]:
        clusters: list[RepeatCluster] = []
        seq = 1
        for left, right in zip(units, units[1:]):
            if left.normalized_text and left.normalized_text == right.normalized_text:
                evidence = self._evidence(
                    seq=seq,
                    evidence_type="exact_repeat",
                    units=[left, right],
                    reason="adjacent edit units have identical normalized text",
                    confidence=0.98,
                    requires_semantic_decision=False,
                )
                clusters.append(
                    self._cluster(
                        seq=seq,
                        repeat_type="exact_repeat",
                        units=[left, right],
                        evidence=evidence,
                        local_recommendation="keep_right_drop_left",
                    )
                )
                seq += 1
        return clusters

    def _boundary_prefix_containment(self, units: list[EditUnit], words_by_id: dict[str, Any]) -> list[RepeatCluster]:
        clusters: list[RepeatCluster] = []
        seq = 4000
        for left, right in zip(units, units[1:]):
            left_text = normalize_text(left.text)
            right_text = normalize_text(right.text)
            if not left_text or not right_text or left_text == right_text:
                continue
            if len(left_text) < 2 or len(left_text) > 12:
                continue
            if not right_text.startswith(left_text):
                continue
            if left.cut_policy == "unsafe":
                continue

            safe_drop = self._safe_boundary_prefix_drop(left, right, words_by_id)
            metadata = {
                "candidate": {
                    "type": "boundary_prefix_containment",
                    "prev_text": left.text,
                    "next_text": right.text,
                    "overlap": left.text,
                    "severity": "fatal",
                    "reason": "right edit unit is a complete prefix extension of the left edit unit",
                },
                "left_unit_id": left.unit_id,
                "right_unit_id": right.unit_id,
                "left_word_ids": left.word_ids,
                "right_word_ids": right.word_ids,
            }
            evidence = self._evidence(
                seq=seq,
                evidence_type="cjk_short_overlap",
                units=[left, right],
                reason="boundary prefix containment: right edit unit is the complete restart",
                confidence=0.96 if safe_drop else 0.65,
                requires_semantic_decision=False,
                metadata=metadata,
            )
            clusters.append(
                self._cluster(
                    seq=seq,
                    repeat_type="cjk_short_overlap",
                    units=[left, right],
                    evidence=evidence,
                    local_recommendation="boundary_prefix_containment_drop_left"
                    if safe_drop
                    else "boundary_prefix_containment_requires_human_review",
                )
            )
            seq += 1
        return clusters

    def _safe_boundary_prefix_drop(self, left: EditUnit, right: EditUnit, words_by_id: dict[str, Any]) -> bool:
        if left.cut_policy == "unsafe":
            return False
        left_words = [words_by_id[word_id] for word_id in left.word_ids if word_id in words_by_id]
        right_words = [words_by_id[word_id] for word_id in right.word_ids if word_id in words_by_id]
        if not left_words or not right_words:
            return False
        left_segments = {getattr(word, "source_segment_id", None) for word in left_words}
        right_segments = {getattr(word, "source_segment_id", None) for word in right_words}
        left_materials = {getattr(word, "source_material_id", "") for word in left_words}
        right_materials = {getattr(word, "source_material_id", "") for word in right_words}
        if len(left_segments | right_segments) != 1 or len(left_materials | right_materials) != 1:
            return False
        gap_us = right.source_start_us - left.source_end_us
        return -80_000 <= gap_us <= 1_500_000

    def _cjk_short_repeats(self, units: list[EditUnit], words_by_id: dict[str, Any]) -> list[RepeatCluster]:
        rows = _unit_display_rows(units)
        candidates = [row for row in detect_cjk_short_repeats(rows) if str(row.get("severity") or "fatal") == "fatal"]
        clusters: list[RepeatCluster] = []
        for seq, candidate in enumerate(candidates, start=1000):
            row_index = int(candidate.get("row_index") or 0)
            next_index = int(candidate.get("next_row_index") or 0)
            related = []
            if 1 <= row_index <= len(units):
                related.append(units[row_index - 1])
            if 1 <= next_index <= len(units):
                related.append(units[next_index - 1])
            if not related:
                continue
            evidence_type = "restart" if str(candidate.get("type") or "") == "restart_disfluency" else "cjk_short_overlap"
            evidence = self._evidence(
                seq=seq,
                evidence_type=evidence_type,
                units=related,
                reason=str(candidate.get("reason") or candidate.get("type") or "CJK short repeat"),
                confidence=0.95,
                requires_semantic_decision=False,
                metadata={"candidate": candidate} | self._split_metadata_for_candidate(candidate, related, words_by_id),
            )
            if len(related) > 1 and self._is_boundary_suffix_prefix_overlap_candidate(candidate):
                recommendation = "compiler_boundary_suffix_prefix_overlap_cleanup"
            else:
                recommendation = "keep_right_drop_left" if len(related) > 1 else "requires_unit_split"
            clusters.append(
                self._cluster(
                    seq=seq,
                    repeat_type=evidence_type,
                    units=related,
                    evidence=evidence,
                    local_recommendation=recommendation,
                )
            )
        return clusters

    def _is_boundary_suffix_prefix_overlap_candidate(self, candidate: dict[str, Any]) -> bool:
        if str(candidate.get("issue_type") or "") != "cjk_adjacent_subtitle_boundary_overlap":
            return False
        left = normalize_text(str(candidate.get("left_text") or candidate.get("prev_text") or ""))
        right = normalize_text(str(candidate.get("right_text") or candidate.get("next_text") or ""))
        overlap = normalize_text(str(candidate.get("overlap") or candidate.get("phrase") or ""))
        if len(overlap) < 2 or not left or not right:
            return False
        return left.endswith(overlap) and right.startswith(overlap) and overlap != left

    def _split_metadata_for_candidate(
        self,
        candidate: dict[str, Any],
        related: list[EditUnit],
        words_by_id: dict[str, Any],
    ) -> dict[str, Any]:
        if len(related) != 1:
            return {}
        unit = related[0]
        metadata: dict[str, Any] = {"word_tokens": self._word_tokens_for_unit(unit, words_by_id)}
        span = candidate.get("span") if isinstance(candidate.get("span"), dict) else {}
        phrase = str(candidate.get("phrase") or candidate.get("overlap") or "")
        if not phrase:
            metadata["split_failed_reason"] = "drop_text_missing_for_unit_split_binding"
            return metadata
        drop_word_ids: list[str] = []
        binding_source = "candidate_char_span"
        if isinstance(span, dict) and span.get("start_char") is not None:
            start_char = int(span.get("start_char") or 0)
            drop_len = len(phrase)
            if str(candidate.get("type") or "") == "restart_disfluency" and len(phrase) >= 3:
                drop_len = len(phrase) - 1
            drop_word_ids = self._word_ids_for_cjk_char_span(unit, words_by_id, start_char, start_char + drop_len)
        if not drop_word_ids:
            token_binding = self._whole_word_split_for_phrase(unit, words_by_id, phrase)
            drop_word_ids = list(token_binding.get("drop_word_ids") or [])
            binding_source = str(token_binding.get("binding_source") or "source_word_token_binding")
        if not drop_word_ids:
            metadata.update(
                {
                    "split_drop_text": phrase,
                    "normalized_drop_text": normalize_text(phrase),
                    "split_failed_reason": "no_safe_whole_word_binding_for_drop_text",
                }
            )
            return metadata
        keep_word_ids = [word_id for word_id in unit.word_ids if word_id not in set(drop_word_ids)]
        metadata.update(
            {
                "split_drop_word_ids": drop_word_ids,
                "split_keep_word_ids": keep_word_ids,
                "split_drop_text": phrase,
                "normalized_drop_text": normalize_text(phrase),
                "split_binding_source": binding_source,
            }
        )
        return metadata

    def _word_ids_for_cjk_char_span(
        self,
        unit: EditUnit,
        words_by_id: dict[str, Any],
        start_char: int,
        end_char: int,
    ) -> list[str]:
        cursor = 0
        selected: list[str] = []
        for word_id in unit.word_ids:
            word = words_by_id.get(word_id)
            text = normalize_text(str(getattr(word, "text", "") or ""))
            if not text:
                continue
            word_start = cursor
            word_end = cursor + len(text)
            if start_char <= word_start and word_end <= end_char:
                selected.append(word_id)
            elif word_start < end_char and start_char < word_end:
                return []
            cursor = word_end
        return selected

    def _modifier_redundancy(self, units: list[EditUnit], words_by_id: dict[str, Any]) -> list[RepeatCluster]:
        rows = _unit_display_rows(units)
        candidates = [row for row in detect_adjacent_modifier_semantic_redundancy(rows) if str(row.get("severity") or "fatal") == "fatal"]
        clusters: list[RepeatCluster] = []
        for seq, candidate in enumerate(candidates, start=2000):
            row_index = int(candidate.get("row_index") or 0)
            if not (1 <= row_index <= len(units)):
                continue
            unit = units[row_index - 1]
            safe_candidate = self._semantic_modifier_candidate(candidate, unit, words_by_id)
            evidence = self._evidence(
                seq=seq,
                evidence_type="modifier_redundancy",
                units=[unit],
                reason=str(candidate.get("reason") or "adjacent modifier semantic redundancy"),
                confidence=0.72,
                requires_semantic_decision=True,
                metadata={"candidate": safe_candidate},
            )
            clusters.append(
                self._cluster(
                    seq=seq,
                    repeat_type="modifier_redundancy",
                    units=[unit],
                    evidence=evidence,
                    local_recommendation="semantic_review",
                )
            )
        return clusters

    def _semantic_modifier_candidate(self, candidate: dict[str, Any], unit: EditUnit, words_by_id: dict[str, Any]) -> dict[str, Any]:
        payload = dict(candidate)
        raw_phrase = str(payload.get("phrase") or "")
        left_modifier = normalize_text(str(payload.get("left_modifier") or ""))
        right_modifier = normalize_text(str(payload.get("right_modifier") or ""))
        head_text = normalize_text(str(payload.get("head_text") or ""))
        redundant_word_ids: list[str] = []
        keep_word_ids: list[str] = list(unit.word_ids)
        if left_modifier and right_modifier and head_text:
            redundant_span_text = f"{left_modifier}的"
            unit_text = normalize_text(unit.text)
            start_char = unit_text.find(redundant_span_text)
            if start_char >= 0:
                redundant_word_ids = self._word_ids_for_cjk_char_span(
                    unit,
                    words_by_id,
                    start_char,
                    start_char + len(redundant_span_text),
                )
                if redundant_word_ids:
                    drop_set = set(redundant_word_ids)
                    keep_word_ids = [word_id for word_id in unit.word_ids if word_id not in drop_set]
        for key in ("left_modifier", "right_modifier", "head_text"):
            payload.pop(key, None)
        payload["raw_phrase"] = raw_phrase
        payload["type"] = "single_variant_modifier_redundancy"
        payload["modifiers"] = [
            {"role": "redundant_modifier", "text": f"{left_modifier}的", "position": "left"},
            {"role": "kept_modifier", "text": f"{right_modifier}的", "position": "right"},
        ] if left_modifier and right_modifier else []
        payload["head"] = head_text
        payload["redundant_modifier_word_ids"] = redundant_word_ids
        payload["keep_word_ids_after_drop"] = keep_word_ids
        payload["segmentation_confidence"] = "trusted_same_head" if redundant_word_ids and head_text else "untrusted"
        payload["requires_human_review"] = not bool(redundant_word_ids and keep_word_ids)
        if payload["requires_human_review"]:
            payload["reason"] = str(payload.get("reason") or "") + "; V21 could not bind redundant modifier to whole word ids"
        return payload

    def _self_repair_aborted_phrases(self, units: list[EditUnit], words_by_id: dict[str, Any]) -> list[RepeatCluster]:
        clusters: list[RepeatCluster] = []
        seq = 5000
        for left, right in zip(units, units[1:]):
            candidate = self_repair_aborted_phrase_candidate(left.text, right.text)
            if candidate is None:
                continue
            if not self._safe_adjacent_same_source(left, right, words_by_id):
                continue
            deterministic = bool(candidate.get("deterministic_drop_left"))
            safe_candidate = dict(candidate)
            safe_candidate["type"] = "self_repair_aborted_phrase"
            safe_candidate["issue_type"] = "self_repair_aborted_phrase"
            safe_candidate["severity"] = "high" if deterministic else "medium"
            safe_candidate["left_unit_id"] = left.unit_id
            safe_candidate["right_unit_id"] = right.unit_id
            evidence = self._evidence(
                seq=seq,
                evidence_type="semantic_retry",
                units=[left, right],
                reason="adjacent edit units look like an aborted phrase followed by a completed restart",
                confidence=0.92 if deterministic else float(candidate.get("similarity") or 0.0),
                requires_semantic_decision=not deterministic,
                metadata={"candidate": safe_candidate},
            )
            clusters.append(
                self._cluster(
                    seq=seq,
                    repeat_type="semantic_retry",
                    units=[left, right],
                    evidence=evidence,
                    local_recommendation="self_repair_drop_aborted"
                    if deterministic
                    else "self_repair_requires_semantic_adjudication",
                )
            )
            seq += 1
        return clusters

    def _safe_adjacent_same_source(self, left: EditUnit, right: EditUnit, words_by_id: dict[str, Any]) -> bool:
        if left.cut_policy == "unsafe":
            return False
        left_words = [words_by_id[word_id] for word_id in left.word_ids if word_id in words_by_id]
        right_words = [words_by_id[word_id] for word_id in right.word_ids if word_id in words_by_id]
        if not left_words or not right_words:
            return False
        left_materials = {str(getattr(word, "source_material_id", "") or "") for word in left_words}
        right_materials = {str(getattr(word, "source_material_id", "") or "") for word in right_words}
        left_segments = {str(getattr(word, "source_segment_id", "") or "") for word in left_words}
        right_segments = {str(getattr(word, "source_segment_id", "") or "") for word in right_words}
        if len(left_materials | right_materials) > 1:
            return False
        if any(left_segments) and any(right_segments) and len(left_segments | right_segments) > 1:
            return False
        gap_us = int(right.source_start_us) - int(left.source_end_us)
        return -80_000 <= gap_us <= 1_500_000

    def _intra_unit_repeats(self, units: list[EditUnit], words_by_id: dict[str, Any]) -> list[RepeatCluster]:
        clusters: list[RepeatCluster] = []
        for seq, unit in enumerate(units, start=3000):
            spans = [
                span
                for span in repeated_phrase_spans(unit.text)
                if not self._is_a_not_a_false_positive(unit.text, span)
                and not self._is_protected_quantity_reduplication(unit.text, span)
            ]
            word_audio_spans = self._word_audio_repeat_spans(unit, words_by_id)
            high_confidence = [span for span in spans if int(span.get("phrase_len") or 0) >= 2] + word_audio_spans
            if not high_confidence:
                continue
            split_metadata = self._split_metadata_for_repeat_spans(unit, words_by_id, high_confidence)
            evidence = self._evidence(
                seq=seq,
                evidence_type="hidden_audio_repeat",
                units=[unit],
                reason="single edit unit contains repeated normalized phrase span",
                confidence=0.9,
                requires_semantic_decision=False,
                metadata={"spans": high_confidence[:10], "word_tokens": self._word_tokens_for_unit(unit, words_by_id)} | split_metadata,
            )
            clusters.append(
                self._cluster(
                    seq=seq,
                    repeat_type="hidden_audio_repeat",
                    units=[unit],
                    evidence=evidence,
                    local_recommendation="requires_unit_split",
                )
            )
        return clusters

    def _word_audio_repeat_spans(self, unit: EditUnit, words_by_id: dict[str, Any]) -> list[dict[str, Any]]:
        tokens = []
        for word_id in unit.word_ids:
            word = words_by_id.get(word_id)
            tokens.append(normalize_text(str(getattr(word, "text", "") or "")))
        tokens = [token for token in tokens if token]
        spans: list[dict[str, Any]] = []
        for size in range(1, min(6, len(tokens) // 2) + 1):
            for start in range(0, len(tokens) - (size * 2) + 1):
                left = tokens[start : start + size]
                right = tokens[start + size : start + (size * 2)]
                if left == right:
                    if self._is_protected_word_audio_quantity_reduplication(tokens, start, size):
                        continue
                    spans.append(
                        {
                            "source": "word_audio_sequence",
                            "phrase": "".join(left),
                            "start_token": start,
                            "token_ngram_size": size,
                        }
                    )
        return spans

    def _split_metadata_for_repeat_spans(
        self,
        unit: EditUnit,
        words_by_id: dict[str, Any],
        spans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        for span in spans:
            phrase = str(span.get("phrase") or "")
            if not phrase:
                continue
            binding = self._whole_word_split_for_phrase(unit, words_by_id, phrase)
            drop_word_ids = list(binding.get("drop_word_ids") or [])
            if drop_word_ids:
                keep_word_ids = [word_id for word_id in unit.word_ids if word_id not in set(drop_word_ids)]
                return {
                    "split_drop_word_ids": drop_word_ids,
                    "split_keep_word_ids": keep_word_ids,
                    "split_drop_text": phrase,
                    "normalized_drop_text": normalize_text(phrase),
                    "split_binding_source": str(binding.get("binding_source") or "source_word_token_binding"),
                }
        return {"split_failed_reason": "no_safe_whole_word_binding_for_repeat_spans"}

    def _whole_word_split_for_phrase(
        self,
        unit: EditUnit,
        words_by_id: dict[str, Any],
        phrase: str,
    ) -> dict[str, Any]:
        target = normalize_text(phrase)
        tokens = self._word_tokens_for_unit(unit, words_by_id)
        if not target or not tokens:
            return {}
        exact_spans = self._exact_whole_word_spans(tokens, target)
        for span in exact_spans:
            if any(int(other["start"]) >= int(span["end"]) for other in exact_spans):
                return {"drop_word_ids": list(span["word_ids"]), "binding_source": "exact_repeated_ngram"}
            if self._later_prefix_repeat_exists(tokens, span, target):
                return {"drop_word_ids": list(span["word_ids"]), "binding_source": "short_phrase_before_longer_prefix_word"}
        return {}

    def _exact_whole_word_spans(self, tokens: list[dict[str, str]], target: str) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        for start in range(len(tokens)):
            joined = ""
            word_ids: list[str] = []
            for end in range(start, len(tokens)):
                joined += normalize_text(tokens[end]["text"])
                word_ids.append(tokens[end]["word_id"])
                if joined == target:
                    spans.append({"start": start, "end": end + 1, "word_ids": list(word_ids)})
                    break
                if len(joined) >= len(target) or not target.startswith(joined):
                    break
        return spans

    def _later_prefix_repeat_exists(self, tokens: list[dict[str, str]], span: dict[str, Any], target: str) -> bool:
        for start in range(int(span["end"]), len(tokens)):
            joined = ""
            for end in range(start, len(tokens)):
                joined += normalize_text(tokens[end]["text"])
                if joined == target or joined.startswith(target):
                    return True
                if not target.startswith(joined):
                    break
        return False

    def _word_tokens_for_unit(self, unit: EditUnit, words_by_id: dict[str, Any]) -> list[dict[str, str]]:
        tokens: list[dict[str, str]] = []
        for word_id in unit.word_ids:
            word = words_by_id.get(word_id)
            text = normalize_text(str(getattr(word, "text", "") or ""))
            if word is None or not text:
                return []
            tokens.append({"word_id": word_id, "text": text})
        return tokens

    def _is_protected_quantity_reduplication(self, text: str, span: dict[str, Any]) -> bool:
        phrase = normalize_text(str(span.get("phrase") or ""))
        if not self._looks_like_quantity_phrase(phrase):
            return False
        start = int(span.get("start_char") or 0)
        phrase_len = int(span.get("phrase_len") or len(phrase))
        norm = normalize_text(text)
        suffix_index = start + (phrase_len * 2)
        if suffix_index < 0 or suffix_index >= len(norm):
            return False
        return norm[suffix_index] in REDUPLICATION_MODIFIER_SUFFIXES

    def _is_protected_word_audio_quantity_reduplication(self, tokens: list[str], start: int, size: int) -> bool:
        phrase = "".join(tokens[start : start + size])
        if not self._looks_like_quantity_phrase(phrase):
            return False
        suffix_index = start + (size * 2)
        if suffix_index >= len(tokens):
            return False
        suffix = normalize_text(tokens[suffix_index])
        return suffix.startswith(REDUPLICATION_MODIFIER_SUFFIXES)

    def _looks_like_quantity_phrase(self, phrase: str) -> bool:
        if len(phrase) < 2 or len(phrase) > 4:
            return False
        if not phrase.startswith(CJK_NUMERAL_PREFIXES):
            return False
        return any(char not in CJK_NUMERAL_PREFIXES for char in phrase[1:])

    def _is_a_not_a_false_positive(self, text: str, span: dict[str, Any]) -> bool:
        phrase = str(span.get("phrase") or "")
        start = int(span.get("start_char") or 0)
        norm = normalize_text(text)
        if len(phrase) != 2 or phrase[1] != "不":
            return False
        if start < 0 or start + 2 >= len(norm):
            return False
        return norm[start] == phrase[0] and norm[start + 1] == "不" and norm[start + 2] == phrase[0]

    def _dedupe(self, clusters: list[RepeatCluster]) -> list[RepeatCluster]:
        deduped: list[RepeatCluster] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for cluster in clusters:
            key = (cluster.repeat_type, tuple(unit.unit_id for unit in cluster.variants))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cluster)
        return deduped
