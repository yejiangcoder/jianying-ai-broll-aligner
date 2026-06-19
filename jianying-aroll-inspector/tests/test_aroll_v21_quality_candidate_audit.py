from __future__ import annotations

import unittest

from aroll_v21.quality.quality_candidate_audit import build_quality_candidate_audit_from_artifacts


def _word(
    word_id: str,
    text: str,
    start_us: int,
    end_us: int,
    *,
    debug_hints: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "word_id": word_id,
        "text": text,
        "normalized_text": text,
        "source_start_us": start_us,
        "source_end_us": end_us,
        "debug_hints": debug_hints or {},
    }


def _segment(segment_id: str, word_ids: list[str], text: str, start_us: int, end_us: int) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "target_start_us": start_us,
        "target_end_us": end_us,
        "source_start_us": start_us,
        "source_end_us": end_us,
        "word_ids": word_ids,
        "text": text,
    }


def _caption(caption_id: str, segment_id: str, word_ids: list[str], text: str, start_us: int, end_us: int) -> dict[str, object]:
    return {
        "caption_id": caption_id,
        "timeline_segment_ids": [segment_id],
        "word_ids": word_ids,
        "text": text,
        "target_start_us": start_us,
        "target_end_us": end_us,
        "spoken_source_start_us": start_us,
        "spoken_source_end_us": end_us,
    }


class ArollQualityCandidateAuditTest(unittest.TestCase):
    def test_detects_hidden_audio_prefix_restart_when_visible_caption_collapsed_repeat(self) -> None:
        source_graph = {
            "words": [
                _word("w001", "拍", 0, 120_000),
                _word("w002", "拍出", 160_000, 420_000),
                _word("w003", "好几张", 420_000, 820_000),
            ]
        }
        final_timeline = [
            _segment("seg001", ["w001", "w002", "w003"], "拍出好几张", 0, 820_000),
        ]
        captions = [
            _caption("cap001", "seg001", ["w001", "w002", "w003"], "拍出好几张展示面照片", 0, 820_000),
        ]

        audit = build_quality_candidate_audit_from_artifacts(
            run_name="run",
            run_dir="D:/auto_clip_runtime/run",
            captions=captions,
            final_timeline=final_timeline,
            source_graph=source_graph,
        )

        candidates = [row for row in audit["candidates"] if row["type"] == "hidden_audio_prefix_restart"]
        self.assertEqual(len(candidates), 1, audit)
        self.assertEqual(candidates[0]["target_word_ids"], ["w001", "w002"])
        self.assertEqual(candidates[0]["evidence"]["prefix_word_text"], "拍")
        self.assertEqual(candidates[0]["evidence"]["restart_word_text"], "拍出")

    def test_detects_visible_internal_prefix_restart(self) -> None:
        source_graph = {"words": [_word("w001", "但哪但人家集美哪怕是去拼", 0, 1_000_000)]}
        final_timeline = [
            _segment("seg001", ["w001"], "但哪但人家集美哪怕是去拼", 0, 1_000_000),
        ]
        captions = [
            _caption("cap001", "seg001", ["w001"], "但哪但人家集美哪怕是去拼", 0, 1_000_000),
        ]

        audit = build_quality_candidate_audit_from_artifacts(
            run_name="run",
            run_dir="D:/auto_clip_runtime/run",
            captions=captions,
            final_timeline=final_timeline,
            source_graph=source_graph,
        )

        candidates = [row for row in audit["candidates"] if row["type"] == "visible_internal_prefix_restart"]
        self.assertEqual(len(candidates), 1, audit)
        self.assertEqual(candidates[0]["target_caption_ids"], ["cap001"])
        self.assertEqual(candidates[0]["evidence"]["prefix_text"], "但")

    def test_detects_intraword_restart_normalized_by_asr_text_cleanup(self) -> None:
        source_graph = {
            "words": [
                _word(
                    "w001",
                    "拍",
                    0,
                    240_000,
                    debug_hints={
                        "intraword_cjk_restart_normalized": True,
                        "original_text": "拍拍",
                        "normalization_reason": "leading_single_char_restart_before_result_complement",
                    },
                ),
                _word("w002", "出", 240_000, 320_000),
                _word("w003", "好几组", 320_000, 620_000),
            ]
        }
        final_timeline = [
            _segment("seg001", ["w001", "w002", "w003"], "拍出好几组", 0, 620_000),
        ]
        captions = [
            _caption("cap001", "seg001", ["w001", "w002", "w003"], "拍出好几组展示面照片", 0, 620_000),
        ]

        audit = build_quality_candidate_audit_from_artifacts(
            run_name="run",
            run_dir="D:/auto_clip_runtime/run",
            captions=captions,
            final_timeline=final_timeline,
            source_graph=source_graph,
        )

        candidates = [row for row in audit["candidates"] if row["type"] == "intraword_audio_restart_normalized"]
        self.assertEqual(len(candidates), 1, audit)
        self.assertEqual(candidates[0]["target_word_ids"], ["w001"])
        self.assertFalse(candidates[0]["evidence"]["word_boundary_safe_to_auto_apply"])
        self.assertEqual(candidates[0]["evidence"]["original_text"], "拍拍")

    def test_prefixed_reduplicated_lexeme_is_not_reported_as_intraword_audio_restart(self) -> None:
        source_graph = {
            "words": [
                _word("w001", "拼", 0, 80_000),
                _word(
                    "w002",
                    "夕",
                    120_000,
                    280_000,
                    debug_hints={
                        "intraword_cjk_restart_normalized": True,
                        "original_text": "夕夕",
                        "normalization_reason": "leading_single_char_restart_before_result_complement",
                    },
                ),
                _word("w003", "上", 320_000, 460_000),
                _word("w004", "毫无", 460_000, 760_000),
            ]
        }
        final_timeline = [
            _segment("seg001", ["w001", "w002", "w003", "w004"], "拼夕上毫无", 0, 760_000),
        ]
        captions = [
            _caption("cap001", "seg001", ["w001", "w002", "w003", "w004"], "你买的全是拼夕上毫无", 0, 760_000),
        ]

        audit = build_quality_candidate_audit_from_artifacts(
            run_name="run",
            run_dir="D:/auto_clip_runtime/run",
            captions=captions,
            final_timeline=final_timeline,
            source_graph=source_graph,
        )

        self.assertFalse(
            any(row["type"] == "intraword_audio_restart_normalized" for row in audit["candidates"]),
            audit,
        )

    def test_ignores_trivial_hidden_audio_particle_prefix(self) -> None:
        source_graph = {
            "words": [
                _word("w001", "的", 0, 120_000),
                _word("w002", "的确", 160_000, 420_000),
            ]
        }
        final_timeline = [_segment("seg001", ["w001", "w002"], "的确", 0, 420_000)]
        captions = [_caption("cap001", "seg001", ["w001", "w002"], "的确", 0, 420_000)]

        audit = build_quality_candidate_audit_from_artifacts(
            run_name="run",
            run_dir="D:/auto_clip_runtime/run",
            captions=captions,
            final_timeline=final_timeline,
            source_graph=source_graph,
        )

        self.assertFalse(any(row["type"] == "hidden_audio_prefix_restart" for row in audit["candidates"]), audit)


if __name__ == "__main__":
    unittest.main()
