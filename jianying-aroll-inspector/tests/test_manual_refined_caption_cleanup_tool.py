from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "manual_refined_caption_cleanup.py"


spec = importlib.util.spec_from_file_location("manual_refined_caption_cleanup", TOOL)
assert spec and spec.loader
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


def _material(material_id: str, text: str, display_text: str | None = None) -> dict:
    visible = display_text or text
    return {
        "id": material_id,
        "type": "subtitle",
        "recognize_text": text,
        "content": json.dumps({"text": visible, "styles": [{"range": [0, len(visible)]}]}, ensure_ascii=False),
        "base_content": json.dumps({"text": visible, "styles": [{"range": [0, len(visible)]}]}, ensure_ascii=False),
    }


def _text_segment(segment_id: str, material_id: str, start: int, duration: int) -> dict:
    return {
        "id": segment_id,
        "material_id": material_id,
        "target_timerange": {"start": start, "duration": duration},
    }


def _draft() -> dict:
    return {
        "tracks": [
            {
                "id": "video_track",
                "type": "video",
                "segments": [
                    {
                        "id": "video_1",
                        "material_id": "video_mat",
                        "target_timerange": {"start": 0, "duration": 5_000_000},
                    }
                ],
            },
            {
                "id": "text_track",
                "type": "text",
                "segments": [
                    _text_segment("text_1", "text_mat_1", 0, 900_000),
                    _text_segment("text_2", "text_mat_2", 1_600_000, 900_000),
                    _text_segment("text_3", "text_mat_3", 2_500_000, 900_000),
                    _text_segment("text_4", "text_mat_4", 3_400_000, 900_000),
                ],
            },
        ],
        "materials": {
            "videos": [{"id": "video_mat", "path": "D:/raw.mp4"}],
            "texts": [
                _material("text_mat_1", "低于6分的立刻滚回A2"),
                _material("text_mat_2", "说个der"),
                _material("text_mat_3", "PRO的阶段从来不是为了乞求被爱"),
                _material("text_mat_4", "把你的体脂压死在12%到15%的区间", "把你的体脂压死在\n12%到15%的区间"),
            ],
        },
    }


def test_terms_profanity_layout_and_gap_fill_preserve_video() -> None:
    draft = _draft()
    before_video_tracks = [track for track in draft["tracks"] if track["type"] != "text"]
    before_videos = list(draft["materials"]["videos"])

    cleaned, report = cleanup.build_cleanup_plan(
        draft,
        script_text="AIR PRO Logo",
        corrections={},
        mask_dirty_words=True,
        layout=True,
        fill_gaps=True,
        gap_threshold_us=80_000,
        max_fill_gap_us=2_000_000,
        max_line_chars=12,
        allow_multiple_text_tracks=False,
    )

    texts = {row["id"]: row for row in cleaned["materials"]["texts"]}
    assert texts["text_mat_1"]["recognize_text"] == "低于6分的立刻滚回AIR"
    assert texts["text_mat_2"]["recognize_text"] == "说个**"
    assert "\n" in json.loads(texts["text_mat_3"]["content"])["text"]
    assert json.loads(texts["text_mat_4"]["content"])["text"] == "把你的体脂压死在\n12%到15%的区间"
    assert cleaned["tracks"][1]["segments"][0]["target_timerange"]["duration"] == 1_600_000
    assert [track for track in cleaned["tracks"] if track["type"] != "text"] == before_video_tracks
    assert cleaned["materials"]["videos"] == before_videos
    assert report["text_correction_count"] == 2
    assert report["display_layout_count"] >= 1
    assert report["gap_fill_count"] == 1
    assert report["remaining_gap_count"] == 0


def test_script_phrase_rescue_fixes_nearby_manual_caption_drift() -> None:
    candidates = cleanup.script_phrase_candidates("皮相的目标是高级、哑光、无瑕疵。")

    fixed, reason = cleanup.rescue_with_script_phrase("干净哑光、无瑕疵", candidates)

    assert fixed == "高级、哑光、无瑕疵"
    assert reason["reason"] == "script_phrase_rescue"
    assert reason["matched_script_unit"] == "皮相的目标是高级、哑光、无瑕疵。"


def test_script_phrase_rescue_keeps_valid_script_slices() -> None:
    candidates = cleanup.script_phrase_candidates("皮相的目标是高级、哑光、无瑕疵。")

    fixed, reason = cleanup.rescue_with_script_phrase("皮相的目标是高级、哑光", candidates)

    assert fixed == "皮相的目标是高级、哑光"
    assert reason is None


def test_script_phrase_rescue_keeps_script_substrings_across_punctuation() -> None:
    script_text = "第三，皮相：肤色均匀，零瑕疵，低噪点。\n这几个硬件参数，少任何一个，直接踢回AIR阶段。"
    candidates = cleanup.script_phrase_candidates(script_text)

    fixed, reason = cleanup.rescue_with_script_phrase(
        "低噪点，这几个硬件参数",
        candidates,
        script_compact=cleanup.compact_match_text(script_text),
    )

    assert fixed == "低噪点，这几个硬件参数"
    assert reason is None


def test_cleanup_plan_suggests_script_phrase_rescue_without_applying_by_default() -> None:
    draft = {
        "tracks": [
            {
                "id": "text_track",
                "type": "text",
                "segments": [_text_segment("text_1", "text_mat_1", 0, 900_000)],
            }
        ],
        "materials": {
            "texts": [_material("text_mat_1", "干净哑光、无瑕疵")],
        },
    }

    cleaned, report = cleanup.build_cleanup_plan(
        draft,
        script_text="皮相的目标是高级、哑光、无瑕疵。",
        corrections={},
        mask_dirty_words=True,
        layout=True,
        fill_gaps=True,
        gap_threshold_us=80_000,
        max_fill_gap_us=2_000_000,
        max_line_chars=16,
        allow_multiple_text_tracks=False,
    )

    text = cleaned["materials"]["texts"][0]
    assert text["recognize_text"] == "干净哑光、无瑕疵"
    assert report["text_correction_count"] == 0
    assert report["script_phrase_suggestion_count"] == 1
    assert report["suggestions"][0]["suggested_text"] == "高级、哑光、无瑕疵"
