from __future__ import annotations

from aroll_v21.ir.models import CaptionRenderUnit, FinalTimelineSegment
from aroll_v21.quality.caption_alignment import build_caption_alignment_report
from aroll_v21.quality.final_caption_visible_repeat import build_final_caption_visible_repeat_gate


def _caption(index: int, start_us: int, end_us: int, text: str) -> CaptionRenderUnit:
    return CaptionRenderUnit(
        caption_id=f"v21_cap_{index:06d}",
        timeline_segment_ids=[f"v21_seg_{index:06d}"],
        word_ids=[f"w{index:03d}"],
        text=text,
        target_start_us=start_us,
        target_end_us=end_us,
        source_subtitle_uids=[f"s{index:03d}"],
        style_template_id="canonical_caption_template",
    )


def test_repeat_gate_distant_topic_containment_not_fatal() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 700_000, "集美"),
            _caption(2, 20_000_000, 21_200_000, "很多集美今天都在讨论这个问题"),
        ]
    )

    assert gate["gate_passed"] is True
    assert gate["containment_repeat_count"] == 0
    assert gate["containment_repeat_raw_count"] == 1
    assert gate["visible_repeat_allow_candidate_count"] == 1
    candidate = gate["repeat_classification_candidates"][0]
    assert candidate["classification"] == "short_concept_reuse"
    assert candidate["semantic_repeat_class"] == "semantic_expansion"
    assert candidate["needs_semantic_arbitration"] is False
    assert candidate["distance_kind"] == "distant"
    assert candidate["severity"] == "allow"


def test_repeat_gate_distant_short_concept_recurrence_not_fatal() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 900_000, "敢展示"),
            _caption(2, 18_000_000, 19_300_000, "敢展示自己并不等于重复表达"),
        ]
    )

    assert gate["gate_passed"] is True
    assert gate["containment_repeat_count"] == 0
    assert gate["visible_repeat_allow_candidate_count"] == 1
    assert gate["repeat_classification_candidates"][0]["classification"] == "short_concept_reuse"


def test_repeat_gate_repeated_address_or_topic_terms_across_video_not_fatal() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 700_000, "朋友们"),
            _caption(2, 10_000_000, 11_000_000, "朋友们先看这个条件"),
            _caption(3, 22_000_000, 23_000_000, "朋友们最后再回到结论"),
        ]
    )

    assert gate["gate_passed"] is True
    assert gate["containment_repeat_count"] == 0
    assert gate["visible_repeat_allow_candidate_count"] >= 1
    assert all(
        candidate["severity"] != "fatal"
        for candidate in gate["repeat_classification_candidates"]
    )


def test_repeat_gate_distant_semantic_recurrence_warns_without_blocking() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 1_000_000, "我们重新开始吧"),
            _caption(2, 16_000_000, 17_200_000, "大家都要重新开始"),
        ]
    )

    assert gate["gate_passed"] is True
    assert gate["ngram_repeat_count"] == 0
    assert gate["ngram_repeat_raw_count"] >= 1
    assert gate["visible_repeat_warning_candidate_count"] >= 1
    candidate = gate["visible_repeat_warning_candidates"][0]
    assert candidate["classification"] == "distant_semantic_recurrence"
    assert candidate["distance_kind"] == "distant"


def test_repeat_gate_adjacent_restart_still_blocks() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 1_000_000, "咳咳就你骂"),
            _caption(2, 1_000_000, 2_600_000, "就你骂集美虚容啊"),
        ]
    )

    assert gate["gate_passed"] is False
    assert gate["prefix_suffix_overlap_count"] >= 1
    assert "V21_FINAL_CAPTION_VISIBLE_REPEAT_GATE_FAILED" in gate["blocker_codes"]
    candidate = gate["visible_repeat_candidates"][0]
    assert candidate["severity"] == "fatal"
    assert candidate["distance_kind"] == "adjacent"


def test_repeat_gate_near_restart_still_blocks_without_threshold_escape() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 900_000, "我们重新开始"),
            _caption(2, 3_000_000, 3_900_000, "重新开始吧"),
        ]
    )

    assert gate["gate_passed"] is False
    assert gate["prefix_suffix_overlap_count"] >= 1
    assert gate["visible_repeat_candidates"][0]["distance_kind"] == "near"


def test_repeat_gate_adjacent_shared_ngram_without_boundary_restart_warns_only() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 1_600_000, "他的购物车全是投资和享受"),
            _caption(2, 1_620_000, 2_300_000, "你的购物车"),
        ]
    )

    assert gate["gate_passed"] is True
    assert gate["ngram_repeat_count"] == 0
    assert gate["ngram_repeat_raw_count"] >= 1
    assert gate["visible_repeat_warning_candidate_count"] >= 1
    candidate = gate["visible_repeat_warning_candidates"][0]
    assert candidate["classification"] == "local_semantic_recurrence"
    assert candidate["shared_text"] == "的购物车"
    assert candidate["left_tail"] == "全是投资和享受"
    assert candidate["right_tail"] == ""
    assert candidate["is_boundary_restart"] is False
    assert candidate["has_progressive_marker"] is False
    assert candidate["has_parallel_structure"] is False
    assert candidate["left_is_fragment"] is False
    assert candidate["right_completes_left"] is False
    assert candidate["caption_gap_us"] == 20_000
    assert candidate["source_gap_us"] is None
    assert candidate["semantic_classification"] == "local_semantic_recurrence"
    assert candidate["semantic_repeat_class"] == "ambiguous_restart_or_progression"
    assert candidate["needs_semantic_arbitration"] is True
    assert candidate["semantic_repeat_evidence"]["shared_text"] == candidate["shared_text"]
    assert candidate["semantic_repeat_evidence"]["needs_semantic_arbitration"] is True
    assert candidate["distance_kind"] == "adjacent"
    assert gate["repeat_semantic_arbitration_candidate_count"] == 1
    assert gate["repeat_semantic_arbitration_request_count"] == 1
    payload = gate["repeat_semantic_arbitration_request_payloads"][0]
    assert payload["issue_type"] == "ambiguous_repeat"
    assert payload["cluster_type"] == "final_visible_ambiguous_repeat"
    assert payload["type"] == "semantic_decision_required"
    assert payload["severity"] == "medium"
    assert payload["warning_only"] is True
    assert payload["left_text"] == "他的购物车全是投资和享受"
    assert payload["right_text"] == "你的购物车"
    assert payload["local_context"]["semantic_repeat_evidence"]["shared_text"] == "的购物车"
    assert payload["local_context"]["semantic_repeat_evidence"]["needs_semantic_arbitration"] is True
    assert payload["recommended_action"] == "no_decision"
    assert "keep_all" in payload["allowed_decisions"]
    assert "requires_human_review" in payload["allowed_decisions"]


def test_repeat_gate_progressive_semantic_expansion_is_warning_only() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 1_200_000, "你更没钱去健身了"),
            _caption(2, 1_200_000, 2_400_000, "也没钱去做YM了"),
            _caption(3, 2_400_000, 3_600_000, "也没钱去投资自己了"),
            _caption(4, 3_600_000, 4_800_000, "你变得更丑了"),
            _caption(5, 4_800_000, 6_000_000, "哈你更没有认识"),
            _caption(6, 6_000_000, 7_200_000, "你更没有你变得"),
            _caption(7, 7_200_000, 8_400_000, "你更没有认识异性的渠道了"),
            _caption(8, 8_400_000, 9_600_000, "就变得更压抑"),
        ]
    )

    assert gate["gate_passed"] is True
    assert gate["repeat_semantic_arbitration_candidate_count"] == 0
    assert gate["repeat_semantic_arbitration_request_payloads"] == []
    assert gate["visible_repeat_candidate_count"] == 0
    assert gate["restart_repeat_visible_count"] == 0
    assert gate["ngram_repeat_count"] == 0
    assert {
        candidate["overlap_text"]
        for candidate in gate["visible_repeat_warning_candidates"]
    } == {"也没钱去", "你更没有"}
    assert all(
        candidate["classification"] == "parallel_progressive_semantic_expansion"
        for candidate in gate["visible_repeat_warning_candidates"]
    )
    assert gate["repeat_semantic_class_counts"]["parallel_progression"] == 3
    progressive_pairs = [
        candidate
        for candidate in gate["visible_repeat_warning_candidates"]
        if candidate["overlap_text"] == "你更没有"
    ]
    assert len(progressive_pairs) == 2
    assert all(candidate["classification"] == "parallel_progressive_semantic_expansion" for candidate in progressive_pairs)
    assert all(candidate["semantic_repeat_class"] == "parallel_progression" for candidate in progressive_pairs)
    assert all(candidate["needs_semantic_arbitration"] is False for candidate in progressive_pairs)
    assert all(candidate["severity"] == "warning" for candidate in progressive_pairs)
    assert all("protected_semantic_structure" in candidate["risk_tags"] for candidate in progressive_pairs)
    first_progressive_pair = next(candidate for candidate in progressive_pairs if candidate["text"] == "哈你更没有认识")
    assert first_progressive_pair["shared_text"] == "你更没有"
    assert first_progressive_pair["left_prefix"] == "哈"
    assert first_progressive_pair["left_tail"] == "认识"
    assert first_progressive_pair["right_tail"] == "你变得"
    assert first_progressive_pair["is_boundary_restart"] is False
    assert first_progressive_pair["has_progressive_marker"] is True
    assert first_progressive_pair["has_parallel_structure"] is True
    assert first_progressive_pair["left_is_fragment"] is False
    assert first_progressive_pair["right_completes_left"] is False
    assert first_progressive_pair["semantic_classification"] == "parallel_progressive_semantic_expansion"
    assert first_progressive_pair["semantic_repeat_evidence"]["has_parallel_structure"] is True
    assert first_progressive_pair["semantic_repeat_evidence"]["caption_gap_us"] == 0


def test_repeat_gate_progressive_fragment_still_blocks_as_restart() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 900_000, "也没钱去"),
            _caption(2, 900_000, 2_100_000, "也没钱去健身了"),
        ]
    )

    assert gate["gate_passed"] is False
    assert gate["repeat_semantic_arbitration_candidate_count"] == 0
    assert gate["repeat_semantic_arbitration_request_payloads"] == []
    assert gate["visible_repeat_candidate_count"] >= 1
    assert any(candidate["severity"] == "fatal" for candidate in gate["visible_repeat_candidates"])
    assert any(candidate["semantic_repeat_class"] == "stutter_restart" for candidate in gate["visible_repeat_candidates"])
    assert all(candidate["needs_semantic_arbitration"] is False for candidate in gate["visible_repeat_candidates"])
    restart_candidate = next(candidate for candidate in gate["visible_repeat_candidates"] if candidate["overlap_text"] == "也没钱去")
    assert restart_candidate["shared_text"] == "也没钱去"
    assert restart_candidate["left_tail"] == ""
    assert restart_candidate["right_tail"] == "健身了"
    assert restart_candidate["is_boundary_restart"] is True
    assert restart_candidate["has_progressive_marker"] is True
    assert restart_candidate["has_parallel_structure"] is False
    assert restart_candidate["left_is_fragment"] is True
    assert restart_candidate["right_completes_left"] is True
    assert restart_candidate["semantic_classification"] == restart_candidate["classification"]
    assert restart_candidate["semantic_repeat_evidence"]["right_completes_left"] is True


def test_repeat_gate_parallel_enumeration_does_not_enter_semantic_arbitration() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 900_000, "他要观察你的结构"),
            _caption(2, 900_000, 1_800_000, "观察你的状态"),
            _caption(3, 1_800_000, 2_700_000, "观察你的表达"),
        ]
    )

    assert gate["gate_passed"] is True
    assert gate["visible_repeat_candidate_count"] == 0
    assert gate["repeat_semantic_arbitration_candidate_count"] == 0
    assert gate["repeat_semantic_arbitration_request_payloads"] == []
    assert all(
        candidate["classification"] == "parallel_progressive_semantic_expansion"
        for candidate in gate["visible_repeat_warning_candidates"]
    )
    assert all(candidate["semantic_repeat_class"] == "parallel_progression" for candidate in gate["visible_repeat_warning_candidates"])
    assert all(candidate["has_parallel_structure"] is True for candidate in gate["visible_repeat_warning_candidates"])


def test_repeat_gate_same_caption_restart_still_blocks() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 900_000, "你是你们是极度恐慌"),
        ]
    )

    assert gate["gate_passed"] is False
    assert gate["restart_repeat_visible_count"] == 1
    assert gate["restart_repeat_visible_candidates"][0]["classification"] == "same_segment_restart"


def test_repeat_gate_medium_repeated_island_shape_not_fatal() -> None:
    gate = build_final_caption_visible_repeat_gate(
        [
            _caption(1, 0, 1_200_000, "我们讨论他们我们继续"),
        ]
    )

    assert gate["gate_passed"] is True
    assert gate["restart_repeat_visible_count"] == 0
    assert gate["visible_repeat_candidate_count"] == 0


def test_repeat_gate_classification_does_not_downgrade_caption_coverage_missing() -> None:
    caption = _caption(1, 0, 1_000_000, "短概念")
    report = build_caption_alignment_report(
        final_timeline=[
            FinalTimelineSegment(
                segment_id="v21_seg_000001",
                source_material_id="video",
                source_segment_id="src",
                source_start_us=0,
                source_end_us=1_000_000,
                target_start_us=0,
                target_end_us=1_000_000,
                word_ids=["w001", "w002"],
                text="短概念",
                decision_ids=[],
            )
        ],
        captions=[caption],
    )

    assert report["gate_passed"] is False
    assert report["missing_final_timeline_caption_word_count"] == 1
    assert "V21_FINAL_TIMELINE_CAPTION_WORD_COVERAGE_FAILED" in report["blocker_codes"]
