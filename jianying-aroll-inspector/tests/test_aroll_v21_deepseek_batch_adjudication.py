from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

from aroll_v21.decision.deepseek_semantic_planner import DeepSeekSemanticProvider
from aroll_v21.decision.semantic_contracts import (
    SemanticAdjudicationDecisionType,
    SemanticAdjudicationRequest,
    SemanticIssueSeverity,
    SemanticIssueType,
)
from aroll_v21.operator import ArollV21OperatorConfig, run_operator
from tests.test_aroll_v21_semantic_adjudication_layer import _two_caption_input, _write_input


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class _BatchResponder:
    def __init__(self, *, response_builder=None, failures: list[Any] | None = None) -> None:  # type: ignore[no-untyped-def]
        self.response_builder = response_builder or self._default_response
        self.failures = list(failures or [])
        self.calls = []
        self.payloads = []

    def __call__(self, request, timeout=0):  # type: ignore[no-untyped-def]
        self.calls.append({"request": request, "timeout": timeout})
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, BaseException):
                raise failure
            return _FakeHttpResponse(failure)
        payload = json.loads(request.data.decode("utf-8"))
        content = json.loads(payload["messages"][1]["content"])
        self.payloads.append(content)
        return _FakeHttpResponse(self.response_builder(content))

    def _default_response(self, content: dict[str, Any]) -> dict[str, Any]:
        decisions = [
            {
                "issue_id": issue["issue_id"],
                "decision_type": "drop_right",
                "action": "drop_right",
                "confidence": 0.91,
                "reason": "batch test decision",
                "drop_word_ids": [],
                "keep_word_ids": [],
                "requires_human_review": False,
            }
            for issue in content["issues"]
        ]
        return _deepseek_envelope({"batch_id": content["batch_id"], "decisions": decisions})


def _deepseek_envelope(content: dict[str, Any] | str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)}}]}


def _request(issue_id: str, *, issue_type: SemanticIssueType = SemanticIssueType.SEMANTIC_CONTAINMENT) -> SemanticAdjudicationRequest:
    return SemanticAdjudicationRequest(
        issue_id=issue_id,
        issue_type=issue_type,
        severity=SemanticIssueSeverity.MEDIUM,
        candidate_segment_ids=[f"seg_{issue_id}"],
        candidate_caption_ids=[f"cap_{issue_id}"],
        word_ids=[f"w_{issue_id}_1", f"w_{issue_id}_2"],
        text_before=f"left {issue_id}",
        text_after=f"right {issue_id}",
        local_context={"cluster_type": issue_id},
        allowed_decisions=["keep_all", "drop_left", "drop_right", "repair_text", "requires_human_review", "no_decision"],
    )


class ArollV21DeepSeekBatchAdjudicationTests(unittest.TestCase):
    def test_deepseek_batch_adjudicates_multiple_provider_required_issues_in_one_call(self) -> None:
        responder = _BatchResponder()
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")
        requests = [_request("issue_1"), _request("issue_2"), _request("issue_3")]

        with patch("urllib.request.urlopen", new=responder):
            decisions = provider.decide(requests)

        self.assertEqual(len(decisions), 3)
        self.assertEqual(len(responder.calls), 1)
        self.assertEqual(provider.provider_called_count, 1)
        self.assertEqual(provider.deepseek_batch_request_count, 1)
        self.assertEqual(provider.deepseek_batch_issue_count, 3)
        self.assertEqual(provider.deepseek_batch_resolved_count, 3)
        self.assertEqual(responder.payloads[0]["mode"], "auto")
        self.assertEqual([issue["issue_id"] for issue in responder.payloads[0]["issues"]], ["issue_1", "issue_2", "issue_3"])

    def test_deepseek_batch_maps_each_response_by_issue_id(self) -> None:
        def reversed_response(content: dict[str, Any]) -> dict[str, Any]:
            decisions = [
                {
                    "issue_id": issue["issue_id"],
                    "decision_type": "drop_left" if issue["issue_id"] == "issue_2" else "drop_right",
                    "action": "drop_left" if issue["issue_id"] == "issue_2" else "drop_right",
                    "confidence": 0.9,
                    "reason": "out of order response",
                }
                for issue in reversed(content["issues"])
            ]
            decisions.append({"issue_id": "unknown_issue", "decision_type": "drop_right", "action": "drop_right"})
            return _deepseek_envelope({"batch_id": content["batch_id"], "decisions": decisions})

        responder = _BatchResponder(response_builder=reversed_response)
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            decisions = provider.decide([_request("issue_1"), _request("issue_2")])

        by_issue = {decision.issue_id: decision.decision for decision in decisions}
        self.assertEqual(by_issue["issue_1"], SemanticAdjudicationDecisionType.DROP_RIGHT)
        self.assertEqual(by_issue["issue_2"], SemanticAdjudicationDecisionType.DROP_LEFT)
        self.assertEqual(provider.deepseek_batch_unknown_issue_ids, ["unknown_issue"])

    def test_deepseek_batch_missing_issue_result_fail_closed(self) -> None:
        def partial_response(content: dict[str, Any]) -> dict[str, Any]:
            first = content["issues"][0]
            return _deepseek_envelope(
                {
                    "batch_id": content["batch_id"],
                    "decisions": [
                        {
                            "issue_id": first["issue_id"],
                            "decision_type": "drop_right",
                            "action": "drop_right",
                            "confidence": 0.9,
                            "reason": "partial",
                        }
                    ],
                }
            )

        responder = _BatchResponder(response_builder=partial_response)
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            decisions = provider.decide([_request("issue_1"), _request("issue_2")])

        self.assertEqual(len(decisions), 1)
        self.assertEqual(provider.provider_called_count, 3)
        self.assertEqual(provider.deepseek_batch_retry_count, 2)
        self.assertEqual(provider.deepseek_batch_missing_issue_ids, ["issue_2"])
        self.assertIn("V21_SEMANTIC_BATCH_PARTIAL_RESPONSE", provider.deepseek_batch_error)

    def test_deepseek_batch_retries_json_parse_failure_then_succeeds(self) -> None:
        responder = _BatchResponder(failures=[_deepseek_envelope("not json")])
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            decisions = provider.decide([_request("issue_1")])

        self.assertEqual(len(decisions), 1)
        self.assertEqual(provider.provider_called_count, 2)
        self.assertEqual(provider.deepseek_batch_retry_count, 1)

    def test_deepseek_batch_retries_429_then_succeeds(self) -> None:
        http_429 = urllib.error.HTTPError("https://example.invalid", 429, "rate limit", hdrs=None, fp=None)
        responder = _BatchResponder(failures=[http_429])
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            decisions = provider.decide([_request("issue_1")])

        self.assertEqual(len(decisions), 1)
        self.assertEqual(provider.provider_called_count, 2)
        self.assertEqual(provider.deepseek_batch_retry_count, 1)

    def test_deepseek_batch_stops_after_three_attempts(self) -> None:
        responder = _BatchResponder(failures=[_deepseek_envelope("not json"), _deepseek_envelope("not json"), _deepseek_envelope("not json")])
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            with self.assertRaises(RuntimeError):
                provider.decide([_request("issue_1")])

        self.assertEqual(provider.provider_called_count, 3)
        self.assertEqual(provider.deepseek_batch_attempt_count, 3)
        self.assertEqual(provider.deepseek_batch_retry_count, 2)
        self.assertIn("V21_SEMANTIC_BATCH_PROVIDER_FAILED", provider.deepseek_batch_error)

    def test_deepseek_batch_rejects_mismatched_batch_id(self) -> None:
        def wrong_batch_id(_content: dict[str, Any]) -> dict[str, Any]:
            return _deepseek_envelope({"batch_id": "wrong_batch", "decisions": []})

        responder = _BatchResponder(response_builder=wrong_batch_id)
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            with self.assertRaises(RuntimeError):
                provider.decide([_request("issue_1")])

        self.assertEqual(provider.provider_called_count, 3)
        self.assertEqual(provider.deepseek_batch_retry_count, 2)
        self.assertIn("batch_id mismatch", provider.deepseek_batch_error)

    def test_deepseek_batch_rejects_duplicate_issue_decisions(self) -> None:
        def duplicate_issue(content: dict[str, Any]) -> dict[str, Any]:
            decisions = []
            for issue in content["issues"]:
                decisions.append(
                    {
                        "issue_id": issue["issue_id"],
                        "decision_type": "drop_right",
                        "action": "drop_right",
                        "confidence": 0.9,
                        "reason": "first",
                    }
                )
            decisions.append(
                {
                    "issue_id": content["issues"][0]["issue_id"],
                    "decision_type": "drop_left",
                    "action": "drop_left",
                    "confidence": 0.9,
                    "reason": "conflicting duplicate",
                }
            )
            return _deepseek_envelope({"batch_id": content["batch_id"], "decisions": decisions})

        responder = _BatchResponder(response_builder=duplicate_issue)
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            with self.assertRaises(RuntimeError):
                provider.decide([_request("issue_1"), _request("issue_2")])

        self.assertEqual(provider.provider_called_count, 3)
        self.assertEqual(provider.deepseek_batch_retry_count, 2)
        self.assertIn("duplicate decision", provider.deepseek_batch_error)

    def test_deepseek_batch_provider_missing_no_retry_fail_closed(self) -> None:
        provider = DeepSeekSemanticProvider(api_key="")
        with self.assertRaises(RuntimeError):
            provider.decide([_request("issue_1")])
        self.assertEqual(provider.provider_called_count, 0)
        self.assertEqual(provider.deepseek_batch_attempt_count, 0)

    def test_deepseek_batch_requires_human_review_blocks_ready(self) -> None:
        def human_review(content: dict[str, Any]) -> dict[str, Any]:
            issue = content["issues"][0]
            return _deepseek_envelope(
                {
                    "batch_id": content["batch_id"],
                    "decisions": [
                        {
                            "issue_id": issue["issue_id"],
                            "decision_type": "requires_human_review",
                            "action": "requires_human_review",
                            "confidence": 0.4,
                            "reason": "ambiguous",
                            "requires_human_review": True,
                        }
                    ],
                }
            )

        provider = DeepSeekSemanticProvider(api_key="unit-test-token")
        with patch("urllib.request.urlopen", new=_BatchResponder(response_builder=human_review)):
            decisions = provider.decide([_request("issue_1")])

        self.assertEqual(decisions[0].decision, SemanticAdjudicationDecisionType.REQUIRES_HUMAN_REVIEW)
        self.assertEqual(provider.deepseek_batch_resolved_count, 0)
        self.assertEqual(provider.deepseek_batch_unresolved_count, 1)

    def test_commit_reuses_batch_semantic_cache_without_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            run_dir = root / "run"
            _write_input(input_json, _two_caption_input("这个问题需要从", "这个问题可以看"))
            (run_dir).mkdir()
            (run_dir / "semantic_decision_cache.json").write_text(
                json.dumps(
                    [
                        {
                            "cluster_id": "cluster_1",
                            "decision": "drop_aborted",
                            "keep_unit_id": "unit_2",
                            "drop_unit_ids": ["unit_1"],
                            "reason": "cached",
                            "confidence": 0.9,
                        }
                    ],
                    ensure_ascii=False,
                ),
                "utf-8",
            )

            with patch("aroll_v21.operator.deepseek_provider_from_env", side_effect=AssertionError("provider must not be called")):
                summary = run_operator(
                    ArollV21OperatorConfig(mode="write", run_dir=run_dir, input_json=input_json, semantic_mode="auto")
                )

        self.assertEqual(summary["deepseek_provider_called_count"], 0)
        self.assertTrue(summary["semantic_decision_cache_used"])
        self.assertTrue(summary["commit_reused_semantic_cache"])

    def test_commit_blocks_when_batch_cache_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            run_dir = root / "run"
            _write_input(input_json, _two_caption_input("这个问题需要从", "这个问题可以看"))

            with patch("aroll_v21.operator.deepseek_provider_from_env", side_effect=AssertionError("provider must not be called")):
                summary = run_operator(
                    ArollV21OperatorConfig(
                        mode="write",
                        run_dir=run_dir,
                        input_json=input_json,
                        semantic_mode="auto",
                        commit=True,
                    )
                )

        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(summary["deepseek_provider_called_count"], 0)

    def test_batch_report_counts_requests_vs_issues_correctly(self) -> None:
        responder = _BatchResponder()
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            provider.decide([_request(f"issue_{index}") for index in range(19)])

        self.assertEqual(provider.provider_called_count, 4)
        self.assertEqual(provider.deepseek_batch_request_count, 4)
        self.assertEqual(provider.deepseek_batch_chunk_sizes, [6, 6, 6, 1])
        self.assertEqual(provider.deepseek_batch_issue_count, 19)
        self.assertEqual(provider.deepseek_batch_resolved_count, 19)

    def test_structural_issues_do_not_enter_deepseek_batch(self) -> None:
        requests = [_request("semantic_issue")]
        responder = _BatchResponder()
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")

        with patch("urllib.request.urlopen", new=responder):
            provider.decide(requests)

        issue_types = {issue["issue_type"] for issue in responder.payloads[0]["issues"]}
        self.assertEqual(issue_types, {"semantic_containment"})
        self.assertNotIn("audio_coverage", issue_types)
        self.assertNotIn("root_mirror", issue_types)

    def test_final_target_repeat_and_restart_take_share_same_batch(self) -> None:
        responder = _BatchResponder()
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")
        requests = [
            _request("final_target_repeat_issue", issue_type=SemanticIssueType.SEMANTIC_CONTAINMENT),
            _request("restart_take_issue", issue_type=SemanticIssueType.AMBIGUOUS_REPEAT),
        ]

        with patch("urllib.request.urlopen", new=responder):
            provider.decide(requests)

        self.assertEqual(provider.provider_called_count, 1)
        self.assertEqual([issue["issue_id"] for issue in responder.payloads[0]["issues"]], ["final_target_repeat_issue", "restart_take_issue"])

    def test_batch_chunking_does_not_degrade_to_one_issue_one_call(self) -> None:
        responder = _BatchResponder()
        provider = DeepSeekSemanticProvider(api_key="unit-test-token")
        requests = [_request(f"issue_{index}") for index in range(6)]

        with patch("aroll_v21.decision.deepseek_semantic_planner.DEFAULT_BATCH_MAX_ISSUES", 3):
            with patch("urllib.request.urlopen", new=responder):
                provider.decide(requests)

        self.assertEqual(provider.provider_called_count, 2)
        self.assertEqual(provider.deepseek_batch_chunk_sizes, [3, 3])
        self.assertLess(provider.provider_called_count, len(requests))


if __name__ == "__main__":
    unittest.main()
