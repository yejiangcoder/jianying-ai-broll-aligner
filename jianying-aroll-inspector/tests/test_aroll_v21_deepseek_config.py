from __future__ import annotations

import json
import uuid
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aroll_v21.decision.deepseek_semantic_planner as deepseek_planner
from aroll_v21.decision.deepseek_semantic_planner import (
    _load_deepseek_config_from_files,
    _parse_deepseek_yaml_config,
    _decision_from_provider_row,
    DeepSeekSemanticProvider,
    deepseek_provider_from_env,
    deepseek_provider_from_runtime_config,
)
from aroll_v21.decision import (
    SemanticAdjudicationDecision,
    SemanticAdjudicationDecisionType,
    SemanticAdjudicationRequest,
    SemanticIssueSeverity,
    SemanticIssueType,
)
from aroll_v21.operator import ArollV21OperatorConfig, run_operator


class _FakeDeepSeekResponse:
    def __init__(self, batch_id: str = "") -> None:
        self.batch_id = batch_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        content = json.dumps({"batch_id": self.batch_id, "decisions": []})
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


class ArollV21DeepSeekConfigTests(unittest.TestCase):
    def _write_deepseek_yaml(self, path: Path, key: str) -> None:
        path.write_text(
            "\n".join(
                [
                    "deepseek:",
                    f"  api-key: {key}",
                    "  base-url: https://api.deepseek.com",
                    "app:",
                    "  ai:",
                    "    semantic:",
                    "      model: deepseek-chat",
                ]
            ),
            "utf-8",
        )

    def _write_operator_input(self, path: Path) -> str:
        words = []
        subtitles = []
        cursor = 0
        left = "这个问题需要从"
        right = "这个问题可以看"
        for subtitle_index, text in enumerate([left, right], start=1):
            word_ids = []
            for char in text:
                word_id = f"w_{len(words) + 1:06d}"
                word_ids.append(word_id)
                words.append(
                    {
                        "word_id": word_id,
                        "word_text": char,
                        "source_start_us": cursor,
                        "source_end_us": cursor + 90_000,
                        "subtitle_uid": f"s{subtitle_index}",
                        "subtitle_index": subtitle_index,
                    }
                )
                cursor += 90_000
            subtitles.append(
                {
                    "subtitle_uid": f"s{subtitle_index}",
                    "subtitle_index": subtitle_index,
                    "text": text,
                    "word_ids": word_ids,
                }
            )
        payload = {
            "source_segments": [{"id": "clip", "material_id": "main_video", "source_start_us": 0, "source_end_us": cursor + 1_000_000}],
            "word_timeline": words,
            "subtitles": subtitles,
            "text_materials": [
                {
                    "material": {
                        "category": "Text",
                        "id": "text_main",
                        "type": "Caption",
                        "text": "dummy",
                        "track_id": "caption_track",
                        "display": {},
                    }
                }
            ],
            "text_segments": [
                {
                    "material_id": "text_main",
                    "segment_text": "dummy",
                    "segment_id": "segment_main",
                    "start_us": 0,
                    "end_us": cursor + 1_000_000,
                }
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        return path

    def test_deepseek_config_loader_reads_application_yaml_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "application.yaml"
            path.write_text(
                "\n".join(
                    [
                        "deepseek:",
                        "  api" + "-key: local-test-token",
                        "  base-url: https://api.deepseek.com",
                        "app:",
                        "  ai:",
                        "    summary:",
                        "      model: deepseek-chat",
                    ]
                ),
                "utf-8",
            )

            config = _parse_deepseek_yaml_config(path)

        self.assertEqual(config["api_key"], "local-test-token")
        self.assertEqual(config["base_url"], "https://api.deepseek.com")
        self.assertEqual(config["model"], "deepseek-chat")

    def test_deepseek_provider_loads_from_runtime_secrets_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "auto_clip_runtime" / "secrets"
            runtime_root.mkdir(parents=True, exist_ok=True)
            runtime_config = runtime_root / "deepseek.local.yaml"
            self._write_deepseek_yaml(runtime_config, "runtime-secret-token")

            repo_root = Path(tmp) / "config"
            repo_root.mkdir()
            self._write_deepseek_yaml(repo_root / "deepseek.local.yaml", "repo-secret-token")

            with patch.dict("os.environ", {}, clear=True), patch(
                "aroll_v21.decision.deepseek_semantic_planner.DEFAULT_DEEPSEEK_CONFIG_PATHS",
                (runtime_config, repo_root / "deepseek.local.yaml", repo_root / "deepseek.yaml"),
            ):
                provider = deepseek_provider_from_runtime_config()

        self.assertIsNotNone(provider)
        self.assertEqual(provider.api_key, "runtime-secret-token")
        self.assertEqual(provider.config_source, "deepseek.local.yaml")

    def test_deepseek_provider_loads_from_repo_local_yaml_when_gitignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "deepseek.local.yaml"
            self._write_deepseek_yaml(config_path, "repo-secret-token")
            with patch.dict("os.environ", {}, clear=True), patch(
                "aroll_v21.decision.deepseek_semantic_planner.DEFAULT_DEEPSEEK_CONFIG_PATHS",
                (config_path,),
            ):
                provider = deepseek_provider_from_runtime_config()

        self.assertIsNotNone(provider)
        self.assertEqual(provider.api_key, "repo-secret-token")
        self.assertEqual(provider.config_source, "deepseek.local.yaml")

    def test_deepseek_provider_falls_back_to_reference_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_config = Path(tmp) / "reference-application.yaml"
            self._write_deepseek_yaml(reference_config, "reference-secret-token")
            with patch.dict(
                "os.environ",
                {"REFERENCE_VIDEO_DATA_CATCHER_DEEPSEEK_CONFIG_PATH": str(reference_config)},
                clear=True,
            ):
                provider = deepseek_provider_from_runtime_config()

        self.assertIsNotNone(provider)
        self.assertEqual(provider.api_key, "reference-secret-token")
        self.assertEqual(provider.config_source, "reference-application.yaml")

    def test_deepseek_provider_falls_back_to_env_last(self) -> None:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-only-secret"}, clear=True):
            provider = deepseek_provider_from_runtime_config()

        self.assertIsNotNone(provider)
        self.assertEqual(provider.api_key, "env-only-secret")
        self.assertEqual(provider.config_source, "env:DEEPSEEK_API_KEY")

    def test_deepseek_placeholder_config_does_not_configure_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "deepseek.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "deepseek:",
                        "  api" + "-key: sk-REPLACE_WITH_DEEPSEEK_API_KEY",
                        "  base-url: https://api.deepseek.com",
                    ]
                ),
                "utf-8",
            )

            with patch.dict("os.environ", {}, clear=True), patch(
                "aroll_v21.decision.deepseek_semantic_planner.DEFAULT_DEEPSEEK_CONFIG_PATHS",
                (config_path,),
            ):
                provider = deepseek_provider_from_runtime_config()

        self.assertIsNone(provider)

    def test_provider_uses_file_config_when_env_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "application.yaml"
            path.write_text(
                "\n".join(
                    [
                        "deepseek:",
                        "  api" + "-key: local-test-token",
                        "  base-url: https://api.deepseek.com",
                    ]
                ),
                "utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                with patch(
                    "aroll_v21.decision.deepseek_semantic_planner._load_deepseek_config_from_files",
                    return_value=_load_deepseek_config_from_files([path]),
                ):
                    provider = deepseek_provider_from_env()

        self.assertIsNotNone(provider)
        self.assertEqual(provider.api_key, "local-test-token")
        self.assertEqual(provider.base_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(provider.model, "deepseek-chat")

    def test_deepseek_prompt_schema_allows_keep_longest_drop_others(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            captured["body"] = request.data
            captured["timeout"] = timeout
            payload = json.loads(request.data.decode("utf-8"))
            user_content = json.loads(payload["messages"][1]["content"])
            return _FakeDeepSeekResponse(batch_id=user_content["batch_id"])

        provider = DeepSeekSemanticProvider(api_key="unit-test-token", timeout_s=7)
        semantic_request = SemanticAdjudicationRequest(
            issue_id="final_target_repeat_tc_0001",
            issue_type=SemanticIssueType.SEMANTIC_CONTAINMENT,
            severity=SemanticIssueSeverity.MEDIUM,
            text_before="红花",
            text_after="蓝月升过长桥",
            allowed_decisions=[
                "keep_all",
                "drop_left",
                "drop_right",
                "keep_longest_drop_others",
                "requires_human_review",
            ],
        )

        with patch("urllib.request.urlopen", new=fake_urlopen):
            provider.decide([semantic_request])

        payload = json.loads(captured["body"].decode("utf-8"))
        user_content = json.loads(payload["messages"][1]["content"])
        decisions = user_content["schema"]["decisions"][0]["decision"].split("|")
        self.assertEqual(captured["timeout"], 7)
        self.assertIn("keep_longest_drop_others", decisions)

    def test_keep_longest_drop_others_decision_type_roundtrip(self) -> None:
        self.assertEqual(
            SemanticAdjudicationDecisionType("keep_longest_drop_others"),
            SemanticAdjudicationDecisionType.KEEP_LONGEST_DROP_OTHERS,
        )

        decision = _decision_from_provider_row(
            {
                "issue_id": "final_target_repeat_tc_0001",
                "decision": "keep_longest_drop_others",
                "reason": "keep the longest final target candidate",
                "confidence": 0.87,
            }
        )

        self.assertEqual(decision.decision, SemanticAdjudicationDecisionType.KEEP_LONGEST_DROP_OTHERS)
        self.assertEqual(decision.decision.value, "keep_longest_drop_others")

    def test_deepseek_reports_config_source_without_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            input_json = root / "input.json"
            config_path = root / "config" / "deepseek.local.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            secret = "local-config-secret-for-test-source-visibility"
            self._write_deepseek_yaml(config_path, secret)
            self._write_operator_input(input_json)

            with patch.dict("os.environ", {}, clear=True), patch(
                "aroll_v21.decision.deepseek_semantic_planner.DEFAULT_DEEPSEEK_CONFIG_PATHS",
                (config_path,),
            ), patch.object(
                deepseek_planner.DeepSeekSemanticProvider,
                "decide",
                lambda self, requests: [
                    SemanticAdjudicationDecision(
                        issue_id=request.issue_id,
                        decision=SemanticAdjudicationDecisionType.DROP_ABORTED,
                        reason="test provider decision",
                        confidence=0.95,
                        provider_name="deepseek_semantic_planner",
                    )
                    for request in requests
                ],
            ):
                summary = run_operator(
                    ArollV21OperatorConfig(mode="dry-run", run_dir=run_dir, input_json=input_json, semantic_mode="auto")
                )

            report_files = [run_dir / name for name in ("run_summary.json", "semantic_adjudication_report.json", "deepseek_decisions.json")]
            for report_file in report_files:
                self.assertTrue(report_file.exists())
                self.assertNotIn(secret, report_file.read_text("utf-8"))
            self.assertEqual(summary["deepseek_provider_config_source"], "deepseek.local.yaml")
            self.assertNotIn(secret, summary["deepseek_provider_config_source"])

    def test_secret_scan_no_key_in_reports_src_tests(self) -> None:
        secret = f"scan-no-secret-token-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            input_json = root / "input.json"
            config_path = root / "config" / "deepseek.local.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_deepseek_yaml(config_path, secret)
            self._write_operator_input(input_json)

            with patch.dict("os.environ", {"DEEPSEEK_TIMEOUT_S": "1"}, clear=True), patch(
                "aroll_v21.decision.deepseek_semantic_planner.DEFAULT_DEEPSEEK_CONFIG_PATHS",
                (config_path, Path("config/deepseek.yaml")),
            ), patch.object(
                deepseek_planner.DeepSeekSemanticProvider,
                "decide",
                lambda self, requests: [
                SemanticAdjudicationDecision(
                        issue_id=request.issue_id,
                        decision=SemanticAdjudicationDecisionType.DROP_ABORTED,
                        reason="test provider decision",
                        confidence=0.95,
                        provider_name="deepseek_semantic_planner",
                    )
                    for request in requests
                ],
            ):
                run_operator(
                    ArollV21OperatorConfig(mode="dry-run", run_dir=run_dir, input_json=input_json, semantic_mode="auto")
                )

            scanned_files = []
            for directory in (Path("src"), Path("tests"), run_dir):
                for artifact in directory.rglob("*"):
                    if not artifact.is_file():
                        continue
                    if artifact.suffix.lower() not in {".json", ".py", ".md", ".txt", ".yaml", ".yml", ".ini", ".toml", ".cfg"}:
                        continue
                    scanned_files.append(artifact)
            leaked = []
            for artifact in scanned_files:
                try:
                    content = artifact.read_text("utf-8", errors="ignore")
                except Exception:
                    continue
                if secret in content:
                    leaked.append(str(artifact))
            self.assertEqual(leaked, [])

    def test_auto_mode_loads_deepseek_provider_from_reference_config_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference-application.yaml"
            path.write_text(
                "\n".join(
                    [
                        "deepseek:",
                        "  api" + "-key: local-test-token",
                        "  base-url: https://api.deepseek.com",
                        "app:",
                        "  ai:",
                        "    semantic:",
                        "      model: deepseek-chat",
                    ]
                ),
                "utf-8",
            )

            with patch.dict(
                "os.environ",
                {"REFERENCE_VIDEO_DATA_CATCHER_DEEPSEEK_CONFIG_PATH": str(path)},
                clear=True,
            ):
                provider = deepseek_provider_from_runtime_config()

        self.assertIsNotNone(provider)
        self.assertEqual(provider.config_source, "reference-application.yaml")
        self.assertEqual(provider.base_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(provider.model, "deepseek-chat")

    def test_auto_mode_provider_configured_when_smoke_config_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "deepseek.local.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "deepseek:",
                        "  api" + "-key: local-test-token",
                        "  base-url: https://api.deepseek.com",
                    ]
                ),
                "utf-8",
            )
            with patch.dict("os.environ", {}, clear=True), patch(
                "aroll_v21.decision.deepseek_semantic_planner.DEFAULT_DEEPSEEK_CONFIG_PATHS",
                (config_path,),
            ):
                provider = deepseek_provider_from_runtime_config()

        self.assertIsNotNone(provider)
        self.assertTrue(bool(provider.api_key))
        self.assertEqual(provider.config_source, "deepseek.local.yaml")


if __name__ == "__main__":
    unittest.main()
