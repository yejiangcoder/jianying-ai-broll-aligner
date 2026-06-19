from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aroll_runtime_paths import get_deepseek_config_path

DEFAULT_CONFIG = get_deepseek_config_path()


@dataclass(frozen=True)
class DeepSeekConfig:
    config_path: Path
    base_url: str
    api_key: str

    def public_dict(self, model: str, response_format: bool) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": model,
            "config_path": str(self.config_path).replace("\\", "/"),
            "api_key_loaded": bool(self.api_key),
            "response_format": response_format,
        }


def _strip_yaml_value(value: str) -> str:
    value = value.strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value.strip()


def load_deepseek_config(path: Path | None = None) -> DeepSeekConfig:
    if path is None:
        path = get_deepseek_config_path()
    if not path.exists():
        raise RuntimeError(f"DeepSeek config not found: {path}")
    lines = path.read_text("utf-8").splitlines()
    in_deepseek = False
    base_indent = -1
    values: dict[str, str] = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip().lstrip("\ufeff")
        if stripped == "deepseek:":
            in_deepseek = True
            base_indent = indent
            continue
        if in_deepseek and indent <= base_indent and re.match(r"^[\w.-]+:", stripped):
            break
        if in_deepseek and ":" in stripped:
            key, value = stripped.split(":", 1)
            values[key.strip()] = _strip_yaml_value(value)
    api_key = values.get("api-key") or values.get("api_key") or ""
    base_url = values.get("base-url") or values.get("base_url") or ""
    if not _is_configured_api_key(api_key):
        raise RuntimeError(f"DeepSeek api-key not found in {path}")
    if not base_url:
        raise RuntimeError(f"DeepSeek base-url not found in {path}")
    return DeepSeekConfig(config_path=path, base_url=base_url.rstrip("/"), api_key=api_key)


def _is_configured_api_key(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and "REPLACE_WITH" not in text


def _redact(text: str, api_key: str) -> str:
    if not text:
        return text
    text = text.replace(api_key, "***REDACTED***") if api_key else text
    text = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer ***REDACTED***", text)
    return text


def post_chat_completions(
    config: DeepSeekConfig,
    payload: dict[str, Any],
    timeout_sec: int = 240,
) -> dict[str, Any]:
    url = config.base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer " + config.api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {_redact(raw, config.api_key)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(_redact(str(exc), config.api_key)) from exc


def extract_message_content(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("DeepSeek response missing choices")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError("DeepSeek response missing message.content")
    meta = {
        "model": response.get("model"),
        "finish_reason": (choices[0] or {}).get("finish_reason"),
        "usage": response.get("usage") or {},
        "reasoning_content_present": bool(message.get("reasoning_content")),
    }
    return str(content), meta


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise RuntimeError("Empty model content")
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise RuntimeError("No JSON object boundary found in model content")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("Extracted JSON is not an object")
    return parsed
