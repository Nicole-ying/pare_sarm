"""OpenAI-compatible LLM client for ASE-MTAGE.

DeepSeek exposes an OpenAI-compatible chat/completions API. This client uses only
Python standard library so the framework can run without adding another SDK.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]


class LLMClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key_env: str = "DEEPSEEK_API_KEY",
        api_key: str | None = None,
        timeout_seconds: int = 120,
        max_tokens: int = 4096,
        fallback_on_error: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = api_key or os.environ.get(api_key_env)
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.fallback_on_error = bool(fallback_on_error)
        if not self.api_key:
            raise RuntimeError(
                f"LLM is enabled but API key is missing. Set environment variable {api_key_env} "
                "or provide llm.api_key in config. Do not commit real keys."
            )

    @classmethod
    def from_config(cls, llm_config: Any) -> "LLMClient | None":
        if not getattr(llm_config, "enabled", False):
            return None
        return cls(
            provider=getattr(llm_config, "provider", "deepseek"),
            model=getattr(llm_config, "model", "deepseek-reasoner"),
            base_url=getattr(llm_config, "base_url", "https://api.deepseek.com"),
            api_key_env=getattr(llm_config, "api_key_env", "DEEPSEEK_API_KEY"),
            api_key=getattr(llm_config, "api_key", None),
            timeout_seconds=int(getattr(llm_config, "timeout_seconds", 120)),
            max_tokens=int(getattr(llm_config, "max_tokens", 4096)),
            fallback_on_error=bool(getattr(llm_config, "fallback_on_error", True)),
        )

    def chat(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int | None = None) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens or self.max_tokens),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection error: {exc}") from exc
        try:
            content = raw["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected LLM response format: {raw}") from exc
        return LLMResponse(content=content, raw=raw)


def load_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent / "prompts" / name
    return prompt_path.read_text(encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from raw LLM text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def extract_python_code(text: str) -> str:
    """Extract Python code from raw LLM text."""
    cleaned = text.strip()
    if "```" not in cleaned:
        return cleaned
    parts = cleaned.split("```")
    for part in parts:
        p = part.strip()
        if p.startswith("python"):
            return p[len("python") :].strip()
    return parts[1].strip() if len(parts) > 1 else cleaned
