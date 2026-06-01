"""OpenAI-compatible LLM client for ASE-MTAGE.

DeepSeek exposes an OpenAI-compatible chat/completions API. This client uses only
Python standard library so the framework can run without adding another SDK.
"""

from __future__ import annotations

import json
import os
import socket
import time as _time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_seconds: float = 5.0
    max_delay_seconds: float = 120.0
    backoff_multiplier: float = 2.0


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
        retry: RetryConfig | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = api_key or os.environ.get(api_key_env)
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.fallback_on_error = bool(fallback_on_error)
        self.retry = retry or RetryConfig()
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

    def chat(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int | None = None, agent_name: str = "unknown") -> LLMResponse:
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
        prompt_bytes = len(data)

        last_error: Exception | None = None
        for attempt in range(self.retry.max_retries + 1):
            try:
                return self._call_api(url, data, agent_name, prompt_bytes)
            except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt >= self.retry.max_retries:
                    break
                delay = min(self.retry.base_delay_seconds * (self.retry.backoff_multiplier ** attempt), self.retry.max_delay_seconds)
                self._log_retry(agent_name, attempt + 1, self.retry.max_retries, delay, str(exc))
                _time.sleep(delay)

        elapsed = 0.0
        self._log_llm(agent_name, prompt_bytes, 0, elapsed, False)
        if isinstance(last_error, urllib.error.HTTPError):
            body = last_error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {last_error.code}: {body} (retries exhausted)") from last_error
        raise RuntimeError(f"LLM call failed after {self.retry.max_retries} retries: {last_error}") from last_error

    def _call_api(self, url: str, data: bytes, agent_name: str, prompt_bytes: int) -> LLMResponse:
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        t0 = _time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError:
            raise
        try:
            content = raw["choices"][0]["message"]["content"]
        except Exception as exc:
            elapsed = _time.time() - t0
            self._log_llm(agent_name, prompt_bytes, 0, elapsed, False)
            raise RuntimeError(f"Unexpected LLM response format: {raw}") from exc
        elapsed = _time.time() - t0
        resp_bytes = len(content.encode("utf-8"))
        self._log_llm(agent_name, prompt_bytes, resp_bytes, elapsed, True)
        return LLMResponse(content=content, raw=raw)

    def _log_retry(self, agent_name: str, attempt: int, max_retries: int, delay: float, error: str) -> None:
        try:
            from ase_mtage.utils.logger import get_logger
            get_logger().info(f"LLM | {agent_name} | retry {attempt}/{max_retries} in {delay:.1f}s | {error[:120]}")
        except Exception:
            pass

    def _log_llm(self, agent_name: str, prompt_bytes: int, resp_bytes: int, elapsed: float, success: bool) -> None:
        try:
            from ase_mtage.utils.logger import get_logger
            get_logger().llm_call(agent_name, self.model, prompt_bytes, resp_bytes, elapsed, success)
        except Exception:
            pass


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
    """Extract Python code from raw LLM text, with truncation detection."""
    cleaned = text.strip()
    if "```" not in cleaned:
        code = cleaned
    else:
        parts = cleaned.split("```")
        code = cleaned  # fallback
        for part in parts:
            p = part.strip()
            if p.startswith("python"):
                code = p[len("python") :].strip()
                break
        else:
            code = parts[1].strip() if len(parts) > 1 else cleaned

    # Detect obviously truncated code
    lines = code.split("\n")
    if lines and not lines[-1].strip().startswith(("#", "import", "def", "class", " ", "\t", "return", "try", "if", "for", "while")):
        # Last line is likely incomplete — try to find the last complete statement
        pass  # return as-is; caller handles retry

    # Remove trailing incomplete lines (e.g. "total_reward = 4.0 * progress +")
    while lines and any(lines[-1].strip().endswith(c) for c in ("+", "-", "*", "/", "(", "[", "{", ",", "=", "and", "or", ":")):
        lines.pop()
    return "\n".join(lines)
