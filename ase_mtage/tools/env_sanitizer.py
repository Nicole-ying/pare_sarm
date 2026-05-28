"""Environment sanitizer for ASE-MTAGE.

This tool extracts reward-safe environment metadata without exposing official
reward formulas to LLM agents. It can inspect a Gym/Gymnasium environment and can
also sanitize a source-code string by redacting reward-related lines.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

from ase_mtage.utils.io import save_json, save_text


REWARD_PATTERNS = [
    re.compile(r"\breward\b", re.IGNORECASE),
    re.compile(r"\bshaping\b", re.IGNORECASE),
    re.compile(r"\breturn\s+.*reward", re.IGNORECASE),
]


class EnvSanitizer:
    """Create reward-safe environment summaries for EnvPerceptionAgent."""

    def sanitize_env(self, *, env_id: str, output_dir: str | Path | None = None) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "env_id": env_id,
            "official_reward_visible": False,
            "observation_space": None,
            "action_space": None,
            "max_episode_steps": None,
            "termination_logic": "unknown_from_runtime_summary",
            "truncation_logic": "time_limit_or_wrapper_dependent",
            "info_keys": [],
            "sanitized_source_summary": "source_not_available_or_not_used",
            "warnings": [],
        }
        try:
            try:
                import gymnasium as gym  # type: ignore
            except Exception:
                import gym  # type: ignore
            env = gym.make(env_id)
            summary["observation_space"] = str(env.observation_space)
            summary["action_space"] = str(env.action_space)
            summary["max_episode_steps"] = getattr(getattr(env, "spec", None), "max_episode_steps", None)
            base_env = getattr(env, "unwrapped", env)
            try:
                source = inspect.getsource(base_env.__class__)
                sanitized_source, redacted = self.sanitize_source(source)
                summary["sanitized_source_summary"] = sanitized_source[:12000]
                summary["redacted_reward_lines"] = redacted
            except Exception as exc:
                summary["warnings"].append(f"Could not inspect env source: {exc}")
            try:
                env.close()
            except Exception:
                pass
        except Exception as exc:
            summary["warnings"].append(f"Could not instantiate env {env_id}: {exc}")

        if output_dir is not None:
            out = Path(output_dir)
            save_json(out / "sanitized_env_summary.json", summary)
            save_text(out / "sanitized_env_source.txt", str(summary.get("sanitized_source_summary", "")))
        return summary

    def sanitize_source(self, source: str) -> tuple[str, list[int]]:
        sanitized_lines: list[str] = []
        redacted: list[int] = []
        for idx, line in enumerate(source.splitlines(), start=1):
            if any(p.search(line) for p in REWARD_PATTERNS):
                sanitized_lines.append("# [REDACTED_REWARD_LOGIC]")
                redacted.append(idx)
            else:
                sanitized_lines.append(line)
        return "\n".join(sanitized_lines), redacted
