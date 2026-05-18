"""
Config loader: YAML config loading with validation for multi_reward framework.
"""

from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict:
    """Load a YAML config file. Falls back to minimal parser if pyyaml unavailable."""
    text = path.read_text("utf-8")

    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        return _fallback_parse(text)


def _fallback_parse(text: str) -> dict:
    """Minimal YAML parser supporting top-level key: value pairs."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        # Try numeric conversion
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def validate_config(config: dict) -> list[str]:
    """Validate experiment config, returning list of issues."""
    issues = []
    required = ["total_timesteps", "n_envs"]
    for key in required:
        if key not in config:
            issues.append(f"Missing required config key: {key}")

    if "ppo" not in config:
        issues.append("Missing 'ppo' section in config")
    else:
        ppo_required = ["policy", "learning_rate", "n_steps", "batch_size"]
        for key in ppo_required:
            if key not in config["ppo"]:
                issues.append(f"Missing required ppo config key: ppo.{key}")

    if "rounds" not in config:
        issues.append("Missing 'rounds' in config (default: 5)")

    if "evaluation" not in config:
        issues.append("Missing 'evaluation' section in config")

    return issues
