"""
File utilities: path helpers, artifact saving, JSON serialization.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

BEIJING = timezone(timedelta(hours=8))


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict | list, indent: int = 2) -> Path:
    """Save data as JSON file. Creates parent dirs if needed."""
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8"
    )
    return path


def load_json(path) -> dict | list | None:
    """Load JSON file. Returns None if not found. Accepts str or Path."""
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, Exception):
        return None


def save_text(path: Path, content: str) -> Path:
    """Save text to file. Creates parent dirs if needed."""
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")
    return path


def load_text(path) -> str:
    """Load text file. Returns empty string if not found. Accepts str or Path."""
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        return ""
    return p.read_text("utf-8")


def beijing_timestamp() -> str:
    """Get current time string in Beijing timezone."""
    return datetime.now(BEIJING).strftime("%y%m%d%H%M%S")


def experiment_dir_name(env_name: str, total_steps: int) -> str:
    """Generate experiment directory name."""
    ts = beijing_timestamp()
    env_slug = env_name.lower().replace(" ", "_")
    return f"{env_slug}_{ts}_{total_steps}"
