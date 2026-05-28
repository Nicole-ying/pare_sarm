"""Utility functions: config loading, file I/O, logging."""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

BEIJING = timezone(timedelta(hours=8))


def load_yaml(path: Path) -> dict:
    """Load a YAML config file. Falls back to JSON if yaml not installed."""
    text = path.read_text("utf-8")
    if yaml:
        return yaml.safe_load(text)
    return json.loads(text)


def save_yaml(path: Path, data: dict) -> None:
    """Save dict as YAML (or JSON if yaml not installed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """Return a compact Beijing-time timestamp string, e.g. 2605271430."""
    return datetime.now(BEIJING).strftime("%y%m%d%H%M")


def experiment_name(env_id: str, steps: int) -> str:
    """Generate a timestamped experiment directory name."""
    env_name = env_id.lower().replace(" ", "-")
    return f"{env_name}_{timestamp()}_{steps}"


def read_jsonl(path: Path) -> list[dict]:
    """Read all JSON records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text("utf-8").strip().split("\n"):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def read_all_jsonl(dir_path: Path, pattern: str = "*.jsonl") -> list[dict]:
    """Read all JSONL files matching a glob pattern in a directory."""
    if not dir_path.exists():
        return []
    records = []
    for f in sorted(dir_path.glob(pattern)):
        records.extend(read_jsonl(f))
    return records


def save_json(path: Path, data: dict) -> None:
    """Write a dict as JSON with indent=2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Read a JSON file, return empty dict if missing."""
    if not path.exists():
        return {}
    return json.loads(path.read_text("utf-8"))


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def die(msg: str) -> None:
    """Print error and exit."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)
