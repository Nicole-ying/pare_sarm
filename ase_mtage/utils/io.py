"""File-system and serialization helpers for ASE-MTAGE.

The Phase 1 pipeline intentionally depends only on the Python standard library.
YAML support is optional: if PyYAML is installed, `.yaml`/`.yml` files are read
normally; otherwise JSON config files are still supported.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when a config file cannot be loaded or parsed."""


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_timestamp() -> str:
    """Return a filesystem-friendly local timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Save JSON with UTF-8 encoding and stable formatting."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return p


def load_json(path: str | Path, default: Any | None = None) -> Any:
    """Load a JSON file; return default if the file does not exist and default is set."""
    p = Path(path)
    if not p.exists():
        if default is not None:
            return default
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def save_text(path: str | Path, text: str) -> Path:
    """Save UTF-8 text, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    """Append one JSON record to a JSONL file."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return p


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file. Missing files return an empty list."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON/YAML config file.

    When path is None, returns an empty dict. The pipeline will then fill defaults.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ConfigError(
                "YAML config requires PyYAML. Install pyyaml or use a .json config."
            ) from exc
        loaded = yaml.safe_load(text)
        return loaded or {}
    raise ConfigError(f"Unsupported config type: {p.suffix}. Use .json, .yaml, or .yml")


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """Copy a file, creating parent directories."""
    src_p = Path(src)
    dst_p = Path(dst)
    ensure_dir(dst_p.parent)
    shutil.copy2(src_p, dst_p)
    return dst_p
