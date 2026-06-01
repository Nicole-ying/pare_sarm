"""Failure-repair memory store for ASE-MTAGE Phase 6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.utils.io import append_jsonl, load_jsonl


class FailureRepairMemory:
    """Append-only memory of failure -> repair -> outcome lessons."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def add(self, record: dict[str, Any]) -> Path:
        return append_jsonl(self.path, record)

    def read_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = load_jsonl(self.path)
        return rows[-limit:]

    def summarize_recent(self, limit: int = 5) -> dict[str, Any]:
        recent = self.read_recent(limit=limit)
        return {
            "num_records": len(load_jsonl(self.path)),
            "recent_records": recent,
        }
