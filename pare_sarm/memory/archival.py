"""Archival Memory: cross-experiment design patterns and principles.

Stores abstract lessons learned across experiments.
Each pattern has: text, importance (manual), recency (auto), source_round.
"""

import json
import time
from pathlib import Path


class ArchivalMemory:
    """Cross-experiment pattern store with importance-based retrieval.

    Patterns are abstract design principles like:
    "Fuel penalties that dominate early training suppress exploration.
     Gate fuel cost behind a progress threshold."
    """

    def __init__(self, exp_dir: Path):
        self._path = Path(exp_dir) / "memory" / "archival.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._patterns: list[dict] = []
        self._load()

    def add(self, pattern: str, source_round: int, importance: float = 1.0):
        """Add a design pattern/principle.

        Args:
            pattern: The abstract principle text.
            source_round: Which round this was learned from.
            importance: Manual importance score (higher = more important).
        """
        self._patterns.append({
            "pattern": pattern,
            "source_round": source_round,
            "importance": importance,
            "timestamp": time.time(),
        })

    def search(self, query: str = "", max_results: int = 5) -> list[str]:
        """Retrieve relevant patterns, sorted by importance × recency.

        If query is provided, filters to patterns containing query keywords.
        """
        if not self._patterns:
            return []

        candidates = self._patterns
        if query:
            terms = set(query.lower().split())
            candidates = [
                p for p in self._patterns
                if any(t in p["pattern"].lower() for t in terms)
            ]

        # Sort by importance (higher first), then recency
        candidates = sorted(candidates, key=lambda p: (p["importance"], p["timestamp"]), reverse=True)
        return [p["pattern"] for p in candidates[:max_results]]

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def save(self):
        """Persist to disk."""
        self._path.write_text(
            json.dumps(self._patterns, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self):
        """Load from disk if exists."""
        if self._path.exists():
            try:
                self._patterns = json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._patterns = []
