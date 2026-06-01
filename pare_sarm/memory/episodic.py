"""Episodic Memory: per-round records with keyword-based retrieval.

Stores what was diagnosed, tried, and what happened in each round.
Provides simple TF-IDF-like keyword search for cross-round context injection.
"""

import json
import math
from collections import Counter
from pathlib import Path


class EpisodicMemory:
    """Stores per-round records with keyword search capability."""

    def __init__(self, exp_dir: Path):
        self._path = Path(exp_dir) / "memory" / "episodic.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rounds: dict[int, dict] = {}
        self._load()

    def store(self, round_num: int, data: dict):
        """Store data for a round. Overwrites if round already exists."""
        self._rounds[round_num] = {
            "round": round_num,
            "summary": data.get("summary", ""),
            "reward_fn_source": data.get("reward_fn_source", "")[:3000],
            "diagnosis": data.get("diagnosis", ""),
            "health_score": data.get("health_score", 0),
            "perception_report": data.get("perception_report", "")[:2000],
            "reflection": data.get("reflection", "")[:2000],
        }

    def get_round(self, round_num: int) -> dict:
        """Get stored data for a specific round."""
        return self._rounds.get(round_num, {})

    def get_all_rounds(self) -> dict[int, dict]:
        """Get all stored rounds."""
        return dict(self._rounds)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search rounds by keyword overlap with the query.

        Simple TF-based scoring: sum of query term frequency in round text.
        """
        if not query or not self._rounds:
            return []

        query_terms = set(query.lower().split())
        scores = {}

        for round_num, data in self._rounds.items():
            text = f"{data.get('summary', '')} {data.get('diagnosis', '')}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scores[round_num] = score

        # Return top results sorted by score
        sorted_rounds = sorted(scores.items(), key=lambda x: -x[1])[:max_results]
        return [self._rounds[r] for r, _ in sorted_rounds]

    def get_history_text(self, max_rounds: int = 10) -> str:
        """Build a cross-round history text for prompt injection."""
        parts = []
        for r in sorted(self._rounds.keys())[-max_rounds:]:
            data = self._rounds[r]
            parts.append(
                f"Round {r}: {data.get('summary', 'N/A')[:200]}"
            )
        return "\n".join(parts) if parts else "(no history)"

    def save(self):
        """Persist to disk."""
        serializable = {
            str(k): v for k, v in self._rounds.items()
        }
        self._path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self):
        """Load from disk if exists."""
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text("utf-8"))
                self._rounds = {int(k): v for k, v in raw.items()}
            except (json.JSONDecodeError, OSError, ValueError):
                self._rounds = {}
