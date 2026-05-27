"""
archival_memory.py — Cross-experiment pattern library and lessons.

Layer 3 of the three-layer memory system. Stores abstract patterns,
causal lessons, and successful reward designs that generalize across
environments.

Key features:
- Pattern storage with environment-type tagging (locomotion, manipulation, ...)
- Retrieval scoring: recency × importance × relevance
- Consolidation: background reflection during training adds new patterns
- Shared across experiments via a central patterns/ directory

Design: inspired by Generative Agents' memory retrieval (Park et al., 2023)
with recency/importance/relevance scoring.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional


# Environment type classification keywords (not env-specific, but archetype)
_ENV_TYPE_KEYWORDS = {
    "locomotion": ["walk", "run", "cheetah", "hopper", "ant", "humanoid", "bipedal",
                   "locomot", "gait", "forward", "speed", "velocity"],
    "landing": ["land", "lunar", "moon", "descent", "touchdown", "thrust", "gravity"],
    "balance": ["balance", "pendulum", "cartpole", "pole", "upright", "angle", "tilt"],
    "navigation": ["maze", "navigate", "goal", "waypoint", "mountain", "car"],
    "manipulation": ["hand", "grasp", "reach", "push", "pick", "place", "franka", "robot"],
    "flight": ["fly", "quadcopter", "drone", "hover", "altitude", "pitch", "roll"],
}


def _classify_env_type(text: str) -> str:
    """Infer environment archetype from description text."""
    text_lower = text.lower()
    scores = {}
    for etype, keywords in _ENV_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[etype] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


class ArchivalMemory:
    """Cross-experiment pattern storage with scored retrieval.

    Patterns are stored in ~/.eureka_llm_patterns/ (shared across experiments).
    Each pattern has:
    - content: The lesson/principle/pattern
    - env_type: Archetype tag (locomotion, landing, ...)
    - importance: 1-10 score (manually set or auto-computed)
    - timestamp: When it was created
    - source_experiment: Where it came from
    """

    def __init__(self, storage_dir: Path = None):
        if storage_dir is None:
            storage_dir = Path.home() / ".eureka_llm_patterns"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._patterns: list[dict] = []
        self._loaded = False

    # ── Storage ─────────────────────────────────────────────────────────────

    def add_pattern(
        self,
        content: str,
        env_type: str = "",
        importance: int = 5,
        source_experiment: str = "",
        tags: list[str] = None,
    ) -> None:
        """Add a pattern to the archival store.

        Args:
            content: The lesson, principle, or pattern description.
            env_type: Archetype tag (auto-detected if empty).
            importance: 1-10 score.
            source_experiment: Experiment ID where this was discovered.
            tags: Optional tags for search filtering.
        """
        self._ensure_loaded()

        if not env_type:
            env_type = _classify_env_type(content)

        pattern = {
            "id": f"pat_{int(time.time())}_{len(self._patterns)}",
            "content": content,
            "env_type": env_type,
            "importance": max(1, min(10, importance)),
            "timestamp": time.time(),
            "source_experiment": source_experiment,
            "tags": tags or [],
            "access_count": 0,
        }
        self._patterns.append(pattern)
        self._save()

    def add_lesson(
        self,
        lesson: str,
        round_num: int = 0,
        experiment_id: str = "",
        env_type: str = "",
    ) -> None:
        """Add a causal lesson from a reflection.

        Shortcut for add_pattern with importance auto-computed from
        lesson structure: "because X → Y" patterns get higher score.
        """
        importance = 5
        # Heuristic: lessons with causal structure are more important
        causal_markers = ["because", "therefore", "caused", "led to", "resulted",
                          "due to", "hence", "thus", "→", "->"]
        if any(m in lesson.lower() for m in causal_markers):
            importance = 7
        if any(m in lesson.lower() for m in ["always", "never", "critical"]):
            importance = 8

        self.add_pattern(
            content=f"[Round {round_num}] {lesson}",
            env_type=env_type,
            importance=importance,
            source_experiment=experiment_id,
        )

    # ── Retrieval ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        env_type: str = "",
        min_importance: int = 1,
    ) -> list[dict]:
        """Search archival memory with recency × importance × relevance scoring.

        Args:
            query: Natural language query.
            k: Maximum results.
            env_type: Filter by environment archetype.
            min_importance: Minimum importance score (1-10).

        Returns:
            List of {content, score, env_type, importance, ...} dicts.
        """
        self._ensure_loaded()
        if not self._patterns:
            return []

        now = time.time()
        query_terms = self._tokenize(query)
        scored = []

        for pat in self._patterns:
            if env_type and pat.get("env_type") != env_type:
                continue
            if pat.get("importance", 5) < min_importance:
                continue

            # Relevance: TF-IDF-like keyword overlap
            relevance = self._keyword_relevance(query_terms, pat["content"])

            # Recency: exponential decay (half-life = 30 days)
            age_days = (now - pat["timestamp"]) / 86400
            recency = math.exp(-age_days / 30)

            # Importance: normalized to [0, 1]
            importance = pat.get("importance", 5) / 10

            # Composite score: weighted product
            score = (
                0.5 * relevance +
                0.2 * recency +
                0.3 * importance
            )

            if score > 0.05:
                scored.append((score, pat))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, pat in scored[:k]:
            pat["access_count"] = pat.get("access_count", 0) + 1
            results.append({
                "content": pat["content"],
                "score": round(score, 4),
                "env_type": pat.get("env_type", "general"),
                "importance": pat.get("importance", 5),
                "source": pat.get("source_experiment", ""),
                "tags": pat.get("tags", []),
            })

        return results

    def get_by_env_type(self, env_type: str, k: int = 10) -> list[dict]:
        """Get top patterns for a specific environment archetype."""
        return self.search("", k=k, env_type=env_type)

    def get_high_importance(self, k: int = 20) -> list[dict]:
        """Get the most important patterns across all types."""
        return self.search("", k=k, min_importance=7)

    # ── Consolidation ───────────────────────────────────────────────────────

    def consolidate_from_reflection(
        self, reflection_text: str, round_num: int,
        experiment_id: str = "", env_description: str = "",
    ) -> int:
        """Extract patterns from a reflection and add to archival memory.

        Returns the number of new patterns added.
        """
        env_type = _classify_env_type(env_description) if env_description else ""
        added = 0

        # Extract standalone lessons (lines that look like lessons)
        for line in reflection_text.split("\n"):
            line = line.strip()
            # Lesson markers
            if any(m in line.lower() for m in [
                "lesson", "learned", "key insight", "important",
                "should", "should not", "avoid", "always",
            ]):
                if len(line) > 30 and len(line) < 500:  # reasonable lesson length
                    self.add_lesson(line, round_num, experiment_id, env_type)
                    added += 1

        return added

    def deduplicate(self) -> int:
        """Remove near-duplicate patterns. Returns count removed."""
        self._ensure_loaded()
        seen = set()
        unique = []
        removed = 0
        for pat in self._patterns:
            # Simple dedup: first 100 chars as fingerprint
            fp = pat["content"][:100].lower().strip()
            if fp not in seen:
                seen.add(fp)
                unique.append(pat)
            else:
                removed += 1
        self._patterns = unique
        self._save()
        return removed

    # ── Internal ────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in re.findall(r'[a-zA-Z_]\w+', text.lower()) if len(t) >= 2]

    def _keyword_relevance(self, query_terms: list[str], doc_text: str) -> float:
        if not query_terms:
            return 0.1
        doc_lower = doc_text.lower()
        hits = sum(1 for t in query_terms if t in doc_lower)
        return hits / max(len(query_terms), 1)

    def _save(self) -> None:
        path = self.storage_dir / "patterns.json"
        path.write_text(json.dumps(self._patterns, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        path = self.storage_dir / "patterns.json"
        if path.exists():
            try:
                self._patterns = json.loads(path.read_text("utf-8"))
            except Exception:
                self._patterns = []
        self._loaded = True

    @property
    def pattern_count(self) -> int:
        self._ensure_loaded()
        return len(self._patterns)

    @property
    def env_types(self) -> list[str]:
        self._ensure_loaded()
        return sorted(set(p.get("env_type", "general") for p in self._patterns))
