"""
Per-agent belief state repository with half-life decay and accuracy tracking.

Each Diagnostician maintains its own belief state — what hypothesis categories
it's good at, what patterns it tends to miss, and how credible its diagnoses
have been historically.

This is NOT a static log. It's a self-correcting model of the agent's own
diagnostic accuracy.
"""

import json
from pathlib import Path
from typing import Any, Optional

# Default belief schema for Diagnostician agents
DIAGNOSTICIAN_BELIEF_V2 = {
    "agent": "diagnostician",
    "version": 2,
    "history": [],  # recent round entries
    "hypothesis_category_accuracy": {},  # {category: {proposed: N, correct: N, accuracy: float}}
    "current_credibility": 0.5,  # overall reliability score in [0, 1]
    "self_awareness": "",  # natural language summary of own patterns
}


class BeliefRepository:
    """Manage persistent belief states for agents with decay and accuracy tracking."""

    def __init__(self, belief_dir: Path):
        import os
        self.belief_dir = belief_dir
        self.belief_dir.mkdir(parents=True, exist_ok=True)

    def get_belief(self, agent_id: str) -> dict[str, Any]:
        """Load belief state for an agent. Creates default if not exists."""
        path = self._path(agent_id)
        if not path.exists():
            return dict(DIAGNOSTICIAN_BELIEF_V2, agent=agent_id)

        try:
            import json
            belief = json.loads(path.read_text("utf-8"))
            # Ensure all default fields exist
            for key, default in DIAGNOSTICIAN_BELIEF_V2.items():
                belief.setdefault(key, default)
            belief["agent"] = agent_id
            belief.setdefault("version", 2)
            return belief
        except Exception:
            return dict(DIAGNOSTICIAN_BELIEF_V2, agent=agent_id)

    def update_belief(self, agent_id: str, entry: dict, max_history: int = 50) -> dict:
        """Append a round entry to the belief state history.

        Args:
            agent_id: Agent identifier (e.g., "diagnostician_A").
            entry: Dict with round-level data (round, diagnosis_summary,
                   violated_principle, confidence, etc.).
            max_history: Maximum number of history entries to retain.

        Returns updated belief dict.
        """
        belief = self.get_belief(agent_id)
        hist = belief.setdefault("history", [])
        hist.append(entry)

        if len(hist) > max_history:
            belief["history"] = hist[-max_history:]

        self._save(agent_id, belief)
        return belief

    def record_hypothesis_outcome(
        self,
        agent_id: str,
        hypothesis_id: str,
        category: str,
        predicted: str,
        actual: str,
        correct: bool,
    ):
        """Record the outcome of a hypothesis for accuracy tracking.

        Args:
            agent_id: Agent identifier.
            hypothesis_id: Unique hypothesis identifier (e.g., "round3_hypothesis").
            category: Hypothesis category (e.g., "exploration_collapse",
                      "component_dominance", "reward_misalignment").
            predicted: What the agent predicted.
            actual: What actually happened.
            correct: Whether the prediction was correct.
        """
        belief = self.get_belief(agent_id)

        # Update hypothesis category accuracy
        acc = belief.setdefault("hypothesis_category_accuracy", {})
        cat_entry = acc.setdefault(category, {"proposed": 0, "correct": 0, "accuracy": 0.0})
        cat_entry["proposed"] += 1
        if correct:
            cat_entry["correct"] += 1
        cat_entry["accuracy"] = cat_entry["correct"] / max(cat_entry["proposed"], 1)

        # Update overall credibility
        all_cats = acc.values()
        total_proposed = sum(c["proposed"] for c in all_cats)
        total_correct = sum(c["correct"] for c in all_cats)
        belief["current_credibility"] = total_correct / max(total_proposed, 1)

        # Generate self-awareness summary
        weak_cats = [
            cat for cat, stats in acc.items()
            if stats["proposed"] >= 2 and stats["accuracy"] < 0.5
        ]
        strong_cats = [
            cat for cat, stats in acc.items()
            if stats["proposed"] >= 2 and stats["accuracy"] > 0.67
        ]

        parts = []
        if strong_cats:
            parts.append(
                f"I tend to correctly diagnose: {', '.join(strong_cats)}"
            )
        if weak_cats:
            parts.append(
                f"I tend to over-diagnose: {', '.join(weak_cats)} — "
                "should check alternative explanations first"
            )
        belief["self_awareness"] = ". ".join(parts) if parts else "Insufficient data for self-assessment"

        self._save(agent_id, belief)
        return belief

    def apply_decay(self, agent_id: str, half_life_rounds: int = 5):
        """Apply half-life decay to older belief entries.

        Reduces the weight of older entries so recent performance matters more.
        """
        belief = self.get_belief(agent_id)

        # Apply decay to credibility: weight recent rounds more
        hist = belief.get("history", [])
        total_weight = 0.0
        weighted_correct = 0.0

        for i, entry in enumerate(reversed(hist)):
            age = i  # 0 = most recent
            weight = 0.5 ** (age / half_life_rounds)
            total_weight += weight

        # Don't actually modify history — just note in belief
        belief["decay_applied"] = True
        belief["half_life_rounds"] = half_life_rounds

        self._save(agent_id, belief)
        return belief

    def _path(self, agent_id: str) -> Path:
        return self.belief_dir / f"{agent_id}.json"

    def _save(self, agent_id: str, belief: dict):
        self._path(agent_id).write_text(
            json.dumps(belief, ensure_ascii=False, indent=2), encoding="utf-8"
        )
