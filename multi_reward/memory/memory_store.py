"""
Central MemoryStore for multi_reward framework.

Three-layer design:
1. TASK_MANIFEST.md — permanent env understanding (from EnvInterpreter)
2. MEMORY.md — cross-round causal lessons (max 200 lines)
3. Per-round artifacts — full diagnosis, evidence board, reward code

Features:
- Similarity-based retrieval (cosine similarity on feature vectors)
- Half-life decay for older beliefs
- Hypothesis accuracy tracking
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from .similarity import cosine_similarity
from .belief_repository import BeliefRepository

BEIJING = timezone(timedelta(hours=8))

MEMORY_HEADER = """# Reward Design Memory

Cross-round causal lessons from reward function iteration.
Each line is a single compressed lesson: what changed -> what happened -> why -> recommendation.
Max 200 lines. Oldest entries are dropped when limit exceeded.

"""

MEMORY_MAX_LINES = 200


class MemoryStore:
    """Cross-round memory management for the multi-agent pipeline."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.memory_dir = self.experiment_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.belief_dir = self.memory_dir / "beliefs"
        self.belief_dir.mkdir(parents=True, exist_ok=True)
        self.belief_repo = BeliefRepository(self.belief_dir)

    # ── Layer 1: Task Manifest ──────────────────────────────────────────

    @property
    def task_manifest_path(self) -> Path:
        return self.memory_dir / "TASK_MANIFEST.md"

    def get_task_manifest(self) -> str:
        p = self.task_manifest_path
        return p.read_text("utf-8") if p.exists() else ""

    def save_task_manifest(self, content: str):
        self.task_manifest_path.write_text(content, encoding="utf-8")

    # ── Layer 2: MEMORY.md ──────────────────────────────────────────────

    @property
    def memory_md_path(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    def get_all_lessons(self) -> str:
        """Get full MEMORY.md content."""
        p = self.memory_md_path
        return p.read_text("utf-8") if p.exists() else ""

    def get_lessons(self, max_lines: int = 200) -> str:
        """Get MEMORY.md content truncated to max_lines."""
        p = self.memory_md_path
        if not p.exists():
            return ""
        lines = p.read_text("utf-8").splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"\n[TRUNCATED at {max_lines} lines]"]
        return "\n".join(lines)

    def add_lesson(self, lesson_text: str):
        """Append a lesson to MEMORY.md. Auto-trims if over limit."""
        p = self.memory_md_path
        header = MEMORY_HEADER if not p.exists() else ""

        # Read existing content
        existing = p.read_text("utf-8") if p.exists() else ""

        # Remove header from existing before counting
        body = existing.replace(MEMORY_HEADER, "").strip()

        # Append new lesson
        new_body = body + "\n\n" + lesson_text.rstrip() if body else lesson_text.rstrip()

        # Trim to max lines
        lines = new_body.splitlines()
        if len(lines) > MEMORY_MAX_LINES:
            lines = lines[-MEMORY_MAX_LINES:]
            new_body = "\n".join(lines)

        p.write_text(MEMORY_HEADER + new_body + "\n", encoding="utf-8")

    def save_reward(self, round_num: int, code: str):
        """Save reward function to memory for cross-round reference."""
        rewards_dir = self.memory_dir / "rewards"
        rewards_dir.mkdir(exist_ok=True)
        (rewards_dir / f"round{round_num}.py").write_text(code, encoding="utf-8")

    def get_reward(self, round_num: int) -> str:
        """Load a past round's reward function. Returns '' if not found."""
        p = self.memory_dir / "rewards" / f"round{round_num}.py"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def add_round_lesson(self, round_num: int, diagnosis_summary: str,
                          actual_outcome: str, learned: str):
        """Add a structured lesson entry for a completed round."""
        lesson = (
            f"**Round {round_num}**: "
            f"Modified reward because {diagnosis_summary}. "
            f"Actual outcome: {actual_outcome}. "
            f"Lesson: {learned}"
        )
        self.add_lesson(lesson)
        return lesson

    def query_lessons(self, keyword: str, max_results: int = 5) -> list[str]:
        """Search MEMORY.md for lessons relevant to keyword."""
        p = self.memory_md_path
        if not p.exists():
            return []

        text = p.read_text("utf-8")
        lessons = text.split("\n\n")

        matches = []
        kw = keyword.lower()
        for lesson in lessons:
            lesson = lesson.strip()
            if not lesson or lesson.startswith("#"):
                continue
            if kw in lesson.lower():
                matches.append(lesson)
        return matches[:max_results]

    # ── Similarity-based retrieval ──────────────────────────────────────

    def find_similar_rounds(self, query_feature_vector: dict[str, float],
                             n: int = 3) -> list[dict[str, Any]]:
        """Return most similar past rounds by cosine similarity of feature vectors.

        If all similarity scores are below 0.3, returns empty list
        (avoids injecting irrelevant memories).
        """
        scored = []

        for r in self.get_available_rounds():
            board = self.get_artifact(r, "evidence_board.json")
            if board is None:
                continue
            fv = board.get("feature_vector", {})
            if not fv:
                continue

            sim = cosine_similarity(query_feature_vector, fv)
            if sim < 0.3:  # Threshold to avoid noise
                continue

            scored.append({
                "round": r,
                "similarity": round(sim, 4),
                "diagnosis_summary": self._get_round_diagnosis_summary(r),
                "lesson": self._get_round_lesson(r),
                "evidence_board": board,
            })

        scored.sort(key=lambda x: -x["similarity"])
        return scored[:n]

    # ── Layer 3: Per-round artifact storage ─────────────────────────────

    def round_path(self, round_num: int) -> Path:
        return self.experiment_dir / f"round{round_num}"

    def get_available_rounds(self) -> list[int]:
        """Return sorted list of round numbers that exist."""
        rounds = []
        for d in self.experiment_dir.iterdir():
            if d.name.startswith("round") and d.name[5:].isdigit():
                try:
                    rounds.append(int(d.name[5:]))
                except ValueError:
                    pass
        return sorted(rounds)

    def store_artifact(self, round_num: int, name: str, data: dict | str):
        """Save an artifact (JSON dict or text) to a round directory."""
        rd = self.round_path(round_num)
        rd.mkdir(parents=True, exist_ok=True)
        p = rd / name

        if isinstance(data, dict):
            p.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            p.write_text(data, encoding="utf-8")

    def get_artifact(self, round_num: int, name: str) -> Optional[dict | str]:
        """Load an artifact from a round directory."""
        p = self.round_path(round_num) / name
        if not p.exists():
            return None

        if name.endswith(".json"):
            try:
                return json.loads(p.read_text("utf-8"))
            except Exception:
                return None
        return p.read_text("utf-8")

    def get_recent_lessons(self, n: int = 3) -> str:
        """Get a summary of the n most recent rounds for prompt injection."""
        rounds = self.get_available_rounds()
        recent = rounds[-n:] if len(rounds) > n else rounds

        parts = []
        for r in recent:
            board = self.get_artifact(r, "evidence_board.json")
            diagnosis = self.get_artifact(r, "diagnosis_A.json") or self.get_artifact(r, "final_diagnosis.json")

            lines = [f"### Round {r}"]
            if diagnosis and isinstance(diagnosis, dict):
                lines.append(f"Diagnosis: {diagnosis.get('diagnosis', 'N/A')[:200]}")
            if board and isinstance(board, dict):
                metrics = board.get("training_result", {}).get("behavior_descriptors", {})
                key_vals = ", ".join(
                    f"{k}={v.get('mean', '?')}" for k, v in list(metrics.items())[:4]
                )
                lines.append(f"Metrics: {key_vals}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    # ── Belief state management ─────────────────────────────────────────

    def get_belief(self, agent_id: str) -> dict:
        return self.belief_repo.get_belief(agent_id)

    def update_belief(self, agent_id: str, entry: dict) -> dict:
        return self.belief_repo.update_belief(agent_id, entry)

    def record_hypothesis_outcome(self, agent_id: str, hypothesis_id: str,
                                   category: str, predicted: str, actual: str,
                                   correct: bool):
        return self.belief_repo.record_hypothesis_outcome(
            agent_id, hypothesis_id, category, predicted, actual, correct
        )

    def format_beliefs_for_prompt(self, agent_id: str, max_history: int = 3) -> str:
        """Format agent beliefs as a compact prompt snippet."""
        belief = self.get_belief(agent_id)
        parts = [f"### Your Track Record ({agent_id})"]

        # Self-awareness
        sa = belief.get("self_awareness", "")
        if sa:
            parts.append(f"Self-awareness: {sa}")
        parts.append(f"Credibility: {belief.get('current_credibility', 0.5):.2f}")

        # Category accuracy
        acc = belief.get("hypothesis_category_accuracy", {})
        if acc:
            parts.append("Category accuracy:")
            for cat, stats in acc.items():
                parts.append(
                    f"  - {cat}: {stats['correct']}/{stats['proposed']} correct "
                    f"({stats['accuracy']:.0%})"
                )

        # Recent history
        hist = belief.get("history", [])[-max_history:]
        if hist:
            parts.append("Recent rounds:")
            for h in hist:
                parts.append(
                    f"  - Round {h.get('round', '?')}: "
                    f"{h.get('diagnosis_summary', h.get('diagnosis', 'N/A'))[:120]}"
                )

        return "\n".join(parts)

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_round_diagnosis_summary(self, round_num: int) -> str:
        diagnosis = self.get_artifact(round_num, "diagnosis_A.json") or self.get_artifact(round_num, "final_diagnosis.json")
        if diagnosis and isinstance(diagnosis, dict):
            return diagnosis.get("diagnosis", "")[:200]
        return ""

    def _get_round_lesson(self, round_num: int) -> str:
        """Extract lesson text for a specific round from MEMORY.md."""
        text = self.get_all_lessons()
        if not text:
            return ""
        for part in text.split("\n\n"):
            if f"**Round {round_num}**" in part:
                return part.strip()
        return ""
