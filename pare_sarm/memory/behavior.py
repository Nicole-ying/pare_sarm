"""Cross-round behavior memory: tracks what the agent DID, not just reward stats.

This is the KEY difference from Eureka's reflection:
Eureka reflects on reward COMPONENT statistics (mean/std per component).
PARE reflects on AGENT BEHAVIOR patterns across rounds (hovering? crashing? improving?).

This enables detection of multi-round patterns like oscillation between failure modes.
"""

import json
from pathlib import Path


class BehaviorMemory:
    """Tracks agent behavior patterns across rounds.

    Each round stores:
    - Episode length trend during training
    - Dominant behavioral pattern (hovering, crashing, improving, oscillating)
    - What was diagnosed and what repair was attempted
    - Whether the repair worked or caused a new failure mode
    """

    PATTERN_HOVERING = "hovering"       # Survives to max steps but doesn't complete task
    PATTERN_CRASHING = "crashing"       # Terminates very early (<20% max steps)
    PATTERN_MODERATE = "moderate"       # Middle range, not clearly good or bad
    PATTERN_IMPROVING = "improving"     # Getting better over training
    PATTERN_DECLINING = "declining"     # Getting worse over training

    def __init__(self, exp_dir: Path):
        self._path = Path(exp_dir) / "memory" / "behavior.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self._load()

    def record(self, round_num: int, **kwargs):
        """Record behavior data for a completed round."""
        record = {
            "round": round_num,
            "final_length": kwargs.get("final_length", 0),
            "max_length": kwargs.get("max_length", 0),
            "max_episode_steps": kwargs.get("max_episode_steps", 1000),
            "length_trend": kwargs.get("length_trend", []),
            "health_score": kwargs.get("health_score", 0),
            "pattern": kwargs.get("pattern", ""),
            "diagnosis": kwargs.get("diagnosis", "")[:300],
            "repair_strategy": kwargs.get("repair_strategy", ""),
        }
        # Overwrite if round already exists
        existing = [i for i, r in enumerate(self._records) if r["round"] == round_num]
        if existing:
            self._records[existing[0]] = record
        else:
            self._records.append(record)

    def get_history(self) -> list[dict]:
        """Return all behavior records sorted by round."""
        return sorted(self._records, key=lambda r: r["round"])

    def detect_patterns(self) -> dict:
        """Detect cross-round behavioral patterns.

        Returns dict with:
        - oscillation: True if alternating between hovering and crashing
        - stagnation: True if health improves but behavior doesn't
        - trend: "improving", "oscillating", "stagnant", "declining"
        - suggestion: what the next repair should consider
        """
        records = self.get_history()
        if len(records) < 2:
            return {"oscillation": False, "stagnation": False, "trend": "insufficient_data",
                    "suggestion": ""}

        patterns = [r["pattern"] for r in records]
        healths = [r["health_score"] for r in records]
        lengths = [r["final_length"] for r in records]
        max_steps = records[0].get("max_episode_steps", 1000)

        # Oscillation: alternating between hovering and crashing
        oscillation = False
        if len(patterns) >= 3:
            # Check if pattern alternates between hovering and crashing
            is_hover = [p == self.PATTERN_HOVERING for p in patterns]
            is_crash = [p == self.PATTERN_CRASHING for p in patterns]
            alternates = all(
                (is_hover[i] and is_crash[i+1]) or (is_crash[i] and is_hover[i+1])
                for i in range(len(patterns) - 1)
            )
            if alternates:
                oscillation = True

        # Stagnation: health improves but behavior doesn't
        stagnation = (
            len(healths) >= 2
            and healths[-1] > healths[0] + 5  # health improved
            and lengths[-1] < max_steps * 0.3  # but still crashing
        )

        # Overall trend
        if oscillation:
            trend = "oscillating"
            suggestion = (
                "WARNING: The reward is oscillating between two failure modes "
                "(hovering ↔ crashing). Each repair overshoots. Do NOT continue "
                "incremental coefficient tuning. Consider a FUNDAMENTALLY different "
                "reward structure: phase-aware shaping (different incentives at "
                "different stages of descent), or reduce ALL per-step magnitudes "
                "and rely primarily on a well-designed terminal bonus."
            )
        elif stagnation:
            trend = "stagnant"
            suggestion = (
                "Health scores are improving but the agent still crashes. "
                "The component structure may be improving but the reward still "
                "lacks a clear success signal. Strengthen the terminal bonus "
                "and ensure at least one per-step component provides a dense "
                "progress gradient."
            )
        elif lengths and lengths[-1] > lengths[0] * 1.2:
            trend = "improving"
            suggestion = ("Behavior is improving. Continue with incremental refinements, "
                         "focusing on the weakest component identified by health metrics.")
        elif lengths and lengths[-1] < lengths[0] * 0.8:
            trend = "declining"
            suggestion = ("Behavior is getting worse. The last repair likely introduced "
                         "a new perverse incentive. Revert the last change's approach "
                         "and try a different strategy.")
        else:
            trend = "stable"
            suggestion = ""

        return {
            "oscillation": oscillation,
            "stagnation": stagnation,
            "trend": trend,
            "suggestion": suggestion,
            "patterns": patterns,
            "healths": healths,
            "lengths": lengths,
        }

    def format_history_table(self) -> str:
        """Format behavior history as a markdown table for the Analyzer prompt."""
        records = self.get_history()
        if not records:
            return "*(no behavior history)*"

        patterns_detected = self.detect_patterns()

        lines = ["## Cross-Round Behavior History", ""]
        lines.append("| Round | Health | Final Len | Pattern | Diagnosis (abbreviated) |")
        lines.append("|-------|--------|-----------|---------|------------------------|")

        for r in records:
            lines.append(
                f"| {r['round']} | {r['health_score']:.0f} | {r['final_length']:.0f} | "
                f"{r['pattern']} | {r['diagnosis'][:80]} |"
            )

        lines.append("")

        # Add pattern detection results
        if patterns_detected.get("oscillation"):
            lines.append("**DETECTED: Reward is OSCILLATING between failure modes.**")
            lines.append("Each repair overshoots — do NOT repeat the same strategy.")
            lines.append("")

        if patterns_detected.get("stagnation"):
            lines.append("**DETECTED: Health improving but behavior STAGNANT.**")
            lines.append("Component metrics look better but the agent still fails.")
            lines.append("")

        if patterns_detected.get("suggestion"):
            lines.append(f"**Guidance for next repair:** {patterns_detected['suggestion']}")
            lines.append("")

        lines.append("**Pattern interpretation:**")
        lines.append(f"- 'hovering' = agent survives near max steps without completing task (per-step reward farming)")
        lines.append(f"- 'crashing' = agent terminates very early (penalties dominate, learns to die fast)")
        lines.append(f"- 'moderate' = in between, not clearly good or bad")
        lines.append("")

        return "\n".join(lines)

    def classify_behavior(self, final_length: float, max_episode_steps: int,
                          length_trend: list = None) -> str:
        """Classify a round's behavior pattern from episode length."""
        if final_length >= max_episode_steps * 0.8:
            return self.PATTERN_HOVERING
        elif final_length < max_episode_steps * 0.2:
            return self.PATTERN_CRASHING
        elif length_trend and len(length_trend) >= 2:
            first = length_trend[0]
            last = length_trend[-1]
            if last > first * 1.2:
                return self.PATTERN_IMPROVING
            elif last < first * 0.8:
                return self.PATTERN_DECLINING
        return self.PATTERN_MODERATE

    def save(self):
        """Persist to disk."""
        self._path.write_text(
            json.dumps(self._records, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self):
        """Load from disk if exists."""
        if self._path.exists():
            try:
                self._records = json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._records = []
