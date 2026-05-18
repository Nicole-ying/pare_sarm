"""
Per-Diagnostician belief state tracking.

Tracks each Diagnostician's hypothesis performance across rounds
so the agent can learn from its own mistakes.
"""


def create_default_belief(agent_id: str) -> dict:
    return {
        "agent": agent_id,
        "version": 2,
        "history": [],
        "hypothesis_category_accuracy": {},
        "current_credibility": 0.5,
        "self_awareness": "",
    }


def format_belief_for_prompt(belief: dict, max_history: int = 3) -> str:
    """Format belief state as a compact prompt snippet for injection."""
    if not belief or not belief.get("history"):
        return ""

    parts = ["### Your Track Record"]

    sa = belief.get("self_awareness", "")
    if sa:
        parts.append(f"Self-awareness: {sa}")

    acc = belief.get("hypothesis_category_accuracy", {})
    if acc:
        parts.append("\nCategory accuracy:")
        for cat, stats in sorted(acc.items()):
            if stats.get("proposed", 0) > 0:
                parts.append(
                    f"  {cat}: {stats['correct']}/{stats['proposed']} correct "
                    f"({stats['accuracy']:.0%})"
                )

    hist = belief.get("history", [])[-max_history:]
    if hist:
        parts.append("\nRecent rounds:")
        for h in hist:
            parts.append(
                f"  R{h.get('round', '?')}: {h.get('diagnosis_summary', '')[:120]}"
            )

    return "\n".join(parts)
