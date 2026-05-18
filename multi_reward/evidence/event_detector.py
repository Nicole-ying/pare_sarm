"""
Critical event detection for evidence layer.

Detects structurally significant events in training dynamics:
- Reward hacking (component with high mean but near-zero std)
- Entropy collapse
- Action saturation
- Component dominance
- Exploration collapse

All detection is algorithmic — no LLM.
"""

from typing import Any


def detect_critical_events(
    component_stats: dict[str, dict],
    behavior_descriptors: dict[str, dict],
    entropy_history: list[dict],
    episode_stats: dict,
) -> list[dict[str, Any]]:
    """Detect critical events from training dynamics.

    Returns list of event dicts, each with:
        - type: event category
        - severity: "high" | "medium" | "low"
        - at_step: approximate timestep of onset
        - description: human-readable summary
        - evidence: dict of relevant metrics
    """
    events = []

    # 1. Component with high mean but near-zero std → potential reward hacking
    for name, stats in component_stats.items():
        cmean = stats.get("mean", 0)
        cstd = stats.get("std", 0)
        if abs(cmean) > 0.1 and cstd < 1e-6:
            events.append({
                "type": "constant_offset_component",
                "severity": "high",
                "at_step": "unknown",
                "description": (
                    f"Component '{name}' has mean={cmean:.4f} but std={cstd:.8f}. "
                    "Provides near-constant reward regardless of behavior — "
                    "can be harvested without learning."
                ),
                "evidence": {"component": name, "mean": cmean, "std": cstd},
            })

    # 2. Entropy collapse
    if entropy_history:
        initial = entropy_history[0].get("entropy", 0)
        final = entropy_history[-1].get("entropy", 0)
        if final < 0.1 and initial > 0.2:
            # Find the step where entropy first dropped below 0.1
            collapse_step = None
            for rec in entropy_history:
                if rec.get("entropy", 0) < 0.1:
                    collapse_step = rec.get("timestep")
                    break
            events.append({
                "type": "entropy_collapse",
                "severity": "high" if final < 0.05 else "medium",
                "at_step": collapse_step,
                "description": (
                    f"Policy entropy collapsed from {initial:.3f} to {final:.3f}. "
                    "Agent has likely converged to a deterministic local optimum."
                ),
                "evidence": {
                    "initial_entropy": initial,
                    "final_entropy": final,
                    "collapse_step": collapse_step,
                },
            })

    # 3. Component dominance (single component > 80% of total)
    shares = {
        name: abs(stats.get("share_of_total", 0))
        for name, stats in component_stats.items()
    }
    if shares:
        max_name = max(shares, key=shares.get)
        max_share = shares[max_name]
        if max_share > 0.80:
            events.append({
                "type": "component_dominance",
                "severity": "high" if max_share > 0.90 else "medium",
                "at_step": "unknown",
                "description": (
                    f"Component '{max_name}' dominates {max_share*100:.0f}% of total reward. "
                    "Agent is optimizing a single objective, ignoring others."
                ),
                "evidence": {"component": max_name, "share": max_share},
            })

    # 4. Action saturation (action magnitude near bounds)
    am = behavior_descriptors.get("action_magnitude", {})
    if am.get("mean", 0) > 0.95:
        events.append({
            "type": "action_saturation",
            "severity": "medium",
            "at_step": "unknown",
            "description": (
                f"Action magnitude mean={am['mean']:.3f} near maximum. "
                "Policy is saturating actuators, wasting energy."
            ),
            "evidence": {"action_magnitude_mean": am["mean"]},
        })

    # 5. Low survival rate
    tb = episode_stats.get("termination_breakdown", {})
    n_term = tb.get("terminated", {}).get("count", 0)
    n_trunc = tb.get("truncated", {}).get("count", 0)
    total_ep = n_term + n_trunc
    if total_ep > 0 and n_term / total_ep > 0.8:
        events.append({
            "type": "high_failure_rate",
            "severity": "high",
            "at_step": "unknown",
            "description": (
                f"{n_term}/{total_ep} episodes ({n_term/total_ep*100:.0f}%) "
                "ended in termination (failure). Agent is frequently dying."
            ),
            "evidence": {
                "terminated_count": n_term,
                "total_episodes": total_ep,
                "failure_rate": n_term / total_ep,
            },
        })

    # 6. Narrow episode length range → potential state coverage issue
    lengths = episode_stats.get("length_distribution", {})
    if lengths:
        q10 = lengths.get("q10", 0)
        q90 = lengths.get("q90", 0)
        if q90 > 0 and (q90 - q10) / q90 < 0.2:
            events.append({
                "type": "narrow_length_distribution",
                "severity": "low",
                "at_step": "unknown",
                "description": (
                    "Episode lengths concentrated in narrow range "
                    f"(q10={q10}, q90={q90}). Policy may be locked in "
                    "a repetitive local optimum."
                ),
                "evidence": {"q10": q10, "q90": q90},
            })

    return events
