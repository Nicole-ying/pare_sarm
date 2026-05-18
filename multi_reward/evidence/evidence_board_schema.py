"""
Evidence board schema and validation for multi_reward framework.

The evidence board is the central data structure — algorithmically generated,
never LLM-generated. All agents read from it. No inter-agent information passing
through it (it's a shared fact source, not a communication channel).
"""

from typing import Any


# Expected schema for validation
EVIDENCE_BOARD_SCHEMA = {
    "meta": dict,
    "environment_context": dict,
    "training_result": dict,
    "previous_proposal": dict,
    "cross_round_trends": dict,
    "feature_vector": dict,
}

TRAINING_RESULT_SCHEMA = {
    "episode_stats": dict,
    "reward_components": dict,
    "behavior_descriptors": dict,
    "health_checks": dict,
    "critical_events": list,
}

HEALTH_CHECK_SCHEMA = {
    "component_activity": dict,
    "component_dominance": dict,
    "entropy_collapse": dict,
    "survival_health": dict,
    "serious_violation": dict,
}

FEATURE_VECTOR_KEYS = [
    "mean_length",
    "action_magnitude_mean",
    "velocity_mean",
    "action_efficiency",
    "component_activity_ratio",
    "max_component_share",
    "termination_ratio",
    "entropy_final",
]


def create_empty_board() -> dict[str, Any]:
    """Create an empty evidence board with default structure."""
    return {
        "meta": {
            "round": 0,
            "experiment_id": "",
            "generated_at": "",
            "n_episodes": 0,
            "total_training_steps": 0,
        },
        "environment_context": {
            "obs_dim": 0,
            "action_dim": 0,
            "action_bounds": {"low": -1.0, "high": 1.0},
            "max_episode_steps": 1000,
            "termination_conditions": [],
            "zero_action_profile": {},
        },
        "training_result": {
            "episode_stats": {},
            "reward_components": {},
            "behavior_descriptors": {},
            "health_checks": {},
            "critical_events": [],
        },
        "previous_proposal": {},
        "cross_round_trends": {},
        "feature_vector": {},
    }


def validate_evidence_board(board: dict) -> list[str]:
    """Validate evidence board structure. Returns list of issues (empty = valid)."""
    issues = []

    for key, expected_type in EVIDENCE_BOARD_SCHEMA.items():
        if key not in board:
            issues.append(f"Missing top-level key: '{key}'")
        elif not isinstance(board[key], expected_type):
            issues.append(
                f"Key '{key}': expected {expected_type.__name__}, "
                f"got {type(board[key]).__name__}"
            )

    # Validate training_result sub-structure
    tr = board.get("training_result", {})
    if isinstance(tr, dict):
        for key, expected_type in TRAINING_RESULT_SCHEMA.items():
            if key not in tr:
                issues.append(f"Missing training_result key: '{key}'")

    return issues


def board_to_feature_vector(board: dict) -> dict[str, float]:
    """Extract a normalized feature vector from an evidence board for similarity computation."""
    fv = {}
    tr = board.get("training_result", {})

    # Episode stats
    es = tr.get("episode_stats", {})
    fv["mean_length"] = float(es.get("mean_length", 0))

    # Behavior descriptors
    bd = tr.get("behavior_descriptors", {})
    fv["action_magnitude_mean"] = float(
        bd.get("action_magnitude", {}).get("mean", 0)
    )
    fv["velocity_mean"] = float(
        bd.get("velocity_x", {}).get("mean", bd.get("velocity_mean", {}).get("mean", 0))
    )
    fv["action_efficiency"] = float(
        bd.get("action_efficiency", {}).get("mean", 0)
    )

    # Component stats
    rc = tr.get("reward_components", {})
    shares = [abs(c.get("share_of_total", 0)) for c in rc.values()]
    total = sum(shares) + 1e-9
    fv["max_component_share"] = max(shares) / total if shares else 0
    active = sum(1 for c in rc.values() if abs(c.get("mean", 0)) > 0.01)
    fv["component_activity_ratio"] = active / max(len(rc), 1)

    # Termination
    tb = es.get("termination_breakdown", {})
    n_term = tb.get("terminated", {}).get("count", 0)
    n_total = n_term + tb.get("truncated", {}).get("count", 0) + 1
    fv["termination_ratio"] = n_term / n_total

    # Health
    hc = tr.get("health_checks", {})
    ec = hc.get("entropy_collapse", {})
    fv["entropy_final"] = float(ec.get("final_entropy", 0))

    return fv
