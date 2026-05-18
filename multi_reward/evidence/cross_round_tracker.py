"""
Cross-round trend computation for evidence layer.

Tracks how key metrics change across rounds.
Pure statistics — no LLM, no "good/bad" judgment.
"""

from pathlib import Path
from typing import Any


def compute_cross_round_trends(
    current_board: dict,
    previous_boards: list[dict],
) -> dict[str, Any]:
    """Compute cross-round trends from current + historical evidence boards.

    Args:
        current_board: Current round's evidence board.
        previous_boards: List of previous evidence boards, earliest first.

    Returns:
        Dict mapping metric name to {roundN: value, ..., direction: "increasing"|"decreasing"|"stable"}.
    """
    # Collect all boards in order
    all_boards = list(previous_boards) + [current_board]

    # Metrics to track across rounds
    metric_extractors = {
        "mean_length": lambda b: b["training_result"]["episode_stats"].get("mean_length", 0),
        "termination_rate": _extract_termination_rate,
        "action_magnitude": lambda b: _extract_behavior_mean("action_magnitude")(b),
        "velocity_x": _extract_velocity,
        "action_efficiency": lambda b: _extract_behavior_mean("action_efficiency")(b),
        "max_component_share": _extract_max_share,
        "entropy_final": _extract_entropy_final,
    }

    trends = {}
    for metric_name, extractor in metric_extractors.items():
        values = {}
        for board in all_boards:
            r = board["meta"]["round"]
            try:
                val = extractor(board)
                if val is not None:
                    values[f"round{r}"] = val
            except Exception:
                values[f"round{r}"] = None

        if len(values) >= 2:
            numeric_vals = [v for v in values.values() if isinstance(v, (int, float))]
            if len(numeric_vals) >= 2:
                first, last = numeric_vals[0], numeric_vals[-1]
                if last > first * 1.05:
                    direction = "increasing"
                elif last < first * 0.95:
                    direction = "decreasing"
                else:
                    direction = "stable"
            else:
                direction = "unknown"
            values["direction"] = direction
        else:
            values["direction"] = "insufficient_data"

        trends[metric_name] = values

    return trends


def _extract_termination_rate(board: dict) -> float:
    tb = board["training_result"]["episode_stats"].get("termination_breakdown", {})
    n_term = tb.get("terminated", {}).get("count", 0)
    n_total = n_term + tb.get("truncated", {}).get("count", 0)
    return n_term / max(n_total, 1)


def _extract_behavior_mean(key: str):
    def _fn(board: dict) -> float:
        bd = board["training_result"]["behavior_descriptors"]
        return bd.get(key, {}).get("mean", 0)
    return _fn


def _extract_velocity(board: dict) -> float:
    bd = board["training_result"]["behavior_descriptors"]
    for k in ("velocity_x", "velocity_mean", "velocity"):
        val = bd.get(k, {}).get("mean")
        if val is not None:
            return float(val)
    return 0.0


def _extract_max_share(board: dict) -> float:
    rc = board["training_result"]["reward_components"]
    shares = [abs(c.get("share_of_total", 0)) for c in rc.values()]
    return max(shares) / max(sum(shares), 1e-9) if shares else 0


def _extract_entropy_final(board: dict) -> float:
    hc = board["training_result"]["health_checks"]
    return float(hc.get("entropy_collapse", {}).get("final_entropy", 0))
