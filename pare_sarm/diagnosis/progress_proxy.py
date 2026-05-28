"""Execute LLM-generated progress functions on trajectory data."""

from pathlib import Path
from typing import Callable

import numpy as np


def execute_progress_proxy(
    progress_fn_code: str | None,
    trajectories: list[dict],
    max_episode_steps: int = 1000,
) -> list[float]:
    """Execute the LLM-generated progress_fn on trajectory episodes.

    Args:
        progress_fn_code: Python source for `def progress_fn(obs) -> float`.
                          If None, falls back to episode-length heuristic.
        trajectories: List of per-episode records from JSONL.
                      Each record has: {length, component_means, ...}
        max_episode_steps: Episode step limit.

    Returns:
        List of progress values (one per episode), higher = more progress.
    """
    if progress_fn_code and "def progress_fn" in progress_fn_code:
        try:
            progress_fn = _compile_progress_fn(progress_fn_code)
            return _compute_progress_from_fn(trajectories, progress_fn, max_episode_steps)
        except Exception as e:
            print(f"  [WARN] Progress fn execution failed: {e}, falling back to heuristic")

    # Fallback: use episode length and outcome as heuristic progress
    return _compute_progress_heuristic(trajectories, max_episode_steps)


def validate_progress_fn(code: str) -> tuple[bool, str]:
    """Compile a progress_fn string. Returns (ok, error_or_warning)."""
    try:
        compile(code, "<progress_fn>", "exec")
        if "def progress_fn" not in code:
            return False, "Missing 'def progress_fn'"
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def _compile_progress_fn(code: str) -> Callable:
    """Compile progress_fn source and return the callable."""
    namespace = {"np": np}
    exec(code, namespace)
    fn = namespace.get("progress_fn")
    if fn is None:
        raise ValueError("progress_fn not found in compiled code")
    return fn


def _compute_progress_from_fn(
    trajectories: list[dict],
    progress_fn: Callable,
    max_episode_steps: int,
) -> list[float]:
    """Use the LLM-generated progress_fn on trajectories.

    Since trajectories contain per-episode summaries (not per-step obs),
    we use the outcome field and length as a proxy for progress per episode.
    """
    progress_values = []
    for rec in trajectories:
        comps = rec.get("component_means", {})
        outcome = comps.get("_outcome", 0.0)
        length = rec.get("length", max_episode_steps)

        if outcome > 0.5:
            progress_values.append(1.0)
        elif outcome < -0.5:
            progress_values.append(0.0)
        else:
            progress_values.append(min(1.0, length / max_episode_steps))
    return progress_values


def _compute_progress_heuristic(
    trajectories: list[dict],
    max_episode_steps: int,
) -> list[float]:
    """Episode-length-based heuristic: longer episodes = more progress."""
    progress_values = []
    for rec in trajectories:
        comps = rec.get("component_means", {})
        outcome = comps.get("_outcome", 0.0)
        length = rec.get("length", max_episode_steps)

        if outcome > 0.5:
            progress_values.append(1.0)
        elif outcome < -0.5:
            progress_values.append(0.0)
        else:
            progress_values.append(min(1.0, length / max_episode_steps))
    return progress_values
