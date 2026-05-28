"""Behavioral diversity check: detect if reward candidates produce equivalent behavior.

Two reward functions may look different in code but produce nearly identical
trajectory preferences. This module detects such duplicates at three levels:

Level 1 — Component Jaccard distance: are the component name sets different?
Level 2 — Reward-vector correlation: do two rewards rank the same trajectories identically?
Level 3 — Code text similarity: are they nearly identical source code?

Level 2 is the GOLD STANDARD: if two reward functions give the same cumulative
reward to the same batch of trajectories (corr > 0.95), they are behaviorally
equivalent regardless of how different the code looks.

Reference: CARD (Sun et al., 2025) uses Trajectory Preference Evaluation to
judge reward quality without full training. We adapt this idea to detect
behavioral equivalence between reward candidates.
"""

from __future__ import annotations

import re
import numpy as np


def check_behavioral_diversity(
    candidates: list[dict],
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Filter out behaviorally duplicate candidates.

    Uses component Jaccard distance as the primary lightweight check.
    When proxy-trained models are available, reward-vector correlation
    provides a stronger behavioral equivalence test.

    Args:
        candidates: List of {idx, code, health, ...} dicts.
        similarity_threshold: Jaccard similarity above which candidates
                              are considered duplicates.

    Returns:
        Filtered list (lower-scoring near-duplicates removed).
    """
    if len(candidates) <= 1:
        return candidates

    to_remove = set()

    # ── Level 1: Component Jaccard ──
    comp_sets = [_extract_component_names(c.get("code", "")) for c in candidates]

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if i in to_remove or j in to_remove:
                continue
            jaccard = _jaccard_similarity(comp_sets[i], comp_sets[j])

            # ── Level 3: Code text similarity ──
            text_sim = _code_similarity(
                candidates[i].get("code", ""),
                candidates[j].get("code", ""),
            )

            # Mark as duplicate if both Jaccard AND text similarity are high
            if jaccard > similarity_threshold and text_sim > 0.85:
                score_i = candidates[i].get("health", {}).get("overall_health", 0)
                score_j = candidates[j].get("health", {}).get("overall_health", 0)
                remove_idx = j if score_i >= score_j else i
                keep_idx = i if score_i >= score_j else j
                to_remove.add(remove_idx)
                print(f"  Diversity: candidate {remove_idx} is duplicate of {keep_idx} "
                      f"(Jaccard={jaccard:.2f}, text_sim={text_sim:.2f}), removing")

    # ── Level 2: Reward-vector correlation (if proxy models available) ──
    # This is the GOLD STANDARD from GPT §6.3
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if i in to_remove or j in to_remove:
                continue
            # Use health score correlation as proxy for behavioral correlation
            # (full reward-vector correlation requires env access)
            hi = candidates[i].get("health", {})
            hj = candidates[j].get("health", {})
            comps_i = hi.get("components", [])
            comps_j = hj.get("components", [])
            if comps_i and comps_j:
                # Compare component-level progress correlation patterns
                rvc = _component_pattern_correlation(comps_i, comps_j)
                if rvc > 0.95:
                    score_i = hi.get("overall_health", 0)
                    score_j = hj.get("overall_health", 0)
                    remove_idx = j if score_i >= score_j else i
                    keep_idx = i if score_i >= score_j else j
                    if remove_idx not in to_remove:
                        to_remove.add(remove_idx)
                        print(f"  Diversity: candidate {remove_idx} behaviorally equivalent to {keep_idx} "
                              f"(pattern_corr={rvc:.3f}), removing")

    if not to_remove:
        print(f"  Diversity: all candidates behaviorally distinct")
        return candidates

    kept = [c for i, c in enumerate(candidates) if i not in to_remove]
    print(f"  Diversity: kept {len(kept)}/{len(candidates)} candidates")
    return kept


def reward_vector_correlation(
    reward_fn_a,
    reward_fn_b,
    trajectories: list[dict],
) -> float:
    """Compute correlation between two reward functions on the same trajectories.

    This is the GOLD-STANDARD diversity check (GPT design §6.3).

    For each trajectory τ in the fixed set, compute:
      V_a[k] = Σ_t R_a(obs_t, action_t, next_obs_t, done_t)
      V_b[k] = Σ_t R_b(obs_t, action_t, next_obs_t, done_t)

    Then: corr(V_a, V_b)

    If corr > 0.95, the two rewards are BEHAVIORALLY EQUIVALENT — they would
    train policies with the same preferences, regardless of code differences.

    Args:
        reward_fn_a: First reward function callable
        reward_fn_b: Second reward function callable
        trajectories: List of trajectory dicts, each with a "steps" list.
                      Each step: {obs, action, next_obs, done}

    Returns:
        Pearson r in [-1, 1]. Values > 0.95 indicate behavioral equivalence.
    """
    va = _reward_vector(reward_fn_a, trajectories)
    vb = _reward_vector(reward_fn_b, trajectories)
    return _pearson_r(va, vb)


def _reward_vector(reward_fn, trajectories: list[dict]) -> np.ndarray:
    """Compute cumulative reward for each trajectory under a reward function."""
    values = []
    for traj in trajectories:
        steps = traj.get("steps", [traj])  # support both formats
        total = 0.0
        for step in steps:
            try:
                r, _ = reward_fn(
                    step["obs"], step.get("action", 0),
                    step.get("next_obs", step["obs"]),
                    step.get("done", False),
                    step.get("info", {}),
                )
                total += float(r)
            except Exception:
                total += 0.0
        values.append(total)
    return np.array(values)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[:n], y[:n]
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_component_names(code: str) -> set[str]:
    """Extract reward component names from Python source code.

    Looks for string keys in the components dict:
      components = {"distance": ..., "angle": ...}
    """
    names = set(re.findall(r'"(\w+)"\s*:', code))
    # Also detect variable names assigned before the dict
    for m in re.finditer(r'(\w+)\s*=', code):
        name = m.group(1)
        if name not in ("components", "total", "_outcome", "state", "obs",
                        "action", "terminated", "truncated", "done", "info",
                        "reward", "result", "x", "y", "theta", "vx", "vy"):
            # Check if this variable appears as a value in the components dict
            if re.search(r'"\w+":\s*' + name, code):
                names.add(name)
    return names


def _jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _component_pattern_correlation(comps_a: list[dict], comps_b: list[dict]) -> float:
    """Compare component progress-correlation patterns between two candidates.

    If two rewards produce similar component-progress correlation patterns,
    they are likely behaviorally equivalent — even if the code differs.

    This is a lightweight proxy for the full reward-vector correlation (§6.3)
    that works without needing to execute reward functions.
    """
    names_a = {c["name"]: c.get("progress_corr", 0) for c in comps_a}
    names_b = {c["name"]: c.get("progress_corr", 0) for c in comps_b}
    common = set(names_a.keys()) & set(names_b.keys())
    if len(common) < 2:
        return 0.0  # not enough overlap to judge
    x = [names_a[n] for n in common]
    y = [names_b[n] for n in common]
    return _pearson_r(np.array(x), np.array(y))


def _code_similarity(code_a: str, code_b: str) -> float:
    """Simple character-level similarity for text comparison."""
    if not code_a or not code_b:
        return 0.0
    if code_a == code_b:
        return 1.0
    # Remove whitespace variation for comparison
    a_clean = re.sub(r'\s+', '', code_a)
    b_clean = re.sub(r'\s+', '', code_b)
    set_a, set_b = set(a_clean), set(b_clean)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
