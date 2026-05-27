"""
progress_diagnosis.py — Progress-Anchored Reward Component Diagnosis.

Computes per-component alignment with task progress (PARE method).

For each reward component r_i, computes:
- P_i: Pearson correlation between r_i and progress delta Δp
- C_i: correlation between r_i and episode failure
- Activation: whether |mean(r_i)| > threshold
- Balance: whether r_i dominates other components

Outputs a structured diagnosis table for the Analyzer's prompt:
  "distance_reward: P=+0.72 (aligned), keep"
  "fuel_penalty: P=-0.51 (misaligned), reduce"
  "angle_penalty: P=+0.03 (no correlation), reconsider"
"""

from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any


def compute_progress_diagnosis(
    trajectory_dir: Path,
    progress_fn_code: str | None,
    max_episode_steps: int = 1000,
) -> dict:
    """Analyze trajectory logs and produce per-component progress diagnosis.

    Args:
        trajectory_dir: Path to trajectory_logs/ directory with JSONL files.
        progress_fn_code: Python code string for a function `progress_fn(obs) -> float`.
                          If None, uses a heuristic based on observation magnitude.
        max_episode_steps: Episode time limit.

    Returns:
        Dict with:
        - components: list of {name, mean, std, progress_corr, failure_corr, diagnosis}
        - summary: one-sentence overall assessment
        - health_score: 0-100 overall score
    """
    records = _load_trajectories(trajectory_dir)
    if not records:
        return {"components": [], "summary": "no trajectory data", "health_score": 0.0}

    # Extract component time series and episode outcomes
    comp_series: dict[str, list[float]] = {}
    progress_series: list[float] = []
    failure_flags: list[float] = []
    obs_samples: list = []

    for rec in records:
        comps = rec.get("component_means", {})
        length = rec.get("length", max_episode_steps)
        for name, val in comps.items():
            if name == "_outcome":
                continue
            comp_series.setdefault(name, []).append(float(val))

        # Episode outcome: 1.0 = failure (short episode), 0.0 = survived
        is_failure = 1.0 if length < max_episode_steps * 0.3 else 0.0
        failure_flags.append(is_failure)

        # Progress proxy: use _outcome if available, else length-based heuristic
        outcome = comps.get("_outcome", 0.0)
        if outcome > 0.5:
            progress_series.append(1.0)
        elif outcome < -0.5:
            progress_series.append(-1.0)
        else:
            # Heuristic: longer episodes = more progress (imperfect but general)
            progress_series.append(min(1.0, length / max_episode_steps))

    if not comp_series:
        return {"components": [], "summary": "no reward components found", "health_score": 0.0}

    # Compute per-component statistics
    components = []
    abs_means = []
    for name, vals in comp_series.items():
        n = len(vals)
        mean_v = sum(vals) / n
        std_v = (sum((x - mean_v)**2 for x in vals) / n) ** 0.5 if n > 1 else 0.0
        abs_means.append(abs(mean_v))

        p_corr = _pearson_r(vals, progress_series) if len(vals) >= 5 else 0.0
        f_corr = _pearson_r(vals, failure_flags) if len(vals) >= 5 else 0.0

        components.append({
            "name": name,
            "mean": round(mean_v, 6),
            "std": round(std_v, 6),
            "n": n,
            "progress_corr": round(p_corr, 4),
            "failure_corr": round(f_corr, 4),
        })

    # Diagnosis per component
    total_abs = sum(abs_means)
    for c in components:
        p = c["progress_corr"]
        f = c["failure_corr"]
        share = abs(c["mean"]) / max(total_abs, 1e-9)
        active = abs(c["mean"]) > 0.01

        diagnoses = []
        if not active:
            diagnoses.append("inactive — remove or rescale")
        elif p > 0.3:
            diagnoses.append("aligned with progress — keep or strengthen")
        elif p < -0.2:
            diagnoses.append(f"misaligned (P={p:.2f}) — reduce or negate")
        elif abs(p) < 0.1:
            diagnoses.append(f"no progress signal (P={p:.2f}) — reconsider this component")

        if f > 0.3:
            diagnoses.append("correlated with failure — likely harmful")
        if share > 0.5:
            diagnoses.append(f"dominates ({share:.0%} of total) — rescale down")

        c["diagnosis"] = "; ".join(diagnoses) if diagnoses else "acceptable"
        c["active"] = active
        c["share"] = round(share, 3)

    # Overall health score
    activation = sum(1 for c in components if c["active"]) / max(len(components), 1)
    balance = max(0.0, 1.0 - max(c["share"] for c in components))
    alignment = sum(max(0, c["progress_corr"]) for c in components) / max(len(components), 1)
    conflict = sum(max(0, c["failure_corr"]) for c in components) / max(len(components), 1)

    health = 100 * (0.25 * activation + 0.25 * balance + 0.40 * alignment - 0.10 * conflict)
    health = max(0, min(100, health))

    aligned = sum(1 for c in components if c["progress_corr"] > 0.2)
    misaligned = sum(1 for c in components if c["progress_corr"] < -0.2)

    summary = (
        f"{len(components)} components: {aligned} aligned, {misaligned} misaligned, "
        f"activation={activation:.0%}, balance={balance:.2f}, "
        f"health={health:.0f}/100"
    )

    return {
        "components": components,
        "summary": summary,
        "health_score": round(health, 1),
    }


def format_diagnosis_table(diagnosis: dict) -> str:
    """Format the diagnosis as a markdown table for the Analyzer prompt."""
    comps = diagnosis.get("components", [])
    if not comps:
        return "*(no component diagnosis available)*"

    lines = [
        f"## Progress-Aligned Component Diagnosis",
        f"Summary: {diagnosis.get('summary', '')}",
        f"",
        f"| Component | Mean | Active | Progress Corr | Failure Corr | Diagnosis |",
        f"|-----------|------|--------|---------------|--------------|-----------|",
    ]
    for c in comps:
        active = "yes" if c["active"] else "no"
        lines.append(
            f"| {c['name']} | {c['mean']:.4f} | {active} | "
            f"{c['progress_corr']:+.3f} | {c['failure_corr']:+.3f} | "
            f"{c['diagnosis']} |"
        )
    lines.append("")
    lines.append("**Key:** Progress Corr > 0 = component helps task progress. Failure Corr > 0 = component linked to early termination.")
    lines.append("Misaligned components (Progress Corr < -0.2) should be reduced, negated, or removed.")
    return "\n".join(lines)


# ── Internal ────────────────────────────────────────────────────────────────

def _load_trajectories(traj_dir: Path) -> list[dict]:
    if not traj_dir.exists():
        return []
    records = []
    for f in sorted(traj_dir.glob("*.jsonl")):
        for line in f.read_text("utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    sx = (sum((xi - mx)**2 for xi in x) / n) ** 0.5
    sy = (sum((yi - my)**2 for yi in y) / n) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n * sx * sy)
