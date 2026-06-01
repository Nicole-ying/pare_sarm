"""PARE health scoring: progress-aligned component diagnosis.

For each reward component r_i:
  H_i = 0.25 * A_i + 0.25 * B_i + 0.40 * P_i - 0.10 * C_i

Where:
  A_i = activation (std > threshold)
  B_i = balance (not dominating other components)
  P_i = progress alignment (correlation with progress delta)
  C_i = failure conflict (correlation with early termination)
"""

from __future__ import annotations


def compute_health_scores(
    components: list[dict],
    progress_values: list[float],
    failure_flags: list[float],
    max_episode_steps: int = 1000,
) -> dict:
    """Compute PARE health scores for each reward component.

    Args:
        components: List of {name, mean, std, ...} from component_stats.
        progress_values: Per-episode progress estimates (higher = better).
        failure_flags: Per-episode failure flags (1.0 = failure).

    Returns:
        Dict with overall_health (0-100), per-component verdicts, and summary.
    """
    if not components:
        return {
            "overall_health": 0.0,
            "components": [],
            "summary": "no components to diagnose",
        }

    # Compute per-component metrics
    total_abs = sum(abs(c["mean"]) for c in components)
    diagnosed = []

    for c in components:
        mean_v = c["mean"]
        std_v = c["std"]
        share = abs(mean_v) / max(total_abs, 1e-9)
        active = c.get("active", abs(mean_v) > 0.01)

        # Activation: min(std/threshold, 1)
        activation_score = min(std_v / 0.05, 1.0) if std_v > 0 else 0.0

        # Balance: 1 - dominance
        if share < 0.01:
            balance_score = 0.1  # too weak
        elif share > 0.7:
            balance_score = 0.1  # too dominant
        else:
            balance_score = 1.0 - abs(share - 0.25)  # ideal ~25% share

        # Progress alignment (placeholder — computed from actual trajectory data)
        progress_corr = 0.0
        failure_corr = 0.0

        # Now compute the actual correlations from trajectory-level data
        # These are computed externally and merged in, or computed here if data available
        progress_corr = c.get("progress_corr", 0.0)
        failure_corr = c.get("failure_corr", 0.0)

        # Progress score: map [-1, 1] to [0, 1]
        progress_score = (progress_corr + 1.0) / 2.0

        # Conflict score: map [-1, 1] to [0, 1], higher = worse
        conflict_score = (failure_corr + 1.0) / 2.0

        # Health score
        health = (
            0.25 * activation_score
            + 0.25 * balance_score
            + 0.40 * progress_score
            - 0.10 * conflict_score
        )
        health = max(0.0, min(1.0, health))

        # Diagnosis
        verdict, reason = _diagnose_component(
            name=c["name"],
            active=active,
            share=share,
            progress_corr=progress_corr,
            failure_corr=failure_corr,
        )

        diagnosed.append({
            "name": c["name"],
            "mean": mean_v,
            "std": std_v,
            "share": round(share, 3),
            "active": active,
            "progress_corr": round(progress_corr, 4),
            "failure_corr": round(failure_corr, 4),
            "health": round(health, 4),
            "verdict": verdict,
            "reason": reason,
        })

    # Overall health: mean of component healths, scaled to 0-100
    overall_health = 100.0 * sum(d["health"] for d in diagnosed) / len(diagnosed)

    n_aligned = sum(1 for d in diagnosed if d["progress_corr"] > 0.2)
    n_misaligned = sum(1 for d in diagnosed if d["progress_corr"] < -0.2)
    n_active = sum(1 for d in diagnosed if d["active"])

    summary = (
        f"{len(diagnosed)} components: {n_active} active, "
        f"{n_aligned} aligned with progress, {n_misaligned} misaligned, "
        f"health={overall_health:.0f}/100"
    )

    return {
        "overall_health": round(overall_health, 1),
        "components": diagnosed,
        "summary": summary,
    }


def _diagnose_component(
    name: str,
    active: bool,
    share: float,
    progress_corr: float,
    failure_corr: float,
) -> tuple[str, str]:
    """Generate a verdict and reason for a single component."""
    if not active:
        return "remove", f"{name} is inactive (|mean| ≈ 0) — remove or rescale"

    reasons = []

    if progress_corr > 0.3:
        reasons.append(f"strongly aligned with progress (P={progress_corr:+.2f})")
    elif progress_corr < -0.2:
        reasons.append(f"misaligned with progress (P={progress_corr:+.2f})")
    elif abs(progress_corr) < 0.1:
        reasons.append(f"no progress signal (P={progress_corr:+.2f})")

    if failure_corr > 0.3:
        reasons.append(f"correlated with failure (F={failure_corr:+.2f})")

    if share > 0.5:
        reasons.append(f"dominates reward ({share:.0%} share)")

    if not reasons:
        return "keep", f"{name} is healthy"

    combined = "; ".join(reasons)

    if progress_corr < -0.2 or failure_corr > 0.3:
        return "reduce", combined
    elif share > 0.5:
        return "reduce", combined
    elif abs(progress_corr) < 0.1:
        return "reconsider", combined
    else:
        return "keep", combined


def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 if insufficient data."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    sx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n * sx * sy)


def compute_progress_correlations(
    component_stats: list[dict],
    trajectory_records: list[dict],
    progress_values: list[float],
    max_episode_steps: int = 1000,
) -> list[dict]:
    """Augment component stats with progress and failure correlations.

    This uses per-episode component means (from JSONL) correlated against
    per-episode progress values and failure flags.

    Returns updated component stats list with added progress_corr and failure_corr fields.
    """
    n_eps = len(trajectory_records)
    if n_eps < 3:
        return component_stats

    # Build per-episode series for each component
    comp_series: dict[str, list[float]] = {}
    failure_flags: list[float] = []

    for rec in trajectory_records:
        length = rec.get("length", max_episode_steps)
        for name, val in rec.get("component_means", {}).items():
            if name == "_outcome":
                continue
            comp_series.setdefault(name, []).append(float(val))
        failure_flags.append(1.0 if length < max_episode_steps * 0.3 else 0.0)

    for c in component_stats:
        name = c["name"]
        vals = comp_series.get(name, [])
        if len(vals) >= 3:
            c["progress_corr"] = round(pearson_r(vals, progress_values[:len(vals)]), 4)
            c["failure_corr"] = round(pearson_r(vals, failure_flags[:len(vals)]), 4)
        else:
            c["progress_corr"] = 0.0
            c["failure_corr"] = 0.0

    return component_stats


def format_health_table(health_result: dict) -> str:
    """Format health scores as a markdown table for LLM prompts."""
    comps = health_result.get("components", [])
    if not comps:
        return "*(no health data)*"

    lines = [
        f"## Component Health Diagnosis",
        f"Overall health: {health_result['overall_health']:.0f}/100",
        f"Summary: {health_result.get('summary', '')}",
        "",
        f"| Component | Mean | Active | Share | Progress Corr | Failure Corr | Health | Verdict |",
        f"|-----------|------|--------|-------|---------------|--------------|--------|---------|",
    ]
    for c in comps:
        lines.append(
            f"| {c['name']} | {c['mean']:.4f} | {c['active']} | {c['share']:.1%} | "
            f"{c['progress_corr']:+.3f} | {c['failure_corr']:+.3f} | "
            f"{c['health']:.2f} | {c['verdict']} |"
        )
    lines.append("")
    lines.append("**Key:** Progress Corr > 0 = helps task progress. "
                  "Failure Corr > 0 = linked to early termination.")
    return "\n".join(lines)
