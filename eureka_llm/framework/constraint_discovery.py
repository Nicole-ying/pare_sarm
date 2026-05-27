"""
constraint_discovery.py — Algorithmic (non-LLM) detection of transferable
training-dynamics constraint violations.

All metric names are auto-discovered from env_metrics and component data.
No metric name is hardcoded — detection is based on statistical profiles.
"""

from __future__ import annotations

from typing import Any
import math


def _auto_profile_metrics(
    env_metrics: dict, eval_history: list[dict]
) -> dict[str, dict]:
    """Build a statistical profile for every metric key found in the data.

    Returns:
        {metric_name: {mean, std, trend, bounded, ...}}
    """
    profiles = {}

    # Collect all metric series from env_metrics in traj_summary
    for key, val in env_metrics.items():
        if isinstance(val, dict) and "mean" in val:
            profiles[key] = {
                "mean": float(val["mean"]),
                "std": float(val.get("std", 0)),
                "source": "traj_summary",
            }

    # Collect metric series from eval_history for trend detection
    eval_series: dict[str, list[float]] = {}
    for row in eval_history:
        for k, v in (row.get("env_metrics") or {}).items():
            m = v.get("mean") if isinstance(v, dict) else None
            if isinstance(m, (int, float)):
                eval_series.setdefault(k, []).append(float(m))

    for key, series in eval_series.items():
        if key not in profiles:
            profiles[key] = {"source": "eval_history"}
        profile = profiles[key]
        profile["eval_points"] = len(series)
        profile["mean"] = sum(series) / len(series) if series else 0.0
        if len(series) >= 2:
            avg = profile["mean"]
            var = sum((x - avg) ** 2 for x in series) / len(series)
            profile["std"] = math.sqrt(var)
        if len(series) >= 2:
            profile["trend"] = series[-1] - series[0]
            profile["trend_sign"] = "increasing" if profile["trend"] > 0 else "decreasing"

    # Classify each metric by statistical profile
    for key, p in profiles.items():
        mean_abs = abs(p.get("mean", 0))
        std = p.get("std", 0)

        # Bounded: low absolute mean + low variance → likely action/power metric
        # High-variance: std > |mean| → likely progress/speed metric
        if mean_abs < 1e-9:
            p["category"] = "dead"
        elif std > mean_abs * 2 and mean_abs > 1e-6:
            p["category"] = "high_variance"
        elif mean_abs < 2.0 and std < mean_abs * 0.5:
            p["category"] = "bounded"
        else:
            p["category"] = "general"

        # Detect if strongly correlated with episode length
        p["ep_length_metric"] = any(
            kw in key.lower()
            for kw in ("length", "len", "step", "duration")
        )

    return profiles


def _is_likely_action_metric(name: str, profile: dict) -> bool:
    """Heuristic: does this metric likely measure action/energy/power?"""
    name_lower = name.lower()
    action_keywords = ("action", "power", "energy", "effort", "force", "control", "magnitude")
    if any(kw in name_lower for kw in action_keywords):
        return True
    if profile.get("category") == "bounded":
        return True
    return False


def _is_likely_progress_metric(name: str, profile: dict) -> bool:
    """Heuristic: does this metric likely measure task progress (speed, distance)?"""
    name_lower = name.lower()
    progress_keywords = ("speed", "velocity", "vel", "distance", "dist", "progress",
                         "forward", "x_", "y_", "z_", "displacement")
    if any(kw in name_lower for kw in progress_keywords):
        return True
    if profile.get("category") == "high_variance":
        return True
    return False


def detect_constraint_violations(
    traj_summary: dict, eval_history: list[dict]
) -> list[dict[str, Any]]:
    """Return structured violations inferred from training dynamics only.

    All metric names are auto-discovered. No hardcoded names.
    """
    violations: list[dict[str, Any]] = []
    envm = traj_summary.get("env_metrics", {}) or {}
    comps = traj_summary.get("components", {}) or {}
    lengths = traj_summary.get("lengths") or {}

    profiles = _auto_profile_metrics(envm, eval_history)

    # ── Action efficiency: any action-like metric vs any progress-like metric ──
    action_metrics = {k: p for k, p in profiles.items() if _is_likely_action_metric(k, p)}
    progress_metrics = {k: p for k, p in profiles.items() if _is_likely_progress_metric(k, p)}

    for a_name, a_prof in action_metrics.items():
        a_mean = abs(a_prof.get("mean", 0))
        if a_mean < 1e-6:
            continue
        for p_name, p_prof in progress_metrics.items():
            p_mean = abs(p_prof.get("mean", 0))
            if p_mean < 1e-6:
                continue
            eff = p_mean / a_mean
            if eff < 0.35:
                violations.append({
                    "principle": "action_efficiency",
                    "severity": "high",
                    "evidence": {
                        "action_metric": a_name,
                        "action_mean": round(a_mean, 4),
                        "progress_metric": p_name,
                        "progress_mean": round(p_mean, 4),
                        "progress_per_action": round(eff, 4),
                    },
                    "diagnosis": (
                        f"High {a_name} ({a_mean:.4f}) but low {p_name} gain ({p_mean:.4f}); "
                        "likely energy-inefficient behavior."
                    ),
                })

    # ── Action saturation: bounded metric with very low variance ──
    for name, prof in action_metrics.items():
        a_mean = abs(prof.get("mean", 0))
        a_std = prof.get("std", 0)
        if a_mean > 1e-6 and a_std < 0.1 * a_mean:
            violations.append({
                "principle": "action_efficiency",
                "severity": "medium",
                "evidence": {
                    "metric": name,
                    "mean": round(a_mean, 4),
                    "std": round(a_std, 4),
                },
                "diagnosis": (
                    f"{name} variance is very low relative to its mean; "
                    "policy may be saturating at a fixed action."
                ),
            })

    # ── Dead reward components (constant offset, near-zero variance) ──
    for name, info in comps.items():
        cmean = float(info.get("mean", 0.0))
        cstd = float(info.get("std", 0.0))
        if abs(cmean) > 0.1 and cstd < 1e-6:
            violations.append({
                "principle": "reward_goal_alignment",
                "severity": "high",
                "evidence": {
                    "component": name,
                    "mean": round(cmean, 4),
                    "std": round(cstd, 8),
                },
                "diagnosis": (
                    "Reward component has strong constant offset with near-zero variance; "
                    "can be harvested without informative learning signal."
                ),
            })

    # ── Stuck episode lengths ──
    if eval_history:
        mls = [
            float(r.get("mean_length", 0.0))
            for r in eval_history
            if r.get("mean_length") is not None
        ]
        if len(mls) >= 2 and max(mls) > 0:
            span = max(mls) - min(mls)
            if span / max(max(mls), 1e-6) < 0.1:
                violations.append({
                    "principle": "state_coverage",
                    "severity": "medium",
                    "evidence": {
                        "mean_length_min": round(min(mls), 2),
                        "mean_length_max": round(max(mls), 2),
                    },
                    "diagnosis": (
                        "Episode lengths are concentrated in a narrow range; "
                        "policy may be locked into a repetitive local optimum."
                    ),
                })

        # Temporal consistency: early vs late evaluation windows
        drift = _eval_window_drift(eval_history)
        if drift["max_relative_drift"] > 0.5:
            violations.append({
                "principle": "temporal_consistency",
                "severity": "medium",
                "evidence": {
                    "metric": drift["metric"],
                    "early_mean": round(drift["early"], 4),
                    "late_mean": round(drift["late"], 4),
                    "relative_drift": round(drift["max_relative_drift"], 4),
                },
                "diagnosis": (
                    f"{drift['metric']} shifted strongly between early and late training; "
                    "intra-policy dynamics may be inconsistent."
                ),
            })

    # ── Termination exploitation ──
    lmin = lengths.get("min")
    lmax = lengths.get("max")
    lmean = lengths.get("mean")
    if (
        isinstance(lmin, (int, float))
        and isinstance(lmax, (int, float))
        and isinstance(lmean, (int, float))
        and lmax > 0
    ):
        if lmean < 0.5 * lmax:
            violations.append({
                "principle": "termination_exploitation",
                "severity": "medium",
                "evidence": {
                    "length_mean": round(lmean, 2),
                    "length_max": round(lmax, 2),
                    "ratio": round(lmean / lmax, 3),
                },
                "diagnosis": (
                    "Average episode length is far below observed max; "
                    "agent may be exploiting early termination dynamics."
                ),
            })

    return violations


def derive_action_cross_metrics(
    traj_summary: dict, eval_history: list[dict]
) -> dict[str, Any]:
    """Derive action/behavior cross metrics for Phase-1 diagnostics.

    Auto-discovers metric names — no hardcoded assumptions.
    """
    envm = traj_summary.get("env_metrics", {}) or {}
    lengths = traj_summary.get("lengths", {}) or {}
    profiles = _auto_profile_metrics(envm, eval_history)

    out: dict[str, Any] = {}

    # Report all discovered metrics with their profiles
    for name, prof in profiles.items():
        if prof.get("mean") is not None:
            out[f"{name}_mean"] = round(float(prof["mean"]), 6)
        if prof.get("std") is not None:
            out[f"{name}_std"] = round(float(prof["std"]), 6)
        if prof.get("trend") is not None:
            out[f"{name}_trend"] = round(float(prof["trend"]), 6)

    # Action efficiency: any action metric paired with any progress metric
    action_metrics = {k: p for k, p in profiles.items() if _is_likely_action_metric(k, p)}
    progress_metrics = {k: p for k, p in profiles.items() if _is_likely_progress_metric(k, p)}

    for a_name, a_prof in action_metrics.items():
        a_mean = abs(a_prof.get("mean", 0))
        if a_mean < 1e-8:
            continue
        for p_name, p_prof in progress_metrics.items():
            p_mean = float(p_prof.get("mean", 0))
            out[f"{p_name}_per_{a_name}"] = round(p_mean / a_mean, 6)

    # Episode length span ratio
    if eval_history:
        vals = [
            float(r.get("mean_length", 0.0))
            for r in eval_history
            if r.get("mean_length") is not None
        ]
        if vals:
            out["mean_length_span_ratio"] = round(
                (max(vals) - min(vals)) / max(max(vals), 1e-8), 6
            )

    # Length utilization
    if {"mean", "max"} <= set(lengths.keys()) and lengths.get("max", 0):
        out["length_utilization_ratio"] = round(
            float(lengths.get("mean", 0.0)) / max(float(lengths["max"]), 1e-8), 6
        )

    return out


def derive_episode_consistency_metrics(
    traj_summary: dict, eval_history: list[dict]
) -> dict[str, Any]:
    """Estimate within-episode behavioral consistency using temporal proxies."""
    envm = traj_summary.get("env_metrics", {}) or {}
    profiles = _auto_profile_metrics(envm, eval_history)
    out: dict[str, Any] = {}

    drift = _eval_window_drift(eval_history)
    rel = float(drift.get("max_relative_drift", 0.0) or 0.0)
    out["early_late_relative_drift"] = round(rel, 6)
    out["early_late_consistency_score"] = round(max(0.0, 1.0 - min(rel, 1.0)), 6)
    if drift.get("metric") != "n/a":
        out["drift_dominant_metric"] = drift.get("metric")

    # Action variance for any bounded metrics
    for name, prof in profiles.items():
        if prof.get("category") == "bounded":
            am = float(prof.get("mean", 0))
            am_std = float(prof.get("std", 0))
            if abs(am) > 1e-8:
                out[f"{name}_cv"] = round(abs(am_std) / abs(am), 6)

    return out


def _eval_window_drift(
    eval_history: list[dict],
) -> dict[str, float | str]:
    if len(eval_history) < 4:
        return {"metric": "n/a", "early": 0.0, "late": 0.0, "max_relative_drift": 0.0}
    metric_series: dict[str, list[float]] = {}
    for row in eval_history:
        for k, v in (row.get("env_metrics") or {}).items():
            m = v.get("mean") if isinstance(v, dict) else None
            if isinstance(m, (int, float)):
                metric_series.setdefault(k, []).append(float(m))
    if not metric_series:
        return {"metric": "n/a", "early": 0.0, "late": 0.0, "max_relative_drift": 0.0}
    best = ("n/a", 0.0, 0.0, 0.0)
    for k, arr in metric_series.items():
        if len(arr) < 4:
            continue
        mid = len(arr) // 2
        early = sum(arr[:mid]) / max(mid, 1)
        late = sum(arr[mid:]) / max(len(arr) - mid, 1)
        rel = abs(late - early) / max(abs(early), 1e-6)
        if rel > best[3]:
            best = (k, early, late, rel)
    return {
        "metric": best[0],
        "early": best[1],
        "late": best[2],
        "max_relative_drift": best[3],
    }
