"""Behavior-aware diagnostics for short and long training runs.

The classifier is intentionally deterministic.  LLM agents may propose or
explain behavior specs, but experiment selection must not depend on hidden LLM
judgment.  The report generated here is a weak, task-aware diagnostic signal; it
is not the environment's official reward and is never used as the training
reward.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


FAILURE_HOVERING = "hovering"
FAILURE_EARLY_CRASH = "early_crash"
MODE_APPROACH_UNSTABLE = "approach_but_unstable"
MODE_LANDING_PROGRESS = "landing_progress"
MODE_IMPROVING = "improving"
MODE_DECLINING = "declining"
MODE_MODERATE = "moderate"
MODE_UNCLEAR = "unclear"


def build_behavior_report_from_dir(
    run_dir: Path,
    env_name: str = "",
    max_episode_steps: int = 1000,
) -> dict[str, Any]:
    """Build a deterministic behavior report from a candidate/run directory.

    The function first reads SB3 evaluation history, then optionally augments it
    with trajectory/step logs if available.  It works for both candidate short
    training directories and ``round/full_training`` directories.
    """
    run_dir = Path(run_dir)
    eval_history = _read_eval_history(run_dir)
    episode_records = _read_episode_logs(run_dir)
    step_records = _read_step_logs(run_dir, limit=25000)

    env_l = env_name.lower()
    if "lunarlander" in env_l:
        report = _classify_lunarlander(eval_history, episode_records, step_records, max_episode_steps)
    elif "cartpole" in env_l:
        report = _classify_cartpole(eval_history, episode_records, step_records, max_episode_steps)
    else:
        report = _classify_generic(eval_history, episode_records, step_records, max_episode_steps)

    report["source_dir"] = str(run_dir)
    report["env_name"] = env_name
    return report


def summarize_behavior_report(report: dict[str, Any]) -> str:
    """Format a behavior report for analyzer/mutator prompts."""
    if not report:
        return "(no behavior report available)"
    metrics = report.get("metrics", {})
    evidence = report.get("evidence", [])
    lines = [
        f"Behavior mode: {report.get('behavior_mode', 'unknown')}",
        f"Behavior quality: {report.get('behavior_quality', 0):.2f}",
        f"Reason: {report.get('reason', '')}",
    ]
    if metrics:
        compact = ", ".join(f"{k}={_fmt(v)}" for k, v in metrics.items() if isinstance(v, (int, float, str, bool)))
        if compact:
            lines.append(f"Metrics: {compact[:800]}")
    if evidence:
        lines.append("Evidence:")
        for item in evidence[:8]:
            lines.append(f"- {item}")
    return "\n".join(lines)


def is_bad_failure_mode(mode: str) -> bool:
    return mode in {FAILURE_HOVERING, FAILURE_EARLY_CRASH}


def _read_eval_history(run_dir: Path) -> list[dict[str, Any]]:
    for p in [run_dir / "evaluations" / "history.csv", run_dir / "full_training" / "evaluations" / "history.csv"]:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    return list(csv.DictReader(f))
            except OSError:
                return []
    return []


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def _read_episode_logs(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [run_dir / "trajectory_logs", run_dir / "full_training" / "trajectory_logs"]
    records: list[dict[str, Any]] = []
    for d in candidates:
        if d.exists():
            for p in sorted(d.glob("*.jsonl")):
                records.extend(_read_jsonl(p))
    return records


def _read_step_logs(run_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    for p in [run_dir / "step_logs" / "steps.jsonl", run_dir / "full_training" / "step_logs" / "steps.jsonl"]:
        if p.exists():
            return _read_jsonl(p, limit=limit)
    return []


def _lengths_from_eval(eval_history: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []
    for row in eval_history:
        for key in ("mean_length", "ep_len_mean", "length"):
            if key in row:
                try:
                    vals.append(float(row[key]))
                    break
                except (TypeError, ValueError):
                    pass
    return vals


def _returns_from_eval(eval_history: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []
    for row in eval_history:
        for key in ("mean_reward", "mean_return", "reward", "ep_rew_mean"):
            if key in row:
                try:
                    vals.append(float(row[key]))
                    break
                except (TypeError, ValueError):
                    pass
    return vals


def _episode_lengths_from_records(records: list[dict[str, Any]], max_episode_steps: int) -> list[float]:
    vals: list[float] = []
    for rec in records:
        if "length" in rec:
            try:
                vals.append(float(rec["length"]))
            except (TypeError, ValueError):
                pass
    return vals


def _obs_from_record(rec: dict[str, Any], key_candidates: tuple[str, ...]) -> list[float] | None:
    for key in key_candidates:
        val = rec.get(key)
        if isinstance(val, list) and len(val) >= 4:
            try:
                return [float(x) for x in val]
            except (TypeError, ValueError):
                return None
    return None


def _last_step_by_episode(step_records: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    for rec in step_records:
        ep = rec.get("episode", len(out))
        out[ep] = rec
    return out


def _classify_lunarlander(eval_history, episode_records, step_records, max_episode_steps: int) -> dict[str, Any]:
    lengths = _lengths_from_eval(eval_history) or _episode_lengths_from_records(episode_records, max_episode_steps)
    returns = _returns_from_eval(eval_history)
    first_len = lengths[0] if lengths else 0.0
    final_len = lengths[-1] if lengths else 0.0
    max_len = max(lengths) if lengths else 0.0
    len_ratio = final_len / max(max_episode_steps, 1)

    last_steps = list(_last_step_by_episode(step_records).values())
    final_obs = [_obs_from_record(r, ("next_obs", "obs", "state", "next_state")) for r in last_steps]
    final_obs = [o for o in final_obs if o is not None and len(o) >= 8]

    final_y = mean([o[1] for o in final_obs]) if final_obs else None
    final_dist = mean([math.sqrt(o[0] ** 2 + o[1] ** 2) for o in final_obs]) if final_obs else None
    final_speed = mean([math.sqrt(o[2] ** 2 + o[3] ** 2) for o in final_obs]) if final_obs else None
    final_angle = mean([abs(o[4]) for o in final_obs]) if final_obs else None
    contact_ratio = mean([(float(o[6]) + float(o[7])) / 2.0 for o in final_obs]) if final_obs else None

    dist_by_episode: dict[Any, list[float]] = {}
    for rec in step_records:
        obs = _obs_from_record(rec, ("obs", "state"))
        if obs is not None and len(obs) >= 2:
            ep = rec.get("episode", 0)
            dist_by_episode.setdefault(ep, []).append(math.sqrt(obs[0] ** 2 + obs[1] ** 2))
    dist_improvements = []
    min_distances = []
    for vals in dist_by_episode.values():
        if len(vals) >= 2:
            dist_improvements.append(vals[0] - vals[-1])
            min_distances.append(min(vals))
    mean_dist_improve = mean(dist_improvements) if dist_improvements else None
    mean_min_dist = mean(min_distances) if min_distances else None

    evidence: list[str] = []
    mode = MODE_UNCLEAR
    quality = 0.35
    reason = "Insufficient LunarLander trajectory evidence; using conservative fallback."

    if final_obs:
        if len_ratio > 0.85 and (final_y is not None and final_y > 0.35) and (mean_dist_improve is None or mean_dist_improve < 0.25):
            mode, quality = FAILURE_HOVERING, 0.10
            reason = "Episode lasts near max steps while lander remains high or shows little approach progress."
        elif len_ratio < 0.20 and (final_speed is None or final_speed > 0.4 or (final_angle is not None and final_angle > 0.4)):
            mode, quality = FAILURE_EARLY_CRASH, 0.10
            reason = "Episode terminates very early with unstable final speed/angle evidence."
        elif mean_min_dist is not None and mean_min_dist < 0.35 and (final_speed is None or final_speed > 0.25):
            mode, quality = MODE_APPROACH_UNSTABLE, 0.55
            reason = "Policy approaches the pad region but remains unstable near the end."
        elif final_dist is not None and final_dist < 0.25 and (final_speed is None or final_speed < 0.25) and (final_angle is None or final_angle < 0.25):
            mode, quality = MODE_LANDING_PROGRESS, 0.85
            reason = "Policy reaches near the pad with low speed and relatively stable angle."
        elif mean_dist_improve is not None and mean_dist_improve > 0.2:
            mode, quality = MODE_APPROACH_UNSTABLE, 0.50
            reason = "Distance-to-pad improves, but there is not enough evidence of stable touchdown."
    elif lengths:
        if len_ratio > 0.85:
            mode, quality = FAILURE_HOVERING, 0.15
            reason = "Length-only fallback: near-max episode length may indicate hovering/stalling."
        elif len_ratio < 0.20:
            mode, quality = FAILURE_EARLY_CRASH, 0.10
            reason = "Length-only fallback: very short episode suggests early crash."
        elif first_len and final_len > first_len * 1.2:
            mode, quality = MODE_IMPROVING, 0.55
            reason = "Episode length is improving, but no state-level evidence is available."
        else:
            mode, quality = MODE_MODERATE, 0.45
            reason = "Moderate episode duration under length-only fallback."

    if lengths:
        evidence.append(f"episode_length {first_len:.1f} -> {final_len:.1f}, max_observed={max_len:.1f}, ratio={len_ratio:.2f}")
    if final_y is not None:
        evidence.append(f"mean final y={final_y:.3f}")
    if final_dist is not None:
        evidence.append(f"mean final distance={final_dist:.3f}")
    if mean_dist_improve is not None:
        evidence.append(f"mean distance improvement={mean_dist_improve:.3f}")
    if mean_min_dist is not None:
        evidence.append(f"mean minimum distance={mean_min_dist:.3f}")
    if final_speed is not None:
        evidence.append(f"mean final speed={final_speed:.3f}")
    if final_angle is not None:
        evidence.append(f"mean final |angle|={final_angle:.3f}")
    if contact_ratio is not None:
        evidence.append(f"mean final leg contact={contact_ratio:.3f}")

    return {
        "behavior_mode": mode,
        "behavior_quality": quality,
        "reason": reason,
        "evidence": evidence,
        "metrics": {
            "first_length": first_len,
            "final_length": final_len,
            "max_length": max_len,
            "length_ratio": len_ratio,
            "final_y": final_y,
            "final_distance": final_dist,
            "final_speed": final_speed,
            "final_abs_angle": final_angle,
            "contact_ratio": contact_ratio,
            "mean_distance_improvement": mean_dist_improve,
            "mean_min_distance": mean_min_dist,
            "final_return": returns[-1] if returns else None,
        },
    }


def _classify_cartpole(eval_history, episode_records, step_records, max_episode_steps: int) -> dict[str, Any]:
    lengths = _lengths_from_eval(eval_history) or _episode_lengths_from_records(episode_records, max_episode_steps)
    first_len = lengths[0] if lengths else 0.0
    final_len = lengths[-1] if lengths else 0.0
    ratio = final_len / max(max_episode_steps, 1)
    if ratio > 0.85:
        mode, quality, reason = MODE_LANDING_PROGRESS, 0.90, "Near-max CartPole survival is the desired behavior."
    elif ratio < 0.20:
        mode, quality, reason = FAILURE_EARLY_CRASH, 0.10, "Very short survival indicates unstable policy."
    elif first_len and final_len > first_len * 1.2:
        mode, quality, reason = MODE_IMPROVING, 0.65, "Episode length improves during training."
    else:
        mode, quality, reason = MODE_MODERATE, 0.45, "Moderate survival without clear convergence."
    return {
        "behavior_mode": mode,
        "behavior_quality": quality,
        "reason": reason,
        "evidence": [f"episode_length {first_len:.1f} -> {final_len:.1f}, ratio={ratio:.2f}"],
        "metrics": {"first_length": first_len, "final_length": final_len, "length_ratio": ratio},
    }


def _classify_generic(eval_history, episode_records, step_records, max_episode_steps: int) -> dict[str, Any]:
    lengths = _lengths_from_eval(eval_history) or _episode_lengths_from_records(episode_records, max_episode_steps)
    returns = _returns_from_eval(eval_history)
    first_len = lengths[0] if lengths else 0.0
    final_len = lengths[-1] if lengths else 0.0
    ratio = final_len / max(max_episode_steps, 1)
    if returns and len(returns) >= 2 and returns[-1] > returns[0]:
        mode, quality, reason = MODE_IMPROVING, 0.60, "Evaluation return improves during training."
    elif ratio < 0.15:
        mode, quality, reason = FAILURE_EARLY_CRASH, 0.15, "Very short episodes under generic fallback."
    else:
        mode, quality, reason = MODE_MODERATE, 0.45, "Generic behavior fallback; no task-specific classifier available."
    return {
        "behavior_mode": mode,
        "behavior_quality": quality,
        "reason": reason,
        "evidence": [f"episode_length {first_len:.1f} -> {final_len:.1f}, ratio={ratio:.2f}"],
        "metrics": {
            "first_length": first_len,
            "final_length": final_len,
            "length_ratio": ratio,
            "first_return": returns[0] if returns else None,
            "final_return": returns[-1] if returns else None,
        },
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
