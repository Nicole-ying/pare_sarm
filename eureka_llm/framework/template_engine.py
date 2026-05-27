"""
template_engine.py — Fills template {placeholders} with actual data.

Usage:
    python template_engine.py --env-dir envs/BipedalWalker-v3/ --template templates/round0_prompt.txt --exploration exploration.json --output round0.txt
"""

import argparse
import csv
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from constraint_discovery import (
    detect_constraint_violations,
    derive_action_cross_metrics,
    derive_episode_consistency_metrics,
)
from shared_rules import render_rules, ROUND0_RULES


# ─────────────────────────────────────────────────────────────────────────────
# Round 0 prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_step_source(env_dir: Path) -> str:
    """Read step.py from environment directory."""
    step_path = env_dir / "step.py"
    if not step_path.exists():
        raise FileNotFoundError(f"step.py not found: {step_path}")
    return step_path.read_text(encoding="utf-8").rstrip("\n")


def extract_compute_reward_signature(step_source: str) -> str:
    """Extract argument list from self.compute_reward(...) call."""
    m = re.search(r'self\.compute_reward\(([^)]+)\)', step_source)
    if m:
        return m.group(1).strip()
    return "action"


def build_action_desc(exploration: dict) -> str:
    """Build human-readable action space description from exploration data."""
    act = exploration.get("spaces", {}).get("action", {})
    if "n" in act and act["n"] is not None:
        return f"Discrete({act['n']})"
    shape = act.get("shape", [])
    if shape:
        return f"Box({shape})"
    return "unknown"


def build_obs_rows(exploration: dict) -> str:
    """Build obs stats table rows from exploration data."""
    rows = []
    for s in exploration.get("obs_dim_stats", []):
        inferred = "continuous"
        lo = s.get("space_low", 0)
        hi = s.get("space_high", 0)
        sample_min = s.get("sample_min", 0)
        sample_max = s.get("sample_max", 0)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if abs(hi - 3.14159) < 0.05 and abs(lo + 3.14159) < 0.05:
                inferred = "angle"
            elif sample_min >= -0.01 and sample_max <= 1.01:
                inferred = "binary/normalized"
        rows.append(
            f"| {s['dim']} | [{s['space_low']}, {s['space_high']}] "
            f"| {s['mean']} ± {s['std']} "
            f"| [{s['sample_min']}, {s['sample_max']}] "
            f"| {inferred} |"
        )
    return "\n".join(rows) if rows else "| — | — | — | — | — |"


def build_term_summary(exploration: dict) -> str:
    """Build termination summary lines."""
    lines = []
    for reason, info in exploration.get("termination_summary", {}).items():
        lines.append(
            f"- `{reason}`: {info['count']} episodes ({info['fraction']*100:.0f}%)"
        )
    return "\n".join(lines) if lines else "- (no termination data)"


def build_info_summary(exploration: dict) -> str:
    """Build info keys summary, excluding reward-like signals."""
    lines = []
    for k, v in exploration.get("info_keys", {}).items():
        if any(x in k.lower() for x in ("reward", "fitness", "score")):
            continue
        lines.append(f"- `{k}`: {v.get('type', '?')}, range=[{v.get('min', '?')}, {v.get('max', '?')}]")
    return "\n".join(lines) if lines else "- (none observed)"


def derive_reward_constraints(exploration: dict) -> str:
    """Auto-derive reward design constraints from environment exploration data.

    Analyzes observation statistics, termination patterns, and zero-action baseline
    to detect environment type (gravity, balance, velocity-driven, etc.) and generate
    specific reward design rules — no hardcoded environment names needed.
    """
    constraints = []
    za = exploration.get("zero_action", {})
    obs_stats = exploration.get("obs_dim_stats", [])
    term = exploration.get("termination_summary", {})
    act = exploration.get("spaces", {}).get("action", {})

    # ── Gravity / passive dynamics from zero-action baseline ──
    gravity = za.get("gravity_hypothesis", "unknown")
    death_rate = za.get("death_rate", 0)

    if gravity == "strong":
        constraints.append(
            "- **Gravity detected**: Zero-action episodes terminate quickly "
            "(>50% die within 20% of episode limit). The agent MUST act to stay alive. "
            "Do NOT penalize the main control actions — "
            "the agent needs them to counteract passive dynamics. "
            "Reward staying alive but not at the cost of task progress."
        )
    elif gravity == "weak":
        constraints.append(
            "- **Weak passive dynamics**: Some zero-action episodes end early. "
            "Mild gravity or friction may be present. "
            "Survival rewards should not dominate task-completion signals."
        )
    elif gravity == "none":
        constraints.append(
            "- **No gravity detected**: Zero-action episodes survive to timeout. "
            "The agent can remain idle indefinitely. "
            "Focus reward on active task progress, not survival."
        )

    # ── Angle/balance dimensions ──
    angle_dims = []
    for s in obs_stats:
        lo = s.get("space_low", 0)
        hi = s.get("space_high", 0)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if abs(hi - 3.14159) < 0.05 and abs(lo + 3.14159) < 0.05:
                angle_dims.append(s["dim"])

    if angle_dims:
        constraints.append(
            f"- **Balance/angle dims** (obs dims {angle_dims}, range ≈ \\u00b1\\u03c0): "
            "These are angular coordinates. Use smallest signed angular difference "
            "(not raw subtraction) for distance calculations. "
            "If the task requires maintaining a specific posture, include "
            "an angular stability reward component."
        )

    # ── Termination patterns ──
    term_counts = {k: v.get("count", 0) for k, v in term.items()}
    total_ep = sum(term_counts.values()) or 1
    term_rate = term_counts.get("terminated", 0) / total_ep

    if term_rate > 0.5:
        constraints.append(
            f"- **High failure rate under random actions** ({term_rate:.0%}): "
            "The task has strong failure conditions. "
            "Termination penalty should be < 100x expected per-step reward "
            "to avoid pathological risk-aversion."
        )

    # ── High-dimensional action space ──
    act_shape = act.get("shape", [])
    if act_shape and len(act_shape) == 1 and act_shape[0] is not None:
        n_acts = act_shape[0]
        if n_acts >= 4:
            constraints.append(
                f"- **High-dim action space** ({n_acts} continuous dims): "
                "Different dimensions likely control different effectors. "
                "A uniform energy penalty on ALL dimensions may penalize necessary movements. "
                "Prefer penalizing action smoothness (delta from previous step) "
                "rather than raw action magnitude."
            )

    # ── Large-range obs dims (unbounded velocity / position) ──
    large_range_dims = []
    for s in obs_stats:
        lo = s.get("space_low", 0)
        hi = s.get("space_high", 0)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if abs(hi - lo) > 10 and abs(hi - 3.14159) >= 0.05:
                large_range_dims.append(s["dim"])

    if large_range_dims:
        constraints.append(
            f"- **Large-range obs dims** (dims {large_range_dims}): "
            "These may be unbounded velocities or positions. "
            "Apply np.tanh or np.clip to any reward component derived from them."
        )

    return "\n".join(constraints) if constraints else "- (no specific constraints derived)"


def build_exploration_summary(exploration: dict) -> str:
    """Build a compact text summary of exploration data for round0 prompt."""
    lines = []

    # Observation dimension stats
    obs_stats = exploration.get("obs_dim_stats", [])
    if obs_stats:
        lines.append("### Observation Dimension Statistics")
        lines.append("| Dim | Mean ± Std | Sample Range |")
        lines.append("|-----|------------|--------------|")
        for s in obs_stats:
            lines.append(f"| {s['dim']} | {s['mean']:.3f} ± {s['std']:.3f} "
                         f"| [{s['sample_min']:.3f}, {s['sample_max']:.3f}] |")

    # Termination summary
    term = exploration.get("termination_summary", {})
    if term:
        lines.append("\n### Termination Analysis")
        for reason, info in term.items():
            lines.append(f"- `{reason}`: {info['count']} episodes ({info['fraction']*100:.0f}%)")

    # Episode length stats
    ep = exploration.get("episode_length_stats", {})
    if ep:
        lines.append(f"\n### Episode Lengths")
        lines.append(f"Mean: {ep.get('mean', '?')} ± {ep.get('std', '?')}, "
                     f"Range: [{ep.get('min', '?')}, {ep.get('max', '?')}]")

    # Zero-action baseline
    za = exploration.get("zero_action", {})
    if za:
        death_rate = za.get("death_rate", 0)
        gravity = za.get("gravity_hypothesis", "unknown")
        lines.append(f"\n### Zero-Action Baseline")
        lines.append(f"Death rate: {death_rate:.0%}, Gravity hypothesis: {gravity}")

    return "\n".join(lines) if lines else "*(no exploration data)*"


def build_round0_prompt(env_dir: Path, template_path: Path, exploration_path: Path,
                        task_manifest: str = None) -> str:
    """Build a complete round0 prompt by filling placeholders.

    Fills {task_manifest}, {exploration_summary}, and {compute_reward_signature}.
    """
    template = template_path.read_text(encoding="utf-8")
    exploration = json.loads(exploration_path.read_text(encoding="utf-8"))

    exploration_summary = build_exploration_summary(exploration)

    # Extract compute_reward signature from step.py
    compute_reward_signature = "state, action"  # default fallback
    step_path = env_dir / "step.py"
    if step_path.exists():
        step_source = step_path.read_text(encoding="utf-8")
        sig = extract_compute_reward_signature(step_source)
        if sig:
            compute_reward_signature = sig

    # Override any "Compute Reward Signature" in the task manifest with the
    # one extracted from actual source code — prevents EnvPerception hallucinations
    # from producing a signature that doesnt match env.step()'s actual call.
    corrected_manifest = task_manifest or ""
    if compute_reward_signature and corrected_manifest:
        corrected_manifest = re.sub(
            r'(?<=## Compute Reward Signature\n).*',
            f"`compute_reward({compute_reward_signature})`",
            corrected_manifest,
        )

    placeholders = {
        "task_manifest": corrected_manifest or "(not available)",
        "exploration_summary": exploration_summary,
        "compute_reward_signature": compute_reward_signature,
        "shared_rules": render_rules(ROUND0_RULES),
    }

    result = template
    for key, value in placeholders.items():
        result = result.replace("{" + key + "}", str(value))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_history(run_dir: Path) -> list[dict]:
    """Load evaluation history CSV. Returns list of {timesteps, mean_length, env_metrics, ...}.

    Handles both the legacy format (with completion_rate/fall_rate/truncation_rate columns)
    and the current format (timesteps + mean_length + env_metrics only).
    """
    csv_path = run_dir / "evaluations" / "history.csv"
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {}
            entry["timesteps"] = int(row["timesteps"])
            entry["mean_length"] = float(row.get("mean_length", 0)) if row.get("mean_length") else 0.0
            # Legacy compat: silently drop completion/fall/truncation fields
            raw = row.get("env_metrics", "{}")
            try:
                entry["env_metrics"] = json.loads(raw)
            except json.JSONDecodeError:
                entry["env_metrics"] = {}
            rows.append(entry)
    return rows


def load_trajectory_summary(run_dir: Path) -> dict:
    """Load all trajectory JSONL files and aggregate component + env_metrics stats."""
    traj_dir = run_dir / "trajectory_logs"
    if not traj_dir.exists():
        return {"n_episodes": 0, "components": {}}

    all_means = defaultdict(list)
    all_env_metrics = defaultdict(list)
    total_episodes = 0
    lengths = []

    for f in sorted(traj_dir.glob("*.trajectory.jsonl")):
        for line in f.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            total_episodes += 1
            lengths.append(record.get("length", 0))
            for comp, mean_val in record.get("component_means", {}).items():
                all_means[comp].append(mean_val)
            for metric, mean_val in record.get("env_metrics_means", {}).items():
                all_env_metrics[metric].append(mean_val)

    if total_episodes == 0:
        return {"n_episodes": 0, "components": {}}

    def _aggregate(data: dict) -> dict:
        result = {}
        for key, vals in data.items():
            arr = vals
            n = len(arr)
            if n == 0:
                continue
            mean = sum(arr) / n
            std = math.sqrt(sum((x - mean)**2 for x in arr) / n) if n > 1 else 0.0
            result[key] = {
                "mean": round(float(mean), 6),
                "std": round(float(std), 6),
            }
        return result

    result = {
        "n_episodes": total_episodes,
        "components": _aggregate(all_means),
        "lengths": {
            "mean": round(float(sum(lengths) / len(lengths)), 1) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }
    if all_env_metrics:
        result["env_metrics"] = _aggregate(all_env_metrics)
    return result


def load_training_data(run_dir: Path) -> dict:
    """Load all training artifacts from a run directory."""
    config_path = run_dir / "config.yaml"
    config = {}
    if config_path.exists():
        import yaml
        config = yaml.safe_load(config_path.read_text("utf-8"))

    run_info = {}
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        run_info = json.loads(info_path.read_text("utf-8"))

    reward_src = ""
    src_path = run_dir / "reward_fn_source.py"
    if src_path.exists():
        reward_src = src_path.read_text("utf-8")

    eval_history = load_eval_history(run_dir)
    traj_summary = load_trajectory_summary(run_dir)

    return {
        "config": config,
        "run_info": run_info,
        "reward_fn_source": reward_src,
        "eval_history": eval_history,
        "traj_summary": traj_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Iteration prompt formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics_table(eval_history: list[dict], max_metrics: int = 6, stats: dict | None = None) -> str:
    """Format eval history as markdown table rows (timesteps + mean_length only)."""
    if not eval_history:
        return "| — | — |"
    rows = []
    headers = ["timesteps", "mean_length"]
    seen = set()
    # Collect any env_metric keys that appear consistently
    for row in eval_history:
        for k in row.get("env_metrics", {}):
            seen.add(k)
    if seen:
        # Keep only most informative metrics by variance across eval points (up to 6)
        metric_scores = []
        for k in seen:
            vals = []
            for row in eval_history:
                m = row.get("env_metrics", {}).get(k, {})
                v = m.get("mean")
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
            else:
                var = 0.0
            metric_scores.append((k, var))
        selected = [k for k, _ in sorted(metric_scores, key=lambda x: x[1], reverse=True)[:max_metrics]]
        if stats is not None:
            stats["metrics_table_total_candidates"] = len(metric_scores)
            stats["metrics_table_selected"] = selected
        headers.extend(sorted(selected))

    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join("---" for _ in headers) + "|")
    for row in eval_history:
        vals = [str(row['timesteps']), f"{row.get('mean_length', 0):.1f}"]
        for k in headers[2:]:
            m = row.get("env_metrics", {}).get(k, {})
            vals.append(f"{m.get('mean', '—')}")
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def format_env_metrics_section(eval_history: list[dict], max_metrics: int = 6, stats: dict | None = None) -> str:
    """Format env-specific metrics from evaluation history as markdown."""
    if not eval_history:
        return "*(none collected)*"

    # Collect all metric names across all eval steps
    all_metrics = set()
    for row in eval_history:
        all_metrics.update(row.get("env_metrics", {}).keys())

    if not all_metrics:
        return "*(none collected)*"

    sections = []
    # limit section length by selecting at most 6 metrics with strongest trend variation
    scores = []
    for metric in all_metrics:
        vals = []
        for row in eval_history:
            m = row.get("env_metrics", {}).get(metric, {})
            v = m.get("mean")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if len(vals) >= 2:
            drift = abs(vals[-1] - vals[0]) / max(abs(vals[0]), 1e-6)
        else:
            drift = 0.0
        scores.append((metric, drift))

    selected = [metric for metric, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:max_metrics]]
    if stats is not None:
        stats["env_metrics_total_candidates"] = len(scores)
        stats["env_metrics_selected"] = selected
    for metric in selected:
        rows = []
        rows.append(f"| timesteps | {metric}_mean | {metric}_std |")
        rows.append("|-----------|-------------|-------------|")
        for row in eval_history:
            m = row.get("env_metrics", {}).get(metric, {})
            mean = m.get("mean", "—")
            std = m.get("std", "—")
            rows.append(f"| {row['timesteps']} | {mean} | {std} |")
        sections.append("\n".join(rows))
    return "\n\n".join(sections)


def format_component_table(traj_summary: dict) -> str:
    """Format component attribution as markdown table."""
    components = traj_summary.get("components", {})
    if not components:
        return "| — | — | — | — |"

    rows = []
    for comp in sorted(components.keys()):
        info = components[comp]
        rows.append(f"| {comp} | {info['mean']} | {info['std']} | step-level |")
    return "\n".join(rows)


def format_traj_env_metrics_table(traj_summary: dict) -> str:
    """Format per-step env metrics from trajectory data as markdown table."""
    env_metrics = traj_summary.get("env_metrics", {})
    if not env_metrics:
        return "*(not available — training env_metrics collection requires MetricsTrackingWrapper)*"
    rows = []
    for metric in sorted(env_metrics.keys()):
        info = env_metrics[metric]
        rows.append(f"| {metric} | {info['mean']} | {info['std']} | per-step |")
    return "\n".join(rows)


def compute_component_dynamics(traj_summary: dict) -> dict:
    """Compute training dynamics metrics from trajectory component data.

    Analyzes component balance (relative contribution), activity (which
    components are alive), and stability (variance across episodes).

    Returns formatted markdown section for the perception prompt.
    """
    components = traj_summary.get("components", {})
    if not components:
        return "*No trajectory component data available.*\n"

    # Component balance: each component's fraction of total absolute magnitude
    total_abs = sum(abs(c["mean"]) for c in components.values())
    if total_abs < 1e-9:
        return "*All component means are near zero — reward function may be dead.*\n"

    lines = []
    lines.append("| Component | Mean | Std | % of Total | Stability | Status |")
    lines.append("|-----------|------|-----|------------|-----------|--------|")

    active_count = 0
    dominant_count = 0
    noisy_count = 0

    for name in sorted(components.keys()):
        info = components[name]
        mean = info["mean"]
        std = info["std"]
        pct = abs(mean) / total_abs * 100

        # Status classification
        status = "active"
        if abs(mean) < 0.01:
            status = "inactive"
        elif pct > 80:
            status = "DOMINANT"
            dominant_count += 1
        elif pct > 50:
            status = "major"

        if abs(mean) > 0.01:
            active_count += 1

        # Stability: coefficient of variation
        stability = "stable"
        if abs(mean) > 0.01:
            cv = std / abs(mean)
            if cv > 1.5:
                stability = "noisy"
                noisy_count += 1
            elif cv > 0.5:
                stability = "variable"

        lines.append(
            f"| {name} | {mean:.4f} | {std:.4f} | {pct:.1f}% | {stability} | {status} |"
        )

    # Summary
    n_total = len(components)
    lines.append("")
    lines.append(f"**Summary:** {active_count}/{n_total} components active, "
                 f"{dominant_count} dominant (>80%), {noisy_count} noisy (CV>1.5).")

    if dominant_count > 0:
        lines.append("**Warning:** A single component dominates. "
                     "The agent may be optimizing for one objective while ignoring others.")
    if active_count < 2 and n_total >= 2:
        lines.append("**Warning:** Most components are inactive (mean ≈ 0). "
                     "The reward function's multi-objective design is not being utilized.")

    return "\n".join(lines)


def load_entropy_history(run_dir: Path) -> list[dict]:
    """Load policy entropy history from training."""
    entropy_path = run_dir / "entropy_history.jsonl"
    if not entropy_path.exists():
        return []
    records = []
    for line in entropy_path.read_text("utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def format_entropy_section(run_dir: Path) -> str:
    """Format policy entropy trend as markdown."""
    entropy = load_entropy_history(run_dir)
    if not entropy:
        return "*(not collected — entropy tracking added in training dynamics update)*"

    initial = entropy[0]["entropy"]
    final = entropy[-1]["entropy"]
    trend = "increasing" if final > initial * 1.1 else (
        "decreasing" if final < initial * 0.9 else "stable"
    )
    assessment = (
        "Policy is converging to deterministic behavior (may be premature)."
        if trend == "decreasing" and final < 0.5 else
        "Policy maintains exploration."
        if trend == "stable" and final > 0.5 else
        "Entropy is increasing — policy may be unstable."
        if trend == "increasing" else
        "Policy entropy is within expected range."
    )

    lines = [
        f"| Step | Entropy |",
        f"|------|---------|",
    ]
    for r in entropy:
        lines.append(f"| {r['timestep']} | {r['entropy']:.4f} |")

    lines.append("")
    lines.append(f"**Trend:** {trend} ({initial:.4f} → {final:.4f})")
    lines.append(f"**Assessment:** {assessment}")
    return "\n".join(lines)


def format_dynamics_section(traj_summary: dict, run_dir: Path = None) -> str:
    """Format the complete training dynamics section for the perception prompt."""
    parts = ["### Component Balance & Stability"]
    parts.append(compute_component_dynamics(traj_summary))

    if run_dir is not None:
        parts.append("")
        parts.append("### Policy Entropy Trend")
        parts.append(format_entropy_section(run_dir))

    return "\n".join(parts)


# DEPRECATED: TDRQ found to be misleading — reward hacking can inflate score
# while task success is zero. Removed from perception prompts as of the
# cross-round-learning fix. Function kept to avoid import breakage.
def compute_tdrq_index(traj_summary: dict, run_dir: Path = None) -> dict:
    """Compute a Training-Dynamics Reward Quality (TDRQ) index in [0, 100].

    TDRQ combines three internal-signal families:
    1) component balance (avoid single-component domination)
    2) component activity (avoid dead/inactive reward decomposition)
    3) exploration health from policy entropy trend (if available)
    """
    components = traj_summary.get("components", {})
    if not components:
        return {"score": 0.0, "balance": 0.0, "activity": 0.0, "exploration": 0.0}

    means = [abs(v.get("mean", 0.0)) for v in components.values()]
    total = sum(means)
    if total < 1e-12:
        return {"score": 0.0, "balance": 0.0, "activity": 0.0, "exploration": 0.0}

    shares = [m / total for m in means]
    max_share = max(shares)
    # 1.0 means well-balanced, 0.0 means one component fully dominates
    balance = max(0.0, 1.0 - max_share)

    # active if absolute mean is large enough to be meaningful
    active = sum(1 for m in means if m > 0.01)
    activity = active / max(1, len(means))

    exploration = 0.5  # neutral default when entropy unavailable
    if run_dir is not None:
        entropy_hist = load_entropy_history(run_dir)
        if entropy_hist:
            initial = float(entropy_hist[0].get("entropy", 0.0))
            final = float(entropy_hist[-1].get("entropy", 0.0))
            if initial > 1e-8:
                ratio = final / initial
                # Absolute floor: if final entropy is near-zero, exploration collapsed
                # (fixes false "healthy" when both initial and final are tiny,
                #  e.g. init=0.01, final=0.008 → ratio=0.8 but no real exploration)
                if final < 0.1:
                    exploration = 0.1
                elif 0.35 <= ratio <= 1.4:
                    exploration = 1.0
                elif 0.2 <= ratio <= 1.8:
                    exploration = 0.6
                else:
                    exploration = 0.2

    score = 100.0 * (0.45 * balance + 0.35 * activity + 0.20 * exploration)
    return {
        "score": round(score, 2),
        "balance": round(balance * 100.0, 2),
        "activity": round(activity * 100.0, 2),
        "exploration": round(exploration * 100.0, 2),
    }


def format_tdrq_section(traj_summary: dict, run_dir: Path = None) -> str:
    """Format TDRQ section for perception prompt."""
    t = compute_tdrq_index(traj_summary, run_dir)
    lines = [
        "| TDRQ | Score |",
        "|------|-------|",
        f"| overall | {t['score']:.2f} / 100 |",
        f"| component_balance | {t['balance']:.2f} |",
        f"| component_activity | {t['activity']:.2f} |",
        f"| exploration_health | {t['exploration']:.2f} |",
        "",
        "Interpretation: <40 = unhealthy reward dynamics, 40-70 = mixed, >70 = healthy.",
    ]
    return "\n".join(lines)


def format_constraint_discovery_section(traj_summary: dict, eval_history: list[dict]) -> str:
    """Format algorithmic constraint-discovery output."""
    violations = detect_constraint_violations(traj_summary, eval_history)
    if not violations:
        return "*(No strong generic constraint violation detected from current dynamics signals.)*"
    lines = ["| Principle | Severity | Evidence | Diagnosis |",
             "|-----------|----------|----------|-----------|"]
    for v in violations:
        ev = ", ".join(f"{k}={val}" for k, val in v.get("evidence", {}).items())
        lines.append(f"| {v['principle']} | {v['severity']} | {ev} | {v['diagnosis']} |")
    return "\n".join(lines)


def format_action_cross_metrics_section(traj_summary: dict, eval_history: list[dict]) -> str:
    """Format Phase-1 action/behavior cross metrics."""
    metrics = derive_action_cross_metrics(traj_summary, eval_history)
    if not metrics:
        return "*(insufficient metrics to derive action cross-analysis)*"
    lines = ["| Metric | Value |", "|--------|-------|"]
    for k in sorted(metrics.keys()):
        lines.append(f"| {k} | {metrics[k]} |")
    return "\n".join(lines)


def format_episode_consistency_section(traj_summary: dict, eval_history: list[dict]) -> str:
    """Format Phase-1 temporal/episode consistency diagnostics."""
    metrics = derive_episode_consistency_metrics(traj_summary, eval_history)
    if not metrics:
        return "*(insufficient metrics to derive episode consistency diagnostics)*"
    lines = ["| Metric | Value |", "|--------|-------|"]
    for k in sorted(metrics.keys()):
        lines.append(f"| {k} | {metrics[k]} |")
    return "\n".join(lines)




# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="round0",
                        choices=["round0"])
    parser.add_argument("--env-dir", help="Path to env directory (contains step.py)")
    parser.add_argument("--template", help="Path to template file")
    parser.add_argument("--exploration", help="Path to exploration JSON")
    parser.add_argument("--task-manifest", default=None, help="Task Manifest markdown")
    parser.add_argument("--output", required=True, help="Output prompt path")
    args = parser.parse_args()

    if args.mode == "round0":
        if not all([args.env_dir, args.template, args.exploration]):
            parser.error("--mode round0 requires --env-dir, --template, --exploration")
        prompt = build_round0_prompt(
            Path(args.env_dir), Path(args.template),
            Path(args.exploration), args.task_manifest,
        )

    Path(args.output).write_text(prompt, encoding="utf-8")
    print(f"Prompt saved → {args.output}")
