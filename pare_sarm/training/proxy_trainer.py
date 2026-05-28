"""Short proxy training for reward candidate evaluation.

Enables per-step progress-aligned diagnosis by:
1. Passing progress_fn to the training subprocess for per-step logging
2. Reading per-step logs (step_logs/steps.jsonl) after training
3. Computing true per-step component-progress correlations
"""

import re
import subprocess
import sys
from pathlib import Path

from pare_sarm.utils import ensure_dir, save_yaml
from pare_sarm.diagnosis.health_score import (
    compute_health_scores, compute_progress_correlations,
    pearson_r,
)


DEFAULT_PPO = {
    "policy": "MlpPolicy", "learning_rate": 3e-4, "n_steps": 1024,
    "batch_size": 64, "n_epochs": 4, "gamma": 0.99,
    "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,
    "vf_coef": 0.5, "max_grad_norm": 0.5,
}


def run_proxy_training(
    env_dir: Path,
    reward_code: str,
    output_dir: Path,
    config: dict,
    env_id_suffix: str,
    total_timesteps: int = 50_000,
    n_envs: int = 4,
    seed: int = 42,
    progress_fn_code: str | None = None,
) -> dict:
    """Run short PPO training to evaluate a reward candidate.

    Args:
        env_dir: Path to env directory (contains env.py)
        reward_code: Python source for compute_reward function
        output_dir: Directory for training outputs
        config: Full experiment config dict
        env_id_suffix: Unique suffix for gym env registration
        total_timesteps: Training steps (proxy budget)
        n_envs: Number of parallel envs
        seed: Random seed
        progress_fn_code: Python source for progress_fn(obs)->float.
                          If provided, injected into training for per-step logging.

    Returns:
        dict with: success, health, component_stats, eval_history, model_path,
                   per_step_data_available (bool)
    """
    ensure_dir(output_dir)

    # Build proxy training config
    proxy_config = _build_proxy_config(config, total_timesteps, seed)

    # Write reward code
    reward_path = output_dir / "reward_fn_source.py"
    cleaned = re.sub(r'^"""LLM[- ].*?"""', '', reward_code.strip(), flags=re.DOTALL)
    cleaned = re.sub(r'^import\s+(math|numpy).*?\n', '', cleaned, flags=re.MULTILINE)
    reward_path.write_text(
        f'"""Proxy reward candidate."""\n\nimport math\nimport numpy as np\n\n{cleaned}\n',
        encoding="utf-8"
    )

    # Write progress_fn if provided
    progress_path = None
    if progress_fn_code and "def progress_fn" in progress_fn_code:
        progress_path = output_dir / "progress_fn.py"
        progress_path.write_text(progress_fn_code + "\n", encoding="utf-8")

    # Write config
    config_path = output_dir / "proxy_config.yaml"
    save_yaml(config_path, proxy_config)

    # Run training subprocess
    train_script = Path(__file__).resolve().parent / "_train_script.py"
    cmd = [
        sys.executable, str(train_script),
        "--env-dir", str(env_dir),
        "--env-id", f"{env_dir.name}-{env_id_suffix}",
        "--config", str(config_path),
        "--run-dir", str(output_dir),
        "--reward-source", str(reward_path),
    ]
    if progress_path and progress_path.exists():
        cmd += ["--progress-source", str(progress_path)]
    max_eps = config.get("max_episode_steps")
    if max_eps:
        cmd += ["--max-episode-steps", str(max_eps)]

    print(f"    Proxy training ({total_timesteps} steps)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    Proxy training FAILED: {result.stderr[-500:]}")
        return {
            "success": False,
            "health": {"overall_health": 0.0, "verdict": "failed"},
            "component_stats": [],
            "eval_history": [],
            "model_path": None,
            "per_step_data_available": False,
        }

    # ── Post-training diagnosis ──

    # Read per-step logs (the key innovation: true step-level correlations)
    step_records = _read_step_logs(output_dir)
    per_step_available = len(step_records) > 0

    # Read episode-level logs (fallback / supplement)
    ep_records = _read_episode_logs(output_dir)

    # Collect component stats from episode-level summaries
    component_stats = _collect_component_stats_from_episodes(output_dir)

    # Compute true per-step progress correlations if available
    max_eps_value = max_eps or config.get("max_episode_steps", 500)

    if per_step_available and len(step_records) >= 10:
        # Per-step diagnosis: corr(component_step_value, progress_delta_step_value)
        component_stats = _augment_with_per_step_correlations(
            component_stats, step_records,
        )
        progress_values = _extract_progress_values(step_records)
        failure_flags = _extract_failure_flags_from_steps(step_records, max_eps_value)
    else:
        # Fallback: episode-level heuristic
        progress_values = _compute_progress_heuristic(ep_records, max_eps_value)
        failure_flags = _compute_failure_flags(ep_records, max_eps_value)
        component_stats = compute_progress_correlations(
            component_stats, ep_records, progress_values, max_eps_value,
        )

    # Health score
    health = compute_health_scores(component_stats, progress_values, failure_flags, max_eps_value)

    # Read eval history
    eval_history = []
    eval_csv = output_dir / "evaluations" / "history.csv"
    if eval_csv.exists():
        import csv
        with eval_csv.open("r") as f:
            eval_history = list(csv.DictReader(f))

    model_path = output_dir / "model.zip"
    verdict = "good" if health["overall_health"] >= 60 else ("ok" if health["overall_health"] >= 35 else "poor")

    return {
        "success": True,
        "health": {**health, "verdict": verdict},
        "component_stats": component_stats,
        "eval_history": eval_history,
        "model_path": model_path if model_path.exists() else None,
        "per_step_data_available": per_step_available,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_proxy_config(base_config: dict, timesteps: int, extra_seed: int) -> dict:
    """Build a config dict for proxy training with sensible PPO defaults."""
    user_ppo = base_config.get("ppo", {})
    full_ppo = {**DEFAULT_PPO, **user_ppo}

    cfg = {
        "total_timesteps": timesteps,
        "n_envs": base_config.get("n_envs_proxy", 4),
        "seed": base_config.get("seed", 42) + extra_seed,
        "device": base_config.get("device", "cpu"),
        "normalize": False,
        "ppo": full_ppo,
        "evaluation": {"freq": timesteps // 2, "episodes": 5},
        "checkpoint": {"freq": timesteps * 10},
    }
    if "max_episode_steps" in base_config:
        cfg["max_episode_steps"] = base_config["max_episode_steps"]
    return cfg


def _read_step_logs(output_dir: Path) -> list[dict]:
    """Read per-step JSONL records from StepLoggerWrapper output."""
    step_log = output_dir / "step_logs" / "steps.jsonl"
    if not step_log.exists():
        return []
    records = []
    for line in step_log.read_text("utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            import json
            records.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return records


def _read_episode_logs(output_dir: Path) -> list[dict]:
    """Read episode-level JSONL records from ComponentTrackerWrapper output."""
    from pare_sarm.utils import read_all_jsonl
    traj_dir = output_dir / "trajectory_logs"
    if traj_dir.exists() and traj_dir.is_dir():
        return read_all_jsonl(traj_dir)
    return []


def _collect_component_stats_from_episodes(output_dir: Path) -> list[dict]:
    """Collect per-component mean/std from episode-level trajectory JSONL."""
    from pare_sarm.diagnosis.component_stats import collect_component_stats
    traj_dir = output_dir / "trajectory_logs"
    if traj_dir.exists():
        return collect_component_stats(traj_dir)
    return []


def _augment_with_per_step_correlations(
    component_stats: list[dict],
    step_records: list[dict],
) -> list[dict]:
    """Compute true per-step progress correlations for each component.

    For each component c_i:
      P_i = corr(c_i_step_values, progress_delta_step_values)
      C_i = corr(c_i_step_values, failure_indicators)

    This is the KEY innovation: component health is measured against
    per-step task progress, not just episode-level aggregates.
    """
    if not step_records:
        return component_stats

    # Collect per-step time series for each component
    comp_series: dict[str, list[float]] = {}
    progress_deltas: list[float] = []
    failure_indicators: list[float] = []

    for rec in step_records:
        comps = rec.get("components", {})
        for name, val in comps.items():
            if name == "_outcome":
                continue
            comp_series.setdefault(name, []).append(float(val))

        pd_val = rec.get("progress_delta")
        if pd_val is not None:
            progress_deltas.append(float(pd_val))
        else:
            progress_deltas.append(0.0)

        # Failure indicator: 1.0 if this step ends the episode via termination
        done = rec.get("done", False)
        terminated = rec.get("terminated", False)
        failure_indicators.append(1.0 if (done and terminated) else 0.0)

    # Augment component stats with per-step correlations
    for c in component_stats:
        name = c["name"]
        vals = comp_series.get(name, [])
        if len(vals) >= 5 and len(progress_deltas) >= 5:
            c["progress_corr"] = round(pearson_r(vals, progress_deltas[:len(vals)]), 4)
            c["failure_corr"] = round(pearson_r(vals, failure_indicators[:len(vals)]), 4)
        else:
            c["progress_corr"] = 0.0
            c["failure_corr"] = 0.0

    return component_stats


def _extract_progress_values(step_records: list[dict]) -> list[float]:
    """Extract per-episode progress estimates from step records."""
    # Group by episode and use the final progress value
    ep_progress: dict[int, list[float]] = {}
    for rec in step_records:
        ep = rec.get("episode", 0)
        np_val = rec.get("next_progress")
        if np_val is not None:
            ep_progress.setdefault(ep, []).append(float(np_val))

    values = []
    for ep in sorted(ep_progress.keys()):
        vals = ep_progress[ep]
        if vals:
            values.append(vals[-1])  # final progress of episode
    return values


def _extract_failure_flags_from_steps(step_records: list[dict], max_steps: int) -> list[float]:
    """Extract per-episode failure flags from step records."""
    ep_failed: dict[int, bool] = {}
    for rec in step_records:
        ep = rec.get("episode", 0)
        if rec.get("terminated", False):
            ep_failed[ep] = True
        elif ep not in ep_failed:
            ep_failed[ep] = False

    return [1.0 if ep_failed.get(ep, False) else 0.0 for ep in sorted(ep_failed.keys())]


def _compute_progress_heuristic(records: list[dict], max_steps: int) -> list[float]:
    """Fallback: episode-length heuristic for progress."""
    values = []
    for rec in records:
        outcome = rec.get("component_means", {}).get("_outcome", 0.0)
        if outcome > 0.5:
            values.append(1.0)
        elif outcome < -0.5:
            values.append(0.0)
        else:
            values.append(min(1.0, rec.get("length", max_steps) / max_steps))
    return values


def _compute_failure_flags(records: list[dict], max_steps: int) -> list[float]:
    """Episode-level failure flags."""
    return [1.0 if rec.get("length", max_steps) < max_steps * 0.3 else 0.0
            for rec in records]
