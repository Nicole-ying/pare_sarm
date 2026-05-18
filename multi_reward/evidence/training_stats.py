"""
Statistical computation helpers for evidence layer.

Loads training data artifacts and computes summary statistics.
Adapted from eureka_llm/framework/template_engine.py.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load_eval_history(run_dir: Path) -> list[dict]:
    """Load evaluation history CSV.

    Returns list of {timesteps, mean_length, std_length,
    termination_breakdown, env_metrics}.
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
            entry["mean_length"] = float(row.get("mean_length", 0) or 0)
            entry["std_length"] = float(row.get("std_length", 0) or 0)

            raw_tb = row.get("termination_breakdown", "{}")
            try:
                entry["termination_breakdown"] = json.loads(raw_tb)
            except (json.JSONDecodeError, TypeError):
                entry["termination_breakdown"] = {}

            raw_em = row.get("env_metrics", "{}")
            try:
                entry["env_metrics"] = json.loads(raw_em)
            except (json.JSONDecodeError, TypeError):
                entry["env_metrics"] = {}

            rows.append(entry)
    return rows


def load_trajectory_summary(run_dir: Path) -> dict:
    """Load all trajectory JSONL files and aggregate component + env_metrics stats.

    Also extracts termination_reason distribution.
    """
    traj_dir = run_dir / "trajectory_logs"
    if not traj_dir.exists():
        return {"n_episodes": 0, "components": {}}

    all_components: dict[str, list] = defaultdict(list)
    all_env_metrics: dict[str, list] = defaultdict(list)
    total_episodes = 0
    lengths = []
    term_reasons = defaultdict(int)

    for fname in sorted(traj_dir.glob("*.trajectory.jsonl")):
        for line in fname.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_episodes += 1
            lengths.append(record.get("length", 0))

            term_reasons[record.get("termination_reason", "unknown")] += 1

            for comp, mean_val in record.get("component_means", {}).items():
                all_components[comp].append(mean_val)

            for metric, mean_val in record.get("env_metrics_means", {}).items():
                all_env_metrics[metric].append(mean_val)

    if total_episodes == 0:
        return {"n_episodes": 0, "components": {}, "termination_reasons": {}}

    def _aggregate(data: dict) -> dict:
        result = {}
        for key, vals in data.items():
            arr = vals
            n = len(arr)
            if n == 0:
                continue
            mean = sum(arr) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in arr) / max(n - 1, 1))
            cv = std / max(abs(mean), 1e-8)
            result[key] = {
                "mean": round(float(mean), 6),
                "std": round(float(std), 6),
                "coeff_of_variation": round(float(cv), 4),
            }
        return result

    return {
        "n_episodes": total_episodes,
        "components": _aggregate(all_components),
        "env_metrics": _aggregate(all_env_metrics) if all_env_metrics else {},
        "lengths": {
            "mean": round(float(sum(lengths) / len(lengths)), 1),
            "std": round(float(float(np_std(lengths))), 1),
            "min": min(lengths),
            "max": max(lengths),
            "q10": _quantile(lengths, 0.10),
            "q25": _quantile(lengths, 0.25),
            "q50": _quantile(lengths, 0.50),
            "q75": _quantile(lengths, 0.75),
            "q90": _quantile(lengths, 0.90),
        },
        "termination_reasons": dict(term_reasons),
    }


def np_std(values: list) -> float:
    """Compute standard deviation (population formula)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def _quantile(values: list, q: float) -> float:
    """Compute quantile of sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(q * len(sorted_vals))
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


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


def load_training_data(run_dir: Path) -> dict:
    """Load all training artifacts from a run directory."""
    config = {}
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text("utf-8")) or {}
        except Exception:
            pass

    run_info = {}
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        try:
            run_info = json.loads(info_path.read_text("utf-8"))
        except Exception:
            pass

    reward_src = ""
    src_path = run_dir / "reward_fn_source.py"
    if src_path.exists():
        reward_src = src_path.read_text("utf-8")

    return {
        "config": config,
        "run_info": run_info,
        "reward_fn_source": reward_src,
        "eval_history": load_eval_history(run_dir),
        "traj_summary": load_trajectory_summary(run_dir),
        "entropy_history": load_entropy_history(run_dir),
    }
