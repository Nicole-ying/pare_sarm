"""Collect per-component statistics from trajectory JSONL files."""

from pathlib import Path

from pare_sarm.utils import read_all_jsonl


def collect_component_stats(trajectory_dir: Path) -> list[dict]:
    """Read trajectory JSONL files and compute per-component mean/std/min/max.

    Returns a list of dicts, each with keys:
        name, mean, std, min, max, n, active
    """
    records = read_all_jsonl(trajectory_dir)
    if not records:
        return []

    all_comps: dict[str, list[float]] = {}

    for rec in records:
        for name, val in rec.get("component_means", {}).items():
            if name == "_outcome":
                continue
            all_comps.setdefault(name, []).append(float(val))

    result = []
    for name, vals in all_comps.items():
        arr = vals
        n = len(arr)
        if n == 0:
            continue
        mean = sum(arr) / n
        std = (sum((x - mean) ** 2 for x in arr) / n) ** 0.5 if n > 1 else 0.0
        active = abs(mean) > 0.01
        result.append({
            "name": name,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(min(arr), 6),
            "max": round(max(arr), 6),
            "n": n,
            "active": active,
        })

    return result


def format_stats_table(stats: list[dict]) -> str:
    """Format component statistics as a markdown table for LLM prompts."""
    if not stats:
        return "*(no component statistics available)*"

    lines = [
        "| Component | Mean | Std | Min | Max | N | Status |",
        "|-----------|------|-----|-----|-----|---|--------|",
    ]
    for s in stats:
        status = "active" if s["active"] else "inactive"
        lines.append(
            f"| {s['name']} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} | {s['n']} | {status} |"
        )
    lines.append("")
    lines.append("**Key:** active = |mean| > 0.01, inactive = negligible contribution")
    return "\n".join(lines)
