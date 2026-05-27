"""
perception_agent.py — Observes raw training data, produces structured perception report.

Role in the multi-agent system:
    Training → Perception Agent → perception_report → Analyst Agent

This agent does NOT write code or propose changes. It only observes and describes.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure framework directory is on path for imports
_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm


def build_perception_prompt(run_dir: Path, template_path: Path) -> str:
    """Build the perception prompt from training results and template."""
    # Dynamic path insert for direct imports (avoid relative import issues)
    _import_dir = Path(__file__).resolve().parent.parent
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))
    from template_engine import (
        load_training_data,
        format_metrics_table,
        format_env_metrics_section,
        format_component_table,
        format_traj_env_metrics_table,
        format_dynamics_section,
        format_constraint_discovery_section,
        format_action_cross_metrics_section,
        format_episode_consistency_section,
    )

    data = load_training_data(run_dir)
    template = template_path.read_text(encoding="utf-8")

    traj = data["traj_summary"]
    lens = traj.get("lengths", {})

    # Load task description from Task Manifest
    # Supports both new format (## Task Goal) and legacy format (## Environment Description)
    task_description = ""
    manifest_path = run_dir.parent / "memory" / "TASK_MANIFEST.md"
    if manifest_path.exists():
        manifest_text = manifest_path.read_text(encoding="utf-8")
        # Try new format first (from env_perception_agent)
        m = re.search(
            r"## Task Goal\s*\n(.*?)(?=\n## |\Z)",
            manifest_text, re.DOTALL
        )
        if m:
            task_description = m.group(1).strip()
        else:
            # Fallback to legacy format
            m = re.search(
                r"## Environment Description\s*\n(.*?)(?=\n## |\Z)",
                manifest_text, re.DOTALL
            )
            if m:
                task_description = m.group(1).strip()

    placeholders = {
        "metrics_table": format_metrics_table(data["eval_history"], max_metrics=6),
        "env_metrics_section": format_env_metrics_section(data["eval_history"], max_metrics=6),
        "component_table": format_component_table(data["traj_summary"]),
        "traj_env_metrics_table": format_traj_env_metrics_table(data["traj_summary"]),
        "dynamics_section": format_dynamics_section(data["traj_summary"], run_dir),
        "constraint_discovery_section": format_constraint_discovery_section(data["traj_summary"], data["eval_history"]),
        "action_cross_metrics_section": format_action_cross_metrics_section(data["traj_summary"], data["eval_history"]),
        "episode_consistency_section": format_episode_consistency_section(data["traj_summary"], data["eval_history"]),
        "task_description": task_description,
        "n_traj_episodes": str(traj.get("n_episodes", 0)),
        "traj_len_mean": str(lens.get("mean", "?")),
        "traj_len_min": str(lens.get("min", "?")),
        "traj_len_max": str(lens.get("max", "?")),
    }

    result = template
    for key, value in placeholders.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def extract_behavior_metrics(perception_report: str) -> dict:
    """Extract key numerical metrics from perception report.

    Only extracts mean_length (environment-agnostic) from perception report.
    Env-specific metrics come from env_metadata.
    """
    metrics = {}

    # Try to extract from "Key Numbers" section
    section_match = re.search(
        r"### 6\. Key Numbers.*?\n(.*?)(?=\n###|\Z)",
        perception_report, re.DOTALL
    )
    if section_match:
        section = section_match.group(1)
        for line in section.split("\n"):
            if "mean_length" in line.lower():
                nums = re.findall(r"[-+]?\d*\.?\d+", line)
                if nums:
                    metrics["mean_length"] = float(nums[0])
    return metrics


def run_perception_agent(run_dir: Path, api_key: str,
                          model: str = "deepseek-reasoner",
                          temperature: float = 0.3) -> str:
    """Run the perception agent on a completed training run.

    Args:
        run_dir: Training run directory (roundN/)
        api_key: LLM API key
        model: Model name
        temperature: Lower temperature for more factual output

    Returns:
        perception_report: Markdown report string
    """
    template_path = Path(__file__).resolve().parent.parent.parent / "templates" / "perception_prompt.txt"
    if not template_path.exists():
        return _generate_fallback_report(run_dir)

    prompt = build_perception_prompt(run_dir, template_path)
    report = call_llm(prompt, api_key, model, temperature)

    # Save artifacts
    (run_dir / "perception_prompt.txt").write_text(prompt, encoding="utf-8")
    try:
        diagnostics = build_perception_diagnostics(run_dir)
        (run_dir / "perception_diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    (run_dir / "perception_response.md").write_text(report, encoding="utf-8")
    (run_dir / "perception_report.md").write_text(report, encoding="utf-8")
    # Perception guard check removed: advisory-only, replaced by shared_rules.py constraints

    # Phase-2: update persistent Perception belief state
    try:
        from memory.memory_system import MemorySystem
        mem = MemorySystem(run_dir.parent if run_dir.name.startswith("round") else run_dir)
        # Extract structured fields from the report
        key_metrics = _extract_key_metrics_from_report(report)
        dynamics_trend = _extract_dynamics_trend(report)
        patterns = _extract_identified_patterns(report)
        anomalies = _extract_anomalies(report)
        mem.update_belief("perception", {
            "round": run_dir.name,
            "behavior_summary": report[:400],
            "key_metrics": key_metrics,
            "dynamics_trend": dynamics_trend,
            "identified_patterns": patterns[:5],
            "anomalies": anomalies[:3],
        })
    except Exception:
        import traceback
        traceback.print_exc()

    return report


def answer_perception_query(run_dir: Path, query: str) -> str:
    """Answer targeted follow-up questions from other agents using structured data.

    Supports direct metric lookup (e.g., "mean_length", "action_magnitude_mean")
    before falling back to themed summaries.
    """
    _import_dir = Path(__file__).resolve().parent.parent
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))
    from template_engine import load_training_data
    from constraint_discovery import detect_constraint_violations, derive_action_cross_metrics

    data = load_training_data(run_dir)
    traj = data.get("traj_summary", {})
    eval_history = data.get("eval_history", [])
    violations = detect_constraint_violations(traj, eval_history)
    cross = derive_action_cross_metrics(traj, eval_history)
    q = (query or "").lower()

    metric_bank = {}
    lengths = traj.get("lengths", {})
    for k in ("mean", "min", "max", "std"):
        if k in lengths:
            metric_bank[f"length_{k}"] = lengths[k]
            if k == "mean":
                metric_bank["mean_length"] = lengths[k]
    for k, v in traj.items():
        if isinstance(v, (int, float)):
            metric_bank[k] = v
    if isinstance(cross, dict):
        for k, v in cross.items():
            if isinstance(v, (int, float)):
                metric_bank[k] = v

    requested = [name for name in metric_bank.keys() if name.lower() in q]
    if requested:
        pairs = {k: metric_bank[k] for k in requested[:8]}
        return f"Perception metric lookup: {pairs}"

    if any(k in q for k in ["constraint", "principle", "violation"]):
        return f"Perception follow-up (violations): {violations}"
    if any(k in q for k in ["efficiency", "action", "velocity"]):
        return f"Perception follow-up (efficiency): {cross}"
    if any(k in q for k in ["length", "termination", "truncation"]):
        return f"Perception follow-up (lengths): {lengths}"
    return (
        "Perception follow-up summary: "
        f"n_episodes={traj.get('n_episodes', 0)}, lengths={lengths}, "
        f"cross_metrics={cross}, violations={violations[:3]}"
    )


def build_perception_diagnostics(run_dir: Path) -> dict:
    """Build structured diagnostics from raw data for downstream agents."""
    _import_dir = Path(__file__).resolve().parent.parent
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))
    from template_engine import load_training_data
    from constraint_discovery import detect_constraint_violations, derive_action_cross_metrics

    data = load_training_data(run_dir)
    traj = data.get("traj_summary", {})
    eval_history = data.get("eval_history", [])
    violations = detect_constraint_violations(traj, eval_history)
    cross = derive_action_cross_metrics(traj, eval_history)
    lengths = traj.get("lengths", {})
    diagnostics = {
        "n_episodes": traj.get("n_episodes", 0),
        "lengths": lengths,
        "mean_length": lengths.get("mean"),
        "cross_metrics": cross,
        "constraint_violations": violations,
    }
    for k, v in traj.items():
        if isinstance(v, (int, float)):
            diagnostics[k] = v
    return diagnostics


def _extract_key_metrics_from_report(report: str) -> dict:
    """Extract numeric key metrics from perception report via regex."""
    metrics = {}

    def _safe_float(v: str) -> float | None:
        """Convert string to float, returning None on failure (e.g. '.' or 'nan')."""
        try:
            return float(v)
        except ValueError:
            return None

    # mean_length
    m = re.search(r'mean[\s_]*length[:\s]*([\d.]+)', report, re.IGNORECASE)
    if m:
        val = _safe_float(m.group(1))
        if val is not None:
            metrics["mean_length"] = val
    # success / completion rate
    m = re.search(r'(success|completion)[\s_]*rate[:\s]*([\d.]+)', report, re.IGNORECASE)
    if m:
        val = _safe_float(m.group(2))
        if val is not None:
            metrics["success_rate"] = val
    # action magnitude
    m = re.search(r'action[\s_]*magnitude[:\s]*([\d.]+)', report, re.IGNORECASE)
    if m:
        val = _safe_float(m.group(1))
        if val is not None:
            metrics["action_magnitude"] = val
    # velocity / efficiency / progress — generic task-level metrics
    m = re.search(r'(?:velocity|efficiency|progress|speed)[:\s]*([\d.]+)', report, re.IGNORECASE)
    if m:
        val = _safe_float(m.group(2))
        if val is not None:
            metrics[m.group(1).lower()] = val
    return metrics


def _extract_dynamics_trend(report: str) -> str:
    """Extract the behavior trend sentence from the report."""
    # Look for "Behavior Trend" or "Trend" section
    for pattern in [r'Behavior[:\s]*Trend[:\s]*[—\-]?\s*(.+?)(?:\n|$)',
                     r'Trend[:\s]*(.+?)(?:\n|$)']:
        m = re.search(pattern, report, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:200]
    return ""


def _extract_identified_patterns(report: str) -> list:
    """Extract listed behavioral patterns from the report."""
    patterns = []
    # Look for bullet points under pattern/behavior/diagnosis sections
    in_section = False
    for line in report.splitlines():
        if re.search(r'(?:identified|behavioral|key)\s*patterns?', line, re.IGNORECASE):
            in_section = True
            continue
        if in_section:
            if line.startswith("##") or (line.startswith("###") and not line.startswith("####")):
                break
            m = re.match(r'[-*]\s*(.+?)$', line)
            if m:
                patterns.append(m.group(1).strip()[:100])
    return patterns[:5]


def _extract_anomalies(report: str) -> list:
    """Extract anomaly/flagged items from the report."""
    anomalies = []
    for line in report.splitlines():
        if re.search(r'(?:anomal|flag|warning|unusual|unexpected|suspicious)', line, re.IGNORECASE):
            m = re.match(r'[-*\d+.]\s*(.+?)$', line)
            if m:
                anomalies.append(m.group(1).strip()[:120])
    return anomalies[:3]


def _generate_fallback_report(run_dir: Path) -> str:
    """Generate a basic report without LLM call (if template is missing)."""
    import csv
    evals = []
    csv_path = run_dir / "evaluations" / "history.csv"
    if csv_path.exists():
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                evals.append(row)

    if not evals:
        return "No evaluation data available."

    last = evals[-1]
    report = (
        f"## Perception Report (Fallback)\n\n"
        f"### Evaluation Summary\n"
        f"- Mean length: {last.get('mean_length', 'N/A')}\n"
        f"- Env metrics: {last.get('env_metrics', 'N/A')}\n\n"
        f"Note: Full perception analysis requires LLM call with template."
    )
    (run_dir / "perception_report.md").write_text(report, encoding="utf-8")
    return report
