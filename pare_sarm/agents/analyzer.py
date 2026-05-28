"""Analyzer Agent: produces structured diagnosis from health scores.

Reads health scores + component stats + current code + cross-round history.
Produces a JSON diagnosis report with per-component verdicts and mutation recommendations.

This is the KEY innovation: progress-aligned component diagnosis.
The Analyzer identifies WHY each component is healthy or broken.
"""

import json
import re
from pathlib import Path

from pare_sarm.llm import call_llm
from pare_sarm.diagnosis.health_score import format_health_table


def run_analyzer(
    health_scores: dict,
    component_stats: list[dict],
    current_reward_code: str,
    task_manifest: str,
    round_num: int,
    history_text: str,
    api_key: str,
    model: str = "deepseek-reasoner",
    temperature: float = 0.4,
    output_dir: Path | None = None,
    behavior_summary: str = "",
) -> dict:
    """Analyze component health and produce a structured diagnosis.

    Args:
        health_scores: Output from compute_health_scores()
        component_stats: Per-component mean/std/min/max from trajectory logs
        current_reward_code: The reward function source code
        task_manifest: Task description
        round_num: Current round number
        history_text: Cross-round history from episodic memory
        api_key: DeepSeek API key
        model: LLM model
        temperature: LLM temperature
        output_dir: Save artifacts here

    Returns:
        dict with: diagnosis, escalation_level, component_verdicts, pipeline_action
    """
    health_table = format_health_table(health_scores)

    prompt = _build_prompt(
        health_scores, health_table, component_stats,
        current_reward_code, task_manifest, round_num, history_text,
        behavior_summary,
    )

    print(f"  [Analyzer] Calling {model}...")
    response = call_llm(prompt, api_key, model, temperature)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / "response.txt").write_text(response, encoding="utf-8")

    result = _parse_response(response)

    if output_dir:
        (output_dir / "analysis.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def _build_prompt(
    health_scores: dict,
    health_table: str,
    component_stats: list[dict],
    current_reward_code: str,
    task_manifest: str,
    round_num: int,
    history_text: str,
    behavior_summary: str = "",
) -> str:
    """Build the analyzer prompt with health data, behavior, and history."""
    overall = health_scores.get("overall_health", 0)
    summary = health_scores.get("summary", "")

    behavior_section = ""
    if behavior_summary:
        behavior_section = f"""=== Observed Agent Behavior ===
{behavior_summary}

IMPORTANT: Use the behavioral observations to understand what the agent is ACTUALLY
doing. The health metrics tell you WHICH component is problematic; the behavior tells
you WHY — what perverse incentive the reward created. Combine both to trace the
causal chain from reward code → agent behavior → task failure.

"""

    return f"""You are the Analyzer Agent. Your job: diagnose WHY the current reward function
produced the observed training dynamics, and produce a structured JSON diagnosis report.

=== Task ===
{task_manifest[:2000]}

{behavior_section}=== Component Health Summary ===
Overall Score: {overall:.0f}/100
{summary}

{health_table}

=== Current Reward Code ===
```python
{current_reward_code[:3000]}
```

=== Previous Rounds History ===
{history_text if history_text else '(Round 0 — no history)'}

=== How the Health Metrics Work (for deeper diagnostic reasoning) ===
These metrics are computed from ~100K per-step training records. Understanding how they are derived helps you diagnose root causes more precisely:

- **Progress Corr**: Pearson correlation between this component's per-step values and the per-step task progress change (Δp = progress_fn(obs_next) - progress_fn(obs_current)).
  * P > 0 = component values move in the SAME direction as task progress (component is HELPING the agent make progress).
  * P ≈ 0 = component values have no consistent relationship with progress (component provides no usable learning signal).
  * P < 0 = component values move OPPOSITE to task progress direction (component is ACTIVELY MISLEADING the agent).
  * Common root cause of P < 0: the component conflates two conflicting goals (e.g., a single term mixing descent and drift), or uses the wrong sign.

- **Failure Corr**: Pearson correlation with episode termination. F > 0 means higher component values tend to co-occur with crashing.

- **Active**: |mean| > 0.01. Inactive components are essentially dead — their magnitude is too small to affect learning.

- **Share**: fraction of total absolute reward contributed by this component. Share > 50% means this component dominates all others, suppressing their gradient signal.

- **Health**: weighted score = 0.25×(activation) + 0.25×(balance) + 0.40×(progress_alignment) − 0.10×(failure_conflict). Progress alignment is weighted highest because a component is only useful if it drives task progress.

=== Instructions ===
1. Identify the ROOT CAUSE of any poor health (not just symptoms).
2. For each component with issues, say WHY:
   - "keep" if progress_corr > 0.2 and active and not dominating
   - "reduce" if progress_corr < -0.2 (misaligned) or share > 0.5 (dominating)
   - "remove" if inactive (|mean| ≈ 0)
   - "strengthen" if active but progress_corr between 0 and 0.2 (weak alignment)
   - "reconsider" if |progress_corr| < 0.1 (no signal at all)
3. Set escalation_level:
   - "coefficient": small weight adjustments will fix the issues
   - "structural": need to add/remove/restructure components
   - "rewrite": fundamental approach is flawed
   IF the same diagnosis appeared in 2+ previous rounds, escalate to "structural" or "rewrite".
4. Set pipeline_action: "continue" (proceed), "regenerate" (restart from scratch), or "stop" (converged).

=== Output Format ===
Output ONLY valid JSON (no markdown, no surrounding text):
{{
  "diagnosis": "Root cause: ... (2-4 sentences)",
  "escalation_level": "coefficient",
  "component_verdicts": [
    {{"component": "name", "verdict": "keep", "reason": "Strong progress alignment (P=+0.72)"}}
  ],
  "pipeline_action": "continue"
}}
"""


def _parse_response(response: str) -> dict:
    """Parse JSON from analyzer response with robust fallbacks."""
    # Strategy 1: JSON code block
    m = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find outermost braces
    start = response.find("{")
    end = response.rfind("}")
    if start >= 0 and end > start:
        json_str = response[start:end + 1]
        # Fix common JSON issues
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "diagnosis": "Failed to parse analyzer response — check response.txt",
        "escalation_level": "coefficient",
        "component_verdicts": [],
        "pipeline_action": "continue",
    }
