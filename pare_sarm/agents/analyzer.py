"""Analyzer Agent: structured failure-mode and component diagnosis."""

from __future__ import annotations

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
    health_table = format_health_table(health_scores)
    prompt = _build_prompt(
        health_scores, health_table, component_stats, current_reward_code,
        task_manifest, round_num, history_text, behavior_summary,
    )
    print(f"  [Analyzer] Calling {model}...")
    response = call_llm(prompt, api_key, model, temperature)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / "response.txt").write_text(response, encoding="utf-8")
    result = _normalize_result(_parse_response(response))
    if output_dir:
        (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _build_prompt(health_scores, health_table, component_stats, current_reward_code, task_manifest, round_num, history_text, behavior_summary="") -> str:
    overall = health_scores.get("overall_health", 0)
    summary = health_scores.get("summary", "")
    behavior_section = f"""=== Observed Agent Behavior ===
{behavior_summary}

IMPORTANT: Behavior can override component health. If health is high but behavior is hovering, early_crash, or oscillating, treat health as a false-positive proxy and recommend structural mutation.

""" if behavior_summary else ""
    return f"""You are the Analyzer Agent for LLM-generated RL reward functions.
Produce structured JSON that the deterministic pipeline can use.

=== Task ===
{task_manifest[:2200]}

{behavior_section}=== Component Health Summary ===
Overall component health: {overall:.0f}/100
{summary}

{health_table}

=== Current Reward Code ===
```python
{current_reward_code[:3200]}
```

=== Previous Rounds / Memory ===
{history_text if history_text else '(no history)'}

=== Diagnostic Rules ===
1. Component health alone is not policy quality.
2. If behavior is hovering, suspect dense positive rewards, contact bonuses, or absolute progress rewards that can be farmed without terminal success.
3. If behavior is early_crash, suspect penalties dominate or the last repair overcorrected.
4. If recent behavior alternates between hovering and early_crash, forbid coefficient-only tuning and recommend progress_gated mutation.
5. Connect reward code -> component evidence -> behavior evidence.

=== Output JSON Schema ===
{{
  "diagnosis": "2-4 sentences connecting reward code, component stats, and behavior failure.",
  "failure_mode": "hovering | early_crash | approach_but_unstable | landing_progress | unclear",
  "root_cause_type": "dense_positive_reward_hacking | penalty_dominance | missing_success_signal | progress_misalignment | component_dominance | coefficient_overcorrection | unclear",
  "behavior_evidence": ["evidence from behavior report"],
  "component_evidence": ["evidence from health table"],
  "component_verdicts": [{{"component": "name", "verdict": "keep|reduce|remove|strengthen|reconsider", "reason": "..."}}],
  "recommended_mutation": "direct_fix | component_edit | progress_gated | regenerate",
  "forbidden_mutation_types": ["global_scale", "coefficient_only"],
  "escalation_level": "coefficient | structural | rewrite",
  "tool_requests": ["search_behavior_memory", "inspect_rollout_summary"],
  "pipeline_action": "continue | regenerate | stop"
}}
Output ONLY valid JSON.
"""


def _parse_response(response: str) -> dict:
    m = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = response.find("{")
    end = response.rfind("}")
    if start >= 0 and end > start:
        js = response[start:end + 1]
        js = re.sub(r",\s*}", "}", js)
        js = re.sub(r",\s*]", "]", js)
        try:
            return json.loads(js)
        except json.JSONDecodeError:
            pass
    return {"diagnosis": "Failed to parse analyzer response", "failure_mode": "unclear", "root_cause_type": "unclear", "component_verdicts": [], "recommended_mutation": "component_edit", "pipeline_action": "continue"}


def _normalize_result(result: dict) -> dict:
    result.setdefault("diagnosis", "")
    result.setdefault("failure_mode", "unclear")
    result.setdefault("root_cause_type", "unclear")
    result.setdefault("behavior_evidence", [])
    result.setdefault("component_evidence", [])
    result.setdefault("component_verdicts", [])
    result.setdefault("forbidden_mutation_types", [])
    result.setdefault("tool_requests", [])
    if result.get("pipeline_action") not in {"continue", "regenerate", "stop"}:
        result["pipeline_action"] = "continue"
    if result.get("escalation_level") not in {"coefficient", "structural", "rewrite"}:
        result["escalation_level"] = "coefficient"
    if result.get("recommended_mutation") not in {"direct_fix", "component_edit", "progress_gated", "regenerate"}:
        result["recommended_mutation"] = "component_edit"
    return result
