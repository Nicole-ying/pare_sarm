"""Mutator Agent: constrained structural reward repair.

The outer workflow deterministically asks for three mutation families:
``direct_fix``, ``component_edit``, and ``progress_gated``.  The LLM is still
responsible for writing the code inside each family, but the family itself is
not left to chance.  This makes ablations and candidate diversity auditable.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from pare_sarm.llm import call_llm, extract_reward_fn, compile_and_check

MUTATION_TYPES = ["direct_fix", "component_edit", "progress_gated"]

MUTATION_GUIDANCE = {
    "direct_fix": """
You are generating Candidate 1: DIRECT-FIX.
- Make the smallest targeted repair consistent with the diagnosis.
- Adjust only the problematic component weights/signs.
- Do not globally scale every coefficient.
- Do not introduce a completely new reward philosophy.
""",
    "component_edit": """
You are generating Candidate 2: COMPONENT-EDIT.
- Change the component structure, not just coefficients.
- Remove low-health or misleading components.
- Add missing components needed by the task.
- Split conflated components and fix sign conflicts.
- Prefer progress_delta terms when absolute-state rewards encourage stalling.
""",
    "progress_gated": """
You are generating Candidate 3: PROGRESS-GATED.
- Use stage/phase-dependent reward logic.
- Different stages should emphasize different objectives.
- For LunarLander-like tasks: far stage encourages approach/descent, middle stage encourages deceleration/uprightness, final stage rewards stable touchdown/contact.
- Gate dense positive rewards so they cannot be farmed while hovering or stalling.
- Do not assume progress_fn is normalized; use relative/delta comparisons or simple thresholds derived from state variables.
""",
}

SYSTEM_PROMPT = """You are a Reward Function Repair Expert.
You receive a diagnosis report and must generate one complete Python reward function.

Strict rules:
- Preserve the exact function signature: def compute_reward(...)
- Return (float(total), components) where components is a non-empty dict.
- Use 3-8 meaningful component names plus optional "_outcome".
- Do not use the environment's official reward.
- Do not call env.step(), gym.make(), file I/O, subprocess, network APIs, or random sampling.
- Avoid global coefficient-only scaling. Fix the causal failure mode.
- If progress_fn(obs) is available, it returns a scalar where larger generally means more task progress. Do not assume [0, 1] normalization.
- Before finalizing, check whether the policy could exploit the reward by hovering, ending early, or farming dense per-step rewards.

Output ONLY a Python code block containing the complete compute_reward function.
"""


def _smoke_test(code: str, params_str: str, progress_fn_code: str | None = None) -> str | None:
    try:
        param_names = [p.strip() for p in params_str.split(",") if p.strip()]
        n_params = len(param_names)
        namespace = {"np": np, "math": __import__("math")}
        if progress_fn_code and "def progress_fn" in progress_fn_code:
            exec(compile(progress_fn_code, "<progress>", "exec"), namespace)
        exec(compile(f"import math\nimport numpy as np\n{code}", "<reward>", "exec"), namespace)
        if "compute_reward" not in namespace:
            return "compute_reward function not found"
        fn = namespace["compute_reward"]
        if n_params == 3:
            result = fn(np.zeros(8, dtype=np.float32), 0, False)
        elif n_params == 4:
            result = fn(np.zeros(8, dtype=np.float32), 0, False, {})
        else:
            args = [np.zeros(8, dtype=np.float32)] + [0] * max(n_params - 1, 0)
            result = fn(*args)
        if not isinstance(result, tuple) or len(result) != 2:
            return f"Returned {type(result).__name__}, expected (float, dict)"
        total, components = result
        if not isinstance(total, (int, float, np.floating, np.integer)):
            return f"total_reward is {type(total).__name__}, expected float"
        if not isinstance(components, dict) or not components:
            return "components must be a non-empty dict"
        bad = {k: type(v).__name__ for k, v in components.items() if not isinstance(v, (int, float, np.floating, np.integer))}
        if bad:
            return f"component values must be numeric, got {bad}"
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def run_mutator(
    analyzer_report: dict,
    current_reward_code: str,
    task_manifest: str,
    reward_signature: str,
    progress_fn_code: str | None,
    api_key: str,
    model: str = "deepseek-reasoner",
    temperature: float = 0.5,
    memory_context: str = "",
    output_dir: Path | None = None,
    mutation_type: str = "direct_fix",
) -> dict:
    params = reward_signature.replace("compute_reward(", "").rstrip(")")
    prompt = _build_agent_prompt(
        analyzer_report, current_reward_code, task_manifest, params,
        progress_fn_code, memory_context, mutation_type,
    )
    conversation = prompt
    for attempt in range(1, 4):
        try:
            response = call_llm(conversation, api_key, model, temperature)
            code = extract_reward_fn(response)
            ok, err = compile_and_check(code)
            if ok and "def compute_reward" in code:
                runtime_err = _smoke_test(code, params, progress_fn_code)
                if runtime_err:
                    conversation += f"\n\nYour code compiled but failed a runtime smoke test:\n{runtime_err}\nFix it and output ONLY corrected Python code."
                    temperature = min(temperature + 0.1, 0.8)
                    continue
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                    (output_dir / "conversation.txt").write_text(conversation, encoding="utf-8")
                    (output_dir / "response.txt").write_text(response, encoding="utf-8")
                    header = '"""LLM-generated reward function.\n"""\n\nimport math\nimport numpy as np\n\n'
                    (output_dir / "reward_fn_source.py").write_text(header + code + "\n", encoding="utf-8")
                return {"code": code, "parse_ok": True, "temperature": temperature, "attempts": attempt, "mutation_type": mutation_type}
            conversation += f"\n\nYour previous output had an error: {err or 'missing def compute_reward'}\nOutput ONLY corrected Python code."
            temperature = min(temperature + 0.1, 0.8)
        except Exception as e:
            conversation += f"\n\nThe call failed with {type(e).__name__}: {e}. Try again with valid code only."
    return {"code": None, "parse_ok": False, "temperature": temperature, "attempts": 3, "mutation_type": mutation_type}


def generate_mutation_candidates(
    analyzer_report: dict,
    current_reward_code: str,
    task_manifest: str,
    reward_signature: str,
    progress_fn_code: str | None,
    api_key: str,
    model: str = "deepseek-reasoner",
    memory_context: str = "",
    output_dir: Path | None = None,
) -> list[dict]:
    candidates = []
    temperatures = {"direct_fix": 0.3, "component_edit": 0.5, "progress_gated": 0.7}
    forbidden = set(analyzer_report.get("forbidden_mutation_types", []) or [])

    for i, mtype in enumerate(MUTATION_TYPES):
        local_report = dict(analyzer_report)
        local_report["forced_mutation_type"] = mtype
        if mtype in forbidden:
            local_report["diagnosis"] = local_report.get("diagnosis", "") + f"\nThis family was listed as risky; if used for comparison, avoid the risky behavior inside the family."
        cand_dir = output_dir / f"candidate_{i}" if output_dir else None
        result = run_mutator(
            local_report, current_reward_code, task_manifest, reward_signature,
            progress_fn_code, api_key, model, temperatures[mtype], memory_context,
            output_dir=cand_dir, mutation_type=mtype,
        )
        result["idx"] = i
        result["style"] = mtype
        candidates.append(result)
        status = f"{len(result['code'])} chars" if result.get("parse_ok") and result.get("code") else "FAILED"
        print(f"  {mtype}: {status}")
    return candidates


def _build_agent_prompt(
    analyzer_report: dict,
    current_reward_code: str,
    task_manifest: str,
    params: str,
    progress_fn_code: str | None,
    memory_context: str,
    mutation_type: str,
) -> str:
    diagnosis = analyzer_report.get("diagnosis", "No diagnosis available")
    escalation = analyzer_report.get("escalation_level", "coefficient")
    failure_mode = analyzer_report.get("failure_mode", "unknown")
    root_cause = analyzer_report.get("root_cause_type", "unknown")
    verdicts = analyzer_report.get("component_verdicts", [])
    verdict_text = "\n".join(
        f"  {v.get('component','?')}: {v.get('verdict','?')} — {v.get('reason','?')}"
        for v in verdicts
    ) or "  (no component verdicts)"
    memory_section = f"## Retrieved Memory and Prior Lessons\n{memory_context}\n\n" if memory_context else ""
    progress_section = ""
    if progress_fn_code:
        progress_section = f"""## Available Progress Function
You may call `progress_fn(obs)`. It returns a scalar where larger generally means better task progress, but it may not be normalized.
Prefer progress deltas or state-derived phase checks over assuming an absolute scale.
```python
{progress_fn_code}
```

"""
    return f"""{SYSTEM_PROMPT}

---

## Forced Mutation Family
{MUTATION_GUIDANCE.get(mutation_type, MUTATION_GUIDANCE['direct_fix'])}

## Task Description
{task_manifest[:3000]}

## Diagnosis Report
Root cause: {diagnosis}
Failure mode: {failure_mode}
Root-cause type: {root_cause}
Escalation level: {escalation}

### Per-Component Verdicts
{verdict_text}

{memory_section}{progress_section}## Current Reward Code
```python
{current_reward_code[:4000]}
```

## Required Function Signature
```python
def compute_reward({params}):
    ...
    return float(total), components
```

Output ONLY the complete Python code block. No explanations.
"""
