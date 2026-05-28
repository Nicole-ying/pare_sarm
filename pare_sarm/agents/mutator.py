"""Mutator Agent: autonomous reward function repair expert.

This is a REAL agent, not a workflow. It:
  1. Reads the diagnosis report and current reward code
  2. Queries memory for past attempts and lessons
  3. Plans its OWN repair strategy (no hardcoded mutation types)
  4. Generates the fixed code
  5. Self-validates syntax before returning

Run 3 times with different temperatures to produce behaviorally diverse candidates.
Each run is an independent agent decision — the agent chooses its own strategy.
"""

import numpy as np
from pathlib import Path

from pare_sarm.llm import call_llm, extract_reward_fn, compile_and_check


# ═══════════════════════════════════════════════════════════════════════════
# Runtime smoke test
# ═══════════════════════════════════════════════════════════════════════════

def _smoke_test(code: str, params_str: str, progress_fn_code: str | None = None) -> str | None:
    """Run the reward function on dummy inputs. Returns error message or None if OK."""
    try:
        param_names = [p.strip() for p in params_str.split(",")]
        n_params = len(param_names)

        full_code = f"import math\nimport numpy as np\n{code}"
        namespace = {"np": np, "math": __import__("math")}

        # Inject progress_fn if available (for Progress-Gated mutations)
        if progress_fn_code and "def progress_fn" in progress_fn_code:
            exec(compile(progress_fn_code, "<progress>", "exec"), namespace)

        # Execute the reward code to define compute_reward
        exec(compile(full_code, "<smoke>", "exec"), namespace)

        if "compute_reward" not in namespace:
            return "compute_reward function not found after exec"

        fn = namespace["compute_reward"]

        # Call with dummy inputs based on param count
        if n_params == 3:
            # state (vector), action (int), terminated (bool)
            state = np.zeros(8, dtype=np.float32)
            result = fn(state, 0, False)
        elif n_params == 4:
            state = np.zeros(8, dtype=np.float32)
            result = fn(state, 0, False, {})
        else:
            args = [np.zeros(8, dtype=np.float32)] + [0] * (n_params - 1)
            result = fn(*args)

        # Validate return format: must be (float, dict)
        if not isinstance(result, tuple) or len(result) != 2:
            return f"Returned {type(result).__name__}, expected (float, dict) tuple"
        total, components = result
        if not isinstance(total, (int, float)):
            return f"total_reward is {type(total).__name__}, expected float"
        if not isinstance(components, dict):
            return f"components is {type(components).__name__}, expected dict"
        if len(components) == 0:
            return "components dict is empty"
        if any(not isinstance(v, (int, float, np.floating, np.integer)) for v in components.values()):
            return f"component values must be numeric, got: { {k: type(v).__name__ for k, v in components.items()} }"

        return None  # OK
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# Agent system prompt — teaches HOW to think, not WHAT to do
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Reward Function Repair Expert. You receive a diagnosis report identifying
specific problems in a reward function, and you must fix them.

## Your Working Method

### Step 1: Understand the Root Cause
Read the diagnosis carefully. Do NOT just treat symptoms (e.g., "reduce coefficient").
Ask yourself: WHY is this component misaligned? Is it:
- Measuring the wrong thing?
- Using the wrong sign?
- Dominating other signals?
- Missing a key dimension?
- Conceptually broken?

### Step 2: Plan Your Fix
Based on the root cause, decide on a strategy.
You are free to choose ANY approach — adjust coefficients, restructure components,
split conflated terms, switch per-step rewards to terminal bonuses, introduce
stage-conditional logic, or rewrite from scratch.

### Step 3: Generate the Code
Write the complete, runnable Python function. Follow these rules:
- Preserve the EXACT function signature: def compute_reward(...)
- Return (float, dict) where dict has 3-6 named components + "_outcome"
- All per-step components should be within 50x magnitude of each other
- Terminal bonuses can be larger (up to 500x per-step)
- Compute total BEFORE adding _outcome
- progress_fn(obs) is available if you need task progress for gating

### Step 4: Self-Verify
Before finalizing, mentally check:
- Did I fix the root cause, not just the symptom?
- If I were the RL agent, could I still exploit this reward?
- Are all component magnitudes balanced?
- Does the terminal bonus handle both success and failure?

Output ONLY the complete Python code block. No explanations outside the code."""


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

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
) -> dict:
    """Run the Mutator agent to fix a reward function.

    The agent receives the full context (diagnosis, code, memory, task)
    and autonomously decides how to fix the reward. No constraints on strategy.

    Self-validation loop: after generating code, the agent checks its own work.
    If syntax is broken, it receives the error and tries again (up to 3 attempts).
    This makes it an AGENT, not a one-shot code generator.

    Returns dict with: code, parse_ok, temperature, attempts.
    """
    params = reward_signature.replace("compute_reward(", "").rstrip(")")
    prompt = _build_agent_prompt(
        analyzer_report, current_reward_code, task_manifest, params,
        progress_fn_code, memory_context,
    )
    conversation = prompt  # accumulates for multi-turn self-correction

    for attempt in range(1, 4):
        try:
            response = call_llm(conversation, api_key, model, temperature)
            code = extract_reward_fn(response)
            ok, err = compile_and_check(code)

            if ok and "def compute_reward" in code:
                # ── Self-validation: runtime smoke test ──
                runtime_err = _smoke_test(code, params, progress_fn_code)
                if runtime_err:
                    print(f"    Agent attempt {attempt}: runtime error — {runtime_err[:100]}")
                    conversation += f"\n\nYour code was compiled but failed at runtime:\n{runtime_err}\nPlease fix and output ONLY the corrected Python code block."
                    temperature = min(temperature + 0.1, 0.8)  # slightly more creative on retry
                    continue

                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                    (output_dir / "conversation.txt").write_text(conversation, encoding="utf-8")
                    (output_dir / "response.txt").write_text(response, encoding="utf-8")
                    header = '"""LLM-generated reward function.\n"""\n\nimport math\nimport numpy as np\n\n'
                    (output_dir / "reward_fn_source.py").write_text(header + code + "\n", encoding="utf-8")
                return {"code": code, "parse_ok": True, "temperature": temperature, "attempts": attempt}

            # ── Self-correction: feed error back to the agent ──
            feedback = f"\n\nYour previous output had an error: {err or 'missing def compute_reward'}\nPlease fix the issue and output ONLY the corrected Python code block."
            conversation += "\n\n" + response[:500] + feedback
            print(f"    Agent attempt {attempt}: {err or 'missing compute_reward'} — self-correcting...")
            temperature = min(temperature + 0.1, 0.8)

        except Exception as e:
            print(f"    Agent attempt {attempt} failed: {e}")

    return {"code": None, "parse_ok": False, "temperature": temperature, "attempts": 3}


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
    """Generate 3 diverse repair candidates via SEQUENTIAL generation.

    Agent 0 generates first. Agent 1 sees Agent 0's code and is told to be different.
    Agent 2 sees both and must find yet another approach.

    This guarantees behavioral diversity — each agent is explicitly pushed away
    from previous solutions, creating genuinely different repair strategies.
    """
    temperatures = [0.3, 0.5, 0.7]
    candidates = []
    previous_codes = []  # Accumulate for sequential diversity

    for i, temp in enumerate(temperatures):
        # Build diversity push: show what previous agents did
        diversity_push = ""
        if i == 1 and previous_codes:
            diversity_push = (
                f"\n\n=== DIVERSITY REQUIREMENT ===\n"
                f"Agent 0 already generated this approach:\n```python\n{previous_codes[0][:800]}\n```\n"
                f"Your solution MUST be FUNDAMENTALLY DIFFERENT. If Agent 0 adjusted coefficients, "
                f"you should restructure components or try a different reward philosophy entirely.\n"
            )
        elif i >= 2 and len(previous_codes) >= 2:
            diversity_push = (
                f"\n\n=== DIVERSITY REQUIREMENT ===\n"
                f"Agent 0's approach:\n```python\n{previous_codes[0][:500]}\n```\n"
                f"Agent 1's approach:\n```python\n{previous_codes[1][:500]}\n```\n"
                f"Your solution MUST be DIFFERENT from BOTH. Find an approach neither has tried. "
                f"Consider a fundamentally different reward philosophy.\n"
            )

        # Append diversity push to the analyzer report for this agent
        modified_report = dict(analyzer_report)
        if diversity_push:
            modified_report["diagnosis"] = analyzer_report.get("diagnosis", "") + diversity_push

        cand_dir = output_dir / f"candidate_{i}" if output_dir else None
        result = run_mutator(
            modified_report, current_reward_code, task_manifest,
            reward_signature, progress_fn_code,
            api_key, model, temp,
            memory_context=memory_context, output_dir=cand_dir,
        )
        result["idx"] = i
        result["style"] = f"agent-{i}" if not result.get("parse_ok") else f"agent-{i}"
        candidates.append(result)

        if result.get("parse_ok") and result.get("code"):
            previous_codes.append(result["code"])
            status = f"{len(result['code'])} chars"
        else:
            status = "FAILED"
        print(f"  Agent {i+1}/3 (temp={temp}): {status}")

    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_agent_prompt(
    analyzer_report: dict,
    current_reward_code: str,
    task_manifest: str,
    params: str,
    progress_fn_code: str | None,
    memory_context: str,
) -> str:
    """Build the agent's prompt with full context and the system prompt."""

    diagnosis = analyzer_report.get("diagnosis", "No diagnosis available")
    escalation = analyzer_report.get("escalation_level", "coefficient")
    verdicts = analyzer_report.get("component_verdicts", [])

    # Format component verdicts as a clear table
    verdict_lines = []
    for v in verdicts:
        verdict_lines.append(
            f"  {v['component']}: {v['verdict']} — {v.get('reason', '?')}"
        )
    verdict_text = "\n".join(verdict_lines) if verdict_lines else "  (no component verdicts)"

    # Memory: past attempts and learned patterns
    memory_section = ""
    if memory_context:
        memory_section = f"""## Past Attempts & Lessons
{memory_context}

"""

    # Progress function reference
    progress_section = ""
    if progress_fn_code:
        progress_section = f"""## Available Progress Function
You may call `progress_fn(obs)` to get a task progress estimate in [0, 1].
This is useful for stage-conditional logic. It is pre-defined and available.

```python
{progress_fn_code}
```

"""

    return f"""{SYSTEM_PROMPT}

---

## Task Description
{task_manifest[:3000]}

## Diagnosis Report
**Root Cause:** {diagnosis}
**Escalation Level:** {escalation}

### Per-Component Verdicts
{verdict_text}

{memory_section}{progress_section}## Current Reward Code (TO FIX)
```python
{current_reward_code[:4000]}
```

---

## Your Turn

Analyze the root cause. Plan your repair strategy. Generate the fixed code.

**Function signature (MUST preserve exactly):**
```python
def compute_reward({params}):
    ...
    return float(total), components
```

Output ONLY the complete Python code block. No markdown, no explanations.
"""
