"""Generator Agent: produces reward function code from task manifest.

Generates K diverse candidates by varying temperature and adding
diversity hints. Integrates with memory to avoid repeating past failures.
"""

from pathlib import Path

from pare_sarm.llm import call_llm, extract_reward_fn, compile_and_check


def run_generator(
    task_manifest: str,
    progress_fn_code: str,
    reward_signature: str,
    exploration_summary: str,
    api_key: str,
    model: str = "deepseek-reasoner",
    temperature: float = 0.6,
    diversity_hint: str = "",
    memory_context: str = "",
    output_dir: Path | None = None,
) -> dict:
    """Generate a reward function for the given environment.

    Args:
        task_manifest: Task description markdown
        progress_fn_code: Progress function source (for context, not to call)
        reward_signature: Function signature e.g. "compute_reward(state, action, terminated)"
        exploration_summary: Exploration data summary
        api_key: DeepSeek API key
        model: LLM model name
        temperature: LLM temperature
        diversity_hint: Additional instruction to encourage different designs
        memory_context: Relevant lessons from past rounds (empty for round 0)
        output_dir: Save artifacts here

    Returns:
        dict with: code, parse_ok, temperature, response
    """
    params = reward_signature.replace("compute_reward(", "").rstrip(")")

    prompt = _build_prompt(
        task_manifest, progress_fn_code, params,
        exploration_summary, diversity_hint, memory_context,
    )

    for attempt in range(1, 4):
        try:
            response = call_llm(prompt, api_key, model, temperature)
            code = extract_reward_fn(response)
            ok, err = compile_and_check(code)
            if ok and "def compute_reward" in code:
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                    (output_dir / "response.txt").write_text(response, encoding="utf-8")
                    header = '"""LLM-generated reward function.\n"""\n\nimport math\nimport numpy as np\n\n'
                    (output_dir / "reward_fn_source.py").write_text(header + code + "\n", encoding="utf-8")
                return {"code": code, "parse_ok": True, "temperature": temperature, "response": response}
            print(f"    Generator attempt {attempt}: {err or 'missing compute_reward'}")
        except Exception as e:
            print(f"    Generator attempt {attempt} failed: {e}")

    return {"code": None, "parse_ok": False, "temperature": temperature, "response": "", "error": "all 3 attempts failed"}


def generate_k_candidates(
    task_manifest: str,
    progress_fn_code: str,
    reward_signature: str,
    exploration_summary: str,
    api_key: str,
    k: int = 3,
    model: str = "deepseek-reasoner",
    base_temperature: float = 0.6,
    memory_context: str = "",
    output_dir: Path | None = None,
) -> list[dict]:
    """Generate K diverse initial reward candidates.

    Diversity comes from:
    1. Varying temperature (0.4, 0.6, 0.8)
    2. Diversity hints for candidates after the first
    3. Memory context about past failures (if available)
    """
    candidates = []
    for i in range(k):
        temp = base_temperature + (i - 1) * 0.2
        temp = max(0.2, min(1.0, temp))

        hint = ""
        if i == 1:
            hint = "Generate a CONSERVATIVE design: simple components, moderate magnitudes, safe choices."
        elif i == 2:
            hint = "Generate a BOLD design: try a fundamentally different approach, novel component structure."

        cand_dir = output_dir / f"candidate_{i}" if output_dir else None
        result = run_generator(
            task_manifest, progress_fn_code, reward_signature,
            exploration_summary, api_key, model, temp, hint,
            memory_context=memory_context, output_dir=cand_dir,
        )
        result["idx"] = i
        candidates.append(result)
        status = f"{len(result['code'])} chars" if result["parse_ok"] else "FAILED"
        print(f"  Candidate {i+1}/{k}: {status} (temp={temp})")
    return candidates


def _build_prompt(
    task_manifest: str,
    progress_fn_code: str,
    params: str,
    exploration_summary: str,
    diversity_hint: str,
    memory_context: str,
) -> str:
    """Build the generator prompt."""
    memory_section = ""
    if memory_context:
        memory_section = f"""=== Lessons from Past Rounds ===
{memory_context}

"""

    diversity_section = ""
    if diversity_hint:
        diversity_section = f"""=== Diversity Instruction ===
{diversity_hint}

"""

    return f"""You are a reward engineer designing a reward function for reinforcement learning.

=== Task Manifest ===
{task_manifest[:4000]}

=== Exploration Data ===
{exploration_summary[:3000]}

=== Progress Function (for understanding only — do NOT call this in your reward) ===
```python
{progress_fn_code[:1000] if progress_fn_code else '# Not available'}
```

{memory_section}{diversity_section}=== Design Principles ===
1. Return (float, dict). Total is a scalar. Component dict maps descriptive names (3-5 components) to per-step float values.
2. Provide DENSE shaping. Every step should give meaningful feedback to the agent.
3. Balance magnitudes. All per-step components should be within 50x of each other.
4. Include "_outcome" in the component dict: +1.0 for success, -1.0 for failure, 0.0 otherwise.
   Do NOT add _outcome to total — it is for diagnosis only.
5. Use raw observation values (obs[0], obs[1], etc.) — do NOT call progress_fn.
6. Use `import math` and `import numpy as np` as needed. They are pre-imported.

=== Exact Function Signature (MUST MATCH) ===
```python
def compute_reward({params}):
    # Compute per-step reward components
    comp1 = ...
    comp2 = ...
    components = {{"name1": comp1, "name2": comp2}}
    total = sum(components.values())  # BEFORE _outcome
    components["_outcome"] = 1.0 if success_condition else (-1.0 if failure else 0.0)
    return float(total), components
```

=== Output ===
Output ONLY one Python code block:
```python
def compute_reward({params}):
    ...
```
"""
