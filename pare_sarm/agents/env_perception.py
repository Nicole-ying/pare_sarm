"""EnvPerception Agent: generates TaskManifest + ProgressProxy from env source code.

Uses ToolRegistry to read env.py, step.py, and exploration data.
Runs once before Round 0. Single LLM call with all context pre-loaded.
Outputs structured task manifest and a progress function for diagnosis.
"""

import re
from pathlib import Path

from pare_sarm.llm import call_llm


def run_env_perception(
    env_dir: Path,
    exploration_path: Path,
    api_key: str,
    model: str = "deepseek-reasoner",
    temperature: float = 0.3,
    output_dir: Path | None = None,
) -> dict:
    """Generate Task Manifest and Progress Proxy from environment source code.

    Args:
        env_dir: Path to env directory (contains env.py, step.py)
        exploration_path: Path to exploration JSON file
        api_key: DeepSeek API key
        model: LLM model name
        temperature: LLM temperature
        output_dir: Optional directory to save artifacts

    Returns:
        dict with keys:
            task_manifest: str       — Markdown task description
            progress_fn_code: str    — Python code for progress_fn(obs) -> float
            reward_signature: str    — e.g. "compute_reward(state, action, terminated)"
            obs_dim: int             — Observation space dimension
            max_episode_steps: int   — Episode step limit
    """
    # Load source files
    env_source = _read(env_dir / "env.py")
    step_source = _read(env_dir / "step.py")
    exploration_raw = _read(exploration_path) if exploration_path.exists() else "{}"

    # Extract signature from step.py
    signature = _extract_signature(step_source) or "state, action, terminated"

    # Extract metadata from exploration
    import json
    exploration = {}
    try:
        exploration = json.loads(exploration_raw)
    except json.JSONDecodeError:
        pass
    obs_dim = exploration.get("obs_dim", 0)
    max_eps = exploration.get("max_episode_steps", 500)

    # Build prompt with all context
    prompt = _build_prompt(env_source, step_source, exploration_raw)

    # Call LLM
    print(f"  [EnvPerception] Calling {model}...")
    response = call_llm(prompt, api_key, model, temperature)

    # Save artifacts
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / "response.txt").write_text(response, encoding="utf-8")
        (output_dir / "task_manifest.md").write_text(response, encoding="utf-8")

    # Parse sections from response
    task_manifest = _extract_section(response, "TASK_MANIFEST") or response
    progress_fn_code = _extract_section(response, "PROGRESS_FN")
    if progress_fn_code:
        m = re.search(r"```python\s*\n(.*?)```", progress_fn_code, re.DOTALL)
        if m:
            progress_fn_code = m.group(1).strip()

    # Save progress function
    if progress_fn_code and output_dir:
        (output_dir / "progress_fn.py").write_text(progress_fn_code + "\n", encoding="utf-8")

    return {
        "task_manifest": task_manifest,
        "progress_fn_code": progress_fn_code,
        "reward_signature": signature,
        "obs_dim": obs_dim,
        "max_episode_steps": max_eps,
    }


def _build_prompt(env_source: str, step_source: str, exploration_raw: str) -> str:
    """Build the environment perception prompt with all context."""
    # Truncate for token budget
    if len(env_source) > 15000:
        env_source = env_source[:15000] + "\n# ... (truncated)"

    return f"""You are an Environment Analyst. Your job is to understand a reinforcement learning
environment from its source code and produce three structured outputs.

=== Environment Source Code (env.py) ===
```python
{env_source}
```

=== Step Function (step.py) ===
```python
{step_source[:8000]}
```

=== Exploration Data (random-policy rollouts) ===
```
{exploration_raw[:6000]}
```

=== Output 1: [TASK_MANIFEST] ===
Write a concise markdown task description with these sections:
- **Task Objective**: What must the agent achieve?
- **Success Criteria**: When is the task complete?
- **Failure Conditions**: What causes termination?
- **Observation Space**: Describe each dimension with its meaning and range
- **Action Space**: What actions are available?
- **Key Dynamics**: Important physics, constraints, or behaviors
- **Reward Design Hints**: Suggestions for shaping a good reward function

=== Output 2: [PROGRESS_FN] ===
Write a Python function that measures task progress from observation alone:
```python
def progress_fn(obs):
    '''obs: numpy array of observation values.
    Returns: float where HIGHER values = more task progress.
    Range should be approximately [0, 1] or [-1, 1].'''
    # Use obs[dim_index] to access each observation dimension
    ...
    return progress_value
```
Rules:
- Use ONLY the observation array (no env internals, no physics)
- Return a single float that monotonically increases toward the goal
- Must be valid, compilable Python
- The function name MUST be `progress_fn`

=== Output 3: [SIGNATURE] ===
Extract the exact parameter list from `self.compute_reward(...)` in step().
Output one line: `compute_reward(param1, param2, ...)`

=== OUTPUT FORMAT ===
Separate sections with `---`:

[TASK_MANIFEST]
(markdown here)

---
[PROGRESS_FN]
```python
def progress_fn(obs):
    ...
```

---
[SIGNATURE]
compute_reward(state, action, terminated)
"""


def _extract_signature(step_source: str) -> str | None:
    """Extract compute_reward call signature from step.py."""
    m = re.search(r'self\.compute_reward\(([^)]+)\)', step_source)
    if m:
        params = [p.strip() for p in m.group(1).split(",")]
        return f"compute_reward({', '.join(params)})"
    return None


def _extract_section(text: str, section_name: str) -> str | None:
    """Extract content between [SECTION_NAME] markers."""
    pattern = rf'\[{section_name}\]\s*\n(.*?)(?=\n\s*\[|\Z)'
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _read(path: Path) -> str:
    """Read a file, return '' if missing."""
    return path.read_text("utf-8") if path.exists() else ""
