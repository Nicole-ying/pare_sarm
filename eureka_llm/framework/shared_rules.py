"""Working methods and self-verification patterns shared across all agents.

This is NOT a list of rules to obey. It's a set of thinking patterns
that help agents verify their own work — like a code review checklist
that tells you WHAT to check and WHY it matters.

Usage:
    from shared_rules import render_rules, ROUND0_RULES
    checks_text = render_rules(ROUND0_RULES)
"""

from __future__ import annotations

SHARED_RULES: dict[str, str] = {
    "component_count": """
### Design Method: Purpose-Built Components

Start with about 3 reward components. For each one, ask yourself:
"What specific behavior does this incentivize?"

If you can't answer in one clear sentence, the component is too vague
— the agent won't know what to optimize either.

Simple reward functions train faster and are easier to debug. You can
always add complexity in later rounds when you know what's missing.
""",

    "terminal_bonus": """
### Terminal Signal: Sparse Success/Failure Bonus

If the environment has a clear terminal success or failure state
(e.g. landing safely vs crashing, reaching a goal vs going out of bounds),
include a sparse terminal bonus.

Self-check after writing it:
  1. Ratio check (upper bound): is |terminal_value| / |per-step component × typical-episode-length| < 50?
     If terminal dominates, the agent ignores per-step shaping and learns to
     optimize purely for the terminal outcome — often by ending episodes
     quickly to minimize negative per-step accumulation.
  2. Ratio check (lower bound): is |terminal_value| > |per-step component × typical-episode-length| × 0.1?
     If terminal is too small (e.g., ±50 vs -2000 over an episode), the agent
     literally cannot feel the difference between success and failure — the
     terminal signal is noise buried in per-step accumulation.
  3. Typical starting values: +100 / -100. Adjust based on both ratio checks.
""",

    "outcome": """
### Tracking Pattern: _outcome Signal

_outcome in the components dict is for evaluation visibility, NOT for the reward.

Working method:
  1. Compute total = sum(components.values()) BEFORE adding _outcome.
  2. Add _outcome only after total is computed — it does NOT contribute to learning.
  3. Use 1.0 for success, -1.0 for failure, 0.0 for neutral/unknown.
  4. Think of _outcome as your independent "did the agent succeed?" label.
     It lets you track success rate across training without interference.
""",

    "component_scaling": """
### Balance Verification: Component Magnitudes

After writing all components, verify their relative magnitudes:

  Ask: "If component A is 100× larger than component B, which objective
  will the agent optimize? Which will it ignore?"

When one component dominates, all weaker signals become noise.
The agent quite literally cannot learn the ignored objective.

Fix method: either rescale small components up or dominant ones down.
Target: all |mean values| within roughly 50× of each other.
""",

    "return_type": """
### Return Type: Scalar Float

Total reward must be a plain Python float.

If your computation uses numpy arrays, extract the scalar with
`.item()` or `float()`. The framework expects (float, dict).

This is a mechanical constraint — if violated, training crashes silently.
""",

    "imports": '''
### Output Boundary: No Header, No Imports

Your output must start directly with `def compute_reward(...)`.

Do NOT include:
  - A docstring header (e.g. """LLM-generated reward function...""")
  - import statements (the framework prepends `import math` + `import numpy` automatically)
  - Any text before or after the ```python code block

Why: the framework wraps your code in its own header + imports. Any extra
headers or imports you include will be duplicated, making the reward file messy.

If you need a function not covered by math/numpy, the framework doesn\'t
support it — restructure your approach.
''',

    "compute_reward_signature": """
### Signature Integrity: compute_reward Parameters

Use the EXACT parameter list from the Task Manifest.

Do NOT add, remove, or reorder parameters. The environment calls your
function with a fixed signature — any mismatch causes a runtime crash
on the very first step, wasting the entire training run.

If in doubt, check the Task Manifest. The manifest is authoritative.
""",

    "simulator_storage": """
### Safe State: No Physics Objects

Never store physics engine objects (bodies, joints, handles, model
instances) as persistent attributes.

Only plain Python values (float, int, np.ndarray) may be stored
between steps. Physics objects become stale references and crash.

If you're not sure whether something is a physics object, it probably is.
""",

    "no_hidden_state": """
### API Boundary: No Environment Internals

Never access internal attributes of the environment object (physics engines,
proprietary data structures, simulator handles). These are implementation
details that change between versions and environments.

Access only: the observation array passed to compute_reward, and
getattr(env, '_last_obs', None) for the most recent observation.

This keeps your reward function portable across environments and simulators.
""",

    "guardrails": """
### Output Completeness Verification

Before finishing, walk through this checklist:

  1. Is compute_reward present with the exact parameter signature?
  2. Does it return (float, dict) — the dict must have named component keys?
  3. Does the logic reference a specific environment by name? If so, remove it.
  4. Are all components interpretable — would a human reader understand each?
  5. Is _outcome computed AFTER total, not included in the sum?
""",

    "dimensional_coverage": """
### Dimensional Coverage: All Critical Dimensions Need Gradients

After writing your reward components, verify that every critical task dimension
has a reward component producing a meaningful gradient for it.

Working method:
  1. Look at the task description. Identify 2-4 observation dimensions that are
     most critical for task success (position, velocity, angle, etc.).
  2. For each critical dimension, name which reward component provides a gradient.
  3. If any critical dimension has no component creating a gradient,
     add or modify a component before finalizing.

Why: If the reward ignores a critical dimension, the agent receives no signal
to optimize it. The dimension is left to chance, producing unreliable behavior.
Examples: ignoring position in a reaching task, ignoring angle in a balancing task.
""",

}


RULE_KEYS = list(SHARED_RULES.keys())

# Named groups — these define which checks apply to each agent.
# Keep them synchronized with actual usage in templates and agents.
ROUND0_RULES = [
    "component_count",
    "terminal_bonus",
    "outcome",
    "component_scaling",
    "dimensional_coverage",
    "return_type",
    "imports",
    "compute_reward_signature",
]

GENERATOR_RULES = [
    "terminal_bonus",
    "outcome",
    "component_scaling",
    "dimensional_coverage",
    "return_type",
    "imports",
    "simulator_storage",
    "no_hidden_state",
    "guardrails",
]


def render_rules(rule_names: list[str]) -> str:
    """Render a subset of shared checks by name.

    Args:
        rule_names: List of rule keys from SHARED_RULES.

    Returns:
        Concatenated check text with blank-line separators.
    """
    parts = []
    for name in rule_names:
        text = SHARED_RULES.get(name)
        if text:
            parts.append(text.strip())
    return "\n\n".join(parts)
