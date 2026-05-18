"""
Safety patterns for CodeValidator.

Regex patterns that detect known anti-patterns in LLM-generated reward code.
"""

import re

# Patterns that indicate risky code
ANTI_PATTERNS = [
    # Storing simulator objects as instance attributes (unpicklable)
    (
        "simulator_object_storage",
        re.compile(
            r"self\.\w+\s*=\s*self\.\w+\.(position|linearVelocity|angle|angularVelocity|"
            r"world|body|fixture|ground_contact)",
        ),
        "Storing simulator object as instance attribute — will crash SubprocVecEnv",
    ),
    # Storing numpy arrays from simulator
    (
        "array_storage",
        re.compile(r"self\.\w+\s*=\s*self\.\w+\[\d+\]"),
        "Storing simulator array element — may not be serializable",
    ),
    # Using Box2D/MuJoCo-specific imports at module level
    (
        "simulator_import",
        re.compile(r"^(?:import|from)\s+(?:Box2D|mujoco|dm_control)", re.MULTILINE),
        "Importing simulator library at module level — must be inside function body",
    ),
    # Using physics engine sleep/awake state
    (
        "physics_sleep_state",
        re.compile(r"\.awake\b|\.sleep\b"),
        "Using physics engine awake/sleep state — unreliable as task metric",
    ),
    # Hardcoded environment-specific constants (absolute thresholds for termination)
    (
        "absolute_threshold",
        re.compile(r"[<>]=?\s*\d{3,}\."),
        "Hardcoded large numeric threshold — may be environment-specific overfitting",
    ),
    # Note: import checks removed — np and math are pre-injected in scope via train.py
]

# Required patterns that must be present
REQUIRED_PATTERNS = [
    (
        "compute_reward_defined",
        re.compile(r"def compute_reward"),
        "compute_reward function must be defined",
    ),
    (
        "return_tuple",
        re.compile(r"return\s+\w+.*,\s*[\w\{\[]"),
        "Must return (float, dict) tuple: return total, components",
    ),
]


def check_all_patterns(code: str) -> list[dict]:
    """Run all anti-pattern and required pattern checks.

    Returns list of issues, each with {category, message, severity}.
    """
    issues = []

    # Anti-pattern checks
    for name, pattern, message in ANTI_PATTERNS:
        if name == "missing_numpy_import":
            # This pattern is complex — skip regex, check directly
            if "import numpy" not in code and "import numpy as np" not in code:
                issues.append({
                    "category": "import",
                    "message": "compute_reward may be missing numpy import — add 'import numpy as np' inside function body",
                    "severity": "high",
                })
        elif pattern.search(code):
            severity = "high" if "simulator" in name or "array_storage" in name else "medium"
            issues.append({
                "category": name,
                "message": message,
                "severity": severity,
            })

    # Required pattern checks
    for name, pattern, message in REQUIRED_PATTERNS:
        if not pattern.search(code):
            issues.append({
                "category": name,
                "message": f"MISSING: {message}",
                "severity": "high",
            })

    return issues
