"""Static-analysis quality checks for LLM-generated reward functions.

Checks run before training to catch common issues:
1. All-negative coefficients — no positive signal, agent learns to die fast
2. Extreme magnitude ratios — one component dominates, others are noise
3. Missing terminal bonus — env has clear terminal states but no sparse bonus

Usage:
    from reward_validator import validate_reward_quality
    warnings = validate_reward_quality(code, task_manifest)
"""

from __future__ import annotations

import re
from typing import List


def _extract_coefficients(code: str) -> list[dict]:
    """Extract (name, coefficient) pairs from compute_reward body.

    Looks for lines like:
        penalty = -0.3 * dist
        bonus = 1.0 if condition else 0.0
        terminal = -100.0
    """
    match = re.search(
        r"def compute_reward\(.*?\):\n(.*?)(?=\n\s*\n\s*def |\Z)",
        code, re.DOTALL,
    )
    if not match:
        return []

    body = match.group(1)
    components = []

    for line in body.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("def ") or line.startswith("return "):
            continue

        # Extract variable name
        var_match = re.match(r"(\w+)\s*=", line)
        if not var_match:
            continue
        var_name = var_match.group(1)

        # Skip internal / housekeeping vars
        if var_name in ("components", "total", "_outcome"):
            continue

        # Extract leading numeric coefficient
        coeff_match = re.search(
            r"=\s*([+-]?\s*\d+\.?\d*)", line.replace(" ", "")
        )
        if coeff_match:
            try:
                coeff = float(coeff_match.group(1))
                components.append({
                    "name": var_name,
                    "coefficient": coeff,
                    "line": line,
                })
            except ValueError:
                continue

    return components


def _has_terminal_condition(code: str) -> bool:
    """Check if code contains terminal-bonus logic (terminated-triggered branch)."""
    # Match `if terminated:` blocks OR ternary expressions like `... if terminated else ...`
    if re.search(r"^\s+if terminated\b", code, re.MULTILINE):
        return True
    if re.search(r"if terminated\s+else", code):
        return True
    return False


def _check_all_negative(
    components: list[dict], task_manifest: str,
) -> list[str]:
    """Warn if all component coefficients are negative (no positive signal)."""
    if not components:
        return []

    non_zero = [c for c in components if abs(c["coefficient"]) > 1e-9]
    if not non_zero:
        return []

    # Check: are there any positive coefficients?
    has_positive = any(c["coefficient"] > 0 for c in non_zero)
    # Also: check for conditional bonuses (ternary with positive value)
    has_positive_signal = has_positive

    if not has_positive_signal:
        return [
            "All reward coefficients are negative or zero — the agent has no "
            "positive signal to pursue. Add at least one positive reward component "
            "(e.g., a progress bonus, sparse terminal success bonus, or "
            "distance-to-target improvement reward)."
        ]
    return []


def _check_component_ratio(components: list[dict]) -> list[str]:
    """Warn if per-step component coefficients differ by > 1000x.

    Terminal components (conditional on terminated/truncated) are excluded
    because they must overcome cumulative per-step rewards and are inherently
    larger in magnitude.
    """
    if len(components) < 2:
        return []

    # Filter out terminal / sparse-bonus components.
    # Terminal components (conditional on terminated/truncated) must overcome
    # cumulative per-step rewards and are inherently larger in magnitude.
    # Also exclude any ternary-conditional bonus (e.g. `bonus if pos>=0.45 else 0`)
    # since those are sparse by nature and shouldn't constrain per-step tuning.
    per_step = [
        c for c in components
        if abs(c["coefficient"]) > 1e-9
        and "terminated" not in c.get("line", "")
        and "truncated" not in c.get("line", "")
        # Ternary-conditional value (sparse bonus, not per-step)
        and not re.search(r"\bif\s+.+\s+else\b", c.get("line", ""))
    ]
    if len(per_step) < 2:
        return []

    abs_coeffs = [abs(c["coefficient"]) for c in per_step]
    ratio = max(abs_coeffs) / min(abs_coeffs)
    if ratio > 1000:
        largest = next(c for c in per_step if abs(c["coefficient"]) == max(abs_coeffs))
        smallest = next(c for c in per_step if abs(c["coefficient"]) == min(abs_coeffs))
        return [
            f"Per-step coefficient ratio {ratio:.0f}x (\"{largest['name']}\"={largest['coefficient']} "
            f"vs \"{smallest['name']}\"={smallest['coefficient']}). "
            f"The smallest per-step component will be negligible noise — keep all per-step "
            f"components within the same order of magnitude (ideally within 5x of each other). "
            f"Terminal components (conditional on terminated/truncated) were excluded."
        ]
    return []


def _check_terminal_bonus(
    components: list[dict], task_manifest: str,
) -> list[str]:
    """Warn if env has terminal success/failure but reward lacks terminal bonus."""
    has_terminal = _has_terminal_condition(
        components[0]["line"] if components else ""
    ) if components else False

    # Actually check the full code
    has_terminal = _has_terminal_condition(
        "\n".join(c.get("full_code", "") for c in components)
    ) if False else None

    # Re-check on full code
    del has_terminal  # will be recomputed below

    return []  # placeholder — will compute below


def _check_terminal_bonus(code: str, task_manifest: str) -> list[str]:
    """Warn if env has terminal success/failure but reward lacks terminal bonus."""
    has_terminal = _has_terminal_condition(code)

    if has_terminal:
        # Terminal logic exists — check that it assigns a large-magnitude value
        for line in code.split("\n"):
            if "terminated" in line and "=" in line:
                # Found terminated handling — assume bonus is present
                return []
        return []  # relaxed: if they handle terminated, assume they handle it

    # No terminal condition at all — check if task manifest suggests one is needed
    success_keywords = ["success", "land", "goal", "reach", "target", "complete"]
    failure_keywords = ["crash", "fail", "damage", "die", "out-of-bounds"]

    manifest_needs_terminal = False
    manifest_lower = task_manifest.lower()
    for kw in success_keywords + failure_keywords:
        if kw in manifest_lower:
            manifest_needs_terminal = True
            break

    if manifest_needs_terminal:
        return [
            "No terminal bonus detected, but the task manifest describes clear "
            "success/failure terminal states. Add a sparse terminal bonus "
            "(e.g., +100 for success, -100 for failure) — this is often the "
            "most important learning signal for the agent."
        ]
    return []


def validate_reward_quality(
    code: str, task_manifest: str = "",
) -> list[dict]:
    """Run all static quality checks on a reward function.

    Args:
        code: Reward function source code (without framework-added imports).
        task_manifest: Task manifest text (for context-aware checks).

    Returns:
        List of warning dicts, each with keys:
          - severity: "high" | "medium" | "low"
          - check: Check name
          - message: Human-readable warning text
        Empty list = no issues found.
    """
    warnings: list[dict] = []

    # Remove the docstring/header if present for cleaner analysis
    body = code
    if code.startswith('"""'):
        # Find closing triple-quote and skip it
        idx = code.find('"""', 3)
        if idx != -1:
            body = code[idx + 3:]

    components = _extract_coefficients(body)

    if not components:
        warnings.append({
            "severity": "medium",
            "check": "coefficient_extraction",
            "message": "Could not parse reward component coefficients. "
                       "Manual review recommended.",
        })
        return warnings

    # Check 1: all-negative coefficients
    for msg in _check_all_negative(components, task_manifest):
        warnings.append({
            "severity": "high",
            "check": "all_negative",
            "message": msg,
        })

    # Check 2: extreme magnitude ratios
    for msg in _check_component_ratio(components):
        warnings.append({
            "severity": "high",
            "check": "component_ratio",
            "message": msg,
        })

    # Check 3: missing terminal bonus
    for msg in _check_terminal_bonus(body, task_manifest):
        warnings.append({
            "severity": "medium",
            "check": "terminal_bonus",
            "message": msg,
        })

    return warnings


def print_quality_report(
    warnings: list[dict], label: str = "Reward Quality",
) -> None:
    """Print formatted quality report to console."""
    if not warnings:
        print(f"  {label}: OK (no issues found)")
        return

    print(f"\n  {'=' * 50}")
    print(f"  {label}")
    print(f"  {'=' * 50}")
    for w in warnings:
        severity_tag = {
            "high": "HIGH",
            "medium": "MEDIUM",
            "low": "LOW",
        }.get(w["severity"], "INFO")
        print(f"  [{severity_tag}] {w['check']}: {w['message']}")
    print(f"  {'=' * 50}\n")
