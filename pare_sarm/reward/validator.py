"""Static quality checks for LLM-generated reward functions (§4.1 in PARE-SARM spec).

Checks (8 total per GPT spec):
  1. Has `def compute_reward` function
  2. Returns tuple(float, dict) format
  3. Components dict is non-empty
  4. Component values are numeric (int/float)
  5. No NaN / Inf in computation patterns
  6. No forbidden API calls (env.reward, subprocess, etc.)
  7. All-negative coefficients check
  8. Extreme magnitude ratio check
  9. Missing terminal bonus check
"""

import math
import re

# GPT spec §4.1 — forbidden patterns for security and correctness
FORBIDDEN_PATTERNS = [
    "env.reward",
    "self.reward",
    "official_reward",
    "env.step(",
    "gym.make",
    "subprocess",
    "os.system",
    "open(",
    "import socket",
    "__import__",
    "eval(",
    "exec(",
]


def validate_reward_quality(code: str, task_manifest: str = "") -> list[dict]:
    """Run all static quality checks on a reward function (§4.1).

    Returns list of warning dicts: {severity, check, message}. Empty = no issues.
    """
    warnings: list[dict] = []

    # ── Check 1: has compute_reward function ──
    if "def compute_reward" not in code:
        warnings.append({
            "severity": "high",
            "check": "missing_function",
            "message": "No 'def compute_reward' function found. Reward code must define this function.",
        })
        return warnings  # can't do further checks without the function

    # ── Check 2: returns tuple(float, dict) ──
    if "return" not in code:
        warnings.append({
            "severity": "high",
            "check": "missing_return",
            "message": "No return statement found in compute_reward.",
        })
    elif not re.search(r"return\s+.*,\s*\{", code) and not re.search(r"return\s+.*,\s*components", code):
        warnings.append({
            "severity": "high",
            "check": "return_format",
            "message": "Must return (float, dict) tuple. Expected: return float(total), components",
        })

    # ── Check 3: components dict is non-empty ──
    comp_names = re.findall(r'"(\w+)"\s*:', code)
    if not comp_names:
        warnings.append({
            "severity": "medium",
            "check": "empty_components",
            "message": "No component names found in dict. Components dict appears empty.",
        })

    # ── Check 4: component values should be numeric ──
    # (Heuristic: check that component assignments involve math operations or obs indexing)
    if "obs[" not in code and "state[" not in code and "abs(" not in code and "math." not in code:
        warnings.append({
            "severity": "low",
            "check": "no_obs_dependency",
            "message": "Reward doesn't reference observation or state. Components may be constant.",
        })

    # ── Check 5: no NaN / Inf patterns ──
    nan_patterns = ["float('nan')", "float('inf')", "np.nan", "np.inf",
                    "math.nan", "math.inf", "NaN", "Infinity"]
    for pat in nan_patterns:
        if pat in code:
            warnings.append({
                "severity": "high",
                "check": "nan_inf",
                "message": f"Found '{pat}' in code. NaN/Inf in rewards will crash training.",
            })
            break

    # Division by zero risk
    if re.search(r"/\s*\([^)]*\)", code) or re.search(r"/\s*\w+\s*$", code, re.MULTILINE):
        if "max(" not in code and "1e-" not in code and "+ 1e-" not in code:
            warnings.append({
                "severity": "medium",
                "check": "division_risk",
                "message": "Division without obvious zero-guard. Consider adding + 1e-8 to denominators.",
            })

    # ── Check 6: no forbidden API calls ──
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            warnings.append({
                "severity": "high",
                "check": "forbidden_api",
                "message": f"Found forbidden pattern '{pattern}'. Reward functions must not access env internals.",
            })
            break  # one is enough

    # ── Checks 7-9: coefficient-based checks (existing) ──
    body = code
    if code.startswith('"""'):
        idx = code.find('"""', 3)
        if idx != -1:
            body = code[idx + 3:]

    components = _extract_coefficients(body)
    if not components:
        if not warnings:
            warnings.append({
                "severity": "medium",
                "check": "coefficient_extraction",
                "message": "Could not parse reward component coefficients. Manual review recommended.",
            })
        return warnings

    for msg in _check_all_negative(components, task_manifest):
        warnings.append({"severity": "high", "check": "all_negative", "message": msg})

    for msg in _check_component_ratio(components):
        warnings.append({"severity": "high", "check": "component_ratio", "message": msg})

    for msg in _check_terminal_bonus(body, task_manifest):
        warnings.append({"severity": "medium", "check": "terminal_bonus", "message": msg})

    return warnings


def print_quality_report(warnings: list[dict], label: str = "Reward Quality") -> None:
    """Print a formatted quality report to console."""
    if not warnings:
        print(f"  {label}: OK (no issues found)")
        return
    print(f"\n  {'=' * 50}")
    print(f"  {label}")
    print(f"  {'=' * 50}")
    for w in warnings:
        tag = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}.get(w["severity"], "INFO")
        print(f"  [{tag}] {w['check']}: {w['message']}")
    print(f"  {'=' * 50}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Internal check functions (unchanged from existing)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_coefficients(code: str) -> list[dict]:
    """Extract (name, coefficient) pairs from compute_reward body."""
    match = re.search(r"def compute_reward\(.*?\):\n(.*?)(?=\n\s*\n\s*def |\Z)", code, re.DOTALL)
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
        var_match = re.match(r"(\w+)\s*=", line)
        if not var_match:
            continue
        var_name = var_match.group(1)
        if var_name in ("components", "total", "_outcome"):
            continue
        coeff_match = re.search(r"=\s*([+-]?\s*\d+\.?\d*)", line.replace(" ", ""))
        if coeff_match:
            try:
                coeff = float(coeff_match.group(1))
                components.append({"name": var_name, "coefficient": coeff, "line": line})
            except ValueError:
                continue
    return components


def _check_all_negative(components: list[dict], task_manifest: str) -> list[str]:
    if not components:
        return []
    non_zero = [c for c in components if abs(c["coefficient"]) > 1e-9]
    if not non_zero:
        return []
    if not any(c["coefficient"] > 0 for c in non_zero):
        return [
            "All reward coefficients are negative or zero — the agent has no "
            "positive signal to pursue. Add at least one positive reward component."
        ]
    return []


def _check_component_ratio(components: list[dict]) -> list[str]:
    if len(components) < 2:
        return []
    per_step = [
        c for c in components
        if abs(c["coefficient"]) > 1e-9
        and "terminated" not in c.get("line", "")
        and "truncated" not in c.get("line", "")
        and not re.search(r"\bif\s+.+\s+else\b", c.get("line", ""))
        # Also skip components whose NAME suggests sparse/terminal reward
        and not any(kw in c.get("name", "").lower()
                    for kw in ("terminal", "outcome", "bonus", "landing", "success", "failure"))
        # Skip components with magnitude >50x the median (likely sparse/terminal)
        and not (
            len(components) >= 3
            and abs(c["coefficient"]) > 50 * sorted([abs(x["coefficient"]) for x in components])[len(components)//2]
        )
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
            f"Keep all per-step components within the same order of magnitude."
        ]
    return []


def _check_terminal_bonus(code: str, task_manifest: str) -> list[str]:
    has_terminal = _has_terminal_condition(code)
    if has_terminal:
        return []
    success_kw = ["success", "land", "goal", "reach", "target", "complete"]
    failure_kw = ["crash", "fail", "damage", "die", "out-of-bounds"]
    manifest_lower = task_manifest.lower()
    needs_terminal = any(kw in manifest_lower for kw in success_kw + failure_kw)
    if needs_terminal:
        return [
            "No terminal bonus detected, but the task manifest describes clear "
            "success/failure terminal states. Add a sparse terminal bonus "
            "(e.g., +100 for success, -100 for failure)."
        ]
    return []


def _has_terminal_condition(code: str) -> bool:
    if re.search(r"^\s+if terminated\b", code, re.MULTILINE):
        return True
    if re.search(r"if terminated\s+else", code):
        return True
    return False
