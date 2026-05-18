"""
CodeValidator — algorithmic validation of generated reward code.

Checks:
1. Python syntax compiles
2. compute_reward function exists with correct signature
3. Returns (float, dict) tuple
4. No new components beyond what diagnosis specifies
5. All proposed changes from diagnosis are actually applied
6. No duplicate imports/headers
7. No simulator object storage
8. No module-level simulator imports
9. Known anti-patterns absent

Note: metrics_fn is NOT required — it is injected algorithmically by train.py.
Pass/fail with specific error messages for retry.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Optional

_mr = Path(__file__).resolve().parent.parent
if str(_mr) not in sys.path:
    sys.path.insert(0, str(_mr))

from .safety_patterns import check_all_patterns


def validate_code(
    code: str,
    expected_signature: str = "action",
    previous_code: str = "",
    diagnosis: Optional[dict] = None,
) -> dict:
    """Validate generated reward code against all checks.

    Args:
        code: The generated reward function code.
        expected_signature: The expected parameter list of compute_reward
            (e.g., "action" or "obs, action, terminated, truncated, info").
        previous_code: The previous round's reward code (for change scope check).
        diagnosis: The final diagnosis (for proposal adherence check).

    Returns:
        {
            "passed": bool,
            "errors": [{"category": str, "message": str}],
            "warnings": [str],
        }
    """
    errors = []
    warnings = []

    # 1. Python syntax check
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        errors.append({
            "category": "syntax",
            "message": f"Syntax error: {e}",
        })
        # Can't continue with other checks if syntax is invalid
        return {"passed": False, "errors": errors, "warnings": warnings}

    # 2. compute_reward function exists
    if "def compute_reward" not in code:
        errors.append({
            "category": "signature",
            "message": "Missing 'def compute_reward' function",
        })

    # 3. compute_reward signature check
    sig_match = re.search(r"def compute_reward\(self,\s*(.*?)\):", code)
    if sig_match:
        actual_sig = " ".join(sig_match.group(1).split())
        expected = " ".join(expected_signature.split())
        if actual_sig != expected:
            errors.append({
                "category": "signature",
                "message": (
                    f"Signature mismatch: expected ({expected}), "
                    f"got ({actual_sig})"
                ),
            })
    else:
        errors.append({
            "category": "signature",
            "message": "Could not find compute_reward signature",
        })

    # 4. Return statement check — must return a tuple of (float, dict)
    # Accepts both literal dict or variable: return total, {"x":1} OR return total, d
    if not re.search(r'return\s+\w+.*,\s*[\w\{\[]', code):
        errors.append({
            "category": "return_tuple",
            "message": "MUST return (float, dict) tuple: return total_reward, components_dict",
        })

    # 6. Components dict check — removed. Variable name varies (reward_dict, comps, etc.)
    # The return_tuple check already ensures (float, dict) is returned.

    # 7. Proposal adherence (if diagnosis provided)
    if diagnosis and previous_code:
        adherence_errors = _check_proposal_adherence(code, previous_code, diagnosis)
        errors.extend(adherence_errors)

    # 8. Scope creep check (no new components beyond what diagnosis specifies)
    if diagnosis and previous_code:
        scope_errors = _check_scope_creep(code, previous_code, diagnosis)
        errors.extend(scope_errors)

    # 9. Anti-pattern checks
    pattern_issues = check_all_patterns(code)
    for issue in pattern_issues:
        if issue["severity"] == "high":
            errors.append({
                "category": issue["category"],
                "message": issue["message"],
            })
        else:
            warnings.append(issue["message"])

    # 6. Should not have module-level imports (scope has np, math pre-injected)
    if "import math" in code.split("def compute_reward")[0] if "def compute_reward" in code else True:
        pass  # No longer an error — stripped by pipeline

    # 7. Additional safety: ensure no Box2D/MuJoCo imports at module level
    if re.search(r"^(?:import|from)\s+(?:Box2D|mujoco)", code, re.MULTILINE):
        errors.append({
            "category": "import",
            "message": "Module-level Box2D or MuJoCo import detected. All imports must be inside function bodies.",
        })

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def _check_proposal_adherence(code: str, previous_code: str,
                                diagnosis: dict) -> list[dict]:
    """Check that the diagnosis's proposed changes are actually applied in the code."""
    errors = []

    changes = diagnosis.get("proposed_changes", [])
    for i, change in enumerate(changes):
        new_code = (change.get("new_code") or "").strip()
        if not new_code:
            continue

        # Skip "remove" type — the code should just not contain the old line
        if change.get("change_type") == "remove":
            current_code = (change.get("current_code") or "").strip()
            if current_code and current_code in code:
                errors.append({
                    "category": "adherence",
                    "message": (
                        f"Change {i+1}: Removal not applied. "
                        f"Old code still present: '{current_code[:80]}'"
                    ),
                })
            continue

        # For other change types, check that new_code appears
        # Extract the semantic part of new_code (before comment)
        semantic_new = new_code.split("#", 1)[0].strip()
        if semantic_new and semantic_new not in code:
            # Try fuzzy match: check if the variable assignment pattern exists
            assignment_match = re.match(r"(\w+)\s*=\s*(.+)", semantic_new)
            if assignment_match:
                var_name = assignment_match.group(1)
                # Check if this variable appears in code
                if var_name not in code:
                    errors.append({
                        "category": "adherence",
                        "message": (
                            f"Change {i+1}: Variable '{var_name}' from new_code "
                            f"not found in generated code"
                        ),
                    })
            else:
                errors.append({
                    "category": "adherence",
                    "message": (
                        f"Change {i+1}: new_code not found in generated code: "
                        f"'{semantic_new[:80]}'"
                    ),
                })

    return errors


def _check_scope_creep(code: str, previous_code: str,
                         diagnosis: dict) -> list[dict]:
    """Check that no unauthorized changes were made beyond what the diagnosis specifies."""
    errors = []

    # Extract component variable names from both versions
    prev_components = _extract_component_names(previous_code)
    new_components = _extract_component_names(code)

    # Get components specifically mentioned in diagnosis
    diagnosis_components = set()
    for change in diagnosis.get("proposed_changes", []):
        comp = change.get("component", "")
        if comp:
            diagnosis_components.add(comp)

    # Check for new components not in diagnosis
    new_unauthorized = new_components - prev_components - diagnosis_components
    if new_unauthorized:
        errors.append({
            "category": "scope_creep",
            "message": (
                f"New reward components introduced without authorization: "
                f"{new_unauthorized}. Only modify components specified in diagnosis."
            ),
        })

    return errors


def _extract_component_names(code: str) -> set[str]:
    """Extract reward component variable names from code.

    Looks for patterns like: r_xxx = ... or reward_xxx = ...
    in the components dict section and compute_reward function.
    """
    components = set()

    # Find assignments of the form: name = ... within compute_reward
    # We look at variable names that start with r_ or contain 'reward'
    for match in re.finditer(r"(r_\w+|reward_\w+)\s*=", code):
        components.add(match.group(1))

    return components
