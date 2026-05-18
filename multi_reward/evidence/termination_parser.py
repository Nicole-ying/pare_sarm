"""
Parse step.py to extract termination conditions WITHOUT LLM.

Uses AST parsing to find conditional blocks that set termination = True.
These are OBJECTIVE constraints of the environment — no interpretation needed.
"""

import ast
import re
from pathlib import Path


def parse_termination_conditions(step_source: str) -> list[dict]:
    """Parse step.py source to extract termination conditions.

    Uses AST to find the conditions under which `terminated = True` is set.

    Returns list of dicts with:
        - condition: the condition expression as a string
        - variables: variable names referenced in the condition
        - source_text: raw text of the condition (for LLM EnvInterpreter)
    """
    conditions = []

    try:
        tree = ast.parse(step_source)
    except SyntaxError:
        # Fall back to regex for non-standard Python
        return _regex_parse_termination(step_source)

    for node in ast.walk(tree):
        # Look for `terminated = True` assignments inside if blocks
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "terminated":
                    # Check if this is inside an if block
                    pass

        # Actually: walk for If nodes that contain terminated = True
        if isinstance(node, ast.If):
            if _contains_terminated_true(node):
                try:
                    cond_text = ast.unparse(node.test)
                except Exception:
                    cond_text = _get_source_segment(step_source, node.test)

                variables = _extract_variables(node.test)
                conditions.append({
                    "condition": cond_text,
                    "variables": list(variables),
                    "source_text": cond_text,
                })

    if not conditions:
        return _regex_parse_termination(step_source)

    return conditions


def _contains_terminated_true(node: ast.If) -> bool:
    """Check if an If node's body contains `terminated = True`."""
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and target.id == "terminated":
                    # Check value is True
                    if isinstance(child.value, ast.Constant) and child.value.value is True:
                        return True
    return False


def _extract_variables(node: ast.AST) -> set[str]:
    """Extract variable names from an AST expression."""
    vars_found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            vars_found.add(child.id)
        elif isinstance(child, ast.Attribute):
            # self.hull.angle -> hull and angle
            parts = []
            obj = child
            while isinstance(obj, ast.Attribute):
                parts.append(obj.attr)
                obj = obj.value
            if isinstance(obj, ast.Name):
                parts.append(obj.id)
            vars_found.update(reversed(parts))
    return vars_found


def _get_source_segment(source: str, node: ast.AST) -> str:
    """Fallback: get source text from line numbers."""
    lines = source.splitlines()
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        start = node.lineno - 1
        end = node.end_lineno
        return "\n".join(lines[start:end]).strip()
    return str(node)


def _regex_parse_termination(step_source: str) -> list[dict]:
    """Regex-based fallback for parsing termination conditions."""
    conditions = []

    # Pattern: terminated = True, preceded by some condition
    # Look for if...terminated = True blocks
    lines = step_source.splitlines()
    in_if = False
    current_cond = ""

    for line in lines:
        stripped = line.strip()

        # Detect if-statement start
        if_match = re.match(r'if\s+(.+):', stripped)
        if if_match:
            in_if = True
            current_cond = if_match.group(1)
            continue

        # Detect terminated = True inside current if block
        if in_if and re.search(r'terminated\s*=\s*True', stripped):
            vars_found = set(re.findall(r'[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*', current_cond))
            # Filter out Python keywords
            keywords = {'if', 'and', 'or', 'not', 'True', 'False', 'None', 'self', 'in', 'is'}
            vars_found -= keywords

            conditions.append({
                "condition": current_cond,
                "variables": list(vars_found),
                "source_text": current_cond,
            })
            in_if = False
            current_cond = ""

        # Detect dedent (end of if block)
        if in_if and stripped and not stripped.startswith((" ", "\t", "#")):
            in_if = False
            current_cond = ""

    return conditions


def extract_compute_reward_signature(step_source: str) -> str:
    """Extract the argument list from self.compute_reward(...) call in step()."""
    m = re.search(r'self\.compute_reward\(([^)]+)\)', step_source)
    if m:
        return m.group(1).strip()
    return "action"


def extract_step_capture_variables(step_source: str) -> dict[str, str]:
    """Extract variables that step.py captures before calling compute_reward.

    These are the variables available to the reward function.
    Returns dict mapping variable name to type hint description.
    """
    captures = {}

    # Common patterns in eureka_llm step.py files:
    # self._last_action = action
    # self._last_pos = (pos[0], pos[1])
    # self.terminated = terminated
    patterns = [
        (r'self\._last_action\s*=\s*(\w+)', "action_array"),
        (r'self\.(\w+)\s*=\s*terminated', "termination_flag"),
    ]

    for pattern, vtype in patterns:
        for m in re.finditer(pattern, step_source):
            captures[m.group(1)] = vtype

    return captures
