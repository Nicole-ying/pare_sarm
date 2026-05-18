"""
Implementor Agent (redesigned — structured edits with auto-indent fix).

LLM only generates replacement code. Framework handles find+replace+indent.
"""
import json, re, sys
from pathlib import Path

_mr = Path(__file__).resolve().parent.parent
if str(_mr) not in sys.path: sys.path.insert(0, str(_mr))

from infra.llm_client import call_llm, parse_json_response
from infra.file_utils import load_text


class Implementor:
    """Plans structured edit operations. Does NOT generate full files."""

    def __init__(self, api_key=None, model="deepseek-chat", temperature=0.15):
        self.api_key = api_key; self.model = model; self.temperature = temperature

    def plan_edits(self, diagnosis, target_code, retry_hint=None):
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        system = load_text(prompts_dir / "implementor_system.txt")
        changes = diagnosis.get("proposed_changes", [])

        code_context = ""
        for c in changes:
            current = c.get("current_code", "")
            if current:
                found = False
                for i, line in enumerate(target_code.splitlines()):
                    if _norm_ws(current) in _norm_ws(line) or _norm_ws(line) in _norm_ws(current):
                        lo = max(0, i - 2)
                        hi = min(len(target_code.splitlines()), i + 4)
                        code_context += (
                            f"\nLines {lo+1}-{hi+1} around '{c.get('component','?')}':\n"
                            + "\n".join(f"  {j+1}: {target_code.splitlines()[j]}" for j in range(lo, hi))
                            + "\n"
                        )
                        found = True; break
                if not found:
                    code_context += f"\n[WARN: current_code not found for '{c.get('component','?')}']\n"

        prompt = f"""{system}

## Changes
```json
{json.dumps(changes, indent=2, ensure_ascii=False)}
```

## Code Context
{code_context if code_context else target_code[:1500]}

{f"## Fix These Errors{chr(10)}{retry_hint}" if retry_hint else ""}

Output JSON edits array."""

        print(f"  [Implementor] Planning edits ({len(prompt)} chars)")
        resp = call_llm(prompt, self.api_key, self.model, self.temperature)
        result = parse_json_response(resp)
        if "_parse_error" in result:
            resp = call_llm(prompt + "\nOutput ONLY valid JSON.", self.api_key, self.model, self.temperature - 0.05)
            result = parse_json_response(resp)

        edits = result.get("edits", [])
        for i, edit in enumerate(edits):
            if i < len(changes) and "current_code" not in edit:
                edit["current_code"] = changes[i].get("current_code", "")
        return result


def _norm_ws(s):
    return ' '.join(s.split())

def _detect_indent(line):
    return line[:len(line) - len(line.lstrip())]

def _fix_indent(code, base_indent):
    lines = code.splitlines()
    if len(lines) <= 1:
        return base_indent + code.lstrip()
    min_ind = min((len(_detect_indent(l)) for l in lines if l.strip()), default=0)
    fixed = []
    for l in lines:
        if not l.strip():
            fixed.append('')
        else:
            rel = len(_detect_indent(l)) - min_ind
            fixed.append(base_indent + ' ' * max(0, rel) + l.lstrip())
    return '\n'.join(fixed)


def _find_and_replace(code, old, new, insert_before="", insert_after=""):
    """Find old in code, replace with new. Auto-fixes indentation. Returns (code, found)."""
    old_norm = _norm_ws(old); lines = code.splitlines(); ol = old.strip().splitlines()

    ms = None; me = None; bi = ""
    # Exact
    if old in code:
        idx = code.index(old); pre = code[:idx].splitlines()
        ms = len(pre) - 1 if pre else 0; me = ms + len(ol)
        bi = _detect_indent(lines[ms]) if ms < len(lines) else "    "
    # Normalized single-line
    if ms is None:
        for i, l in enumerate(lines):
            if _norm_ws(l) == old_norm: ms = i; me = i + 1; bi = _detect_indent(l); break
    # First-token + multi-line block
    old_first = old_norm.split()[0] if old_norm.split() else ""
    if ms is None and old_first:
        for i, l in enumerate(lines):
            if _norm_ws(l).startswith(old_first):
                if len(ol) > 1:
                    ok = all(j < len(lines) - i and _norm_ws(lines[i+j]) == _norm_ws(ol[j]) for j in range(len(ol)))
                    if ok: ms = i; me = i + len(ol); bi = _detect_indent(l)
                else: ms = i; me = i + 1; bi = _detect_indent(l)
                if ms is not None: break
    if ms is None: return code, False

    rep = _fix_indent(new, bi) if bi else new
    if insert_before: rep = _fix_indent(insert_before, bi) + "\n" + rep
    if insert_after: rep = rep + "\n" + _fix_indent(insert_after, bi)

    lines[ms:me] = [rep]
    return "\n".join(lines), True


def apply_edits(code, edit_plan):
    edits = edit_plan.get("edits", [])
    edit_plan["total_count"] = len(edits)
    if not edits: edit_plan["applied_count"] = 0; return code
    modified = code; applied = 0
    for i, edit in enumerate(edits):
        old_code = edit.get("current_code", edit.get("find_pattern", ""))
        new_code = edit.get("new_code", edit.get("replacement", ""))
        ib = edit.get("insert_before", ""); ia = edit.get("insert_after", "")
        if not old_code: continue
        modified, ok = _find_and_replace(modified, old_code, new_code, ib, ia)
        if ok: applied += 1
        else: print(f"  [apply_edits] Edit {i}: NOT FOUND: '{old_code.strip()[:80]}'")
    edit_plan["applied_count"] = applied
    print(f"  [apply_edits] {applied}/{len(edits)} edits applied ({applied/max(len(edits),1):.0%})")
    return modified


def implement(api_key, model, temperature, diagnosis, current_code, task_manifest, retry_hint=None):
    impl = Implementor(api_key=api_key, model=model, temperature=temperature)
    plan = impl.plan_edits(diagnosis, current_code, retry_hint)
    return apply_edits(current_code, plan)
