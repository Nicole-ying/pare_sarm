"""
generator_agent.py — Generator with optional ReAct verification.

Architecture (when evidence_citations are present):
    Phase 1 — Verification: ReAct loop. Generator verifies each claim in the
               proposal by reading cited sources and querying memory.
               Output: ACCEPT or REJECT with reasoning.

    Phase 2 — Code generation: Existing single-call LLM flow. Receives proposal
               and current code, outputs updated Python module.

    When evidence_citations are absent (backward compat), Phase 1 is skipped
    and the original single-call flow runs as-is.

    Retry logic: up to 3 attempts with syntax validation.
    Fallback: returns None → pipeline uses previous round's code.
"""

import json
import re
import sys
from pathlib import Path

_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm
from react_agent import setup_default_tools, run_react_loop


def _build_system_prompt() -> str:
    """Build generator system prompt from shared checks + generator-specific instructions.

    Teaches methodology (how to verify your own work) rather than listing rules.
    """
    from shared_rules import render_rules, GENERATOR_RULES
    checks = render_rules(GENERATOR_RULES)
    return f'''You are the Generator Agent. Your job: take the Analyzer\'s proposed changes and apply them to the reward function code precisely.

Your working method: read each change, apply it one at a time, then verify the complete output.

== Step-by-Step Working Method ==

### Step 1: Parse Each Change
For each proposed change, identify:
- What EXACT lines in the current code need to be replaced?
- What is the replacement code?
- Does the replacement preserve the function\'s overall structure?

### Step 2: Apply Changes (One at a Time)
Apply each change to the current reward code. After each change, verify:
- Did the edit land correctly? (no partial replacements, no doubled lines)
- Are unchanged parts preserved exactly?

### Step 3: Verify the Complete Output
Before finalizing, walk through the ENTIRE output:

- [ ] Is def compute_reward(...) present with the EXACT same parameter list?
- [ ] Does it return (float, dict)?
- [ ] Is _outcome handled correctly (not added to total)?
- [ ] Are both functions COMPLETE (not abbreviated, not "..." sections)?

== Self-Verification Checks ==

Run through these checks on your output:

{checks}

== Output Format ==

Output ONLY a single ```python block. No explanation before or after.
The block must contain the complete compute_reward function.

IMPORTANT — start directly from `def compute_reward(...)`. Do NOT include:
  - A docstring header (`"""LLM-generated reward function..."""`)
  - `import` statements (the framework adds them automatically)
  - Any text before `def compute_reward`
Starting with anything other than `def compute_reward` will cause code duplication.'''


GENERATOR_SYSTEM_PROMPT = _build_system_prompt()


def build_generator_prompt(proposal: dict, current_code: str) -> str:
    """Build a focused code-generation prompt from proposal + current code."""
    sections = [GENERATOR_SYSTEM_PROMPT, ""]

    # Proposal summary
    diagnosis = proposal.get("diagnosis", "No diagnosis provided.")
    sections.append("## Diagnosis")
    sections.append(diagnosis)
    sections.append("")

    changes = proposal.get("proposed_changes", [])
    if changes:
        sections.append("## Proposed Changes")
        for i, change in enumerate(changes, 1):
            component = change.get("component", "unknown")
            new_code = change.get("new_code", "")
            reason = change.get("reason", "")
            sections.append(f"### Change {i}: {component}")
            if new_code:
                sections.append(f"```python\n{new_code}\n```")
            if reason:
                sections.append(f"Reason: {reason}")
            sections.append("")
    else:
        sections.append("## No Changes Proposed")
        sections.append("Output the current code as-is with no modifications.")
        sections.append("")

    # Current code
    sections.append("## Current Reward Function Code")
    sections.append("```python")
    sections.append(current_code)
    sections.append("```")
    sections.append("")

    # Extract and inject exact parameter signature from current code
    sig_match = __import__("re").search(
        r"def compute_reward\(([^)]+)\)",
        current_code
    )
    if sig_match:
        params = sig_match.group(1)
        sections.append("## CRITICAL: Preserve This Function Signature")
        sections.append(
            f'Your output MUST use `def compute_reward({params}):` — '
            f"exactly as shown above. No added or removed parameters. "
            f"The environment calls this function with a fixed argument list. "
            f"Any mismatch will crash training on the first step."
        )
        sections.append("")

    # Output instruction
    sections.append("---")
    sections.append("Output ONLY a single ```python block with the COMPLETE updated module.")
    sections.append("Do NOT include any explanation before or after the code block.")

    return "\n".join(sections)


# ── Phase 1: Verification ─────────────────────────────────────────────────

def _build_verification_prompt(proposal: dict) -> str:
    """Build a verification prompt for the ReAct loop.

    Asks the Generator to verify each evidence citation from the proposal
    by reading the cited files and querying cross-round memory.
    """
    evidence = proposal.get("evidence_citations", [])
    sections = [
        "You are a **second opinion**, not a typist. The Meta-Analyzer has proposed",
        "changes to the reward function and provided evidence citations. Your job is",
        "NOT to blindly apply changes — it is to VERIFY whether the proposal's claims",
        "are actually supported by the experiment records.",
        "",
        "A good second opinion is valuable whether it agrees OR disagrees:",
        "- If the evidence checks out → VERIFICATION: ACCEPT, then write the code.",
        "- If the evidence is wrong or incomplete → VERIFICATION: REJECT,",
        "  explain what's wrong. Rejecting a bad proposal prevents wasting a",
        "  training cycle on a flawed idea.",
        "",
        "You have tools available to read files and search cross-round memory.",
        "For each evidence citation, use read_file to check the cited source",
        "directly. Do NOT take the citation's detail text at face value —",
        "go read the actual file yourself.",
        "Use query_memory to check if similar approaches have been tried before",
        "and what happened."
        "",
        "=== Proposal Diagnosis ===",
        proposal.get("diagnosis", ""),
        "",
        "=== Evidence Citations ===",
    ]
    for i, cit in enumerate(evidence, 1):
        sections.append(f"{i}. Claim: {cit.get('claim', '')}")
        sections.append(f"   Source: {cit.get('source', '')}")
        sections.append(f"   Detail: {cit.get('detail', '')}")
        sections.append("")

    sections.extend([
        "After verifying all claims, output EXACTLY one of these lines:",
        "",
        'VERIFICATION: ACCEPT',
        "  (all key claims are supported by the cited evidence)",
        "",
        'VERIFICATION: REJECT',
        "  (one or more claims are inconsistent with what the actual files show)",
        "",
        "Then briefly explain which claims passed and which (if any) failed.",
    ])
    return "\n".join(sections)


def _run_verification(
    proposal: dict,
    experiment_dir: Path,
    api_key: str,
    model: str,
    temperature: float,
    memory_system=None,
) -> dict:
    """Run the ReAct verification loop.

    Returns:
        {"action": "accept", "reason": "..."}
        or
        {"action": "reject", "reason": "..."}
    """
    evidence = proposal.get("evidence_citations", [])
    if not evidence:
        return {"action": "accept", "reason": "No evidence citations to verify."}

    prompt = _build_verification_prompt(proposal)
    tools = setup_default_tools(experiment_dir, memory_system)

    result = run_react_loop(
        system_prompt=prompt,
        tools=tools,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_steps=8,
        max_idle=2,
        log_fn=lambda msg: print(f"  [Generator-Verify] {msg}"),
    )

    final = result.get("final_output", "")

    if "VERIFICATION: REJECT" in final.upper():
        idx = final.upper().find("VERIFICATION: REJECT")
        reason = final[idx + len("VERIFICATION: REJECT"):].strip()[:400]
        return {"action": "reject", "reason": reason}

    if "VERIFICATION: ACCEPT" in final.upper():
        return {"action": "accept", "reason": "All evidence citations verified."}

    # Fallback: if parsing fails, accept with a warning
    print("  [Generator-Verify] Could not parse verdict — accepting as fallback")
    return {"action": "accept", "reason": "Verification inconclusive — proceeding."}


# ── Phase 2: Code generation helpers ──────────────────────────────────────


def _extract_code_from_response(text: str) -> str | None:
    """Extract Python code block from LLM response."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def run_generator_agent(proposal: dict, current_reward_path: Path,
                        run_dir: Path, api_key: str,
                        model: str = "deepseek-reasoner",
                        temperature: float = 0.3,
                        max_retries: int = 3,
                        memory_system=None) -> str | None:
    """Run the generator agent: optional verification + code generation.

    Phase 1 (if evidence_citations exist): ReAct verification loop.
        Verifies the proposal's evidence claims by reading actual files.
        If claims are rejected, saves the rejection and returns None.

    Phase 2: Code generation with retry logic (existing flow).
        Translates verified proposal into updated reward function code.

    Args:
        proposal: Analyzer/Meta-Analyzer's JSON proposal.
        current_reward_path: Path to current reward function source.
        run_dir: Output directory (for saving artifacts).
        api_key: LLM API key.
        model: LLM model name.
        temperature: Sampling temperature.
        max_retries: Max code generation attempts.
        memory_system: Optional MemorySystem for knowledge base queries.

    Returns:
        Generated code string, or None if verification rejected or all retries failed.
    """
    if not current_reward_path.exists():
        print(f"  [Generator] No current reward code found at {current_reward_path}")
        return None

    # ── Phase 1: Verification (if proposal has evidence citations) ──
    evidence_citations = proposal.get("evidence_citations", [])
    if evidence_citations:
        experiment_dir = run_dir.parent
        print(f"  [Generator] Verifying {len(evidence_citations)} evidence citations...")
        ver = _run_verification(
            proposal, experiment_dir, api_key, model, temperature, memory_system,
        )
        (run_dir / "generator_verification.json").write_text(
            json.dumps(ver, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if ver.get("action") == "reject":
            reason = ver.get("reason", "")[:200]
            print(f"  [Generator] Proposal REJECTED: {reason}")
            return None
        print(f"  [Generator] Proposal ACCEPTED after verification")
    else:
        print(f"  [Generator] No evidence citations — skipping verification")

    # ── Phase 2: Code generation (existing flow) ──
    current_code = current_reward_path.read_text("utf-8")

    for attempt in range(1, max_retries + 1):
        prompt = build_generator_prompt(proposal, current_code)
        if attempt == 1:
            (run_dir / "generator_prompt.txt").write_text(prompt, encoding="utf-8")

        print(f"  [Generator] Calling LLM (attempt {attempt}/{max_retries}) ...")
        try:
            response = call_llm(prompt, api_key, model, temperature)
        except Exception as e:
            print(f"  [Generator] LLM call failed: {e}")
            continue

        code = _extract_code_from_response(response)
        if not code:
            print(f"  [Generator] No code block in response (attempt {attempt})")
            continue

        # Strip leading/trailing whitespace
        code = code.strip()

        # Validate: must have compute_reward
        if "def compute_reward" not in code:
            print(f"  [Generator] Missing compute_reward function (attempt {attempt})")
            continue

        # Syntax check
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            print(f"  [Generator] Syntax error: {e} (attempt {attempt})")
            continue

        # Success
        print(f"  [Generator] Code generated ({len(code)} chars)")
        (run_dir / "generator_response.txt").write_text(response, encoding="utf-8")
        return code

    print(f"  [Generator] All {max_retries} attempts failed. Falling back to previous round's code.")
    return None
