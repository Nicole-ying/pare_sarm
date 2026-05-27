"""
reflection_agent.py — Post-round reflection: compares predictions with actual
outcomes and generates causal lessons for cross-round memory.

This closes the loop:
    Training → Perception → Analysis → Generation → Train → REFLECTION → Memory
"""

import json
import re
import sys
from pathlib import Path

# Ensure framework directory is on path for imports
_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm


def build_reflection_prompt(run_dir: Path, round_num: int,
                             memory_system) -> str:
    """Build reflection prompt comparing prediction vs reality."""
    template_path = Path(__file__).resolve().parent.parent.parent / "templates" / "reflection_prompt.txt"
    template = template_path.read_text("utf-8") if template_path.exists() else _fallback_reflection_prompt()

    # Load analyst proposal (prediction) — the proposal that was tested this round
    # For round N, the relevant proposal is in round(N-1)/analyzer_proposal.json
    proposal = ""
    if round_num > 0:
        proposal_path = run_dir.parent / f"round{round_num - 1}" / "analyzer_proposal.json"
        if proposal_path.exists():
            proposal = proposal_path.read_text("utf-8")

    # Load perception report (reality)
    perception = ""
    perception_path = run_dir / "perception_report.md"
    if perception_path.exists():
        perception = perception_path.read_text("utf-8")

    # Load previous round's reflection if exists
    prev_reflection = ""
    if round_num > 0:
        prev_path = run_dir.parent / f"round{round_num - 1}" / "reflection.md"
        if prev_path.exists():
            prev_reflection = prev_path.read_text("utf-8")

    # Round number placeholder
    prompt = template.replace("{round_num}", str(round_num))

    # Add context sections
    sections = [prompt]

    if prev_reflection:
        sections.append(f"\n## Previous Round Reflection\n{prev_reflection}")

    if proposal:
        sections.append(f"\n## Analyst Proposal (Prediction)\n{proposal}")

    if perception:
        sections.append(f"\n## Perception Report (Actual)\n{perception}")

    sections.append(
        "\n## Instructions\n"
        "Compare the prediction with the actual outcome.\n"
        "1. Was the analyst's diagnosis correct?\n"
        "2. Did the proposed changes have the expected effect?\n"
        "3. What was unexpected?\n"
        "4. What lesson should future rounds learn from this?\n"
        "Output a single, concise causal lesson (2-3 sentences)."
    )

    return "\n".join(sections)


def run_reflection_agent(run_dir: Path, round_num: int,
                          memory_system, api_key: str,
                          model: str = "deepseek-reasoner",
                          temperature: float = 0.3) -> str:
    """Run reflection and store lesson in memory.

    Returns:
        Reflection markdown text.
    """
    prompt = build_reflection_prompt(run_dir, round_num, memory_system)
    response = call_llm(prompt, api_key, model, temperature)

    # Save reflection
    output_path = run_dir / "reflection.md"
    output_path.write_text(response, encoding="utf-8")

    # Extract structured checklist items for the Analyst
    checklist = _extract_checklist(response)
    (run_dir / "reflection_checklist.json").write_text(
        json.dumps(checklist, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Extract causal lesson and store in cross-round memory
    lesson = _extract_lesson(response, round_num)
    if lesson:
        memory_system.add_lesson(lesson)

    return response


def _extract_lesson(reflection: str, round_num: int) -> str:
    """Extract a concise causal lesson from reflection text."""
    # Look for "What We Learned" section
    import re
    match = re.search(
        r"What We Learned\s*\n(.*?)(?=\n#|\Z)",
        reflection, re.DOTALL
    )
    if match:
        lesson = match.group(1).strip()
        return f"**Round {round_num}**: {lesson}"

    # Fallback: use first few lines
    lines = [l for l in reflection.splitlines() if l.strip() and not l.startswith("#")]
    if lines:
        return f"**Round {round_num}**: {lines[0][:200]}"

    return f"**Round {round_num}**: (reflection generated, but no extractable lesson)"


def _extract_checklist(reflection: str) -> list[dict]:
    """Extract structured checklist items from the 'For Next Round' section.

    Each item follows the format:
        - [ ] Action: <instruction>
          Rationale: <why>
          Expected impact: <metric>

    Returns list of dicts: [{"action": "...", "rationale": "...", "expected_impact": "..."}]
    """
    match = re.search(r"### For Next Round\s*\n(.*?)(?=\n#|\Z)", reflection, re.DOTALL)
    if not match:
        return []
    section = match.group(1)
    items = []
    # Split on checklist markers
    blocks = re.split(r'(?=^\s*-\s*\[\s*[ xX]?\s*\]\s+Action:)', section, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        action_m = re.search(r"Action:\s*(.+?)(?:\n|$)", block)
        rationale_m = re.search(r"Rationale:\s*(.+?)(?:\n|$)", block)
        impact_m = re.search(r"Expected impact:\s*(.+?)(?:\n|$)", block)
        if action_m:
            items.append({
                "action": action_m.group(1).strip(),
                "rationale": rationale_m.group(1).strip() if rationale_m else "",
                "expected_impact": impact_m.group(1).strip() if impact_m else "",
            })
    return items


def _fallback_reflection_prompt() -> str:
    """Fallback prompt if template file is missing."""
    return """# Round {round_num} Reflection

Compare the analyst's predictions with the actual training outcome.

## What We Learned
(Write a single causal lesson: what changed, what happened, why)
"""
