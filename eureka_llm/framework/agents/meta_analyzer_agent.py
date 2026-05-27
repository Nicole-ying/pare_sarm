"""
meta_analyzer_agent.py — Cross-round researcher that replaces Analyzer + Reflection.

Architecture:
    Single LLM call with ALL prior rounds' data pre-loaded into context.
    The Meta-Analyzer reads the full evolutionary arc (not just round N-1),
    produces a JSON proposal (compatible with Generator), and generates
    reflection updates for cross-round memory.

    Philosophy (from user discussion):
        "Train a researcher, not a repairman."
        No prescribed thinking steps — the agent decides how to reason.
        The prompt provides: (1) all available data, (2) a clear output contract.
        How to think is the agent's job.

    Replaces:
        - analyzer_agent.py (proposal generation)
        - reflection_agent.py (lesson extraction)
        - distill.py (pattern extraction)

    Old agents' prompts are preserved under templates/ for rollback.
"""

import json
import re
from pathlib import Path
import sys

_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_round_data(round_dir: Path, round_num: int) -> dict:
    """Load all available artifacts from a single round directory.

    Returns dict with any of these keys that exist:
        reward_code, perception_report, proposal, reflection, eval_history, reflection_checklist
    """
    result = {"round": round_num}
    path_map = {
        "reward_code": round_dir / "reward_fn_source.py",
        "perception_report": round_dir / "perception_report.md",
        "proposal": round_dir / "analyzer_proposal.json",
        "reflection": round_dir / "reflection.md",
        "reflection_checklist": round_dir / "reflection_checklist.json",
    }
    for key, path in path_map.items():
        if path.exists():
            try:
                result[key] = path.read_text("utf-8")
            except Exception:
                result[key] = None

    # Evaluation history (truncated: keep last 5 rows for compactness)
    eval_path = round_dir / "evaluations" / "history.csv"
    if eval_path.exists():
        try:
            lines = eval_path.read_text("utf-8").strip().split("\n")
            if len(lines) > 6:
                result["eval_summary"] = "\n".join(lines[:1] + lines[-5:])
            else:
                result["eval_summary"] = "\n".join(lines)
        except Exception:
            result["eval_summary"] = None

    return result


def _summarize_proposal_what(proposal_text: str | None) -> str:
    """Short one-line summary of what a proposal changed."""
    if not proposal_text:
        return "(no proposal)"
    try:
        proposal = json.loads(proposal_text)
        changes = proposal.get("proposed_changes", [])
        if not changes:
            return "No changes"
        names = [c.get("component", "?")[:40] for c in changes if c.get("component")]
        return "; ".join(names[:3])
    except (json.JSONDecodeError, TypeError):
        return "(parse error)"


def _summarize_proposal_expected(proposal_text: str | None) -> str:
    """Extract expected outcome from proposal."""
    if not proposal_text:
        return ""
    try:
        proposal = json.loads(proposal_text)
        diag = proposal.get("diagnosis", "")
        first = diag.split(".")[0] if "." in diag else diag
        return first[:80] + ("..." if len(first) > 80 else "")
    except (json.JSONDecodeError, TypeError):
        return ""


def _summarize_perception(perception_text: str | None) -> str:
    """Short behavior summary from perception report."""
    if not perception_text:
        return "(no perception)"
    # Look for a behavior/trend section
    m = re.search(
        r"(?:Behavior Trend|Behavior Summary|Summary)[:\s]*(.*?)(?:\n\n|\n#)",
        perception_text, re.DOTALL | re.IGNORECASE
    )
    if m:
        summary = m.group(1).strip()[:100]
        return summary + ("..." if len(summary) >= 100 else "")
    # Fallback: use first non-header line
    for line in perception_text.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and len(s) > 20:
            return s[:100] + ("..." if len(s) >= 100 else "")
    return "(report exists)"


def _summarize_metrics(round_data: dict) -> str:
    """Extract key metrics: mean episode length, success rate if available."""
    parts = []
    m = m1 = None
    for field in ("eval_summary", "perception_report"):
        text = round_data.get(field)
        if not text:
            continue
        # Mean episode length
        match = re.search(r"mean[\s_]*(?:episode[\s_]*)?len(?:gth)?[:\s]*([\d.]+)",
                          text, re.IGNORECASE)
        if match and m is None:
            m = match.group(1)
        # Success rate
        match = re.search(r"(?:success|completion)\s*rate[:\s]*([\d.]+)",
                          text, re.IGNORECASE)
        if match and m1 is None:
            m1 = match.group(1)
    if m:
        parts.append(f"len={m}")
    if m1:
        parts.append(f"success={m1}")
    return ", ".join(parts) if parts else ""


def _build_round_arc_history(exp_dir: Path, max_round: int) -> list[dict]:
    """Load all prior rounds' data and build a compact per-round table.

    Returns:
        List of round dicts (reverse order), each with structured summaries.
    """
    rounds = []
    for r in range(max_round + 1):
        round_dir = exp_dir / f"round{r}"
        if not round_dir.exists():
            continue
        data = _load_round_data(round_dir, r)
        rounds.append(data)
    return rounds


def _build_cross_round_table(rounds: list[dict]) -> str:
    """Build markdown table: Round | What Changed | Expected | Actual | Metrics."""
    if not rounds:
        return "*(no prior rounds)*"
    lines = [
        "| Round | What Changed | Expected Outcome | Actual Behavior | Key Metrics |",
        "|-------|-------------|-----------------|----------------|-------------|",
    ]
    for rd in rounds:
        r = rd["round"]
        what = _summarize_proposal_what(rd.get("proposal"))
        expected = _summarize_proposal_expected(rd.get("proposal"))
        actual = _summarize_perception(rd.get("perception_report"))
        metrics = _summarize_metrics(rd)
        lines.append(f"| Round {r} | {what} | {expected} | {actual} | {metrics} |")
    return "\n".join(lines)


def _build_eval_comparison_table(rounds: list[dict]) -> str:
    """Build comparison of eval metrics across rounds for key indicators.

    If component_means are available in perception reports, shows how mean
    episode length and success rate evolved across rounds.
    """
    lines = ["### Evaluation Comparison Across Rounds", ""]
    # Collect mean_length from perception reports
    rows = []
    for rd in rounds:
        r = rd["round"]
        m = None
        for text in (rd.get("perception_report"), rd.get("eval_summary")):
            if not text:
                continue
            match = re.search(r"mean[\s_]*(?:episode[\s_]*)?len(?:gth)?[:\s]*([\d.]+)",
                              text, re.IGNORECASE)
            if match:
                m = match.group(1)
                break
        rows.append((r, m))
    if rows:
        lines.append("| Round | Mean Episode Length |")
        lines.append("|-------|-------------------|")
        for r, m in rows:
            lines.append(f"| Round {r} | {m or '—'} |")
    else:
        lines.append("*(comparison data not available)*")
    return "\n".join(lines)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_meta_analyzer_prompt(
    run_dir: Path, round_num: int, memory_system,
    skill_manager=None,
) -> str:
    """Build the complete Meta-Analyzer prompt with all cross-round data pre-loaded."""
    exp_dir = run_dir.parent
    max_round = round_num - 1  # the last completed round (not including current)

    # Load template
    template_path = (
        Path(__file__).resolve().parent.parent.parent
        / "templates" / "meta_analyzer_prompt.txt"
    )
    template = template_path.read_text("utf-8") if template_path.exists() else _fallback_prompt()

    # Load all prior rounds' data
    all_rounds = _build_round_arc_history(exp_dir, max_round)

    # Build the per-round history section
    per_round_sections = []
    for rd in all_rounds:
        r = rd["round"]
        parts = [f"### Round {r}"]
        # Reward code snippet (first/last 15 lines)
        code = rd.get("reward_code", "")
        if code:
            code_lines = code.strip().split("\n")
            snippet = "\n".join(code_lines[:15])
            if len(code_lines) > 30:
                snippet += f"\n  ... ({len(code_lines) - 30} lines omitted) ...\n"
                snippet += "\n".join(code_lines[-15:])
            parts.append(f"```python\n{snippet}\n```")

        # Perception report summary
        perception = rd.get("perception_report", "")
        if perception:
            # Truncate to ~500 chars
            summary = perception[:600]
            if len(perception) > 600:
                summary += "\n... (truncated)"
            parts.append(f"**Perception Summary:**\n{summary}")

        # Proposal summary
        proposal = rd.get("proposal")
        if proposal:
            try:
                p = json.loads(proposal)
                diag = p.get("diagnosis", "(no diagnosis)")[:300]
                parts.append(f"**Proposal Diagnosis:** {diag}")
                changes = p.get("proposed_changes", [])
                if changes:
                    for c in changes:
                        comp = c.get("component", "?")
                        reason = c.get("reason", "")[:150]
                        parts.append(f"  - Changed `{comp}`: {reason}")
            except json.JSONDecodeError:
                pass

        # Reflection
        reflection = rd.get("reflection", "")
        if reflection:
            # Extract What We Learned
            m = re.search(r"What We Learned[:\s]*(.*?)(?:\n\n|\n#|\Z)",
                          reflection, re.DOTALL)
            if m:
                parts.append(f"**Lesson:** {m.group(1).strip()[:200]}")

        per_round_sections.append("\n\n".join(parts))

    per_round_history = "\n\n---\n\n".join(per_round_sections) if per_round_sections else "*(no previous rounds)*"

    # Cross-round comparison table
    cross_round_table = _build_cross_round_table(all_rounds)

    # Eval comparison
    eval_comparison = _build_eval_comparison_table(all_rounds)

    # Current perception report
    current_perception = ""
    perception_path = run_dir / "perception_report.md"
    if perception_path.exists():
        current_perception = perception_path.read_text("utf-8")

    # Task manifest
    task_manifest = memory_system.get_task_manifest() if memory_system else ""

    # Memory
    memory_text = ""
    if memory_system and memory_system.memory_dir:
        mem_path = memory_system.memory_dir / "MEMORY.md"
        if mem_path.exists():
            memory_text = mem_path.read_text("utf-8")

    # Active skills
    skill_text = ""
    if skill_manager:
        skill_text = skill_manager.active_docs

    # Fill placeholders — not a simple replace; we build the sections explicitly
    sections = [template]

    sections.append("\n\n=== BEGIN DATA ===\n")

    if task_manifest:
        sections.append(f"### 1. Task Manifest\n{task_manifest}\n")

    if memory_text:
        sections.append(f"### 2. Cross-Round Memory (Accumulated Lessons)\n{memory_text}\n")

    sections.append(f"### 3. Per-Round History\n{per_round_history}\n")

    sections.append(f"### Cross-Round Comparison\n{cross_round_table}\n")

    sections.append(f"### Evaluation Comparison\n{eval_comparison}\n")

    if current_perception:
        sections.append(f"### 4. Current Perception Report\n{current_perception}\n")

    if skill_text:
        sections.append(f"### 6. Relevant Design Techniques\n{skill_text}\n")

    sections.append("=== END DATA ===\n")

    sections.append(
        "Begin your analysis. Remember: you are a cross-round researcher, not a repairman."
    )

    return "\n".join(sections)


def _fallback_prompt() -> str:
    """Fallback if template file is missing."""
    return """You are the Meta-Analyzer — a cross-round reward design researcher.

Your data is provided in the sections above. Analyze the full evolutionary arc
across all rounds, then produce:

### Analysis
(Free-form reasoning)

FINAL ANSWER

```json
{
  "diagnosis": "...",
  "changed_count": 0,
  "proposed_changes": []
}
```

### Reflection
What We Learned: ...
Abstract Principle: ...
For Next Round:
- [ ] Action: ...
  Rationale: ...
  Expected impact: ...
```
"""
# ── Output extraction ─────────────────────────────────────────────────────────

def _extract_json_proposal(text: str) -> dict | None:
    """Extract JSON proposal from LLM response.

    Searches for ```json blocks, then falls back to walking braces.
    """
    # Strategy 1: ```json ... ``` blocks in the response
    json_blocks = re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    for m in json_blocks:
        candidate = _try_parse_json(m.group(1).strip())
        if candidate and candidate.get("diagnosis") and candidate.get("proposed_changes") is not None:
            return candidate

    # Strategy 2: Look for Proposal section with inline JSON
    m = re.search(
        r"Proposal.*?(\{(?:[^{}]|\"(?:\\.|[^\"\\])*\")*\})",
        text, re.DOTALL
    )
    if m:
        candidate = _try_parse_json(m.group(1))
        if candidate and candidate.get("diagnosis") and candidate.get("proposed_changes") is not None:
            return candidate

    # Strategy 3: Fallback — walk braces
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = _try_parse_json(text[start:i+1])
                if candidate and candidate.get("diagnosis") and candidate.get("proposed_changes") is not None:
                    return candidate
                start = -1

    return None


def _try_parse_json(s: str) -> dict | None:
    """Try to parse JSON, with common LLM error recovery."""
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fix trailing commas
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return None


def _extract_reflection_section(text: str) -> str | None:
    """Extract the Reflection section from LLM response.

    Looks for the ### Reflection or Reflection section header
    and returns everything from that header onward.
    """
    m = re.search(
        r"#{1,3}\s*Reflection\s*\n(.*?)(?=\Z)",
        text, re.DOTALL
    )
    if m:
        return "## Reflection\n" + m.group(1).strip()

    # Fallback: look for "What We Learned"
    m = re.search(
        r"What We Learned[:\s]*(.*?)(?=\n\n|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        return f"## Reflection\n\nWhat We Learned: {m.group(1).strip()}"

    return None


def _extract_lesson(reflection_text: str, round_num: int) -> str:
    """Extract a compact causal lesson from reflection text for MEMORY.md."""
    m = re.search(
        r"What We Learned[:\s]*(.*?)(?:\n\s*\n|\n#|\Z)",
        reflection_text, re.DOTALL | re.IGNORECASE
    )
    if m:
        lesson = m.group(1).strip()
        return f"**Round {round_num}**: {lesson}"
    return f"**Round {round_num}**: (reflection generated, but no extractable lesson)"


def _extract_checklist(reflection_text: str) -> list[dict]:
    """Extract structured checklist items from the For Next Round section."""
    m = re.search(
        r"For Next Round[:\s]*\n(.*?)(?=\n#|\Z)",
        reflection_text, re.DOTALL | re.IGNORECASE
    )
    if not m:
        return []
    section = m.group(1)
    items = []
    blocks = re.split(r'(?=^\s*-\s*\[\s*[ xX]?\s*\]\s+Action:)',
                      section, flags=re.MULTILINE)
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


# ── Main entry point ──────────────────────────────────────────────────────────

def run_meta_analyzer_agent(
    run_dir: Path, round_num: int,
    memory_system, api_key: str,
    model: str = "deepseek-reasoner",
    temperature: float = 0.4,
    skill_manager=None,
) -> dict:
    """Run Meta-Analyzer: single LLM call with cross-round context.

    Loads ALL prior rounds' data, builds a comprehensive prompt,
    calls the LLM once, and produces:
      1. meta_analyzer_proposal.json (compatible with Generator)
      2. meta_analysis.md (full reasoning output with reflection)

    Args:
        run_dir: Current round directory (round{round_num}/).
        round_num: Current round number.
        memory_system: For task manifest and cross-round memory access.
        api_key: LLM API key.
        model: Model name.
        temperature: Sampling temperature.
        skill_manager: Optional skill catalog for technique injection.

    Returns:
        dict with "proposal" key (compatible with Generator).
    """
    output_dir = run_dir  # the current round's directory
    prev_round_dir = run_dir.parent / f"round{round_num - 1}"

    # Detect if this is LiteAnalyzer fallback mode: no prior rounds at all
    all_rounds = memory_system.get_available_rounds() if memory_system else []
    prior_exist = [r for r in all_rounds if r < round_num]

    if not prior_exist:
        # Fallback: no prior rounds data — use Analyzer-style single-round analysis
        print("  [Meta-Analyzer] No prior rounds found. Falling back to single-round analysis.")
        from agents.analyzer_agent import run_analyzer_agent
        result = run_analyzer_agent(
            prev_round_dir, round_num, memory_system,
            api_key, model, temperature, skill_manager,
        )
        # Still save with meta_analyzer prefix for consistency
        _save_meta_output(output_dir, result.get("proposal", {}), "")
        return result

    # Build prompt with cross-round data
    prompt = _build_meta_analyzer_prompt(
        prev_round_dir, round_num, memory_system, skill_manager,
    )

    # Save prompt for debugging
    prompt_path = output_dir / "meta_analyzer_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"  [Meta-Analyzer] Prompt → {prompt_path}")

    # Single LLM call
    print(f"  [Meta-Analyzer] Starting cross-round analysis (rounds 0..{round_num - 1})...")
    try:
        response = call_llm(prompt, api_key, model, temperature)
        response_path = output_dir / "meta_analyzer_response.txt"
        response_path.write_text(response, encoding="utf-8")
    except Exception as e:
        print(f"  [Meta-Analyzer] LLM call failed: {e}")
        _save_fallback(output_dir, f"Meta-Analyzer LLM call failed: {e}")
        return {"proposal": _fallback_proposal(f"LLM call failed: {e}")}

    # Extract JSON proposal
    proposal = _extract_json_proposal(response)

    if not proposal:
        print(f"  [Meta-Analyzer] No valid JSON found. Retrying with structured prompt...")
        retry_prompt = (
            f"Your previous analysis did not produce a valid JSON proposal.\n\n"
            f"Output ONLY a FINAL ANSWER with a valid JSON block. "
            f"Use exactly this format with double quotes and no trailing commas:\n"
            f"```json\n{{\n"
            f'  "diagnosis": "Brief root cause",\n'
            f'  "changed_count": 0,\n'
            f'  "proposed_changes": []\n'
            f"}}\n```\n"
            f"No text before or after the code block."
        )
        try:
            retry_response = call_llm(retry_prompt, api_key, model, temperature)
            proposal = _extract_json_proposal(retry_response)
            if proposal:
                print(f"  [Meta-Analyzer] Retry succeeded!")
        except Exception as e:
            print(f"  [Meta-Analyzer] Retry failed: {e}")

    if not proposal:
        print(f"  [Meta-Analyzer] No valid JSON extracted. Using fallback.")
        proposal = _fallback_proposal("Failed to extract JSON proposal.")
        proposal["analysis_status"] = "failed"
    else:
        proposal["analysis_status"] = "ok"

    # Normalize evidence_citations (backward compat)
    proposal.setdefault("evidence_citations", [])
    n_cit = len(proposal["evidence_citations"])
    if n_cit:
        print(f"  [Meta-Analyzer] {n_cit} evidence citations included")

    # Enforce max 3 changes
    if proposal.get("changed_count", 0) > 3:
        changes = proposal.get("proposed_changes", [])
        proposal["changed_count"] = min(len(changes), 3)
        proposal["proposed_changes"] = changes[:3]

    # Save proposal
    proposal_path = output_dir / "meta_analyzer_proposal.json"
    proposal_path.write_text(
        json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Extract and save reflection
    reflection_text = _extract_reflection_section(response)
    if reflection_text:
        (output_dir / "meta_reflection.md").write_text(reflection_text, encoding="utf-8")
        # Also save the standard reflection.md for backward compatibility
        (output_dir / "reflection.md").write_text(reflection_text, encoding="utf-8")

        # Extract checklist
        checklist = _extract_checklist(reflection_text)
        (output_dir / "reflection_checklist.json").write_text(
            json.dumps(checklist, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Update cross-round memory
        lesson = _extract_lesson(reflection_text, round_num)
        if lesson and memory_system:
            memory_system.add_lesson(lesson)
    else:
        print(f"  [Meta-Analyzer] No reflection section extracted.")
        # Generate a minimal reflection from the analysis
        minimal_reflection = _generate_minimal_reflection(response, proposal, round_num)
        (output_dir / "meta_reflection.md").write_text(minimal_reflection, encoding="utf-8")

    # Update belief state
    if memory_system:
        memory_system.update_belief("meta_analyzer", {
            "round": round_num,
            "diagnosis": proposal.get("diagnosis", "")[:200],
            "changed_count": proposal.get("changed_count", 0),
        })

    print(f"  [Meta-Analyzer] Proposal: {proposal.get('diagnosis', 'N/A')[:120]}")
    print(f"  [Meta-Analyzer] Changes: {proposal.get('changed_count', 0)}")
    return {"proposal": proposal}


def _save_meta_output(output_dir: Path, proposal: dict, response: str):
    """Save Meta-Analyzer outputs for the fallback case."""
    (output_dir / "meta_analyzer_proposal.json").write_text(
        json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _save_fallback(output_dir: Path, reason: str):
    """Save a fallback note when Meta-Analyzer fails."""
    (output_dir / "META_ANALYZER_ERROR").write_text(reason, encoding="utf-8")


def _fallback_proposal(reason: str = "") -> dict:
    """Return a zero-change fallback proposal."""
    return {
        "diagnosis": f"Meta-Analyzer failed. {reason}".strip(),
        "changed_count": 0,
        "proposed_changes": [],
        "analysis_status": "failed",
    }


def _generate_minimal_reflection(response: str, proposal: dict, round_num: int) -> str:
    """Generate a minimal reflection when the structured section is missing."""
    diag = proposal.get("diagnosis", "(not available)")
    return (
        f"## Meta-Analysis (Round {round_num})\n\n"
        f"### What We Learned\n"
        f"The analysis proposed changes based on cross-round evidence. "
        f"Diagnosis: {diag[:200]}\n\n"
        f"### Abstract Principle\n"
        f"Cross-round pattern detection enables more targeted interventions "
        f"than single-round analysis.\n"
    )
