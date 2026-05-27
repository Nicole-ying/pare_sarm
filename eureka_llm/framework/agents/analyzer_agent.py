"""
analyzer_agent.py — Single-call LLM agent that translates Perception findings into
reward code change proposals.

Architecture:
    All context (Perception report, reward code, evaluation history, task manifest,
    previous proposal, reflection, memory) is pre-loaded into the system prompt.
    Single LLM call → JSON change proposal. No ReAct loop, no read_file tool.

    Retry logic: 1 retry attempt if JSON extraction fails.
    Fallback: returns zero-change proposal if all attempts fail.

Output:
    Structured JSON proposal (dict with changed_count <= 3)
"""

import json
import re
from pathlib import Path
import sys

_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm


def _summarize_proposal_component(proposal_json_str: str | None) -> str:
    """Extract component names from a proposal JSON string."""
    if not proposal_json_str:
        return "(no proposal)"
    try:
        proposal = json.loads(proposal_json_str)
        changes = proposal.get("proposed_changes", [])
        if not changes:
            return "No changes"
        names = [c.get("component", "?")[:30] for c in changes if c.get("component")]
        return "; ".join(names[:2])
    except (json.JSONDecodeError, TypeError):
        return "(parse error)"


def _summarize_proposal_expected(proposal_json_str: str | None) -> str:
    """Extract expected outcome from a proposal's first change reason."""
    if not proposal_json_str:
        return ""
    try:
        proposal = json.loads(proposal_json_str)
        changes = proposal.get("proposed_changes", [])
        if changes and changes[0].get("reason"):
            reason = changes[0]["reason"]
            first_sentence = reason.split(".")[0] if "." in reason else reason
            return first_sentence[:60] + ("..." if len(first_sentence) > 60 else "")
        return ""
    except (json.JSONDecodeError, TypeError):
        return ""


def _summarize_perception_behavior(perception_text: str | None) -> str:
    """Extract key behavior signal from perception report (Behavior Trend section)."""
    if not perception_text:
        return ""
    match = re.search(
        r"### 1\. Behavior Trend Summary\s*\n(.*?)(?=\n###|\Z)",
        perception_text, re.DOTALL
    )
    if match:
        section = match.group(1).strip()
        for line in section.split("\n"):
            line = line.strip()
            if line and not line.startswith("- ") and not line.startswith("|") and len(line) > 10:
                return line[:80] + ("..." if len(line) > 80 else "")
    for line in perception_text.split("\n"):
        if line.strip() and not line.startswith("#") and len(line.strip()) > 10:
            return line.strip()[:80] + ("..." if len(line.strip()) > 80 else "")
    return "(report available)"


def _load_file_content(run_dir: Path, memory_system=None, round_num: int = 0) -> dict:
    """Load ALL available files into a single dict for pre-injection.

    Returns dict: file_key → content string (or None if not found).
    """
    path_map = {
        "reward_code": run_dir / "reward_fn_source.py",
        "history_csv": run_dir / "evaluations" / "history.csv",
        "perception_report": run_dir / "perception_report.md",
        "reflection": run_dir / "reflection.md",
    }
    result = {}
    for key, path in path_map.items():
        result[key] = path.read_text("utf-8") if path.exists() else None

    # Task manifest via memory system
    result["task_manifest"] = memory_system.get_task_manifest() if memory_system else None

    # Cross-round history: proposals + perception outcomes from ALL previous rounds
    result["cross_round_history"] = []
    result["prev_proposal"] = None
    if round_num > 0:
        for r in range(0, round_num):
            r_dir = run_dir.parent / f"round{r}"
            if not r_dir.exists():
                continue
            entry = {"round": r, "proposal": None, "perception": None}
            pp = r_dir / "analyzer_proposal.json"
            if pp.exists():
                entry["proposal"] = pp.read_text("utf-8")
            pr = r_dir / "perception_report.md"
            if pr.exists():
                entry["perception"] = pr.read_text("utf-8")
            rc = r_dir / "reflection_checklist.json"
            if rc.exists():
                entry["reflection_checklist"] = rc.read_text("utf-8")
            result["cross_round_history"].append(entry)
        # Keep prev_proposal for backward compatibility
        if result["cross_round_history"]:
            result["prev_proposal"] = result["cross_round_history"][-1]["proposal"]

    # Cross-round memory index
    memory_text = ""
    if memory_system and memory_system.memory_dir:
        memory_path = memory_system.memory_dir / "MEMORY.md"
        if memory_path.exists():
            memory_text = memory_path.read_text("utf-8")
    result["memory"] = memory_text or None

    return result


def build_system_message(run_dir: Path, round_num: int,
                          file_store: dict | None = None,
                          skill_manager=None) -> str:
    """Build system message with ALL context pre-loaded.

    Every available file is injected as a named section. The agent reads
    everything in order — no read_file tool needed.
    """
    # Base system prompt from template
    sys_prompt_path = (
        Path(__file__).resolve().parent.parent.parent
        / "templates" / "analyst_system_prompt.txt"
    )
    sys_prompt = sys_prompt_path.read_text("utf-8") if sys_prompt_path.exists() else ""

    sections = [
        sys_prompt,
        "",
        "## Experiment Context",
        f"Run directory: {run_dir.parent.name}",
        f"Round: {round_num} (evaluating round {round_num - 1})",
        "",
    ]

    if not file_store:
        sections.append("# WARNING: No data files could be loaded.\n")
        return "\n".join(sections)

    # Task Manifest
    if file_store.get("task_manifest"):
        sections.extend([
            "## Task Manifest",
            file_store["task_manifest"],
            "",
        ])

    # Perception Report (full, not truncated)
    if file_store.get("perception_report"):
        sections.extend([
            "## Perception Report",
            file_store["perception_report"],
            "",
        ])

    # Evaluation History CSV
    if file_store.get("history_csv"):
        sections.extend([
            "## Evaluation History",
            "```",
            file_store["history_csv"],
            "```",
            "",
        ])

    # Current Reward Code
    if file_store.get("reward_code"):
        sections.extend([
            "## Current Reward Code",
            "```python",
            file_store["reward_code"],
            "```",
            "",
        ])

    # Previous Proposal
    if file_store.get("prev_proposal"):
        sections.extend([
            "## Previous Proposal",
            "```json",
            file_store["prev_proposal"],
            "```",
            "",
        ])

    # Cross-Round History Table
    if file_store.get("cross_round_history"):
        history_entries = file_store["cross_round_history"]
        if len(history_entries) > 1:
            table_lines = [
                "## Cross-Round History (Proposals vs Outcomes)",
                "",
                "| Round | What Changed | Expected Outcome | Actual Outcome |",
                "|-------|-------------|-----------------|----------------|",
            ]
            for entry in history_entries:
                r = entry["round"]
                what = _summarize_proposal_component(entry.get("proposal"))
                expected = _summarize_proposal_expected(entry.get("proposal"))
                actual = _summarize_perception_behavior(entry.get("perception"))
                table_lines.append(f"| Round {r} | {what} | {expected} | {actual} |")
            sections.extend(table_lines)
            sections.append("")

    # Reflection
    if file_store.get("reflection"):
        sections.extend([
            "## Previous Round Reflection",
            file_store["reflection"],
            "",
        ])

    # Cross-Round Memory
    if file_store.get("memory"):
        sections.extend([
            "## Cross-Round Memory (Accumulated Lessons)",
            file_store["memory"],
            "",
        ])

    # Active skills
    if skill_manager:
        active_docs = skill_manager.active_docs
        if active_docs:
            sections.extend([
                "## Relevant Techniques",
                active_docs,
                "",
            ])

    sections.append("## Begin Analysis")
    sections.append(
        "Read through each section above in order. "
        "When ready, output FINAL ANSWER with your JSON proposal."
    )

    return "\n".join(sections)


def _try_parse_json(s: str) -> dict | None:
    """Try to parse a JSON string with common LLM error recovery."""
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fix trailing commas before } and ]
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return None


def _extract_json(text: str) -> dict | None:
    """Extract JSON proposal from LLM response.

    Tries multiple strategies:
    1. ```json code blocks
    2. JSON after FINAL ANSWER marker
    3. Any top-level JSON object with diagnosis + proposed_changes keys
    """
    raw_candidates = []

    # Strategy 1: Collect from ```json blocks
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        candidate = _try_parse_json(m.group(1).strip())
        if candidate:
            raw_candidates.append(candidate)

    # Strategy 2: JSON after FINAL ANSWER
    m = re.search(
        r'FINAL ANSWER.*?(\{(?:[^{}]|"(?:\\.|[^"\\])*")*\})(?:\s*```)?',
        text, re.DOTALL
    )
    if m:
        candidate = _try_parse_json(m.group(1))
        if candidate:
            raw_candidates.append(candidate)

    # Strategy 3: Fallback — walk braces to find any JSON with required keys
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
                if candidate:
                    raw_candidates.append(candidate)
                start = -1

    # Validate: must have diagnosis + proposed_changes
    seen = set()
    for candidate in raw_candidates:
        key = str(candidate.get("diagnosis", "")) + str(candidate.get("changed_count"))
        if key in seen:
            continue
        seen.add(key)
        if candidate.get("diagnosis") and candidate.get("proposed_changes") is not None:
            return candidate

    return None


def run_analyzer_agent(run_dir: Path, round_num: int,
                       memory_system, api_key: str,
                       model: str = "deepseek-reasoner",
                       temperature: float = 0.4,
                       skill_manager=None) -> dict:
    """Run analyzer: single LLM call with all context pre-loaded → JSON proposal.

    No ReAct loop. No read_file tool. All data is pre-injected into the prompt.

    Args:
        run_dir: Previous round directory.
        round_num: Current round number.
        memory_system: For task_manifest and belief state access.
        api_key: LLM API key.
        model: LLM model name.
        temperature: Sampling temperature.
        skill_manager: Optional skill catalog for technique injection.

    Returns:
        dict with "proposal" key.
    """
    # Load all data
    file_store = _load_file_content(run_dir, memory_system, round_num)
    system_msg = build_system_message(run_dir, round_num, file_store=file_store, skill_manager=skill_manager)

    # Save prompt for debugging
    (run_dir / "analyzer_prompt.txt").write_text(system_msg, encoding="utf-8")
    print(f"  [Analyzer] Starting analysis (all data pre-loaded)...")

    # Single LLM call
    proposal = None
    try:
        response = call_llm(system_msg, api_key, model, temperature)
        (run_dir / "analyzer_response.txt").write_text(response, encoding="utf-8")
        proposal = _extract_json(response)
    except Exception as e:
        print(f"  [Analyzer] LLM call failed: {e}")

    # Retry if JSON extraction failed
    if not proposal:
        print(f"  [Analyzer] No valid JSON found. Retrying with stricter prompt...")
        last_response = ""
        resp_path = run_dir / "analyzer_response.txt"
        if resp_path.exists():
            last_response = resp_path.read_text("utf-8")[:2000]

        corrective_prompt = (
            f"Your previous analysis did not produce a valid JSON proposal.\n\n"
            f"Your analysis was:\n{last_response}\n\n"
            f"---\n"
            f"Now output ONLY a FINAL ANSWER with a valid JSON block. "
            f"Use exactly this format with double quotes and no trailing commas:\n"
            f"```json\n{{\n"
            f'  "diagnosis": "Brief root cause",\n'
            f'  "changed_count": 0,\n'
            f'  "proposed_changes": []\n'
            f"}}\n```\n"
            f"No text before or after the code block."
        )
        try:
            retry_response = call_llm(corrective_prompt, api_key, model, temperature)
            proposal = _extract_json(retry_response)
            if proposal:
                print(f"  [Analyzer] Retry succeeded!")
        except Exception as e:
            print(f"  [Analyzer] Retry failed: {e}")

    # Fallback
    if not proposal:
        print(f"  [Analyzer] No valid JSON found. Using fallback.")
        proposal = {
            "diagnosis": "Failed to extract JSON proposal from analyzer response.",
            "changed_count": 0,
            "proposed_changes": [],
        }
        proposal["analysis_status"] = "failed"
    else:
        proposal["analysis_status"] = "ok"

    # Enforce max 3 changes
    if proposal.get("changed_count", 0) > 3:
        changes = proposal.get("proposed_changes", [])
        proposal["changed_count"] = min(len(changes), 3)
        proposal["proposed_changes"] = changes[:3]

    # Save proposal
    (run_dir / "analyzer_proposal.json").write_text(
        json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Update belief state
    if memory_system:
        memory_system.update_belief("analyst", {
            "round": round_num,
            "diagnosis": proposal.get("diagnosis", "")[:200],
            "changed_count": proposal.get("changed_count", 0),
        })

    print(f"  [Analyzer] Diagnosis: {proposal.get('diagnosis', 'N/A')[:120]}")
    print(f"  [Analyzer] Changes proposed: {proposal.get('changed_count', 0)}")
    return {"proposal": proposal}
