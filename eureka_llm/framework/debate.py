"""
debate.py — Dual-analyst debate: Exploration vs Exploitation + Judge.

Two specialized Analyzer agents independently diagnose the same round from
complementary RL perspectives, then a Judge arbitrates.

- Exploration Analyst: does the reward provide gradients everywhere, or are
  there dead zones trapping the agent in local optima?
- Exploitation Analyst: is the reward well-balanced, aligned with task success,
  and robust against reward hacking?

The Judge selects the stronger proposal or synthesizes both.

Inspired by MACRM's staged explore-exploit, EPO's cascade failure analysis,
and LEARN-Opt's council-of-analysts voting.
"""

from __future__ import annotations

import json, re, sys
from pathlib import Path

_framework_dir = Path(__file__).resolve().parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from llm_call import call_llm


EXPLORATION_LENS = """You are the Exploration Analyst. Your lens: does this reward function help the agent DISCOVER how to solve the task?

You evaluate reward quality through the exploration perspective of RL. A good reward provides informative gradients everywhere the agent might wander. A bad reward has flat regions where the agent receives no signal and stagnates in local optima.

In your analysis, focus on:
- **Gradient coverage**: Are there components with near-zero values across ALL episodes? These indicate state-space dead zones — regions where the agent gets no learning signal.
- **Novelty**: Does the reward contain any component that encourages reaching NEW states, or does it only reinforce states the agent already visits? Pure exploitation rewards trap the agent.
- **Local optima**: If episode lengths or metrics are flat across training, the reward landscape has a basin the agent cannot escape. The fix must reshape the landscape, not tweak coefficients.

Propose changes that EXPAND the agent's behavioral repertoire: add shaping for unexplored dimensions, increase reward density in dead zones, restructure to create escape routes from local optima."""


EXPLOITATION_LENS = """You are the Exploitation Analyst. Your lens: does this reward function help the agent MASTER the task efficiently?

You evaluate reward quality through the exploitation perspective of RL. A good reward has a clear optimum aligned with task success, well-balanced components, and no loopholes the agent can exploit.

In your analysis, focus on:
- **Optimum alignment**: Does the reward's maximum correspond to actual task completion? Check _outcome signal. If reward is high but _outcome negative, the reward is misaligned.
- **Component balance**: Is one component dominating (>50% of total magnitude)? The agent optimizes that single objective at the expense of everything else.
- **Efficiency**: Does the reward encourage COMPLETING the task, or does it reward lingering? If the agent survives full episodes without completing, the structure incentivizes survival over success.
- **Reward hacking**: If reward increases over training but task metrics stagnate, the agent found a loophole. Close it with task-progress terms.

Propose changes that REFINE the policy toward mastery: rescale for balance, tighten alignment, add efficiency incentives, or close reward-hacking loopholes."""


JUDGE_PROMPT = """You are the Debate Judge. Two analysts examined the same training data from complementary perspectives.

## Exploration Analyst ({xr_count} changes proposed)
Diagnosis: {xr_diag}
Changes: {xr_detail}

## Exploitation Analyst ({xt_count} changes proposed)
Diagnosis: {xt_diag}
Changes: {xt_detail}

## Training Context
{ctx}

Decide:
- EXPLORATION: the exploration proposal is stronger
- EXPLOITATION: the exploitation proposal is stronger
- SYNTHESIS: combine the best elements from both

Consider: causal quality, expected impact, complementarity, and safety.

Output exactly:
DECISION: <EXPLORATION | EXPLOITATION | SYNTHESIS>
REASONING: <2-4 sentences>"""


def run_explore_exploit_debate(
    prev_round_dir: Path,
    round_num: int,
    memory_system,
    api_key: str,
    model: str = "deepseek-reasoner",
) -> dict:
    """Run dual-analyst debate and return the winning proposal."""
    from agents.analyzer_agent import _load_file_content, build_system_message, _extract_json

    # Load round data once, reuse for both analysts
    file_store = _load_file_content(prev_round_dir, memory_system, round_num)

    # --- Exploration Analyst ---
    print("  [Debate] Exploration analyst...")
    explore_result = _run_lens_analyst(
        EXPLORATION_LENS, file_store, prev_round_dir, round_num,
        api_key, model, temperature=0.4, label="exploration",
    )

    # --- Exploitation Analyst ---
    print("  [Debate] Exploitation analyst...")
    exploit_result = _run_lens_analyst(
        EXPLOITATION_LENS, file_store, prev_round_dir, round_num,
        api_key, model, temperature=0.4, label="exploitation",
    )

    # Unpack
    xr_p = explore_result or {}
    xr_changes = xr_p.get("proposed_changes", [])
    xr_diag = xr_p.get("diagnosis", "(no diagnosis)")

    xt_p = exploit_result or {}
    xt_changes = xt_p.get("proposed_changes", [])
    xt_diag = xt_p.get("diagnosis", "(no diagnosis)")

    # Fallbacks
    if not xr_changes and not xt_changes:
        print("  [Debate] Both failed. Falling back to default analyzer.")
        return _fallback_analyzer(prev_round_dir, round_num, memory_system, api_key, model)
    if not xr_changes:
        print("  [Debate] Exploration had no changes. Using exploitation.")
        return {"proposal": xt_p, "_debate_meta": {"winner": "exploitation_fallback"}}
    if not xt_changes:
        print("  [Debate] Exploitation had no changes. Using exploration.")
        return {"proposal": xr_p, "_debate_meta": {"winner": "exploration_fallback"}}

    # --- Judge ---
    ctx = _build_ctx(prev_round_dir)
    judge_text = JUDGE_PROMPT.format(
        xr_count=len(xr_changes), xr_diag=xr_diag[:300],
        xr_detail=json.dumps(xr_changes, indent=2)[:1500],
        xt_count=len(xt_changes), xt_diag=xt_diag[:300],
        xt_detail=json.dumps(xt_changes, indent=2)[:1500],
        ctx=ctx,
    )

    print("  [Debate] Judge deliberating...")
    try:
        judge_resp = call_llm(judge_text, api_key, model, temperature=0.2)
    except Exception as e:
        print(f"  [Debate] Judge failed: {e}. Using exploration.")
        return {"proposal": exp_p, "_debate_meta": {"winner": "exploration_judge_fail"}}

    dec = re.search(r'DECISION:\s*(EXPLOITATION|EXPLORATION|SYNTHESIS)', judge_resp, re.IGNORECASE)
    reason = re.search(r'REASONING:\s*(.+?)(?:\n\n|\Z)', judge_resp, re.IGNORECASE | re.DOTALL)
    decision = dec.group(1).upper() if dec else "EXPLORATION"
    reasoning = reason.group(1).strip()[:300] if reason else ""

    print(f"  [Debate] Judge: {decision} — {reasoning[:100]}")

    if decision == "EXPLOITATION":
        return {"proposal": xt_p, "_debate_meta": {"winner": "exploitation", "reasoning": reasoning}}
    elif decision == "SYNTHESIS":
        combined = (xr_changes + xt_changes)[:3]
        levels = ["coefficient", "structural", "rewrite"]
        a_lvl = levels.index(xr_p.get("escalation_level", "coefficient")) if xr_p.get("escalation_level") in levels else 0
        b_lvl = levels.index(xt_p.get("escalation_level", "coefficient")) if xt_p.get("escalation_level") in levels else 0
        return {
            "proposal": {
                "diagnosis": f"[SYNTHESIS] Explore: {xr_diag[:200]}. Exploit: {xt_diag[:200]}. Judge: {reasoning[:200]}",
                "changed_count": len(combined),
                "proposed_changes": combined,
                "escalation_level": levels[max(a_lvl, b_lvl)],
            },
            "_debate_meta": {"winner": "synthesis", "reasoning": reasoning},
        }
    else:
        return {"proposal": xr_p, "_debate_meta": {"winner": "exploration", "reasoning": reasoning}}


def _run_lens_analyst(
    lens_prompt: str,
    file_store: dict,
    run_dir: Path,
    round_num: int,
    api_key: str,
    model: str,
    temperature: float,
    label: str,
) -> dict | None:
    """Run a single analyst with a specialized lens prompt prepended."""
    from agents.analyzer_agent import build_system_message, _extract_json

    # Build the standard system message with all pre-loaded data
    base_prompt = build_system_message(run_dir, round_num, file_store=file_store)

    # Prepend the lens prompt before the standard template
    full_prompt = lens_prompt + "\n\n---\n\n" + base_prompt

    # Save for audit
    (run_dir / f"analyzer_prompt_{label}.txt").write_text(full_prompt, encoding="utf-8")

    print(f"    [{label}] Calling LLM...")
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = call_llm(full_prompt, api_key, model, temperature)
            (run_dir / f"analyzer_response_{label}.txt").write_text(response, encoding="utf-8")
            proposal = _extract_json(response)
            if proposal and proposal.get("diagnosis"):
                print(f"    [{label}] Diagnosis: {proposal.get('diagnosis','')[:100]}")
                return proposal
            print(f"    [{label}] JSON extraction failed (attempt {attempt+1})")
        except Exception as e:
            print(f"    [{label}] LLM call failed (attempt {attempt+1}): {e}")

    # Fallback: try with the standard analyzer
    print(f"    [{label}] All attempts failed. Falling back to default analyzer.")
    return None


def _fallback_analyzer(prev_round_dir, round_num, memory_system, api_key, model):
    from agents.analyzer_agent import run_analyzer_agent
    return run_analyzer_agent(prev_round_dir, round_num, memory_system, api_key, model, temperature=0.4)


def _build_ctx(prev_round_dir: Path) -> str:
    parts = []
    csv_path = prev_round_dir / "evaluations" / "history.csv"
    if csv_path.exists():
        import csv
        with csv_path.open("r") as f:
            rows = list(csv.DictReader(f))
        if rows:
            parts.append(f"mean_length={rows[-1].get('mean_length','?')}")
    cs_path = prev_round_dir / "component_stats.md"
    if cs_path.exists():
        text = cs_path.read_text("utf-8")
        active = text.count("| active")
        dead = text.count("| dead")
        parts.append(f"components: {active} active, {dead} dead")
    return "; ".join(parts) if parts else "(no data)"
