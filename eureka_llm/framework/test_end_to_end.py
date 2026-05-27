"""
test_end_to_end.py — REAL LLM-based end-to-end tests for Phase 1-5.

Tests actual LLM calls (not mocks). Requires DEEPSEEK_API_KEY.
Run: python eureka_llm/framework/test_end_to_end.py
"""

import json
import os
import sys
from pathlib import Path

_framework_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_framework_dir))
sys.path.insert(0, str(_framework_dir / "agents"))

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MODEL = "deepseek-reasoner"

# Real experiment data
EXP_DIR = (_framework_dir.parent / "runs" / "lunarlander-v2_2605141802_1000000").resolve()

# The Meta-Analyzer's real output from round5 (with evidence citations from previous run)
META_PROPOSAL_PATH = EXP_DIR / "round5" / "meta_analyzer_test_output.json"


def test_react_loop_read_file():
    """Phase 1: ReAct engine with real LLM — ask it to read a file and answer."""
    print(f"\n{'='*60}")
    print("  TEST: ReAct loop — read_file + FINAL ANSWER")
    print(f"{'='*60}")

    from react_agent import ToolRegistry, run_react_loop

    tools = ToolRegistry(EXP_DIR)
    tools.register(
        "read_file",
        "Read any file from the experiment by relative path.",
        lambda rel_path: _read_safe(rel_path),
        param_name="rel_path",
        param_description="Relative path from experiment root",
    )

    system_prompt = (
        "You are a test agent. Read the file 'round4/perception_report.md' and tell me: "
        "what is the main behavioral pattern described? "
        "Call read_file to get the content, then output FINAL ANSWER with your finding."
    )

    result = run_react_loop(
        system_prompt=system_prompt,
        tools=tools,
        api_key=API_KEY,
        model=MODEL,
        temperature=0.3,
        max_steps=5,
        log_fn=lambda msg: print(f"  {msg}"),
    )

    print(f"\n  Steps used: {result['steps']}")
    print(f"  Tool calls: {len(result['tool_calls'])}")
    print(f"  Idle terminated: {result['idle_terminated']}")
    final = result["final_output"][:300] if result["final_output"] else "(empty)"
    print(f"  Final output: {final}...")

    assert result["final_output"] and len(result["final_output"]) > 20, \
        "FINAL ANSWER should contain meaningful output"
    assert result["tool_calls"], "Should have made at least one tool call"
    assert not result["idle_terminated"], "Should not idle-terminate"

    print(f"\n  ✅ PASS: ReAct loop completed {result['steps']} steps, "
          f"{len(result['tool_calls'])} tool calls")

    # Save result for review
    _save_test_result("react_loop_read_file", {
        "final_output": result["final_output"],
        "tool_calls": result["tool_calls"],
        "steps": result["steps"],
        "idle_terminated": result["idle_terminated"],
    })
    return result


def test_react_loop_shortcuts():
    """Phase 1: Test read_perception shortcut with ReAct."""
    print(f"\n{'='*60}")
    print("  TEST: ReAct loop — read_perception shortcut")
    print(f"{'='*60}")

    from react_agent import ToolRegistry, setup_default_tools, run_react_loop

    tools = setup_default_tools(EXP_DIR)
    system_prompt = (
        "Use the tool 'read_perception' with argument '4' to read the perception report "
        "for round 4. Then output FINAL ANSWER followed by a one-sentence summary "
        "of what the agent is doing behaviorally."
    )

    result = run_react_loop(
        system_prompt=system_prompt,
        tools=tools,
        api_key=API_KEY,
        model=MODEL,
        temperature=0.2,
        max_steps=5,
        log_fn=lambda msg: print(f"  {msg}"),
    )

    print(f"\n  Steps: {result['steps']}, Tool calls: {len(result['tool_calls'])}")
    for tc in result["tool_calls"]:
        print(f"    - {tc['tool']}({tc['arg']!r}) → {tc['result_preview'][:80]}...")

    assert result["final_output"] and len(result["final_output"]) > 20
    print(f"  ✅ PASS")

    _save_test_result("react_loop_shortcuts", {
        "final_output": result["final_output"],
        "tool_calls": result["tool_calls"],
    })
    return result


def test_generator_verification():
    """Phase 3: Generator verification with real proposal evidence citations."""
    print(f"\n{'='*60}")
    print("  TEST: Generator verification with real evidence citations")
    print(f"{'='*60}")

    if not META_PROPOSAL_PATH.exists():
        print("  SKIP: Meta-Analyzer proposal not found")
        return None

    test_data = json.loads(META_PROPOSAL_PATH.read_text("utf-8"))
    proposal = test_data.get("proposal", {})

    # Inject evidence citations into the real proposal
    proposal["evidence_citations"] = [
        {
            "claim": "Agent hovers at y≈0.54",
            "source": "round4/perception_report.md",
            "detail": "Behavior Trend: stable hover at y=0.54",
        },
        {
            "claim": "Ground penalty uses y_vel²",
            "source": "round4/reward_fn_source.py",
            "detail": "ground_speed_penalty = -0.1 * (y_vel ** 2)",
        },
    ]

    from agents.generator_agent import _run_verification

    result = _run_verification(
        proposal=proposal,
        experiment_dir=EXP_DIR,
        api_key=API_KEY,
        model=MODEL,
        temperature=0.3,
        memory_system=None,
    )

    print(f"\n  Action: {result['action']}")
    print(f"  Reason: {result.get('reason', '')[:200]}")

    # Either accept or reject is valid — what matters is the LLM made a decision
    assert result["action"] in ("accept", "reject"), \
        f"Unexpected action: {result['action']}"

    print(f"  ✅ PASS: Generator decided to {result['action'].upper()} the proposal")

    _save_test_result("generator_verification", result)
    return result


def test_generator_react_evidence_verification():
    """Phase 3: Full ReAct verification with tools."""
    print(f"\n{'='*60}")
    print("  TEST: Generator ReAct verification — read files to check claims")
    print(f"{'='*60}")

    from agents.generator_agent import _build_verification_prompt, _run_verification

    # Build a more complex scenario requiring tool use
    proposal = {
        "diagnosis": "Round 4 agent hovers at y≈0.54 because ground_speed_penalty creates a barrier.",
        "evidence_citations": [
            {
                "claim": "The ground_speed_penalty is -0.1 * (y_vel ** 2)",
                "source": "round4/reward_fn_source.py",
                "detail": "Check line containing 'ground_speed_penalty'",
            },
            {
                "claim": "Agent's mean episode length is 1000 (timeout)",
                "source": "round4/evaluations/history.csv",
                "detail": "Check the mean_length column in the last row",
            },
        ],
        "changed_count": 2,
        "proposed_changes": [],
    }

    prompt = _build_verification_prompt(proposal)
    assert "second opinion" in prompt.lower()
    assert "not a typist" in prompt

    result = _run_verification(
        proposal=proposal,
        experiment_dir=EXP_DIR,
        api_key=API_KEY,
        model=MODEL,
        temperature=0.3,
    )

    print(f"\n  Action: {result['action']}")
    print(f"  Reason: {result.get('reason', '')[:300]}")

    # Must have made a clear decision
    assert result["action"] in ("accept", "reject")

    print(f"  ✅ PASS: Verification decision = {result['action'].upper()}")

    _save_test_result("react_verification_complex", result)
    return result


def test_generator_full_flow():
    """Phase 3: Full Generator flow — verify + generate code."""
    print(f"\n{'='*60}")
    print("  TEST: Full Generator — verify + code generation")
    print(f"{'='*60}")

    # Use the real proposal from Meta-Analyzer test output
    if not META_PROPOSAL_PATH.exists():
        print("  SKIP: Meta-Analyzer proposal not found")
        return None

    test_data = json.loads(META_PROPOSAL_PATH.read_text("utf-8"))
    proposal = test_data.get("proposal", {})
    proposal["evidence_citations"] = [
        {
            "claim": "height_decrease reward is active at y>0.3",
            "source": "round4/reward_fn_source.py",
            "detail": "Find 'height_decrease' in the code",
        }
    ]

    from agents.generator_agent import run_generator_agent

    reward_path = EXP_DIR / "round4" / "reward_fn_source.py"
    if not reward_path.exists():
        print("  SKIP: reward_fn_source.py not found")
        return None

    code = run_generator_agent(
        proposal=proposal,
        current_reward_path=reward_path,
        run_dir=EXP_DIR / "round5",
        api_key=API_KEY,
        model=MODEL,
        temperature=0.3,
        max_retries=3,
        memory_system=None,
    )

    if code is not None:
        assert "def compute_reward" in code
        assert "def metrics_fn" in code
        print(f"  ✅ PASS: Code generated ({len(code)} chars)")
        _save_test_result("generator_full_flow", {
            "action": "accept",
            "code_length": len(code),
            "has_compute_reward": "def compute_reward" in code,
            "has_metrics_fn": "def metrics_fn" in code,
        })
    else:
        print(f"  ⚠️  Generator returned None (rejected or failed)")
        _save_test_result("generator_full_flow", {
            "action": "reject",
            "code": None,
        })

    return code


# ── Helpers ───────────────────────────────────────────────────────────────

def _read_safe(rel_path):
    """Safe read for test."""
    full = (EXP_DIR / rel_path).resolve()
    try:
        full.relative_to(EXP_DIR.resolve())
    except ValueError:
        return f"ERROR: Path escapes: {rel_path}"
    if not full.exists():
        return f"ERROR: File not found: {rel_path}"
    content = full.read_text("utf-8")
    if len(content) > 5000:
        content = content[:5000] + f"\n... (truncated, {len(content)} total)"
    return content


def _save_test_result(name: str, data: dict):
    """Save test result to output directory."""
    out_dir = _framework_dir / "test_output"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Result saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    if not EXP_DIR.exists():
        print(f"ERROR: Experiment directory not found: {EXP_DIR}")
        sys.exit(1)

    results = {}

    # Phase 1 tests
    results["react_loop_read_file"] = test_react_loop_read_file()
    results["react_loop_shortcuts"] = test_react_loop_shortcuts()

    # Phase 3 tests (Generator verification)
    results["generator_verification"] = test_generator_verification()
    results["react_verification_complex"] = test_generator_react_evidence_verification()
    results["generator_full_flow"] = test_generator_full_flow()

    # Summary
    print(f"\n{'='*60}")
    print("  END-TO-END TEST SUMMARY")
    print(f"{'='*60}")
    passed = 0
    failed = 0
    for name, result in results.items():
        if result is not None:
            status = "✅ PASS" if result.get("action") != "error" else "❌ FAIL"
            if name == "generator_full_flow" and result is not None:
                if isinstance(result, str):
                    status = "✅ PASS"
                elif isinstance(result, dict) and result.get("code") is None and result.get("action") == "reject":
                    status = "⚠️  REJECT"
            passed += 1
        else:
            status = "⏭️  SKIP"
            failed += 1
        print(f"  {status}: {name}")
    print(f"\n  {passed} executed, {failed} skipped/errors")
    print(f"  Outputs: {_framework_dir / 'test_output'}/")


if __name__ == "__main__":
    main()
