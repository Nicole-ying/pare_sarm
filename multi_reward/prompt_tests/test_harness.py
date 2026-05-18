#!/usr/bin/env python3
"""
Prompt testing harness — tests every LLM agent with DeepSeek API.

Layers:
  1. Schema conformance — output parses as valid JSON matching expected schema
  2. Anomaly detection — Diagnostician identifies injected anomaly correctly
  3. End-to-end code validity — Implementor code passes CodeValidator

Usage:
  DEEPSEEK_API_KEY=sk-xxx python prompt_tests/test_harness.py
"""

import json, os, sys
from pathlib import Path
from collections import defaultdict

_mr = Path(__file__).resolve().parent.parent
if str(_mr) not in sys.path: sys.path.insert(0, str(_mr))

from infra.llm_client import call_llm, parse_json_response
from infra.file_utils import load_text, load_json, save_json

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = "deepseek-chat"
RESULTS = []

# ═══════════════════════ Test fixtures ═══════════════════════

TASK_UNDERSTANDING = {
    "task_identity": {
        "task_type": "locomotion",
        "primary_objective": "Move forward efficiently while maintaining balance",
        "success_condition": "Survive to task horizon (1000 steps)",
        "failure_conditions": ["Hull contacts ground (fall)", "Agent body crashes"]
    },
    "physical_constraints": {
        "gravity_present": True, "gravity_strength": "strong",
        "balance_critical": True, "contact_required": True
    },
    "design_implications": {
        "must_reward_forward_motion": True, "must_penalize_falling": True,
        "should_penalize_inefficiency": True
    },
    "reward_trap_warnings": [
        "Do not reward high velocity without stability constraint",
        "Do not penalize action uniformly on all joints"
    ]
}

# Scenario: reward hacking — constant-offset component
EVIDENCE_A = {
    "meta": {"round": 2, "experiment_id": "test_hack", "n_episodes": 120},
    "environment_context": {"obs_dim": 24, "action_dim": 4, "max_episode_steps": 1000},
    "training_result": {
        "episode_stats": {
            "mean_length": 320.0, "n_episodes": 120,
            "termination_breakdown": {
                "terminated": {"count": 85, "fraction": 0.71},
                "truncated": {"count": 35, "fraction": 0.29}
            }
        },
        "reward_components": {
            "r_progress": {"mean": 0.25, "std": 0.10, "share_of_total": 0.18},
            "r_alive": {"mean": 0.95, "std": 0.00005, "share_of_total": 0.72},
            "r_energy": {"mean": -0.12, "std": 0.04, "share_of_total": 0.09},
            "r_fall_pen": {"mean": -0.01, "std": 0.50, "share_of_total": 0.01}
        },
        "behavior_descriptors": {
            "velocity_x": {"mean": 0.23, "std": 0.15, "trend": "decreasing"},
            "action_magnitude": {"mean": 0.12, "std": 0.05, "trend": "decreasing"}
        },
        "health_checks": {
            "component_dominance": {"max_share": 0.72, "passed": True, "detail": "r_alive at 72% but still under 80% threshold"},
            "entropy_collapse": {"final_entropy": 0.18, "passed": True},
            "survival_health": {"termination_rate": 0.71, "passed": True}
        },
        "critical_events": [
            {"type": "constant_offset_component", "severity": "high",
             "description": "r_alive std=0.00005 — functionally constant, provides no gradient"}
        ]
    },
    "cross_round_trends": {"mean_length": {"round0": 420, "round1": 350, "round2": 320, "direction": "decreasing"}},
    "feature_vector": {"mean_length": 320, "velocity_x_mean": 0.23, "action_magnitude_mean": 0.12, "max_component_share": 0.72}
}

SAMPLE_REWARD_CODE = """
import math
import numpy as np

def compute_reward(self, action):
    vel = self.hull.linearVelocity
    velocity_x = vel.x
    forward_reward = velocity_x * 1.0
    alive_bonus = 0.05
    energy_penalty = -0.001 * np.sum(np.square(action))
    fallen = getattr(self, '_terminated', False)
    fall_penalty = -5.0 if fallen else 0.0
    components = {
        "r_progress": forward_reward,
        "r_alive": alive_bonus,
        "r_energy": energy_penalty,
        "r_fall_pen": fall_penalty,
    }
    total = sum(components.values())
    return total, components

def metrics_fn(env, action, obs):
    return {"velocity_x": float(env.hull.linearVelocity.x),
            "action_magnitude": float(np.mean(np.abs(action)))}
"""

# ═══════════════════════ Test runner ═══════════════════════

def header(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def record(test_name, passed, detail=""):
    RESULTS.append({"test": test_name, "passed": passed, "detail": str(detail)[:300]})
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {test_name}")
    if detail: print(f"       {detail[:200]}")

# ═══════════════════════ Layer 1: Schema conformance ═══════════════════════

def test_env_interpreter_schema():
    """EnvInterpreter must output valid JSON with required keys."""
    header("Layer 1: EnvInterpreter schema")
    prompts_dir = _mr / "env_interpreter" / "prompts"
    sys_prompt = load_text(prompts_dir / "env_interpreter_system.txt")

    prompt = f"""{sys_prompt}

You are analyzing a bipedal walker environment. No step.py is available.
Based on typical bipedal walker dynamics: 4-joint walker, gravity present,
hull falls = termination, lidar sensors for terrain. Output JSON."""

    resp = call_llm(prompt, API_KEY, MODEL, 0.2)
    print(f"  Response: {len(resp)} chars")

    data = parse_json_response(resp)
    required = ["task_identity", "physical_constraints", "design_implications"]
    for k in required:
        record(f"EnvInterpreter has '{k}'", k in data,
               f"type={type(data.get(k)).__name__}" if k in data else "missing")


def test_diagnostician_schema():
    """Diagnostician must output valid diagnosis JSON."""
    header("Layer 1: Diagnostician A schema")
    prompts_dir = _mr / "diagnostician" / "prompts"
    sys_prompt = load_text(prompts_dir / "diagnostician_a_system.txt")

    prompt = f"""{sys_prompt}

## Current Context (Round 2)
### Evidence Summary
**Episodes**: 120 episodes, mean length=320, max=1000
**Termination**: 85 terminated, 35 truncated

**Reward Components**:
  - r_progress: mean=0.25, std=0.10, share=18%
  - r_alive: mean=0.95, std=0.00005, share=72%
  - r_energy: mean=-0.12, std=0.04, share=9%

**Critical Events**:
  - [high] constant_offset_component: r_alive std near zero

**Health Checks**:
  - [PASS] component_dominance: max_share=72% (under 80%)

### Task Requirements
**Objective**: Move forward efficiently while maintaining balance
**Failure**: Hull contacts ground

### Current Reward Function
```python
def compute_reward(self, action):
    alive_bonus = 0.05
    forward_reward = self.hull.linearVelocity.x * 1.0
    energy_penalty = -0.001 * np.sum(np.square(action))
    fall_penalty = -5.0 if getattr(self, '_terminated', False) else 0.0
    total = forward_reward + alive_bonus + energy_penalty + fall_penalty
    return total, {{"r_progress": forward_reward, "r_alive": alive_bonus, "r_energy": energy_penalty, "r_fall_pen": fall_penalty}}
```

## Begin Your Analysis
Diagnose the main problem and output FINAL ANSWER as JSON.
"""

    resp = call_llm(prompt, API_KEY, MODEL, 0.4)
    print(f"  Response: {len(resp)} chars")
    print(f"  Preview: {resp[:300]}...")

    data = parse_json_response(resp)
    record("Diagnostician returns dict", isinstance(data, dict) and "_parse_error" not in data)
    record("Has 'diagnosis'", "diagnosis" in data or "primary_hypothesis" in str(data))
    record("Has 'proposed_changes'", "proposed_changes" in data or "proposed_change" in str(data).lower())


def test_moderator_schema():
    """Moderator Phase 1 must produce debate agenda JSON."""
    header("Layer 1: Moderator Phase 1 schema")
    prompts_dir = _mr / "moderator" / "prompts"
    sys_prompt = load_text(prompts_dir / "moderator_phase1_system.txt")

    diag_a = {
        "diagnosis": {"primary_hypothesis": "r_alive provides constant offset with no gradient"},
        "proposed_changes": [{"component": "r_alive", "change_type": "remove", "rationale": "Inactive component"}]
    }
    diag_b = {
        "diagnosis": {"primary_hypothesis": "r_alive should be increased to strengthen survival incentive"},
        "proposed_changes": [{"component": "r_alive", "change_type": "reparameterize", "rationale": "Increase coefficient 5x"}]
    }

    prompt = f"""{sys_prompt}

## Evidence Summary
Round: 2. r_alive std near zero. Many terminations.

## Diagnostician A
```json
{json.dumps(diag_a)}
```

## Diagnostician B
```json
{json.dumps(diag_b)}
```

Output debate agenda as JSON.
"""

    resp = call_llm(prompt, API_KEY, MODEL, 0.3)
    print(f"  Response: {len(resp)} chars")

    data = parse_json_response(resp)
    record("Moderator returns dict", isinstance(data, dict) and "_parse_error" not in data)
    record("Has agreements/disagreements", "agreement" in str(data).lower() or "disagreement" in str(data).lower())


def test_implementor_schema():
    """Implementor must produce valid Python code."""
    header("Layer 1: Implementor code validity")
    prompts_dir = _mr / "implementor" / "prompts"
    sys_prompt = load_text(prompts_dir / "implementor_system.txt")

    diagnosis = {
        "diagnosis": {"primary_hypothesis": "r_alive provides no gradient — should be removed and folded into fall_penalty"},
        "proposed_changes": [{
            "component": "r_alive", "change_type": "remove",
            "current_code": "alive_bonus = 0.05",
            "new_code": "# r_alive removed — survival handled by fall_penalty",
            "rationale": "Constant offset provides no learning signal"
        }],
        "changed_count": 1
    }

    prompt = f"""{sys_prompt}

## Current Reward Function
```python
{SAMPLE_REWARD_CODE}
```

## Final Diagnosis
```json
{json.dumps(diagnosis)}
```

Translate to code. Output ONLY the modified reward function in ```python block.
"""

    resp = call_llm(prompt, API_KEY, MODEL, 0.15)
    print(f"  Response: {len(resp)} chars")

    # Check code output
    has_compute = "def compute_reward" in resp
    record("Implementor outputs compute_reward", has_compute)

    # Check that the change was applied
    has_removal = "r_alive" not in resp or "removed" in resp.lower() or "# r_alive" in resp.lower()
    record("Implementor applied removal change", not ("alive_bonus = 0.05" in resp and "r_alive" in resp.split("components")[-1] if "components" in resp else True),
           "Old code removed or commented out")


# ═══════════════════════ Layer 2: Anomaly detection ═══════════════════════

def test_anomaly_detection():
    """Diagnostician must detect the constant-offset reward component."""
    header("Layer 2: Anomaly detection — constant offset")

    prompts_dir = _mr / "diagnostician" / "prompts"
    sys_a = load_text(prompts_dir / "diagnostician_a_system.txt")
    sys_b = load_text(prompts_dir / "diagnostician_b_system.txt")

    board_json = json.dumps(EVIDENCE_A, indent=2, ensure_ascii=False)

    for agent_id, sys_prompt in [("A", sys_a), ("B", sys_b)]:
        prompt = f"""{sys_prompt}

## Current Context (Round 2)

```json
{board_json}
```

### Task Requirements
Objective: Move forward efficiently while maintaining balance.
Failure: Hull contacts ground, agent crashes.

### Current Reward Function
```python
{SAMPLE_REWARD_CODE}
```

## Begin Your Analysis
Start with a Thought. Use tools (detect_principle_violation, analyze_efficiency).
When ready, output FINAL ANSWER as JSON.
"""

        resp = call_llm(prompt, API_KEY, MODEL, 0.4)
        data = parse_json_response(resp)

        detected = (
            "constant" in str(data).lower() or
            "offset" in str(data).lower() or
            "inactive" in str(data).lower() or
            "dead" in str(data).lower() or
            "gradient" in str(data).lower() or
            "r_alive" in str(data).lower()
        )
        record(f"Anomaly detection Agent {agent_id}", detected,
               str(data).replace('\n',' ')[:200])


# ═══════════════════════ Layer 3: End-to-end ═══════════════════════

def test_code_validator():
    """Generated code must pass all CodeValidator checks."""
    header("Layer 3: CodeValidator checks")

    prompts_dir = _mr / "implementor" / "prompts"
    sys_prompt = load_text(prompts_dir / "implementor_system.txt")

    # Generate code for a simple fix
    diagnosis = {
        "proposed_changes": [{
            "component": "r_energy",
            "change_type": "reparameterize",
            "current_code": "energy_penalty = -0.001 * np.sum(np.square(action))",
            "new_code": "energy_penalty = -0.0005 * np.sum(np.square(action))",
            "rationale": "Reduce energy penalty"
        }],
        "changed_count": 1
    }

    prompt = f"""{sys_prompt}

## Current Reward Function
```python
{SAMPLE_REWARD_CODE}
```

## Final Diagnosis
```json
{json.dumps(diagnosis)}
```

Output modified ```python code.
"""

    resp = call_llm(prompt, API_KEY, MODEL, 0.1)

    # Extract code
    import re
    code_matches = re.findall(r"```python\s*\n(.*?)```", resp, re.DOTALL)
    if not code_matches:
        record("CodeValidator: code extraction", False, "No python block found")
        return

    code = code_matches[0]

    from code_validator.code_validator import validate_code
    v = validate_code(code, "action", SAMPLE_REWARD_CODE, diagnosis)

    record("CodeValidator: syntax", "syntax" not in str(v.get("errors", [])).lower())
    record("CodeValidator: compute_reward exists", "def compute_reward" in code)
    record("CodeValidator: overall PASS", v.get("passed", False),
           f"{len(v.get('errors',[]))} errors, {len(v.get('warnings',[]))} warnings")
    if v.get("errors"):
        for e in v["errors"]:
            print(f"       Error: [{e['category']}] {e['message'][:120]}")


# ═══════════════════════ Main ═══════════════════════

def main():
    global API_KEY
    if not API_KEY:
        API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    print(f"multi_reward Prompt Test Harness")
    print(f"Model: {MODEL}")
    print(f"API Key: {API_KEY[:12]}...")

    try:
        test_env_interpreter_schema()
    except Exception as e:
        record("EnvInterpreter schema", False, str(e))

    try:
        test_diagnostician_schema()
    except Exception as e:
        record("Diagnostician schema", False, str(e))

    try:
        test_moderator_schema()
    except Exception as e:
        record("Moderator schema", False, str(e))

    try:
        test_implementor_schema()
    except Exception as e:
        record("Implementor schema", False, str(e))

    try:
        test_anomaly_detection()
    except Exception as e:
        record("Anomaly detection", False, str(e))

    try:
        test_code_validator()
    except Exception as e:
        record("CodeValidator", False, str(e))

    # Summary
    passed = sum(1 for r in RESULTS if r["passed"])
    total = len(RESULTS)
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} PASSED")
    print(f"{'='*60}")
    for r in RESULTS:
        icon = "PASS" if r["passed"] else "FAIL"
        print(f"  [{icon}] {r['test']}")
        if not r["passed"]:
            print(f"       {r['detail'][:200]}")

if __name__ == "__main__":
    main()
