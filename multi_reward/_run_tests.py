#!/usr/bin/env python3
"""Run prompt tests and save results to file."""
import sys, os, json, re, traceback, time
MR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MR_DIR)

# Set API key directly
os.environ["DEEPSEEK_API_KEY"] = "YOUR_DEEPSEEK_API_KEY"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

RESULTS_FILE = os.path.join(MR_DIR, "_test_results.json")
LOG_FILE = os.path.join(MR_DIR, "_test_output.txt")
API_KEY = os.environ["DEEPSEEK_API_KEY"]
MODEL = "deepseek-chat"

def absp(rel):
    return os.path.join(MR_DIR, rel)

results = []
log_lines = []

def log(msg):
    log_lines.append(msg)
    print(msg, flush=True)

def record(test_name, passed, detail=""):
    results.append({"test": test_name, "passed": passed, "detail": str(detail)[:300]})
    icon = "[PASS]" if passed else "[FAIL]"
    log(f"  {icon} {test_name}: {str(detail)[:150]}")

# ══════════════════════════════════════════════════════════════
# Test 1: Basic LLM connectivity
# ══════════════════════════════════════════════════════════════
log("=" * 60)
log("Test 1: Basic LLM connectivity")

from infra.llm_client import call_llm, parse_json_response, extract_code_from_response

try:
    resp = call_llm("Say hello in one word.", api_key=API_KEY, model=MODEL, temperature=0.0)
    record("LLM connectivity", len(resp) > 0, f"Got {len(resp)} chars: {resp[:100]}")
except Exception as e:
    record("LLM connectivity", False, str(e))
    log(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# Test 2: EnvInterpreter schema
# ══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Test 2: EnvInterpreter output schema")

from infra.file_utils import load_text

env_sys_prompt = load_text(absp("env_interpreter/prompts/env_interpreter_system.txt"))

prompt = f"""{env_sys_prompt}

You are analyzing a bipedal walker with 4-joint legs, gravity, lidar sensors.
Hull contact with ground = termination. Terrain with varying height.
Output ONLY a JSON object with these keys: task_identity, physical_constraints, design_implications."""

try:
    resp = call_llm(prompt, API_KEY, MODEL, 0.2)
    data = parse_json_response(resp)
    has_identity = "task_identity" in data
    has_physics = "physical_constraints" in data
    has_design = "design_implications" in data
    record("EnvInterpreter has task_identity", has_identity)
    record("EnvInterpreter has physical_constraints", has_physics)
    record("EnvInterpreter has design_implications", has_design)
    if "_parse_error" in data:
        record("EnvInterpreter JSON parse", False, data.get("_parse_error",""))
except Exception as e:
    record("EnvInterpreter test", False, str(e))
    log(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# Test 3: Diagnostician diagnosis quality
# ══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Test 3: Diagnostician diagnosis quality")

diag_sys = load_text(absp("diagnostician/prompts/diagnostician_a_system.txt"))

evidence_text = """
**Episodes**: 120 episodes, mean length=320, max=1000
**Termination**: 85 terminated (71%), 35 truncated

**Reward Components**:
  - r_progress: mean=0.25, std=0.10, share=18%
  - r_alive: mean=0.95, std=0.00005, share=72%  <-- NEAR-ZERO STD
  - r_energy: mean=-0.12, std=0.04, share=9%
  - r_fall_pen: mean=-0.01, std=0.50, share=1%

**Health Checks**:
  - [PASS] component_dominance: max_share=72% (under 80%)

**Critical Events**:
  - [high] constant_offset_component: r_alive std=0.00005 (near zero)

**Cross-Round**: Mean length decreasing: 420 -> 350 -> 320

**Task**: Move forward while maintaining balance. Failure: hull contacts ground.
"""

reward_code = """
def compute_reward(self, action):
    alive_bonus = 0.05
    forward_reward = self.hull.linearVelocity.x * 1.0
    energy_penalty = -0.001 * np.sum(np.square(action))
    fall_penalty = -5.0 if getattr(self, '_terminated', False) else 0.0
    components = {"r_progress": forward_reward, "r_alive": alive_bonus, "r_energy": energy_penalty, "r_fall_pen": fall_penalty}
    total = sum(components.values())
    return total, components
"""

prompt = f"""{diag_sys}

## Current Context (Round 2)
{evidence_text}

## Current Reward Function
```python
{reward_code}
```

## Begin Your Analysis
Diagnose the MAIN problem. Output FINAL ANSWER as JSON.
"""

try:
    resp = call_llm(prompt, API_KEY, MODEL, 0.3)
    data = parse_json_response(resp)
    log(f"  Raw response preview: {resp[:300]}")

    # Check key diagnosis quality signals
    data_str = json.dumps(data, ensure_ascii=False).lower()
    has_diag = "diagnosis" in data_str or "hypothesis" in data_str
    record("Diagnostician has diagnosis", has_diag)

    has_changes = "proposed_changes" in data_str or "proposed_change" in data_str
    record("Diagnostician has proposed changes", has_changes)

    mentions_alive = "r_alive" in data_str or "alive" in data_str
    record("Diagnostician mentions r_alive", mentions_alive, "Identifies the anomalous component")

    mentions_std = "std" in data_str or "constant" in data_str or "offset" in data_str
    record("Diagnostician identifies constant/offset issue", mentions_std)

    if has_changes:
        record("Diagnostician proposes action", True)
except Exception as e:
    record("Diagnostician test", False, str(e))
    log(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# Test 4: Moderator debate agenda
# ══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Test 4: Moderator debate agenda")

mod_sys = load_text(absp("moderator/prompts/moderator_phase1_system.txt"))

diag_a = {"diagnosis": {"primary_hypothesis": "r_alive is a constant offset — should be removed"},
          "proposed_changes": [{"component": "r_alive", "change_type": "remove"}]}
diag_b = {"diagnosis": {"primary_hypothesis": "r_alive coefficient should be increased to strengthen survival incentive"},
          "proposed_changes": [{"component": "r_alive", "change_type": "reparameterize"}]}

prompt = f"""{mod_sys}

## Evidence Summary
Round 2. r_alive has near-zero std. Many terminations. Mean length decreasing.

## Diagnostician A
```json
{json.dumps(diag_a)}
```

## Diagnostician B
```json
{json.dumps(diag_b)}
```

Output debate agenda as JSON."""

try:
    resp = call_llm(prompt, API_KEY, MODEL, 0.2)
    data = parse_json_response(resp)
    log(f"  Moderator response: {resp[:300]}")

    data_str = json.dumps(data, ensure_ascii=False).lower()
    has_agree = "agreement" in data_str
    has_disagree = "disagree" in data_str
    record("Moderator identifies agreements", has_agree)
    record("Moderator identifies disagreements", has_disagree or "challenge" in data_str)
except Exception as e:
    record("Moderator test", False, str(e))
    log(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# Test 5: Implementor code generation
# ══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Test 5: Implementor code generation")

impl_sys = load_text(absp("implementor/prompts/implementor_system.txt"))

diagnosis_json = """{
  "diagnosis": {"primary_hypothesis": "r_alive component has near-zero standard deviation, providing constant reward with no gradient signal for learning"},
  "proposed_changes": [{
    "component": "r_alive",
    "change_type": "remove",
    "current_code": "alive_bonus = 0.05",
    "new_code": "# r_alive removed — survival incentive from fall_penalty",
    "rationale": "Constant offset provides no learning gradient and biases total reward"
  }],
  "changed_count": 1
}"""

prompt = f"""{impl_sys}

## Current Reward Function
```python
{reward_code}
```

## Final Diagnosis
```json
{diagnosis_json}
```

Output ONLY the complete modified reward function in a ```python block. Do NOT add any explanation outside the code block."""

try:
    resp = call_llm(prompt, API_KEY, MODEL, 0.1)
    code = extract_code_from_response(resp)

    if code:
        has_compute = "def compute_reward" in code
        record("Implementor has compute_reward", has_compute)

        has_no_alive = "alive_bonus = 0.05" not in code or "# r_alive" in code or "removed" in code.lower()
        record("Implementor removed r_alive", has_no_alive, "r_alive constant removal")
    else:
        record("Implementor code extraction", False, "No python code block")
except Exception as e:
    record("Implementor test", False, str(e))
    log(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# Test 6: Diagnostician B (different bias)
# ══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Test 6: Diagnostician B (exploitation bias)")

diag_b_sys = load_text(absp("diagnostician/prompts/diagnostician_b_system.txt"))

prompt = f"""{diag_b_sys}

## Current Context (Round 2)
{evidence_text}

## Current Reward Function
```python
{reward_code}
```

## Begin Your Analysis
Diagnose based on reward STRUCTURE. Output FINAL ANSWER as JSON."""

try:
    resp = call_llm(prompt, API_KEY, MODEL, 0.25)
    data = parse_json_response(resp)
    data_str = json.dumps(data, ensure_ascii=False).lower()
    log(f"  Diagnostician B preview: {str(data)[:300]}")

    # B should focus on structure, not just coefficients
    is_structural = "struct" in data_str or "remove" in data_str or "incentive" in data_str or "design" in data_str
    record("Diagnostician B structural focus", is_structural, "Structural analysis")
    record("Diagnostician B has changes", "proposed_change" in data_str)
except Exception as e:
    record("Diagnostician B test", False, str(e))
    log(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════
passed = sum(1 for r in results if r["passed"])
total = len(results)

log(f"\n{'='*60}")
log(f"RESULTS: {passed}/{total} PASSED")
log(f"{'='*60}")
for r in results:
    icon = "[PASS]" if r["passed"] else "[FAIL]"
    log(f"  {icon} {r['test']}")
    if not r["passed"]:
        log(f"       {r['detail'][:200]}")

with open(RESULTS_FILE, "w") as f:
    json.dump({"passed": passed, "total": total, "results": results}, f, indent=2, ensure_ascii=False)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print(f"\nResults saved to {RESULTS_FILE}")
print(f"Log saved to {LOG_FILE}")
