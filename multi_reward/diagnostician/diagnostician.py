"""
Diagnostician Agent — ReAct-loop LLM agent for reward function diagnosis.

Two instances run in PARALLEL with different cognitive biases:
- Diagnostician A: Exploration bias (higher temperature)
- Diagnostician B: Exploitation/structural bias (lower temperature)

ReAct loop: Thought → Action (tool call) → Observation → iterate → Final Answer

Output: diagnosis.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

_mr = Path(__file__).resolve().parent.parent
if str(_mr) not in sys.path:
    sys.path.insert(0, str(_mr))

from infra.llm_client import call_llm
from infra.file_utils import save_json, load_text, load_json
from memory.memory_store import MemoryStore
from memory.retrieval import retrieve_relevant_context, format_memory_for_prompt
from .tools import DiagnosticianTools
from .belief_state import format_belief_for_prompt


class Diagnostician:
    """ReAct-loop agent that diagnoses RL training problems from evidence.

    Usage:
        diag = Diagnostician(
            agent_id="A",
            evidence_board=board,
            task_understanding=task_understanding,
            reward_code=reward_code,
            memory_store=memory,
            api_key=api_key,
        )
        diagnosis = diag.diagnose()
    """

    def __init__(
        self,
        agent_id: str,
        evidence_board: dict,
        task_understanding: dict,
        reward_code: str,
        memory_store: MemoryStore,
        api_key: str = None,
        model: str = "deepseek-reasoner",
        temperature: float = 0.6,
        max_react_steps: int = 5,
    ):
        self.agent_id = agent_id  # "A" or "B"
        self.board = evidence_board
        self.task_understanding = task_understanding
        self.reward_code = reward_code
        self.memory = memory_store
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_react_steps = max_react_steps

        self.tools = DiagnosticianTools(
            memory_store, evidence_board, reward_code, task_understanding
        )

    def diagnose(self) -> dict:
        """Run the ReAct loop and return a diagnosis dict.

        Returns dict conforming to diagnosis.json schema.
        """
        system_prompt = self._build_system_prompt()
        context_prompt = self._build_context_prompt()
        tools_description = self._build_tools_description()

        # Build initial user prompt
        user_prompt = (
            f"{system_prompt}\n\n"
            f"---\n\n"
            f"## Current Context (Round {self.board['meta']['round']})\n\n"
            f"{context_prompt}\n\n"
            f"---\n\n"
            f"{tools_description}\n\n"
            f"---\n\n"
            f"## Begin Your Analysis\n\n"
            f"Start with a Thought about what you observe. "
            f"Use tools to gather evidence. "
            f"When ready, output FINAL ANSWER.\n"
        )

        print(
            f"  [Diagnostician-{self.agent_id}] Starting ReAct loop "
            f"(prompt: {len(user_prompt)} chars, temp: {self.temperature})"
        )

        conversation = []
        current_prompt = user_prompt

        for step in range(self.max_react_steps):
            # Append conversation history
            for msg in conversation:
                role_prefix = "User" if msg["role"] == "user" else "Assistant"
                current_prompt += f"\n\n{role_prefix}: {msg['content']}"

            current_prompt += "\n\nAssistant:"

            response = call_llm(
                current_prompt, self.api_key, self.model, self.temperature
            )

            conversation.append({"role": "assistant", "content": response})

            # Check for final answer
            if self._is_final_answer(response):
                diagnosis = self._extract_diagnosis_json(response)
                if diagnosis:
                    print(
                        f"  [Diagnostician-{self.agent_id}] "
                        f"Final answer at step {step + 1}"
                    )
                    return self._finalize_diagnosis(diagnosis)

            # Parse action
            action = self._parse_action(response)
            if action is None:
                # No tool call — prompt continuation
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Step {step+1}/{self.max_react_steps}. Do you have enough evidence? "
                        "If YES: output FINAL ANSWER. "
                        "If NO: what SPECIFIC missing evidence would one more tool provide?"
                    ),
                })
                continue

            # Execute tool
            observation = self.tools.dispatch(action["name"], action["input"])
            conversation.append({
                "role": "user",
                "content": f"Observation:\n{observation}\n\nStep {step+1}/{self.max_react_steps}. If sufficient, FINAL ANSWER. Otherwise, what specific tool next?",
            })

        # Max steps exhausted — try to extract partial diagnosis
        print(
            f"  [Diagnostician-{self.agent_id}] "
            f"Max ReAct steps ({self.max_react_steps}) exhausted"
        )
        return self._fallback_diagnosis(conversation)

    def _build_system_prompt(self) -> str:
        """Load the system prompt for this agent."""
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        filename = f"diagnostician_{self.agent_id.lower()}_system.txt"
        prompt = load_text(prompts_dir / filename)
        if prompt:
            return prompt

        # Fallback
        return (
            f"You are Diagnostician Agent {self.agent_id}. "
            "Diagnose RL training problems and propose ONE targeted reward change."
        )

    def _build_context_prompt(self) -> str:
        """Build the context section: evidence board summary + task understanding + memory."""
        parts = []

        # Evidence board summary
        board = self.board
        round_num = board["meta"]["round"]
        tr = board.get("training_result", {})

        parts.append(f"### Round {round_num} Evidence Summary\n")

        # Episode stats
        es = tr.get("episode_stats", {})
        parts.append(
            f"**Episodes**: {es.get('n_episodes', 0)} episodes, "
            f"mean length={es.get('mean_length', 0):.0f}, "
            f"max={es.get('max_length', 0)}"
        )
        tb = es.get("termination_breakdown", {})
        t_count = tb.get("terminated", {}).get("count", 0)
        tr_count = tb.get("truncated", {}).get("count", 0)
        parts.append(f"**Termination**: {t_count} terminated, {tr_count} truncated")

        # Reward components
        rc = tr.get("reward_components", {})
        if rc:
            parts.append("\n**Reward Components**:")
            for name, stats in rc.items():
                parts.append(
                    f"  - {name}: mean={stats.get('mean', 0):.4f}, "
                    f"std={stats.get('std', 0):.4f}, "
                    f"share={stats.get('share_of_total', 0):.1%}"
                )

        # Behavior descriptors
        bd = tr.get("behavior_descriptors", {})
        if bd:
            parts.append("\n**Behavior Descriptors**:")
            for name, stats in list(bd.items())[:6]:
                parts.append(
                    f"  - {name}: mean={stats.get('mean', 0):.4f}, "
                    f"trend={stats.get('trend', 'unknown')}"
                )

        # Health checks
        hc = tr.get("health_checks", {})
        if hc:
            parts.append("\n**Health Checks**:")
            for check, result in hc.items():
                if isinstance(result, dict):
                    icon = "PASS" if result.get("passed", True) else "FAIL"
                    parts.append(f"  - [{icon}] {check}: {result.get('detail', '')}")

        # Critical events
        events = tr.get("critical_events", [])
        if events:
            parts.append("\n**Critical Events**:")
            for e in events:
                parts.append(
                    f"  - [{e.get('severity', '?')}] {e.get('type')}: "
                    f"{e.get('description', '')[:200]}"
                )

        # Previous proposal comparison
        prev = board.get("previous_proposal", {})
        if prev:
            parts.append(f"\n**Previous Round Prediction vs Actual**:")
            parts.append(f"  Diagnosis was: {prev.get('diagnosis_summary', 'N/A')[:200]}")
            avp = prev.get("actual_vs_predicted", {})
            for comp, result in avp.items():
                parts.append(f"  - {comp}: current mean={result.get('current_mean', '?')}")

        # Cross-round trends
        trends = board.get("cross_round_trends", {})
        if trends:
            parts.append("\n**Cross-Round Trends**:")
            for metric, data in trends.items():
                direction = data.get("direction", "unknown")
                parts.append(f"  - {metric}: {direction}")

        # Task understanding (key points)
        tu = self.task_understanding
        if tu:
            parts.append("\n### Task Requirements\n")
            ti = tu.get("task_identity", {})
            parts.append(f"**Objective**: {ti.get('primary_objective', 'N/A')}")
            parts.append(f"**Failure conditions**: {ti.get('failure_conditions', [])}")
            pc = tu.get("physical_constraints", {})
            parts.append(
                f"**Physical**: gravity={pc.get('gravity_strength', '?')}, "
                f"balance_critical={pc.get('balance_critical', False)}"
            )
            di = tu.get("design_implications", {})
            if di:
                implications = [k for k, v in di.items() if v]
                parts.append(f"**Design implications**: {', '.join(implications)}")

        # Reward traps
        traps = tu.get("reward_trap_warnings", [])
        if traps:
            parts.append("\n**Known Reward Design Traps**:")
            for t in traps:
                parts.append(f"  - {t}")

        # Current reward function (truncated)
        parts.append("\n### Current Reward Function\n```python")
        code_lines = self.reward_code.splitlines()
        parts.append("\n".join(code_lines[:120]))
        if len(code_lines) > 120:
            parts.append(f"# ... ({len(code_lines) - 120} more lines)")
        parts.append("```")

        return "\n".join(parts)

    def _build_tools_description(self) -> str:
        """Build the tools description section."""
        return """## Available Tools

You have access to these tools. To use one, write:
```
Action: tool_name: input_text
```

Available tools:
- `query_memory: keyword` — Search MEMORY.md for past lessons
- `compare_rounds: N M` — Compare round N vs round M evidence
- `check_constraint_consistency:` — Check reward vs termination conditions
- `trace_variable: var_name` — Find all occurrences of a variable in reward code
- `analyze_efficiency:` — Analyze action efficiency metrics
- `detect_principle_violation:` — Run automated constraint violation detection"""

    def _build_memory_section(self) -> str:
        """Build the memory section with similarity-based retrieval."""
        fv = self.board.get("feature_vector", {})
        retrieval = retrieve_relevant_context(self.memory, fv)
        return format_memory_for_prompt(
            retrieval, agent_id=f"diagnostician_{self.agent_id}",
            memory_store=self.memory,
        )

    def _is_final_answer(self, response: str) -> bool:
        """Check if the response contains a final answer."""
        return (
            "FINAL ANSWER" in response.upper()
            or "```json" in response
        )

    def _parse_action(self, response: str) -> Optional[dict]:
        """Parse a tool call from the LLM response."""
        # Look for Action: tool_name: args pattern
        action_match = re.search(
            r"Action:\s*(\w+):\s*(.*?)(?:\n|$)",
            response, re.IGNORECASE,
        )
        if action_match:
            name = action_match.group(1).strip().lower()
            inp = action_match.group(2).strip()
            # Validate tool name
            valid_tools = [
                "query_memory", "compare_rounds",
                "check_constraint_consistency", "trace_variable",
                "analyze_efficiency", "detect_principle_violation",
            ]
            if name in valid_tools:
                return {"name": name, "input": inp}
        return None

    def _extract_diagnosis_json(self, response: str) -> Optional[dict]:
        """Extract JSON diagnosis from LLM response."""
        # Try ```json block first
        json_match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try bare JSON object
        json_match = re.search(r"\{[\s\S]*\"diagnosis\"[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try the whole response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        return None

    def _finalize_diagnosis(self, diagnosis: dict) -> dict:
        diagnosis.setdefault("confidence", {"self_assessment": 0.5, "rationale": ""})
        diagnosis.setdefault("alternative_considered", "")
        changes = diagnosis.get("proposed_changes", [])
        diagnosis["changed_count"] = len(changes)
        return diagnosis

    def _fallback_diagnosis(self, conversation: list) -> dict:
        """Generate a fallback diagnosis when ReAct loop exhausts steps."""
        # Try to extract partial analysis from last assistant response
        last_response = ""
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                last_response = msg["content"]
                break

        # Try to extract JSON anyway
        diagnosis = self._extract_diagnosis_json(last_response)
        if diagnosis:
            return self._finalize_diagnosis(diagnosis)

        # Absolute fallback
        round_num = self.board["meta"]["round"]
        prev_code = self.reward_code[:500]

        return {
            "diagnosis": {
                "primary_hypothesis": (
                    f"Fallback: Diagnostician-{self.agent_id} did not produce "
                    f"a structured proposal within {self.max_react_steps} steps. "
                    f"Partial analysis: {last_response[:300]}"
                ),
                "violated_principles": [],
                "root_cause_category": "other",
                "causal_chain": "Insufficient analysis",
            },
            "proposed_changes": [
                {
                    "component": "general",
                    "change_type": "reparameterize",
                    "current_code": prev_code,
                    "new_code": prev_code,
                    "rationale": "Fallback: no changes proposed",
                    "predicted_effect": "No prediction",
                    "max_risk": "Unknown",
                    "risk_mitigation": "Monitor training metrics",
                }
            ],
            "changed_count": 0,
            "confidence": {"self_assessment": 0.1, "rationale": "Fallback"},
            "alternative_considered": "",
        }


def run_diagnostician(
    agent_id: str,
    round_dir: Path,
    evidence_board: dict,
    task_understanding: dict,
    memory_store: MemoryStore,
    api_key: str = None,
    model: str = "deepseek-reasoner",
    temperature: float = 0.6,
    max_react_steps: int = 5,
) -> dict:
    """Convenience function to run a Diagnostician and save artifacts.

    Args:
        agent_id: "A" or "B"
        round_dir: Round directory (for saving artifacts).
        evidence_board: Evidence board dict.
        task_understanding: Task understanding dict.
        memory_store: MemoryStore instance.
        api_key: DeepSeek API key.
        model: LLM model name.
        temperature: Sampling temperature.
        max_react_steps: Maximum ReAct loop iterations.

    Returns:
        Diagnosis dict.
    """
    # Load reward code
    reward_path = round_dir / "reward_fn_source.py"
    if not reward_path.exists():
        prev_round = round_dir.parent / f"round{evidence_board['meta']['round'] - 1}"
        reward_path = prev_round / "reward_fn_source.py"

    reward_code = load_text(reward_path) if reward_path.exists() else ""

    diag = Diagnostician(
        agent_id=agent_id,
        evidence_board=evidence_board,
        task_understanding=task_understanding,
        reward_code=reward_code,
        memory_store=memory_store,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_react_steps=max_react_steps,
    )

    diagnosis = diag.diagnose()

    # Save artifact
    output_path = round_dir / f"diagnosis_{agent_id}.json"
    save_json(output_path, diagnosis)
    print(f"  [Diagnostician-{agent_id}] diagnosis → {output_path}")

    return diagnosis
