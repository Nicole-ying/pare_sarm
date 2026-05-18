"""
Tool definitions for Diagnostician ReAct loop.

Tools available:
- query_memory: Search cross-round MEMORY.md for relevant lessons
- compare_rounds: Compare two rounds' evidence boards
- check_constraint_consistency: Check if reward aligns with termination conditions
- trace_variable: Trace a variable through the reward function
- analyze_efficiency: Analyze action efficiency metrics
- detect_principle_violation: Run constraint discovery on evidence
"""

import json
from typing import Any, Optional


class DiagnosticianTools:
    """Tool dispatcher for Diagnostician ReAct loop."""

    def __init__(self, memory_store, evidence_board: dict,
                 reward_code: str, task_understanding: dict):
        self.memory = memory_store
        self.board = evidence_board
        self.reward_code = reward_code
        self.task_understanding = task_understanding

    def dispatch(self, tool_name: str, tool_input: str) -> str:
        """Dispatch a tool call by name. Returns observation string."""
        method = getattr(self, f"_tool_{tool_name}", None)
        if method is None:
            return f"Unknown tool: {tool_name}. Available: {self._available_tools()}"

        try:
            return method(tool_input)
        except Exception as e:
            return f"Tool '{tool_name}' error: {e}"

    def _available_tools(self) -> str:
        return ", ".join([
            "query_memory", "compare_rounds", "check_constraint_consistency",
            "trace_variable", "analyze_efficiency", "detect_principle_violation",
        ])

    # ── Tool implementations ────────────────────────────────────────

    def _tool_query_memory(self, inp: str) -> str:
        """Search MEMORY.md for lessons relevant to a keyword."""
        lessons = self.memory.query_lessons(inp)
        if lessons:
            return "Relevant past lessons:\n" + "\n---\n".join(lessons)
        return f"No matching lessons found for '{inp}'."

    def _tool_compare_rounds(self, inp: str) -> str:
        """Compare two rounds' evidence boards. Input: 'N M'."""
        import re
        nums = re.findall(r"\d+", inp)
        if len(nums) < 2:
            return "Please specify two round numbers, e.g., 'compare_rounds: 0 1'"

        r1, r2 = int(nums[0]), int(nums[1])
        b1 = self.memory.get_artifact(r1, "evidence_board.json")
        b2 = self.memory.get_artifact(r2, "evidence_board.json")

        if b1 is None or b2 is None:
            return f"Evidence board for round {r1 if b1 is None else r2} not found."

        # Summarize key differences
        b1 = b1 if isinstance(b1, dict) else {}
        b2 = b2 if isinstance(b2, dict) else {}

        lines = [f"## Round {r1} vs Round {r2}"]
        lines.append(self._compare_metric(b1, b2, "mean_length", "Episode Length"))
        lines.append(self._compare_metric(b1, b2, "termination_rate", "Termination Rate"))
        lines.append(self._compare_behavior(b1, b2, "action_magnitude"))
        lines.append(self._compare_behavior(b1, b2, "velocity_mean"))
        lines.append(self._compare_behavior(b1, b2, "action_efficiency"))
        lines.append(self._compare_entropy(b1, b2))

        return "\n".join(lines)

    def _tool_check_constraint_consistency(self, inp: str) -> str:
        """Check if current reward's incentives are consistent with termination conditions."""
        term_conds = self.board.get("environment_context", {}).get("termination_conditions", [])
        if not term_conds:
            return "No termination conditions found in evidence board."

        lines = ["## Constraint Consistency Check"]
        for tc in term_conds:
            cond = tc.get("condition", "")
            vars_involved = tc.get("variables", [])
            consistency = "OK"
            note = ""

            # Check if any involved variable appears in reward code
            for var in vars_involved:
                if var in self.reward_code:
                    consistency = "PRESENT"
                    note = f"Variable '{var}' referenced in reward"
                    break
                else:
                    # Check for partial matches
                    var_parts = var.split(".")
                    for part in var_parts:
                        if part in self.reward_code:
                            consistency = "PARTIAL"
                            note = f"Partial match: '{part}' found in reward (full var: {var})"
                            break

            if consistency == "OK":
                note = f"Variables {vars_involved} NOT found in reward — termination condition may be unaddressed"

            lines.append(
                f"- Condition `{cond}`: {consistency} — {note}"
            )

        return "\n".join(lines)

    def _tool_trace_variable(self, inp: str) -> str:
        """Trace a variable through the reward function. Shows all occurrences."""
        var_name = inp.strip()
        if not var_name:
            return "Specify a variable name to trace."

        lines = [f"## Tracing '{var_name}' in Reward Function"]

        if var_name in self.reward_code:
            # Find all lines containing this variable
            code_lines = self.reward_code.splitlines()
            hits = [
                (i + 1, line.strip())
                for i, line in enumerate(code_lines)
                if var_name in line
            ]
            if hits:
                for lineno, line in hits[:10]:
                    lines.append(f"  L{lineno}: {line[:120]}")
            else:
                lines.append(f"  Variable '{var_name}' NOT found in reward function")
        else:
            lines.append(f"  Variable '{var_name}' NOT found in reward function")

        # Check if it's in termination conditions
        for tc in self.board.get("environment_context", {}).get("termination_conditions", []):
            if var_name in tc.get("variables", []) or var_name in tc.get("condition", ""):
                lines.append(f"  ⚠ This variable IS part of a termination condition: `{tc['condition']}`")

        return "\n".join(lines)

    def _tool_analyze_efficiency(self, inp: str) -> str:
        """Analyze action efficiency from behavior descriptors."""
        bd = self.board.get("training_result", {}).get("behavior_descriptors", {})
        am = bd.get("action_magnitude", {})
        vel = None
        for k in ("velocity_x", "velocity_mean", "velocity"):
            vel = bd.get(k)
            if vel:
                break

        eff = bd.get("action_efficiency", {})

        lines = ["## Efficiency Analysis"]
        if am:
            lines.append(f"Action Magnitude: mean={am.get('mean', '?')}, std={am.get('std', '?')}, trend={am.get('trend', '?')}")
        if vel:
            lines.append(f"Velocity: mean={vel.get('mean', '?')}, std={vel.get('std', '?')}, trend={vel.get('trend', '?')}")
        if eff:
            lines.append(f"Efficiency (velocity/action): {eff.get('mean', '?')}")

        # Heuristic analysis
        am_mean = am.get("mean", 0)
        vel_mean = vel.get("mean", 0) if vel else 0
        if abs(am_mean) > 0.9 and abs(vel_mean) < 0.3 * abs(am_mean):
            lines.append("⚠ High action magnitude but low velocity gain → inefficient behavior")
        if eff and eff.get("mean", 0) > 5:
            lines.append("⚠ Very high efficiency ratio — check if velocity measurement is appropriate")

        return "\n".join(lines) if len(lines) > 1 else "Insufficient efficiency metrics available."

    def _tool_detect_principle_violation(self, inp: str) -> str:
        """Run constraint discovery on the evidence board."""
        events = self.board.get("training_result", {}).get("critical_events", [])
        health = self.board.get("training_result", {}).get("health_checks", {})

        lines = ["## Principle Violation Detection"]

        # Health check results
        for check_name, result in health.items():
            if isinstance(result, dict):
                passed = result.get("passed", True)
                detail = result.get("detail", "")
                icon = "✓" if passed else "✗"
                lines.append(f"{icon} {check_name}: {detail}")

        # Critical events
        if events:
            lines.append("\n### Critical Events")
            for e in events:
                lines.append(f"- [{e.get('severity', '?')}] {e.get('type')}: {e.get('description', '')[:200]}")

        if not events and all(
            isinstance(v, dict) and v.get("passed", True)
            for v in health.values() if isinstance(v, dict)
        ):
            lines.append("\nNo violations detected. All health checks passed.")

        return "\n".join(lines)

    # ── Comparison helpers ──────────────────────────────────────────

    @staticmethod
    def _compare_metric(b1: dict, b2: dict, key: str, label: str) -> str:
        v1 = b1.get("training_result", {}).get("episode_stats", {}).get(key, "?")
        v2 = b2.get("training_result", {}).get("episode_stats", {}).get(key, "?")
        return f"- {label}: Round {b1['meta']['round']}={v1} → Round {b2['meta']['round']}={v2}"

    @staticmethod
    def _compare_behavior(b1: dict, b2: dict, key: str) -> str:
        bd1 = b1.get("training_result", {}).get("behavior_descriptors", {})
        bd2 = b2.get("training_result", {}).get("behavior_descriptors", {})
        v1 = bd1.get(key, {}).get("mean", "?")
        v2 = bd2.get(key, {}).get("mean", "?")
        return f"- {key}: Round {b1['meta']['round']}={v1} → Round {b2['meta']['round']}={v2}"

    @staticmethod
    def _compare_entropy(b1: dict, b2: dict) -> str:
        hc1 = b1.get("training_result", {}).get("health_checks", {}).get("entropy_collapse", {})
        hc2 = b2.get("training_result", {}).get("health_checks", {}).get("entropy_collapse", {})
        v1 = hc1.get("final_entropy", "?")
        v2 = hc2.get("final_entropy", "?")
        return f"- Final Entropy: Round {b1['meta']['round']}={v1} → Round {b2['meta']['round']}={v2}"
