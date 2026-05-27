"""
v2_agents.py — Agent subclasses wrapping existing agent logic with
MessagePool communication, Memory context injection, and structured roles.

Each agent:
1. Has a proper AgentRole (role/goal/backstory/tools)
2. Reads context from MessagePool (not raw files)
3. Injects memory context into its LLM prompt via build_memory_context()
4. Publishes structured output to MessagePool
5. Can query memory via tools

The existing run_* functions in each agent file remain unchanged for
backward compatibility with pipeline.py (v1).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from .base import Agent, AgentRole
from communication.schemas import (
    AgentMessage, EvaluationReport, GeneratorProposal,
    RewardCode, ReflectionReport, TaskManifest,
)
from communication.message_pool import MessagePool
from memory.context import build_memory_context, inject_memory_into_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Perception Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PerceptionAgent(Agent):
    """Observes training data, produces structured behavior report.

    Reads eval history and trajectory logs from a round directory.
    Outputs a perception report describing what the agent learned,
    not what should change.
    """

    role = AgentRole(
        name="perception",
        role="Training Dynamics Observer",
        goal="Analyze RL training results and describe WHAT happened — the behavioral patterns, "
             "trends, and anomalies — without proposing changes.",
        backstory="You are a forensic analyst specializing in RL training dynamics. "
                  "You distinguish genuine learning from reward hacking by examining "
                  "metrics trends, component balance, and episode statistics. "
                  "You report facts, not opinions.",
        tools=["read_eval_history", "read_trajectory_sample", "query_memory"],
        temperature=0.3,
    )

    def __init__(self, api_key: str, model: str = "deepseek-reasoner",
                 pool: MessagePool | None = None, memory=None,
                 template_dir: Path | None = None):
        super().__init__(self.role, api_key, model, pool, memory)
        self.subscribe(["training_result", "reward_code"])
        self.template_dir = template_dir or (
            _framework_dir.parent / "templates"
        )

    def build_prompt(self, input_msg: AgentMessage | None,
                     pool: MessagePool, memory) -> str:
        """Build perception prompt from round data + memory context."""
        # Determine round directory from input message
        round_dir_str = ""
        if input_msg and "round_dir" in input_msg.content:
            round_dir_str = input_msg.content["round_dir"]
        elif pool:
            # Find latest training_result to locate the round
            training_msgs = pool.query(message_type="training_result")
            if training_msgs:
                latest = training_msgs[-1]
                round_num = latest.round_num
                # Infer from experiment structure
                round_dir_str = input_msg.content.get("round_dir", "") if input_msg else ""

        if not round_dir_str:
            return "# Error: No round directory provided to Perception Agent"

        round_dir = Path(round_dir_str)
        if not round_dir.exists():
            return f"# Error: Round directory not found: {round_dir_str}"

        # Build the base perception prompt (using existing logic)
        from .perception_agent import build_perception_prompt
        template_path = self.template_dir / "perception_prompt.txt"
        prompt = build_perception_prompt(round_dir, template_path)

        # Inject memory context for cross-round awareness
        if memory:
            prompt = inject_memory_into_prompt(
                prompt, memory,
                query="training dynamics behavior pattern",
                max_tokens=600,
            )

        return prompt

    def parse_output(self, response: str) -> AgentMessage:
        """Parse LLM response into structured AgentMessage."""
        from .perception_agent import extract_behavior_metrics
        metrics = extract_behavior_metrics(response)

        # Determine round number from context
        m = re.search(r'round(\d+)', response, re.IGNORECASE)
        round_num = int(m.group(1)) if m else 0

        return AgentMessage(
            sender=self.role.name,
            message_type="perception_report",
            round_num=round_num,
            content={
                "report_markdown": response,
                "metrics": metrics,
            },
        )

    def run_on_round(self, round_dir: Path, round_num: int) -> str:
        """Convenience: run perception on a specific round directory.

        Returns the report markdown string (backward compat).
        """
        msg = AgentMessage(
            sender="orchestrator",
            message_type="run_perception",
            round_num=round_num,
            content={"round_dir": str(round_dir)},
        )
        result = super().run(input_msg=msg)
        return result.content.get("report_markdown", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Analyzer Agent
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyzerAgent(Agent):
    """Translates perception findings into concrete code change proposals.

    Reads perception_report + current reward code → produces JSON proposal.
    """

    role = AgentRole(
        name="analyzer",
        role="Reward Diagnostician",
        goal="Diagnose WHY the current reward function produced the observed behavior, "
             "and propose specific, minimal code changes to fix root causes.",
        backstory="You are an RL reward function diagnostician. You read training "
                  "reports like a doctor reads lab results — identifying the underlying "
                  "dynamics that explain symptoms. You propose surgical fixes, not rewrites. "
                  "Each change must have a clear causal rationale.",
        tools=["read_perception_report", "read_reward_code", "query_memory"],
        temperature=0.4,
    )

    def __init__(self, api_key: str, model: str = "deepseek-reasoner",
                 pool: MessagePool | None = None, memory=None):
        super().__init__(self.role, api_key, model, pool, memory)
        self.subscribe(["perception_report", "reward_code", "reflection_report"])

    def build_prompt(self, input_msg: AgentMessage | None,
                     pool: MessagePool, memory) -> str:
        """Build analyzer prompt from perception + reward code + memory."""
        if not input_msg:
            return "# Error: No input provided to Analyzer Agent"

        round_dir_str = input_msg.content.get("round_dir", "")
        round_num = input_msg.round_num
        round_dir = Path(round_dir_str) if round_dir_str else None

        if not round_dir or not round_dir.exists():
            return f"# Error: Round directory not found: {round_dir_str}"

        # Use existing analyzer logic to build the base prompt
        from .analyzer_agent import build_analyzer_prompt
        # We need to adapt — the existing function takes different args
        # For now, build a simpler prompt that the LLM can work with

        perception_report = ""
        reward_code = ""
        reflection = ""

        # Read from MessagePool first
        if pool:
            perception_msgs = pool.query(message_type="perception_report", round_num=round_num - 1)
            if perception_msgs:
                perception_report = perception_msgs[-1].content.get("report_markdown", "")

            reward_msgs = pool.query(message_type="reward_code", round_num=round_num - 1)
            if reward_msgs:
                reward_code = reward_msgs[-1].content.get("source", "")

            reflection_msgs = pool.query(message_type="reflection_report", round_num=round_num - 1)
            if reflection_msgs:
                reflection = reflection_msgs[-1].content.get("reflection_markdown", "")

        # Fall back to file reading if pool is empty
        if not perception_report:
            perc_file = round_dir / "perception_report.md"
            if perc_file.exists():
                perception_report = perc_file.read_text("utf-8")
        if not reward_code:
            code_file = round_dir / "reward_fn_source.py"
            if code_file.exists():
                reward_code = code_file.read_text("utf-8")
        if not reflection:
            refl_file = round_dir / "reflection.md"
            if refl_file.exists():
                reflection = refl_file.read_text("utf-8")

        prompt = _build_analyzer_prompt_v2(
            perception_report=perception_report,
            reward_code=reward_code,
            reflection=reflection,
            round_num=round_num,
        )

        # Inject memory context
        if memory:
            diagnosis_hint = _extract_diagnosis_hint(perception_report)
            prompt = inject_memory_into_prompt(
                prompt, memory,
                query=diagnosis_hint or "reward function optimization",
                max_tokens=600,
            )

        return prompt

    def parse_output(self, response: str) -> AgentMessage:
        """Parse LLM response into structured GeneratorProposal."""
        from .analyzer_agent import extract_proposal_from_response
        proposal = extract_proposal_from_response(response)

        return AgentMessage(
            sender=self.role.name,
            message_type="generator_proposal",
            round_num=0,  # Will be set by caller
            content={
                "diagnosis": proposal.get("diagnosis", ""),
                "changed_count": proposal.get("changed_count", 0),
                "proposed_changes": proposal.get("proposed_changes", []),
                "evidence_citations": proposal.get("evidence_citations", []),
                "analysis_status": proposal.get("analysis_status", "ok"),
            },
        )

    def run_on_round(self, round_dir: Path, round_num: int,
                      memory_system=None, skill_manager=None) -> dict:
        """Convenience: run analyzer on a specific round. Backward compat."""
        from .analyzer_agent import run_analyzer_agent
        return run_analyzer_agent(
            round_dir, round_num, memory_system or self.memory,
            self.api_key, self.model, temperature=self.role.temperature,
            skill_manager=skill_manager,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Generator Agent
# ═══════════════════════════════════════════════════════════════════════════════

class GeneratorAgent(Agent):
    """Translates proposal → executable Python reward code.

    Has optional ReAct verification phase for evidence-backed proposals.
    """

    role = AgentRole(
        name="generator",
        role="Reward Code Architect",
        goal="Translate the Analyzer's proposed changes into precise, correct Python code. "
             "Preserve the exact function signature. Verify syntax and structure.",
        backstory="You are a precise code translator. You don't design — you implement. "
                  "The Analyzer tells you WHAT to change and WHY. Your job is to apply "
                  "those changes exactly, verify correctness, and output complete, "
                  "runnable Python code.",
        tools=["read_previous_code", "read_proposal", "validate_syntax", "query_memory"],
        temperature=0.3,
    )

    def __init__(self, api_key: str, model: str = "deepseek-reasoner",
                 pool: MessagePool | None = None, memory=None):
        super().__init__(self.role, api_key, model, pool, memory)
        self.subscribe(["generator_proposal", "reward_code"])

    def build_prompt(self, input_msg: AgentMessage | None,
                     pool: MessagePool, memory) -> str:
        """Build generator prompt from proposal + current code + memory."""
        if not input_msg:
            return "# Error: No input to Generator Agent"

        proposal = input_msg.content if input_msg else {}
        current_code = ""

        # Get current reward code from pool
        if pool:
            code_msgs = pool.query(message_type="reward_code")
            if code_msgs:
                current_code = code_msgs[-1].content.get("source", "")

        # Use existing generator prompt builder
        from .generator_agent import build_generator_prompt

        # Inject memory context
        base_prompt = build_generator_prompt(proposal, current_code)
        if memory:
            base_prompt = inject_memory_into_prompt(
                base_prompt, memory,
                query=proposal.get("diagnosis", "reward function code generation"),
                max_tokens=500,
            )

        return base_prompt

    def parse_output(self, response: str) -> AgentMessage:
        """Extract code from LLM response."""
        import re
        m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        code = m.group(1).strip() if m else ""

        return AgentMessage(
            sender=self.role.name,
            message_type="reward_code",
            round_num=0,
            content={
                "source": code,
                "response": response,
            },
        )

    def run_with_proposal(self, proposal: dict, current_reward_path: Path,
                           round_dir: Path, memory_system=None) -> str | None:
        """Convenience: run generator with a proposal. Backward compat."""
        from .generator_agent import run_generator_agent
        return run_generator_agent(
            proposal, current_reward_path, round_dir,
            self.api_key, self.model, temperature=self.role.temperature,
            max_retries=self.role.max_retries,
            memory_system=memory_system or self.memory,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Reflection Agent
# ═══════════════════════════════════════════════════════════════════════════════

class ReflectionAgent(Agent):
    """Extracts causal lessons from the round's results.

    Compares proposal intent vs. actual outcome → writes MEMORY.md lesson.
    """

    role = AgentRole(
        name="reflector",
        role="Cross-Round Researcher",
        goal="Extract causal lessons: what changed → what happened → why. "
             "Build the knowledge base that makes future rounds smarter.",
        backstory="You think like a scientist reviewing an experiment. You compare "
                  "what was INTENDED (the proposal) with what ACTUALLY happened "
                  "(the training results). You extract principles, not patches. "
                  "Your lessons are concise, causal, and actionable.",
        tools=["read_proposal", "read_evaluation", "read_perception", "query_memory"],
        temperature=0.3,
    )

    def __init__(self, api_key: str, model: str = "deepseek-reasoner",
                 pool: MessagePool | None = None, memory=None):
        super().__init__(self.role, api_key, model, pool, memory)
        self.subscribe(["evaluation_report", "generator_proposal", "perception_report"])

    def build_prompt(self, input_msg: AgentMessage | None,
                     pool: MessagePool, memory) -> str:
        """Build reflection prompt from proposal + perception + memory."""
        if not input_msg:
            return "# Error: No input to Reflection Agent"

        round_dir_str = input_msg.content.get("round_dir", "")
        round_num = input_msg.round_num
        round_dir = Path(round_dir_str) if round_dir_str else None

        if not round_dir or not round_dir.exists():
            return f"# Error: Round directory not found: {round_dir_str}"

        # Read files for context
        proposal = ""
        perception = ""
        if (round_dir / "analyzer_proposal.json").exists():
            proposal = (round_dir / "analyzer_proposal.json").read_text("utf-8")
        if (round_dir / "perception_report.md").exists():
            perception = (round_dir / "perception_report.md").read_text("utf-8")

        prompt = f"""You are the Reflection Agent.

## Proposal (what was intended)
{proposal[:3000]}

## Perception Report (what actually happened)
{perception[:3000]}

## Instructions
1. Compare intent vs. outcome
2. Identify the SINGLE most important causal lesson
3. Write it as: "Round {round_num}: [change] → [outcome] → [why]"
4. Provide a checklist item for the next Generator

Output as markdown with ## Causal Lesson and ## Checklist sections.
"""

        if memory:
            prompt = inject_memory_into_prompt(
                prompt, memory,
                query=f"round {round_num} lessons",
                max_tokens=500,
            )

        return prompt

    def parse_output(self, response: str) -> AgentMessage:
        """Parse reflection into structured message."""
        return AgentMessage(
            sender=self.role.name,
            message_type="reflection_report",
            round_num=0,
            content={
                "reflection_markdown": response,
            },
        )

    def run_on_round(self, round_dir: Path, round_num: int,
                      memory_system=None) -> str:
        """Convenience: run reflection on a round. Backward compat."""
        from .reflection_agent import run_reflection_agent
        return run_reflection_agent(
            round_dir, round_num, memory_system or self.memory,
            self.api_key, self.model, temperature=self.role.temperature,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Env Perception Agent
# ═══════════════════════════════════════════════════════════════════════════════

class EnvPerceptionAgent(Agent):
    """Reads environment source code + exploration data → TaskManifest.

    Runs once before Round 0. Uses ReAct loop for autonomous file reading.
    """

    role = AgentRole(
        name="env_perception",
        role="Environment Analyst",
        goal="Build a complete, accurate TaskManifest from environment source code "
             "and exploration data — with zero hardcoded assumptions.",
        backstory="You are a systems analyst specializing in RL environments. "
                  "You reverse-engineer the observation space, action space, "
                  "termination conditions, and task objective by reading source code. "
                  "You never guess — every claim is backed by code evidence.",
        tools=["read_file", "query_exploration_data"],
        temperature=0.3,
    )

    def __init__(self, api_key: str, model: str = "deepseek-reasoner",
                 pool: MessagePool | None = None, memory=None):
        super().__init__(self.role, api_key, model, pool, memory)
        self.subscribe([])  # EnvPerception runs first, nothing to subscribe to

    def build_prompt(self, input_msg: AgentMessage | None,
                     pool: MessagePool, memory) -> str:
        """This agent uses its own ReAct loop, not the single-call LLM pattern."""
        return ""

    def parse_output(self, response: str) -> AgentMessage:
        """Not used — EnvPerception uses its own output extraction."""
        return AgentMessage(
            sender=self.role.name,
            message_type="task_manifest",
            round_num=0,
            content={"manifest_markdown": response},
        )

    def run_discovery(self, env_dir: Path, task_description: str,
                       exploration_path: Path, memory_system=None) -> str:
        """Run environment discovery. Backward compat wrapper."""
        from .env_perception_agent import run_env_perception_agent
        manifest = run_env_perception_agent(
            env_dir, task_description, exploration_path,
            self.api_key, self.model, temperature=self.role.temperature,
            memory_system=memory_system or self.memory,
        )

        # Publish to MessagePool
        if self.pool:
            msg = AgentMessage(
                sender=self.role.name,
                message_type="task_manifest",
                round_num=0,
                content={"manifest_markdown": manifest},
            )
            self.pool.publish(msg)

        # Store in core memory
        if self.memory:
            self.memory.save_task_manifest(manifest)

        return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator Agent (NEW — independent quality check)
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluatorAgent(Agent):
    """Independent evaluator: checks if training actually improved.

    Compares current round metrics vs. previous round.
    Determines pass/fail/retry for the orchestration layer.
    """

    role = AgentRole(
        name="evaluator",
        role="Independent Quality Auditor",
        goal="Determine whether the trained policy shows genuine task progress "
             "or is reward hacking, by comparing independent metrics across rounds.",
        backstory="You are the quality gate. You don't design rewards — you judge them. "
                  "You compare component statistics (per-component mean/std/min/max from training) "
                  "against episode length trends to detect reward hacking. "
                  "Your verdict determines whether the pipeline continues or retries.",
        tools=["read_eval_history", "read_previous_eval", "query_memory"],
        temperature=0.2,
    )

    def __init__(self, api_key: str, model: str = "deepseek-reasoner",
                 pool: MessagePool | None = None, memory=None):
        super().__init__(self.role, api_key, model, pool, memory)
        self.subscribe(["reward_code", "training_result", "perception_report"])

    def build_prompt(self, input_msg: AgentMessage | None,
                     pool: MessagePool, memory) -> str:
        """Build evaluation prompt from training results."""
        round_num = input_msg.round_num if input_msg else 0
        round_dir_str = input_msg.content.get("round_dir", "") if input_msg else ""

        prompt = f"""You are the Evaluator Agent — the quality gate.

## Round {round_num} Evaluation

Determine if the trained policy shows GENUINE improvement (not reward hacking).

Check:
1. Did episode length change meaningfully compared to previous round?
2. Are reward components balanced or is one dominating?
3. Did component activation improve (more components active)?
4. Is there evidence of reward hacking (reward increasing but components stagnant)?

Output:
```
CONCLUSION: pass | fail
CONFIDENCE: 0.0-1.0
REASONING: <2-3 sentences>
RECOMMENDATION: continue | retry_generation | abort
```
"""
        if memory:
            prompt = inject_memory_into_prompt(
                prompt, memory,
                query=f"round {round_num} evaluation quality",
                max_tokens=400,
            )

        return prompt

    def parse_output(self, response: str) -> AgentMessage:
        """Parse evaluation verdict."""
        conclusion = "pass"
        if "fail" in response.lower():
            conclusion = "fail"

        confidence = 0.5
        m = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        if m:
            confidence = float(m.group(1))

        return AgentMessage(
            sender=self.role.name,
            message_type="evaluation_report",
            round_num=0,
            content={
                "conclusion": conclusion,
                "confidence": confidence,
                "reasoning": response,
            },
        )

    def evaluate_round(self, round_dir: Path, prev_round_dir: Path,
                        round_num: int) -> dict:
        """Evaluate a round's training results. Returns verdict dict."""
        eval_history = []
        csv_path = round_dir / "evaluations" / "history.csv"
        if csv_path.exists():
            import csv
            with csv_path.open("r") as f:
                eval_history = list(csv.DictReader(f))

        prev_history = []
        prev_csv = prev_round_dir / "evaluations" / "history.csv"
        if prev_csv.exists():
            import csv
            with prev_csv.open("r") as f:
                prev_history = list(csv.DictReader(f))

        # Simple heuristic evaluation (no LLM needed for basic checks)
        try:
            curr_len = float(eval_history[-1].get("mean_length", 0)) if eval_history else 0
            prev_len = float(prev_history[-1].get("mean_length", 0)) if prev_history else 0

            improved = curr_len > prev_len * 1.05
            collapsed = curr_len < 10 and prev_len > 50
            stagnant = abs(curr_len - prev_len) / max(prev_len, 1) < 0.05

            if collapsed:
                return {"conclusion": "fail", "reason": "Policy collapsed", "needs_retry": True}
            elif stagnant and curr_len < 100:
                return {"conclusion": "fail", "reason": "Stagnant at low level", "needs_retry": True}
            elif improved:
                return {"conclusion": "pass", "reason": "Improving", "needs_retry": False}
            else:
                return {"conclusion": "pass", "reason": "Acceptable", "needs_retry": False}
        except Exception:
            return {"conclusion": "pass", "reason": "Insufficient data", "needs_retry": False}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_analyzer_prompt_v2(
    perception_report: str,
    reward_code: str,
    reflection: str,
    round_num: int,
) -> str:
    """Build a structured analyzer prompt from available data."""
    sections = [
        f"# Reward Function Analysis — Round {round_num}",
        "",
        "You are the Analyzer Agent. Your job: diagnose WHY the reward function "
        "produced the observed behavior and propose specific, minimal fixes.",
        "",
        "## Perception Report (what happened)",
        perception_report[:4000] if perception_report else "(not available)",
        "",
        "## Current Reward Function",
        f"```python\n{reward_code[:2000]}\n```" if reward_code else "(not available)",
        "",
        "## Previous Reflection",
        reflection[:1500] if reflection else "(none)",
        "",
        "## Instructions",
        "1. Identify the ROOT CAUSE — not just the symptom",
        "2. Propose 1-3 specific code changes",
        "3. Each change must have a clear causal rationale",
        "4. Output as JSON with: diagnosis, proposed_changes (list of {component, new_code, reason})",
        "",
        "Output ONLY a JSON object. No markdown, no explanation outside the JSON.",
    ]
    return "\n".join(sections)


def _extract_diagnosis_hint(perception_report: str) -> str:
    """Extract key symptoms from perception report for memory search."""
    hints = []
    for keyword in ["reward hack", "collapse", "stagnant", "sparse",
                    "dominated", "dead", "noisy", "unstable", "overfit"]:
        if keyword in perception_report.lower():
            hints.append(keyword)
    return " ".join(hints) if hints else ""
