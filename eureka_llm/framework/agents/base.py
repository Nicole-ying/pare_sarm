"""Agent base class with structured role definition.

Every agent in the framework inherits from this class. It provides:
- Structured AgentRole (role, goal, backstory) for prompt construction
- Tool registry for function calling
- Input/output schema enforcement
- Memory system integration
- Message pool integration

Design: CrewAI-inspired role-goal-backstory + AutoGen-style tool white-listing.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = object  # type: ignore

# Import communication layer
import sys
_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from communication.schemas import AgentMessage
from communication.message_pool import MessagePool


@dataclass
class AgentRole:
    """Immutable agent identity.

    Inspired by CrewAI's Role-Goal-Backstory pattern:
    - Role: scoped job title that constrains behavior
    - Goal: one-sentence outcome the agent optimizes for
    - Backstory: context, design philosophy, and operating constraints

    Each agent has EXACTLY one narrow competency. A narrow agent
    that does one thing well is better than a broad agent that
    does everything poorly.
    """

    name: str
    role: str
    goal: str
    backstory: str
    tools: list[str] = field(default_factory=list)
    temperature: float = 0.3
    max_retries: int = 3

    def as_system_prompt(self, extra_context: str = "") -> str:
        """Render the role as a system prompt block."""
        parts = [
            f"# Role: {self.role}",
            f"",
            f"## Goal",
            self.goal,
            f"",
            f"## Backstory",
            self.backstory,
        ]
        if self.tools:
            parts.append(f"")
            parts.append(f"## Available Tools")
            parts.append(f"You have access to: {', '.join(self.tools)}")
            parts.append(f"Use them to gather information before making decisions.")
        if extra_context:
            parts.append(f"")
            parts.append(extra_context)
        return "\n".join(parts)


class Agent(ABC):
    """Base class for all framework agents.

    Subclasses must implement:
    - build_prompt(input_msg, pool, memory) → str
    - parse_output(response) → AgentMessage

    The run() method handles the full execution cycle:
    - Read relevant messages from pool
    - Build prompt from role + input + memory
    - Call LLM with retry logic
    - Parse structured output
    - Publish output message to pool
    """

    def __init__(
        self,
        role: AgentRole,
        api_key: str,
        model: str = "deepseek-reasoner",
        message_pool: MessagePool | None = None,
        memory_system=None,
    ):
        self.role = role
        self.api_key = api_key
        self.model = model
        self.pool = message_pool or MessagePool()
        self.memory = memory_system

        # Subscribe to relevant message types (set by subclass)
        self._subscribed_types: list[str] = []

    def subscribe(self, message_types: list[str]) -> None:
        """Register interest in specific message types."""
        self._subscribed_types = message_types
        if self.pool:
            self.pool.subscribe(self.role.name, message_types)

    @abstractmethod
    def build_prompt(
        self,
        input_msg: AgentMessage | None,
        pool: MessagePool,
        memory,
    ) -> str:
        """Build the LLM prompt from role, input, and context.

        Subclasses must implement this — it's where the agent's
        reasoning logic lives.
        """
        ...

    @abstractmethod
    def parse_output(self, response: str) -> AgentMessage:
        """Parse LLM response into a structured AgentMessage.

        Subclasses must implement this — it enforces the output schema.
        """
        ...

    def run(
        self,
        input_msg: AgentMessage | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> AgentMessage:
        """Execute the full agent cycle.

        1. Read messages from pool
        2. Build prompt
        3. Call LLM (with retry)
        4. Parse output
        5. Publish to pool

        Args:
            input_msg: Optional direct input message (bypasses pool read).
            log_fn: Optional logging function.

        Returns:
            The output AgentMessage published to the pool.
        """
        log = log_fn or print

        # Read context from pool
        context_msgs = []
        if self.pool and self._subscribed_types:
            context_msgs = self.pool.get_for(self.role.name)

        log(f"  [{self.role.name}] Starting — {len(context_msgs)} context messages available")

        # Build prompt
        prompt = self.build_prompt(input_msg, self.pool, self.memory)

        # Call LLM with retry
        response = None
        last_error = None
        for attempt in range(1, self.role.max_retries + 1):
            try:
                from llm_call import call_llm
                response = call_llm(
                    prompt, self.api_key, self.model, self.role.temperature
                )
                break
            except Exception as e:
                last_error = str(e)
                log(f"  [{self.role.name}] LLM call failed (attempt {attempt}): {e}")

        if response is None:
            # All retries exhausted — return error message
            error_msg = AgentMessage(
                sender=self.role.name,
                message_type="error",
                round_num=input_msg.round_num if input_msg else 0,
                content={"error": last_error or "Unknown LLM error"},
            )
            if self.pool:
                self.pool.publish(error_msg)
            return error_msg

        # Parse output
        output_msg = self.parse_output(response)

        # Publish to pool
        if self.pool:
            self.pool.publish(output_msg)

        log(f"  [{self.role.name}] Done → {output_msg.message_type}")
        return output_msg

    def query_memory(self, query_text: str, max_results: int = 5) -> str:
        """Search cross-round memory for relevant lessons.

        Args:
            query_text: Natural language query.
            max_results: Maximum number of results.

        Returns:
            Formatted string of relevant memory entries.
        """
        if not self.memory:
            return "(memory system not available)"
        results = self.memory.query_lessons(query_text, max_results=max_results)
        if not results:
            return f"No memory entries found for '{query_text}'."
        return "\n\n".join(
            f"**Memory Match {i}:** {lesson}"
            for i, lesson in enumerate(results, 1)
        )

    def read_file(self, rel_path: str, experiment_dir: Path | None = None) -> str:
        """Read a file relative to the experiment directory.

        Safe path-restricted read — cannot escape experiment directory.
        """
        if experiment_dir is None:
            return "ERROR: No experiment directory set."

        full_path = (experiment_dir / rel_path).resolve()
        try:
            full_path.relative_to(experiment_dir.resolve())
        except ValueError:
            return f"ERROR: Path escapes experiment directory: {rel_path}"

        if not full_path.exists():
            return f"ERROR: File not found: {rel_path}"

        try:
            content = full_path.read_text("utf-8")
            if len(content) > 10000:
                content = content[:10000] + f"\n... (truncated, {len(content)} total chars)"
            return content
        except Exception as e:
            return f"ERROR: Cannot read {rel_path}: {e}"
