"""
core_memory.py — Token-limited core memory always in agent context.

Layer 1 of the three-layer memory system. Holds the most critical
information that every agent needs in its prompt. Self-managing:
agents can read/write via tools, and the system auto-trims to stay
under the token budget.

Design: MemGPT-inspired — small, immutable-core items plus a few
agent-managed slots for the current task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Rough token estimator: ~4 chars per token for English text, ~2 for code
_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


class CoreMemory:
    """Fixed-size core memory that fits in the agent's context window.

    Contains:
    - task_manifest: Permanent environment understanding (auto-set once)
    - agent_working_memory: Agent-managed scratchpad (read/write)
    - key_facts: Immutable short facts injected by the system
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.task_manifest: str = ""
        self.agent_working_memory: str = ""
        self.key_facts: list[str] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def set_task_manifest(self, manifest: str) -> None:
        """Set the permanent task manifest. Called once by EnvPerception."""
        self.task_manifest = manifest

    def add_key_fact(self, fact: str) -> None:
        """Add an immutable system fact (e.g., 'compute_reward takes 3 args')."""
        if fact not in self.key_facts:
            self.key_facts.append(fact)

    def set_working_memory(self, text: str) -> None:
        """Agent writes to its own scratchpad."""
        self.agent_working_memory = text

    def get_working_memory(self) -> str:
        return self.agent_working_memory

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(self, max_output_tokens: int = None) -> str:
        """Render core memory as a compact string for prompt injection.

        Prioritizes: task_manifest > key_facts > working_memory.
        Truncates to fit within token budget.
        """
        budget = max_output_tokens or self.max_tokens
        parts = []

        # Task manifest (permanent, highest priority)
        if self.task_manifest:
            parts.append("## Task Context (from Task Manifest)")
            parts.append(self._truncate_summary(self.task_manifest, 600))
            parts.append("")

        # Key facts (system-injected truths)
        if self.key_facts:
            parts.append("## Key Facts")
            for fact in self.key_facts:
                parts.append(f"- {fact}")
            parts.append("")

        # Working memory (agent-managed)
        if self.agent_working_memory:
            parts.append("## Working Memory")
            parts.append(self.agent_working_memory[:800])
            parts.append("")

        rendered = "\n".join(parts)
        if _estimate_tokens(rendered) > budget:
            rendered = self._trim_to_budget(rendered, budget)
        return rendered

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task_manifest": self.task_manifest[:4000],
            "key_facts": self.key_facts,
            "agent_working_memory": self.agent_working_memory[:2000],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text("utf-8"))
        self.task_manifest = data.get("task_manifest", "")
        self.key_facts = data.get("key_facts", [])
        self.agent_working_memory = data.get("agent_working_memory", "")

    # ── Internal ───────────────────────────────────────────────────────────

    def _truncate_summary(self, text: str, max_chars: int) -> str:
        """Extract the most important parts of a long text."""
        if len(text) <= max_chars:
            return text
        # Keep first section and key sections, drop details
        lines = text.split("\n")
        kept = []
        chars = 0
        in_important_section = True
        for line in lines:
            if chars + len(line) > max_chars:
                kept.append(f"\n... (truncated, {len(text)} total chars)")
                break
            # Skip detailed table rows when near budget
            if chars > max_chars * 0.7 and line.strip().startswith("|"):
                continue
            kept.append(line)
            chars += len(line) + 1
            # Stop keeping non-header lines once we hit the limit
            if chars > max_chars * 0.9 and not line.startswith("#"):
                in_important_section = False
            if not in_important_section and line.startswith("#"):
                in_important_section = True
            if not in_important_section:
                continue
        return "\n".join(kept)

    def _trim_to_budget(self, text: str, max_tokens: int) -> str:
        lines = text.split("\n")
        result = []
        tokens = 0
        for line in lines:
            t = _estimate_tokens(line)
            if tokens + t > max_tokens:
                break
            result.append(line)
            tokens += t
        return "\n".join(result)
