"""
tools.py — Real function-calling tool system for all agents.

Provides:
- ToolRegistry: Register tools with JSON schemas, execute them, format results
- Built-in tools: read_file, query_memory, read_eval_history, validate_code
- Integration with OpenAI-compatible function calling API

Every agent gets a ToolRegistry scoped to its allowed tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable


class Tool:
    """A callable tool with JSON schema for function calling."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: dict = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {"type": "object", "properties": {}, "required": []}

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: dict) -> str:
        """Execute the tool with parsed arguments. Returns string result."""
        try:
            result = self.func(**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"ERROR executing {self.name}: {e}"


class ToolRegistry:
    """Registry of tools available to an agent.

    Each agent gets a ToolRegistry scoped to its allowed tools.
    Supports OpenAI-compatible function calling.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schemas(self, names: list[str] = None) -> list[dict]:
        """Get OpenAI-compatible tool schemas for the given tool names."""
        if names is None:
            names = list(self._tools.keys())
        return [
            self._tools[name].to_openai_schema()
            for name in names
            if name in self._tools
        ]

    def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"ERROR: Tool '{name}' not found. Available: {list(self._tools.keys())}"
        return tool.execute(arguments)

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Tool factory — creates a ToolRegistry with standard built-in tools
# ═══════════════════════════════════════════════════════════════════════════════

def create_tool_registry(
    experiment_dir: Path,
    memory_system=None,
    allowed: list[str] = None,
) -> ToolRegistry:
    """Create a ToolRegistry with built-in tools scoped to allowed list.

    Built-in tools:
    - read_file: Read any file within the experiment directory
    - read_eval_history: Read evaluation history CSV for a round
    - query_memory: Search cross-round memory for lessons
    - query_archival: Search archival patterns
    - validate_python: Check Python code syntax
    - read_reward_code: Read the reward function source for a round
    - read_perception_report: Read perception report for a round

    Args:
        experiment_dir: Root experiment directory (for path safety).
        memory_system: MemorySystem instance for memory queries.
        allowed: List of tool names to include. None = all tools.

    Returns:
        ToolRegistry with the requested tools.
    """
    registry = ToolRegistry()
    exp_dir = Path(experiment_dir).resolve()
    allowed_set = set(allowed) if allowed else None

    def _ok(name: str) -> bool:
        return allowed_set is None or name in allowed_set

    # ── read_file ──────────────────────────────────────────────────────────
    if _ok("read_file"):
        def _read_file(rel_path: str) -> str:
            full = (exp_dir / rel_path).resolve()
            try:
                full.relative_to(exp_dir)
            except ValueError:
                return f"ERROR: Path '{rel_path}' escapes experiment directory"
            if not full.exists():
                return f"ERROR: File not found: {rel_path}"
            content = full.read_text("utf-8")
            if len(content) > 8000:
                content = content[:8000] + f"\n... (truncated, {len(content)} chars total)"
            return content

        registry.register(Tool(
            name="read_file",
            description="Read a file within the experiment directory. Returns file contents.",
            func=_read_file,
            parameters={
                "type": "object",
                "properties": {
                    "rel_path": {
                        "type": "string",
                        "description": "Relative path from experiment root, e.g. 'round3/perception_report.md'",
                    },
                },
                "required": ["rel_path"],
            },
        ))

    # ── read_eval_history ──────────────────────────────────────────────────
    if _ok("read_eval_history"):
        def _read_eval_history(round_num: int) -> str:
            csv_path = exp_dir / f"round{round_num}" / "evaluations" / "history.csv"
            if not csv_path.exists():
                return f"No eval history for round {round_num}"
            import csv
            rows = []
            with csv_path.open("r") as f:
                for row in csv.DictReader(f):
                    rows.append({
                        "timesteps": row.get("timesteps", "?"),
                        "mean_length": row.get("mean_length", "?"),
                        "env_metrics_summary": str(row.get("env_metrics", ""))[:200],
                    })
            return json.dumps(rows[-5:], indent=2)

        registry.register(Tool(
            name="read_eval_history",
            description="Read the last 5 evaluation checkpoints for a round. Returns mean_length and env_metrics trends.",
            func=_read_eval_history,
            parameters={
                "type": "object",
                "properties": {
                    "round_num": {"type": "integer", "description": "Round number to query"},
                },
                "required": ["round_num"],
            },
        ))

    # ── query_memory ───────────────────────────────────────────────────────
    if _ok("query_memory") and memory_system:
        def _query_memory(query: str, max_results: int = 5) -> str:
            results = memory_system.query_lessons(query, max_results=max_results)
            if not results:
                return f"No memory entries found for '{query}'"
            return "\n\n".join(
                f"[Match {i}] {r}" for i, r in enumerate(results, 1)
            )

        registry.register(Tool(
            name="query_memory",
            description="Search the memory system for relevant lessons and patterns from past rounds.",
            func=_query_memory,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'reward hacking' or 'sparse reward'"},
                    "max_results": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["query"],
            },
        ))

    # ── query_archival ─────────────────────────────────────────────────────
    if _ok("query_archival") and memory_system:
        def _query_archival(query: str, env_type: str = "", k: int = 3) -> str:
            results = memory_system.archival.search(query, k=k, env_type=env_type)
            if not results:
                return "No archival patterns found."
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] [{r.get('env_type', 'general')}] importance={r.get('importance', '?')} "
                    f"score={r.get('score', 0):.3f}\n{r['content'][:300]}"
                )
            return "\n\n".join(lines)

        registry.register(Tool(
            name="query_archival",
            description="Search cross-experiment archival patterns for generalizable design principles.",
            func=_query_archival,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "env_type": {"type": "string", "description": "Optional env type filter: locomotion, landing, balance, navigation, manipulation, flight"},
                    "k": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        ))

    # ── validate_python ─────────────────────────────────────────────────────
    if _ok("validate_python"):
        def _validate_python(code: str) -> str:
            try:
                compile(code, "<validate>", "exec")
                return "Syntax OK — code compiles successfully."
            except SyntaxError as e:
                return f"SyntaxError: {e}"

        registry.register(Tool(
            name="validate_python",
            description="Check Python code for syntax errors. Use before finalizing generated code.",
            func=_validate_python,
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to validate"},
                },
                "required": ["code"],
            },
        ))

    # ── read_reward_code ────────────────────────────────────────────────────
    if _ok("read_reward_code"):
        def _read_reward_code(round_num: int) -> str:
            path = exp_dir / f"round{round_num}" / "reward_fn_source.py"
            if not path.exists():
                return f"No reward code for round {round_num}"
            content = path.read_text("utf-8")
            if len(content) > 4000:
                content = content[:4000] + "\n... (truncated)"
            return content

        registry.register(Tool(
            name="read_reward_code",
            description="Read the reward function source code for a specific round.",
            func=_read_reward_code,
            parameters={
                "type": "object",
                "properties": {
                    "round_num": {"type": "integer", "description": "Round number"},
                },
                "required": ["round_num"],
            },
        ))

    # ── read_perception_report ──────────────────────────────────────────────
    if _ok("read_perception_report"):
        def _read_perception_report(round_num: int) -> str:
            path = exp_dir / f"round{round_num}" / "perception_report.md"
            if not path.exists():
                return f"No perception report for round {round_num}"
            content = path.read_text("utf-8")
            if len(content) > 4000:
                content = content[:4000] + "\n... (truncated)"
            return content

        registry.register(Tool(
            name="read_perception_report",
            description="Read the perception report (behavior analysis) for a specific round.",
            func=_read_perception_report,
            parameters={
                "type": "object",
                "properties": {
                    "round_num": {"type": "integer", "description": "Round number"},
                },
                "required": ["round_num"],
            },
        ))

    return registry
