"""
react_agent.py — Reusable ReAct (Reasoning + Acting) engine for all agents.

Architecture:
    ToolRegistry registers tools with name, description, and implementation.
    run_react_loop drives the LLM through a tool-use cycle:

        LLM output → parse tool call → execute → append result → repeat
                    → parse FINAL ANSWER → return output

    Tool call protocol (LLM → system):
        <tool_name>argument</tool_name>
        or
        tool_name("argument")

    LLM signals completion with: FINAL ANSWER
        Everything after FINAL ANSWER is returned as final_output.

    Path safety:
        read_file is restricted to the experiment directory.
        Shortcuts: read_eval(N), read_perception(N), read_reward(N)
                   resolve to round-relative paths.

Usage:
    from react_agent import ToolRegistry, setup_default_tools, run_react_loop

    tools = setup_default_tools(experiment_dir, memory_system)
    result = run_react_loop(
        system_prompt="Your task is to analyze...",
        tools=tools,
        api_key=api_key,
        model="deepseek-reasoner",
    )
    print(result["final_output"])
"""

import json
import re
import sys
from pathlib import Path

_framework_dir = Path(__file__).resolve().parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
from llm_call import call_llm


# ── Tool definition ───────────────────────────────────────────────────────

class Tool:
    """A callable tool available to the LLM in the ReAct loop."""

    def __init__(self, name: str, description: str, fn,
                 param_name: str = "arg",
                 param_description: str = ""):
        self.name = name
        self.description = description
        self.fn = fn
        self.param_name = param_name
        self.param_description = param_description or f"({param_name})"


class ToolRegistry:
    """Registry of tools available to the ReAct loop.

    Provides:
        - Tool registration with descriptions
        - Formatted tool list for prompt injection
        - Safe execution with error handling
        - Path-restricted file access
    """

    def __init__(self, experiment_dir: Path):
        self._tools: dict[str, Tool] = {}
        self._experiment_dir = experiment_dir.resolve()

    def register(self, name: str, description: str, fn,
                 param_name: str = "arg",
                 param_description: str = ""):
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        self._tools[name] = Tool(
            name, description, fn, param_name, param_description,
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> str:
        """Return markdown-formatted tool list for prompt injection."""
        lines = ["## Available Tools\n"]
        for name in sorted(self._tools.keys()):
            t = self._tools[name]
            lines.extend([
                f"### {name}",
                t.description,
                f"  Parameter: `{t.param_name}` — {t.param_description}",
                "",
            ])
        lines.extend([
            "### Protocol",
            'Call a tool with: `tool_name("argument")` or `<tool_name>argument</tool_name>`',
            "Signal completion with: `FINAL ANSWER` followed by your final output.",
            "",
        ])
        return "\n".join(lines)

    def call(self, name: str, arg: str) -> str:
        """Execute a tool by name with a string argument. Returns result string."""
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(sorted(self._tools.keys()))
            return f"ERROR: Unknown tool '{name}'. Available: {available}"
        try:
            result = tool.fn(arg)
            if result is None:
                return "(no output)"
            return str(result)
        except Exception as e:
            return f"ERROR: Tool '{name}' failed: {e}"

    def resolve_path(self, rel_path: str) -> Path:
        """Resolve a relative path and verify it stays within experiment dir."""
        full = (self._experiment_dir / rel_path).resolve()
        if not _path_is_within(full, self._experiment_dir):
            raise PermissionError(
                f"Path escapes experiment directory: {rel_path} → {full}"
            )
        return full


# ── Helpers ───────────────────────────────────────────────────────────────

def _path_is_within(path: Path, parent: Path) -> bool:
    """Check if path is within (or equal to) parent directory."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _detect_tool_call(text: str) -> tuple[str, str] | None:
    """Detect a tool call in LLM output.

    Supports two formats:
        1. <tool_name>argument</tool_name>
        2. tool_name("argument") or tool_name('argument')

    Returns:
        (tool_name, argument) tuple, or None if no tool call detected.
    """
    if not text:
        return None

    # Format 1: <tool_name>argument</tool_name>
    m = re.search(r'<([a-zA-Z_]\w*)>([^<]*)</\1>', text)
    if m:
        return (m.group(1), m.group(2).strip())

    # Format 2: tool_name("argument") or tool_name('argument')
    m = re.search(r'([a-zA-Z_]\w*)\s*\(\s*["\']([^"\']+)["\']\s*\)', text)
    if m:
        return (m.group(1), m.group(2).strip())

    return None


# ── Built-in tool factories ──────────────────────────────────────────────

def _make_read_file(experiment_dir: Path):
    """Create a path-restricted file reader tool."""
    exp_dir = experiment_dir.resolve()

    def read_file(rel_path: str) -> str:
        full_path = (exp_dir / rel_path).resolve()
        if not _path_is_within(full_path, exp_dir):
            return f"ERROR: Path escapes experiment directory: {rel_path}"
        if not full_path.exists():
            return f"ERROR: File not found: {rel_path}"
        try:
            content = full_path.read_text("utf-8")
            max_chars = 10000
            if len(content) > max_chars:
                content = (content[:max_chars]
                           + f"\n... (truncated, {len(content)} total chars)")
            return content
        except Exception as e:
            return f"ERROR: Cannot read {rel_path}: {e}"

    return read_file


def _make_read_shortcut(experiment_dir: Path, template: str):
    """Create a shortcut tool: round number → file path.

    template: e.g. "round{n}/evaluations/history.csv"
    """
    exp_dir = experiment_dir.resolve()

    def shortcut(round_str: str) -> str:
        try:
            round_num = int(round_str.strip())
        except ValueError:
            return f"ERROR: Invalid round number: '{round_str}'"
        rel_path = template.replace("{n}", str(round_num))
        full_path = (exp_dir / rel_path).resolve()
        if not _path_is_within(full_path, exp_dir):
            return "ERROR: Path escapes experiment directory"
        if not full_path.exists():
            fname = template.split("/")[-1]
            return f"ERROR: File not found: round{round_num}/{fname}"
        try:
            content = full_path.read_text("utf-8")
            max_chars = 8000
            if len(content) > max_chars:
                content = (content[:max_chars]
                           + f"\n... (truncated, {len(content)} total chars)")
            return content
        except Exception as e:
            return f"ERROR: Cannot read {rel_path}: {e}"

    return shortcut


def _make_query_memory(memory_system):
    """Create a cross-round memory search tool."""

    def query_memory(keyword: str) -> str:
        if not memory_system:
            return "ERROR: Memory system not available."
        results = memory_system.query_lessons(keyword)
        if not results:
            return f"No memory entries match '{keyword}'."
        lines = [f"=== Memory Results for '{keyword}' ===", ""]
        for i, lesson in enumerate(results, 1):
            lines.append(f"{i}. {lesson}")
            lines.append("")
        return "\n".join(lines)

    return query_memory


# ── Default tool setup ────────────────────────────────────────────────────

def setup_default_tools(
    experiment_dir: Path,
    memory_system=None,
) -> ToolRegistry:
    """Create a ToolRegistry with all default tools registered.

    Default tools:
        - read_file(path): Generic file reader (path-constrained)
        - read_eval(n): Shortcut to round{n}/evaluations/history.csv
        - read_perception(n): Shortcut to round{n}/perception_report.md
        - read_reward(n): Shortcut to round{n}/reward_fn_source.py
        - query_memory(keyword): Search cross-round memory (if memory_system provided)
    """
    registry = ToolRegistry(experiment_dir)

    registry.register(
        "read_file",
        "Read any file from the experiment by relative path.",
        _make_read_file(experiment_dir),
        param_name="rel_path",
        param_description="Relative path from experiment root, e.g. 'round4/perception_report.md'",
    )

    registry.register(
        "read_eval",
        "Read evaluation history CSV for a given round number.",
        _make_read_shortcut(experiment_dir, "round{n}/evaluations/history.csv"),
        param_name="round",
        param_description="Round number, e.g. '4'",
    )
    registry.register(
        "read_perception",
        "Read perception report for a given round number.",
        _make_read_shortcut(experiment_dir, "round{n}/perception_report.md"),
        param_name="round",
        param_description="Round number, e.g. '4'",
    )
    registry.register(
        "read_reward",
        "Read reward function source code for a given round number.",
        _make_read_shortcut(experiment_dir, "round{n}/reward_fn_source.py"),
        param_name="round",
        param_description="Round number, e.g. '4'",
    )

    if memory_system is not None:
        registry.register(
            "query_memory",
            "Search cross-round memory for relevant lessons and past experiences.",
            _make_query_memory(memory_system),
            param_name="keyword",
            param_description="Search keyword, e.g. 'hover' or 'landing'",
        )

    return registry


# ── ReAct loop ────────────────────────────────────────────────────────────

def run_react_loop(
    system_prompt: str,
    tools: ToolRegistry,
    api_key: str,
    model: str = "deepseek-reasoner",
    temperature: float = 0.4,
    max_steps: int = 10,
    max_idle: int = 3,
    log_fn=None,
) -> dict:
    """Run a ReAct loop: LLM ↔ tool execution until FINAL ANSWER.

    The loop:
        1. Present system prompt (with tool descriptions) to the LLM
        2. Parse response for tool calls or FINAL ANSWER
        3. On tool call: execute, append result, go to step 2
        4. On FINAL ANSWER: return the output
        5. On idle (no tool call for max_idle consecutive steps): force-terminate

    Args:
        system_prompt: Instructions + task description for the LLM.
                       Tool descriptions are auto-prepended.
        tools: ToolRegistry with registered tools.
        api_key: LLM API key.
        model: LLM model name.
        temperature: Sampling temperature.
        max_steps: Max LLM calls before forced termination.
        max_idle: Consecutive no-tool-call steps before idle termination.
        log_fn: Optional logging function (defaults to print).

    Returns:
        dict with keys:
            final_output: str — text after FINAL ANSWER (or last response)
            full_response: str — complete LLM output from the final step
            tool_calls: list[dict] — history of all tool calls
            steps: int — number of LLM calls made
            idle_terminated: bool — whether idle termination triggered
            conversation: list[dict] — full message history
    """
    log = log_fn or print

    full_system = tools.list_tools() + "\n\n---\n\n" + system_prompt
    conversation = []
    tool_calls = []
    idle_count = 0
    final_output = None
    final_response = None

    for step in range(max_steps):
        # Build prompt for this iteration
        if step == 0:
            prompt = (
                full_system
                + "\n\nWhat would you like to do? "
                  "Call a tool to examine data, or output "
                  "FINAL ANSWER if you already have enough information."
            )
        else:
            parts = [full_system]
            for msg in conversation:
                parts.append(f"\n\n{msg['role'].title()}: {msg['content']}")
            parts.append("\n\nAssistant:")
            prompt = "".join(parts)

        # LLM call
        log(f"  [ReAct] Step {step + 1}/{max_steps} — calling {model}...")
        try:
            response = call_llm(prompt, api_key, model, temperature)
        except Exception as e:
            log(f"  [ReAct] LLM call failed at step {step}: {e}")
            break

        conversation.append({"role": "assistant", "content": response})
        final_response = response

        # Check for FINAL ANSWER
        final_idx = response.upper().rfind("FINAL ANSWER")
        if final_idx != -1:
            after = response[final_idx + len("FINAL ANSWER"):].strip()
            # Only accept if there's no tool call after FINAL ANSWER
            if not _detect_tool_call(after):
                final_output = after
                log(f"  [ReAct] FINAL ANSWER received (step {step + 1})")
                break
            else:
                log("  [ReAct] FINAL ANSWER followed by tool call — continuing")

        # Detect tool call
        tool_call = _detect_tool_call(response)

        if tool_call:
            idle_count = 0
            name, arg = tool_call
            log(f"  [ReAct] Tool call: {name}({arg!r})")

            result = tools.call(name, arg)
            tool_calls.append({
                "tool": name,
                "arg": arg,
                "result_preview": result[:200],
            })

            result_msg = (
                f"### Result of {name}({arg!r})\n\n"
                f"```\n{result}\n```"
            )
            conversation.append({"role": "user", "content": result_msg})
        else:
            idle_count += 1
            log(f"  [ReAct] No tool call (idle={idle_count}/{max_idle})")

            if idle_count >= max_idle:
                log(f"  [ReAct] Idle termination at step {step + 1}")
                final_output = response
                break

            reminder = (
                "I didn't detect a tool call or FINAL ANSWER. "
                "You can either:\n"
                f'1. Call a tool: `read_file("path")` or `<read_file>path</read_file>`\n'
                "2. Conclude with: FINAL ANSWER followed by your final output\n"
            )
            conversation.append({"role": "user", "content": reminder})

    # Fallback: use last response if no FINAL ANSWER was given
    if final_output is None and final_response is not None:
        final_output = final_response

    log(f"  [ReAct] Done — {step + 1} steps, {len(tool_calls)} tool calls, "
        f"idle={idle_count >= max_idle}")

    return {
        "final_output": final_output or "(no output)",
        "full_response": final_response or "",
        "tool_calls": tool_calls,
        "steps": step + 1,
        "idle_terminated": idle_count >= max_idle,
        "conversation": conversation,
    }
