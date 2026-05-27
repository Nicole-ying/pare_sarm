"""
context.py — Memory context builder for agent prompt injection.

Provides a single function that builds a "memory context" block
for injection into agent prompts. This block includes:
1. Core memory (TaskManifest summary)
2. Recent round history (from EpisodicMemory)
3. Relevant archival patterns (from ArchivalMemory)
4. Relevant cross-round lessons (from MEMORY.md)

Usage:
    mem = MemorySystem(run_dir)
    ctx = build_memory_context(mem, query="reward hacking", max_tokens=800)
    prompt = f"{system_prompt}\\n\\n{ctx}\\n\\n{task_prompt}"
"""

from __future__ import annotations

from .memory_system import MemorySystem


def build_memory_context(
    memory: MemorySystem,
    query: str = "",
    max_tokens: int = 800,
    n_recent_rounds: int = 2,
    n_archival: int = 3,
) -> str:
    """Build a compact memory context block for agent prompt injection.

    Prioritizes:
    1. Core memory (TaskManifest key facts) — always included
    2. Recent round summaries — for continuity
    3. Archival patterns relevant to query — for cross-experiment wisdom

    Args:
        memory: The MemorySystem instance.
        query: What the agent is trying to do (for archival search).
        max_tokens: Rough token budget for the context block.
        n_recent_rounds: Number of recent rounds to summarize.
        n_archival: Number of archival patterns to include.

    Returns:
        Markdown string for injection into LLM prompt.
    """
    if memory is None:
        return ""

    sections = []
    token_budget = max_tokens

    # 1. Core memory (always, high priority)
    core_text = memory.core.render(max_output_tokens=min(400, token_budget))
    if core_text.strip():
        sections.append(core_text)
        token_budget -= _estimate_tokens(core_text)

    # 2. Recent round history
    if token_budget > 100:
        recent = memory.get_recent_lessons(n=n_recent_rounds)
        if recent.strip():
            recent_text = f"## Recent Round History\n{recent}"
            sections.append(_truncate(recent_text, min(400, token_budget)))
            token_budget -= min(400, token_budget)

    # 3. Archival patterns (cross-experiment)
    if token_budget > 100 and query:
        patterns = memory.archival.search(query, k=n_archival)
        if patterns:
            lines = ["## Relevant Patterns from Past Experiments"]
            for i, p in enumerate(patterns, 1):
                lines.append(
                    f"{i}. [{p.get('env_type', 'general')}, "
                    f"importance={p.get('importance', 5)}] "
                    f"{p['content'][:200]}"
                )
            pattern_text = "\n".join(lines)
            sections.append(_truncate(pattern_text, min(300, token_budget)))
            token_budget -= min(300, token_budget)

    # 4. Relevant cross-round lessons
    if token_budget > 100 and query:
        lessons = memory.query_lessons(query, max_results=2)
        lessons = [l for l in lessons if not l.startswith("[Archival")]  # Skip archival dupes
        if lessons:
            lines = ["## Relevant Cross-Round Lessons"]
            for l in lessons[:2]:
                lines.append(f"- {l[:250]}")
            lessons_text = "\n".join(lines)
            sections.append(_truncate(lessons_text, min(200, token_budget)))

    return "\n\n".join(sections)


def inject_memory_into_prompt(
    prompt: str,
    memory: MemorySystem,
    query: str = "",
    max_tokens: int = 800,
) -> str:
    """Inject memory context into an existing prompt.

    Inserts the memory block after any system prompt header (marked by
    "== Context ==" or similar), or at the beginning if no marker found.

    Args:
        prompt: The original prompt text.
        memory: MemorySystem instance.
        query: What the agent is trying to do.
        max_tokens: Token budget for memory context.

    Returns:
        Prompt with memory context injected.
    """
    ctx = build_memory_context(memory, query=query, max_tokens=max_tokens)
    if not ctx:
        return prompt

    # Try to insert after context markers
    markers = ["== Context ==", "## Context", "Context:", "== Task =="]
    inserted = False
    for marker in markers:
        if marker in prompt:
            idx = prompt.index(marker) + len(marker)
            prompt = prompt[:idx] + "\n\n" + ctx + prompt[idx:]
            inserted = True
            break

    if not inserted:
        # Insert after first heading or at beginning
        first_heading = None
        for line in prompt.split("\n"):
            if line.startswith("## ") or line.startswith("# "):
                first_heading = line
                break
        if first_heading:
            idx = prompt.index(first_heading)
            prompt = prompt[:idx] + ctx + "\n\n" + prompt[idx:]
        else:
            prompt = ctx + "\n\n" + prompt

    return prompt


# ── Internal ────────────────────────────────────────────────────────────────

_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _truncate(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget."""
    lines = text.split("\n")
    result = []
    tokens = 0
    for line in lines:
        t = _estimate_tokens(line)
        if tokens + t > max_tokens:
            result.append(f"... (truncated)")
            break
        result.append(line)
        tokens += t
    return "\n".join(result)
