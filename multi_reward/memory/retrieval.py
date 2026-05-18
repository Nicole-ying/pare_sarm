"""
Memory retrieval engine for multi_reward framework.

Given an evidence board, retrieves the most relevant past memories
using cosine similarity on feature vectors.

Avoids injecting noise by:
1. Only returning memories with similarity > 0.3
2. Weighting recent rounds higher
3. Truncating results to max_n
"""

from typing import Any


def retrieve_relevant_context(
    memory_store: 'MemoryStore',
    current_feature_vector: dict[str, float],
    max_similar: int = 3,
    max_recent: int = 2,
) -> dict[str, Any]:
    """Retrieve relevant past context for a Diagnostician agent.

    Returns dict with:
        - similar_rounds: rounds with similar training dynamics (cosine sim)
        - recent_rounds: most recent rounds regardless of similarity
        - both deduplicated
    """
    similar = memory_store.find_similar_rounds(
        current_feature_vector, n=max_similar
    )

    recent_rounds = memory_store.get_available_rounds()[-max_recent:]
    recent_summary = memory_store.get_recent_lessons(n=max_recent)

    return {
        "similar_rounds": similar,
        "recent_summary": recent_summary,
    }


def format_memory_for_prompt(
    retrieval_result: dict[str, Any],
    agent_id: str = None,
    memory_store: 'MemoryStore' = None,
) -> str:
    """Format retrieved memories as a compact prompt snippet for LLM injection.

    Designed to be injected at the END of the Diagnostician prompt
    (not the beginning — avoids diluting attention on the current task).
    """
    parts = []

    # Similar rounds (most relevant first)
    similar = retrieval_result.get("similar_rounds", [])
    if similar:
        parts.append("## Lessons from Similar Past States\n")
        parts.append(
            "The following past rounds had similar training dynamics "
            "(same direction of entropy, component balance, survival rate):\n"
        )
        for i, entry in enumerate(similar):
            r = entry["round"]
            sim = entry["similarity"]
            lesson = entry.get("lesson", "")
            parts.append(
                f"### Most Relevant #{i+1}: Round {r} (similarity: {sim:.2f})"
            )
            if lesson:
                parts.append(lesson)
            parts.append("")

    # Recent rounds
    recent = retrieval_result.get("recent_summary", "")
    if recent and not similar:
        parts.append("## Recent Round History\n")
        parts.append(recent)

    # Agent's own track record
    if agent_id and memory_store:
        belief_text = memory_store.format_beliefs_for_prompt(agent_id)
        if belief_text:
            parts.append("---\n")
            parts.append(belief_text)

    return "\n".join(parts) if parts else ""


# Import at module level for type hints
from .memory_store import MemoryStore
