"""Pydantic message schemas for inter-agent communication.

Every message in the system is a typed Pydantic model wrapped in an
AgentMessage envelope. This replaces file-based communication with
structured, validatable messages.

Design principle: each agent publishes and subscribes to specific
message types. No agent reads another agent's raw files.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Literal
from pathlib import Path

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object  # type: ignore
    Field = None  # type: ignore

BEIJING = timezone(timedelta(hours=8))


def _now() -> str:
    return datetime.now(BEIJING).strftime("%Y-%m-%dT%H:%M:%S+08:00")


# ── Message Envelope ──────────────────────────────────────────────────────

class AgentMessage(BaseModel if BaseModel is not object else object):
    """Universal envelope for all inter-agent messages.

    Every agent publishes messages wrapped in this envelope. The content
    field carries the typed payload matching message_type.
    """

    message_id: str = Field(default_factory=lambda: f"msg_{id(object())}")
    sender: str
    recipient: str | None = None  # None = broadcast to pool
    message_type: str
    round_num: int
    timestamp: str = Field(default_factory=_now)
    content: dict = Field(default_factory=dict)
    parent_id: str | None = None  # For reply chains


# ── Content Schemas ────────────────────────────────────────────────────────

class TaskManifest(BaseModel if BaseModel is not object else object):
    """Structured understanding of the RL environment.

    Produced by EnvPerception Agent before Round 0.
    Consumed by Generator Agent as permanent context.
    """

    task_goal: str = ""
    success_conditions: str = ""
    failure_conditions: str = ""
    observation_dims: list[dict] = Field(default_factory=list)
    action_space: dict = Field(default_factory=dict)
    compute_reward_signature: str = ""
    critical_dimensions: str = ""
    raw_markdown: str = ""  # Full markdown for prompt injection


class GeneratorProposal(BaseModel if BaseModel is not object else object):
    """Generator's proposed reward function changes.

    Produced by Generator Agent.
    Consumed by Evaluator Agent and Reflector Agent.
    """

    diagnosis: str = ""
    changed_count: int = 0
    proposed_changes: list[dict] = Field(default_factory=list)
    evidence_citations: list[dict] = Field(default_factory=list)
    analysis_status: str = "ok"  # "ok" | "failed"


class RewardCode(BaseModel if BaseModel is not object else object):
    """Generated reward function code.

    Produced by Generator Agent.
    Consumed by Trainer and Evaluator Agent.
    """

    source: str = ""  # Full Python source
    compute_reward_snippet: str = ""  # First N lines for quick inspection
    component_dict_info: str = ""
    file_path: str = ""  # Where it's saved on disk


class TrainingResult(BaseModel if BaseModel is not object else object):
    """Summary of a completed training run.

    Produced by the Trainer (train.py).
    Consumed by Evaluator Agent.
    """

    round_num: int = 0
    total_timesteps: int = 0
    elapsed_minutes: float = 0.0
    success: bool = True
    error_log: str = ""
    eval_history_path: str = ""  # Path to evaluations/history.csv


class EvaluationReport(BaseModel if BaseModel is not object else object):
    """Structured evaluation of a trained policy.

    Produced by Evaluator Agent.
    Consumed by Reflector Agent and Generator Agent (feedback loop).
    """

    round_num: int = 0
    behavior_summary: str = ""
    component_health: dict = Field(default_factory=dict)
    constraint_violations: list[dict] = Field(default_factory=list)
    metrics_analysis: dict = Field(default_factory=dict)
    recommendation: str = ""  # "continue" | "retry" | "abort"
    confidence: float = 0.5  # 0..1
    raw_markdown: str = ""


class ReflectionReport(BaseModel if BaseModel is not object else object):
    """Cross-round causal analysis and lessons.

    Produced by Reflector Agent.
    Consumed by Generator Agent and Memory System.
    """

    round_num: int = 0
    causal_lesson: str = ""
    abstract_principle: str = ""
    checklist: list[dict] = Field(default_factory=list)
    what_was_right: str = ""
    what_was_wrong: str = ""
    raw_markdown: str = ""


class MemoryQuery(BaseModel if BaseModel is not object else object):
    """Query to the Memory System.

    Produced by any agent.
    Consumed by Memory System.
    """

    query_type: Literal["keyword", "semantic", "recent", "pattern", "similar_reward"] = "keyword"
    query_text: str = ""
    max_results: int = 5
    round_num: int | None = None


class MemoryResult(BaseModel if BaseModel is not object else object):
    """Result from the Memory System.

    Produced by Memory System.
    Consumed by the querying agent.
    """

    query_type: str = ""
    results: list[dict] = Field(default_factory=list)
    summary: str = ""
