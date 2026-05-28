"""Structured dataclasses and scoring helpers for PARE-SARM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardCandidate:
    """A reward function candidate generated or mutated by an agent."""
    candidate_id: str
    iteration: int
    mutation_type: str  # initial, direct_fix, component_edit, progress_gated
    code: str
    component_names: list[str] = field(default_factory=list)
    parent_candidate_id: str | None = None
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    health_score: float = 0.0
    behavior_quality: float = 0.0
    promotion_score: float = 0.0
    proxy_result: dict | None = None


@dataclass
class ComponentStats:
    name: str
    mean: float = 0.0
    std: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    n: int = 0
    mean_abs_ratio: float = 0.0
    activation_score: float = 0.0
    balance_score: float = 0.0
    progress_corr: float = 0.0
    failure_corr: float = 0.0
    health_score: float = 0.0
    status: str = "unknown"
    verdict: str = ""
    reason: str = ""


@dataclass
class DiagnosisReport:
    candidate_id: str = ""
    overall_judgment: str = ""
    overall_health: float = 0.0
    failure_mode: str = "unknown"
    root_cause_type: str = "unknown"
    behavior_evidence: list[str] = field(default_factory=list)
    component_evidence: list[str] = field(default_factory=list)
    component_stats: list[ComponentStats] = field(default_factory=list)
    component_diagnosis: list[dict] = field(default_factory=list)
    mutation_recommendations: list[str] = field(default_factory=list)
    suggested_mutation_types: list[str] = field(default_factory=list)
    forbidden_mutation_types: list[str] = field(default_factory=list)
    tool_requests: list[str] = field(default_factory=list)
    escalation_level: str = "coefficient"
    pipeline_action: str = "continue"


@dataclass
class DiversityReport:
    candidate_ids: list[str] = field(default_factory=list)
    component_jaccard_matrix: list[list[float]] = field(default_factory=list)
    reward_vector_correlation_matrix: list[list[float]] = field(default_factory=list)
    code_similarity_matrix: list[list[float]] = field(default_factory=list)
    duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)
    passed: bool = True


def compute_candidate_score(
    health: dict[str, Any],
    eval_history: list[dict[str, Any]],
    diversity_bonus: float = 0.5,
    max_episode_steps: int = 1000,
    behavior_report: dict[str, Any] | None = None,
) -> float:
    """Backward-compatible candidate score.

    New code should prefer ``selection.selector.compute_promotion_score``.  This
    helper remains so older pipeline code and tests keep working, but it no
    longer treats episode length alone as a reliable behavior signal when a
    behavior report is available.
    """
    overall_health = float(health.get("overall_health", 0)) / 100.0
    comps = health.get("components", []) or []

    if behavior_report:
        behavior_quality = float(behavior_report.get("behavior_quality", 0.35) or 0.35)
    else:
        behavior_quality = _length_fallback_behavior_quality(eval_history, max_episode_steps)

    if comps:
        progress_align = sum(max(0.0, _float(c.get("progress_corr", 0.0))) for c in comps) / len(comps)
        component_activation_ratio = sum(1 for c in comps if c.get("active", False)) / max(len(comps), 1)
    else:
        progress_align = 0.5
        component_activation_ratio = 0.0

    score = (
        0.30 * behavior_quality
        + 0.20 * overall_health
        + 0.20 * progress_align
        + 0.15 * component_activation_ratio
        + 0.15 * float(diversity_bonus)
    )
    return round(max(0.0, min(1.0, score)), 4)


def _length_fallback_behavior_quality(eval_history: list[dict[str, Any]], max_episode_steps: int) -> float:
    if not eval_history:
        return 0.35
    try:
        last_len = float(eval_history[-1].get("mean_length", 0))
        ratio = last_len / max(max_episode_steps, 1)
    except (ValueError, IndexError, KeyError, TypeError):
        return 0.35
    if ratio > 0.85:
        return 0.15  # generic fallback: could be stalling unless env-specific classifier says otherwise
    if ratio < 0.15:
        return 0.10
    if 0.30 <= ratio <= 0.70:
        return 0.55
    return 0.35


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
