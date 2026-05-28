"""Structured dataclasses for the PARE-SARM framework (§9 in spec).

Provides type-safe data structures for RewardCandidate, ComponentStats,
DiagnosisReport, and DiversityReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# §9.1 RewardCandidate
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RewardCandidate:
    """A reward function candidate generated or mutated by an agent."""
    candidate_id: str
    iteration: int
    mutation_type: str  # "initial", "direct_fix", "component_edit", "progress_gated"
    code: str
    component_names: list[str] = field(default_factory=list)
    parent_candidate_id: str | None = None
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    health_score: float = 0.0
    proxy_result: dict | None = None


# ═══════════════════════════════════════════════════════════════════════════
# §9.2 ComponentStats
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentStats:
    """Per-component statistics computed from training trajectories."""
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
    status: str = "unknown"  # "active", "inactive"
    verdict: str = ""        # "keep", "reduce", "remove", "strengthen", "reconsider"
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# §9.3 DiagnosisReport
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosisReport:
    """Structured diagnosis output from the Analyzer agent."""
    candidate_id: str = ""
    overall_judgment: str = ""
    overall_health: float = 0.0
    component_stats: list[ComponentStats] = field(default_factory=list)
    component_diagnosis: list[dict] = field(default_factory=list)
    mutation_recommendations: list[str] = field(default_factory=list)
    suggested_mutation_types: list[str] = field(default_factory=list)
    escalation_level: str = "coefficient"  # "coefficient", "structural", "rewrite"
    pipeline_action: str = "continue"      # "continue", "regenerate", "stop"


# ═══════════════════════════════════════════════════════════════════════════
# §9.4 DiversityReport
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiversityReport:
    """Diversity check results between reward candidates."""
    candidate_ids: list[str] = field(default_factory=list)
    component_jaccard_matrix: list[list[float]] = field(default_factory=list)
    reward_vector_correlation_matrix: list[list[float]] = field(default_factory=list)
    code_similarity_matrix: list[list[float]] = field(default_factory=list)
    duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)
    passed: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# Update diversity to use reward-vector correlation + weighted selection
# ═══════════════════════════════════════════════════════════════════════════

def compute_candidate_score(
    health: dict,
    eval_history: list[dict],
    diversity_bonus: float = 1.0,
    max_episode_steps: int = 1000,
) -> float:
    """Compute candidate selection score.

    KEY INSIGHT: Pure component health doesn't capture reward quality.
    A reward with health=66 can produce hovering (bad), while health=48
    might actually train a landing policy (good).

    The behavior penalty fixes this: hovering and crashing are both penalized.
    Only moderate-length episodes (actual task progress) get full credit.

    Score = 0.30 * behavior_quality    ← NEW: penalizes hovering AND crashing
          + 0.25 * component_health
          + 0.20 * progress_alignment
          + 0.15 * episode_quality
          + 0.10 * diversity_bonus
    """
    overall_health = health.get("overall_health", 0) / 100.0
    comps = health.get("components", [])

    # ── Behavior quality: penalize both extremes ──
    # Hovering (near max steps, no task completion) = reward hacking
    # Crashing (very short episodes) = penalties dominate
    # Ideal: moderate length, actual task progress
    behavior_quality = 0.5  # default neutral
    if eval_history:
        try:
            last_len = float(eval_history[-1].get("mean_length", 0))
            ratio = last_len / max_episode_steps

            if ratio > 0.85:
                # Hovering/stalling: agent found a way to survive without completing task
                # The closer to max, the more likely it's reward hacking
                behavior_quality = 0.1
            elif ratio < 0.15:
                # Crashing immediately: negative rewards dominate
                behavior_quality = 0.1
            elif 0.30 <= ratio <= 0.70:
                # Sweet spot: likely making genuine progress
                behavior_quality = 1.0
            else:
                # Transitional: between crashing and making progress
                behavior_quality = 0.5
        except (ValueError, IndexError, KeyError):
            behavior_quality = 0.5

    # ── Component health ──
    component_health = overall_health

    # ── Progress alignment ──
    if comps:
        progress_align = sum(max(0, c.get("progress_corr", 0)) for c in comps) / len(comps)
    else:
        progress_align = 0.5

    # ── Episode quality ──
    n_active = sum(1 for c in comps if c.get("active", False))
    episode_quality = n_active / max(len(comps), 1)

    score = (
        0.30 * behavior_quality
        + 0.25 * component_health
        + 0.20 * progress_align
        + 0.15 * episode_quality
        + 0.10 * diversity_bonus
    )
    return round(score, 4)
