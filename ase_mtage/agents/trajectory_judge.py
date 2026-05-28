"""Trajectory Judge Agent for ASE-MTAGE Phase 4.

This is a guarded deterministic/optional-LLM placeholder. Phase 4 keeps the
artifact protocol stable: every trajectory receives a final_label object. If a
rule label is confident, it is accepted after consistency checks. If it is weak,
this agent currently marks it as ambiguous instead of hallucinating. A real LLM
judge can later be plugged in behind the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ase_mtage.tools.label_consistency import LabelConsistencyChecker


@dataclass(slots=True)
class TrajectoryJudgment:
    trajectory_id: str
    final_label: dict[str, Any]
    agree_with_rule: bool
    evidence_used: list[str]
    conflict_warnings: list[str]
    use_for_memory: bool
    use_for_tage_pair: bool
    allowed_preference_role: str
    do_not_use_reason: str
    judge_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "final_label": self.final_label,
            "agree_with_rule": self.agree_with_rule,
            "evidence_used": self.evidence_used,
            "conflict_warnings": self.conflict_warnings,
            "use_for_memory": self.use_for_memory,
            "use_for_tage_pair": self.use_for_tage_pair,
            "allowed_preference_role": self.allowed_preference_role,
            "do_not_use_reason": self.do_not_use_reason,
            "judge_mode": self.judge_mode,
        }


class TrajectoryJudgeAgent:
    """Guarded trajectory label finalizer."""

    def __init__(self, *, confidence_threshold: float = 0.70, llm_enabled: bool = False) -> None:
        self.confidence_threshold = confidence_threshold
        self.llm_enabled = llm_enabled
        self.consistency_checker = LabelConsistencyChecker(confidence_threshold=confidence_threshold)

    def judge(self, card: dict[str, Any]) -> TrajectoryJudgment:
        """Return a final guarded judgment for one trajectory card."""
        trajectory_id = str(card.get("trajectory_id", "unknown"))
        rule_label = dict(card.get("rule_label") or {})
        episode = dict(card.get("episode") or {})
        features = dict(card.get("features") or {})
        rule_conf = float(rule_label.get("confidence", 0.0) or 0.0)
        evidence = list(rule_label.get("evidence") or [])

        if rule_conf >= self.confidence_threshold:
            label = {
                "coarse_label": rule_label.get("coarse_label", "ambiguous"),
                "detail_label": rule_label.get("detail_label", "rule_high_confidence"),
                "confidence": rule_conf,
                "use_for_tage_pair": True,
            }
            judge_mode = "rule_high_confidence"
            agree_with_rule = True
            do_not_use_reason = ""
        else:
            # Phase 4 does not call a real LLM yet. Low-confidence labels are kept
            # in memory but excluded from strong preference pairs.
            label = {
                "coarse_label": "ambiguous",
                "detail_label": "low_confidence_rule_label",
                "confidence": min(rule_conf, 0.49),
                "use_for_tage_pair": False,
            }
            judge_mode = "phase_4_guarded_no_llm"
            agree_with_rule = False
            do_not_use_reason = "Rule confidence is low and LLM judge is not enabled in Phase 4."

        checked = self.consistency_checker.check(label=label, episode=episode, features=features)
        warnings = list(checked.get("consistency_warnings") or [])
        use_for_tage_pair = bool(checked.get("use_for_tage_pair", False))
        preference_role = self._preference_role(str(checked.get("coarse_label", "ambiguous"))) if use_for_tage_pair else "none"
        if not use_for_tage_pair and not do_not_use_reason:
            do_not_use_reason = "Label is ambiguous, low confidence, or conflicts with hard evidence."

        return TrajectoryJudgment(
            trajectory_id=trajectory_id,
            final_label={
                "coarse_label": checked.get("coarse_label", "ambiguous"),
                "detail_label": checked.get("detail_label", "unknown"),
                "confidence": float(checked.get("confidence", 0.0) or 0.0),
            },
            agree_with_rule=agree_with_rule and not warnings,
            evidence_used=evidence,
            conflict_warnings=warnings,
            use_for_memory=True,
            use_for_tage_pair=use_for_tage_pair,
            allowed_preference_role=preference_role,
            do_not_use_reason=do_not_use_reason,
            judge_mode=judge_mode,
        )

    def _preference_role(self, coarse_label: str) -> str:
        if coarse_label in {"early_failure", "low_progress_survival"}:
            return "negative_reference"
        if coarse_label == "partial_progress":
            return "mid_reference"
        if coarse_label == "success_like":
            return "positive_reference"
        return "none"
