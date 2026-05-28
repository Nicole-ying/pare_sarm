"""Rule-based trajectory labeling for ASE-MTAGE Phase 4.

The rule labeler is the first, deterministic pass for assigning coarse outcome
labels. It does not use the official environment reward. It only uses observable
trajectory statistics extracted from evaluation rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


COARSE_LABELS = {
    "early_failure",
    "low_progress_survival",
    "partial_progress",
    "success_like",
    "ambiguous",
}


@dataclass(slots=True)
class RuleLabel:
    coarse_label: str
    detail_label: str
    confidence: float
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coarse_label": self.coarse_label,
            "detail_label": self.detail_label,
            "confidence": float(self.confidence),
            "evidence": self.evidence,
        }


class RuleLabeler:
    """Assign a conservative rule label from trajectory features."""

    def label(self, *, env_id: str, episode: dict[str, Any], features: dict[str, Any]) -> RuleLabel:
        env = env_id.lower()
        if "lunarlander" in env:
            return self._label_lunarlander(episode, features)
        if "cartpole" in env:
            return self._label_cartpole(episode, features)
        if "bipedalwalker" in env:
            return self._label_bipedalwalker(episode, features)
        return self._label_generic(episode, features)

    def _label_lunarlander(self, episode: dict[str, Any], features: dict[str, Any]) -> RuleLabel:
        length = int(episode.get("length", 0) or 0)
        max_steps = int(episode.get("max_episode_steps", 1000) or 1000)
        terminated = bool(episode.get("terminated", False))
        truncated = bool(episode.get("truncated", False))
        final_distance = self._num(features.get("final_distance_to_target"), default=None)
        min_distance = self._num(features.get("min_distance_to_target"), default=None)
        improvement = self._num(features.get("distance_improvement"), default=0.0)
        final_speed = self._num(features.get("final_speed"), default=None)
        final_angle = self._num(features.get("final_angle_abs"), default=None)
        contact_ratio = self._num(features.get("contact_ratio_last20"), default=0.0)

        evidence: list[str] = []
        if terminated and length < 0.25 * max_steps and improvement < 0.20:
            evidence.append(f"terminated early: length={length}, max_steps={max_steps}, improvement={improvement:.3f}")
            return RuleLabel("early_failure", "early_terminal_low_progress", 0.86, evidence)

        if truncated and improvement < 0.25:
            evidence.append(f"truncated with low distance improvement={improvement:.3f}")
            return RuleLabel("low_progress_survival", "timeout_low_progress", 0.82, evidence)

        if length > 0.70 * max_steps and improvement < 0.35:
            evidence.append(f"long episode but low improvement: length={length}, improvement={improvement:.3f}")
            return RuleLabel("low_progress_survival", "long_low_progress", 0.76, evidence)

        stable_near = (
            final_distance is not None
            and final_speed is not None
            and final_angle is not None
            and final_distance < 0.20
            and final_speed < 0.35
            and final_angle < 0.25
            and contact_ratio > 0.40
        )
        if stable_near and not terminated:
            evidence.append("near target, low speed, stable angle, and sustained contact")
            return RuleLabel("success_like", "stable_near_target", 0.78, evidence)
        if stable_near and terminated:
            evidence.append("terminal state appears near target and stable by observable features")
            return RuleLabel("success_like", "terminal_stable_near_target", 0.74, evidence)

        if improvement >= 0.35 or (min_distance is not None and min_distance < 0.35):
            if final_speed is not None and final_speed > 0.70:
                evidence.append(f"approached target but final_speed={final_speed:.3f} is high")
                return RuleLabel("partial_progress", "approach_unstable", 0.76, evidence)
            evidence.append(f"observable approach progress: improvement={improvement:.3f}, min_distance={min_distance}")
            return RuleLabel("partial_progress", "approach_progress", 0.72, evidence)

        evidence.append("insufficient or conflicting LunarLander evidence")
        return RuleLabel("ambiguous", "insufficient_lunarlander_evidence", 0.45, evidence)

    def _label_cartpole(self, episode: dict[str, Any], features: dict[str, Any]) -> RuleLabel:
        length = int(episode.get("length", 0) or 0)
        max_steps = int(episode.get("max_episode_steps", 500) or 500)
        terminated = bool(episode.get("terminated", False))
        truncated = bool(episode.get("truncated", False))
        final_angle = self._num(features.get("final_angle_abs"), default=None)
        max_angle = self._num(features.get("max_angle_abs"), default=None)

        if length >= 0.90 * max_steps or truncated:
            return RuleLabel("success_like", "long_stable_balance", 0.82, [f"episode length {length} is close to max_steps {max_steps}"])
        if terminated and length < 0.30 * max_steps:
            return RuleLabel("early_failure", "early_balance_failure", 0.86, [f"terminated early at length={length}"])
        if length >= 0.50 * max_steps:
            return RuleLabel("partial_progress", "moderate_balance", 0.72, [f"moderate balance duration length={length}"])
        if final_angle is not None or max_angle is not None:
            return RuleLabel("low_progress_survival", "short_unstable_balance", 0.70, ["short episode with observable pole angle evidence"])
        return RuleLabel("ambiguous", "insufficient_cartpole_evidence", 0.45, ["insufficient CartPole evidence"])

    def _label_bipedalwalker(self, episode: dict[str, Any], features: dict[str, Any]) -> RuleLabel:
        length = int(episode.get("length", 0) or 0)
        max_steps = int(episode.get("max_episode_steps", 1600) or 1600)
        terminated = bool(episode.get("terminated", False))
        truncated = bool(episode.get("truncated", False))
        displacement = self._num(features.get("forward_displacement"), default=0.0)
        final_height = self._num(features.get("final_height"), default=None)

        if terminated and length < 0.25 * max_steps and displacement < 0.5:
            return RuleLabel("early_failure", "early_fall", 0.84, [f"early termination and low displacement={displacement:.3f}"])
        if length > 0.60 * max_steps and displacement < 1.0:
            return RuleLabel("low_progress_survival", "standing_or_stalling", 0.78, [f"long episode with low displacement={displacement:.3f}"])
        if displacement >= 3.0 and not terminated:
            return RuleLabel("success_like", "stable_forward_progress", 0.74, [f"large displacement={displacement:.3f} without terminal fall"])
        if displacement >= 1.0:
            return RuleLabel("partial_progress", "unstable_forward_progress", 0.72, [f"some forward displacement={displacement:.3f}"])
        return RuleLabel("ambiguous", "insufficient_bipedalwalker_evidence", 0.45, ["insufficient BipedalWalker evidence"])

    def _label_generic(self, episode: dict[str, Any], features: dict[str, Any]) -> RuleLabel:
        length = int(episode.get("length", 0) or 0)
        max_steps = int(episode.get("max_episode_steps", 1000) or 1000)
        terminated = bool(episode.get("terminated", False))
        truncated = bool(episode.get("truncated", False))
        improvement = self._num(features.get("progress_improvement"), default=0.0)
        if terminated and length < 0.25 * max_steps and improvement <= 0.0:
            return RuleLabel("early_failure", "generic_early_failure", 0.70, ["early termination with no progress improvement"])
        if truncated and improvement <= 0.0:
            return RuleLabel("low_progress_survival", "generic_timeout_low_progress", 0.68, ["timeout/truncation with weak progress"])
        if improvement > 0.0:
            return RuleLabel("partial_progress", "generic_progress", 0.62, ["positive generic progress improvement"])
        return RuleLabel("ambiguous", "generic_ambiguous", 0.40, ["generic rules could not classify confidently"])

    def _num(self, value: Any, default: float | None = 0.0) -> float | None:
        try:
            v = float(value)
        except Exception:
            return default
        return v
