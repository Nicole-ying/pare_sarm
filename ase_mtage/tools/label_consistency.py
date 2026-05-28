"""Consistency checks for ASE-MTAGE trajectory labels.

The checker prevents low-confidence or physically conflicting labels from being
used to construct Memory-TAGE preference pairs.
"""

from __future__ import annotations

from typing import Any


class LabelConsistencyChecker:
    """Validate final labels against simple hard evidence constraints."""

    def __init__(self, *, confidence_threshold: float = 0.70) -> None:
        self.confidence_threshold = confidence_threshold

    def check(self, *, label: dict[str, Any], episode: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
        final_label = dict(label)
        warnings: list[str] = []
        coarse = str(final_label.get("coarse_label", "ambiguous"))
        confidence = float(final_label.get("confidence", 0.0) or 0.0)
        length = int(episode.get("length", 0) or 0)
        max_steps = int(episode.get("max_episode_steps", 1000) or 1000)
        terminated = bool(episode.get("terminated", False))
        final_speed = self._num(features.get("final_speed"), default=None)
        final_angle = self._num(features.get("final_angle_abs"), default=None)
        improvement = self._num(features.get("distance_improvement", features.get("progress_improvement")), default=None)

        if coarse == "success_like":
            if final_speed is not None and final_speed > 0.85:
                warnings.append(f"success_like conflicts with high final_speed={final_speed:.3f}")
            if final_angle is not None and final_angle > 0.80:
                warnings.append(f"success_like conflicts with high final_angle_abs={final_angle:.3f}")
            if terminated and length < 0.25 * max_steps:
                warnings.append("success_like conflicts with early terminal episode")

        if coarse == "early_failure" and length > 0.75 * max_steps:
            warnings.append("early_failure conflicts with long episode length")

        if coarse == "low_progress_survival" and improvement is not None and improvement > 0.60:
            warnings.append(f"low_progress_survival conflicts with high improvement={improvement:.3f}")

        use_for_tage_pair = bool(final_label.get("use_for_tage_pair", True))
        if confidence < self.confidence_threshold:
            use_for_tage_pair = False
            warnings.append(f"confidence {confidence:.3f} is below threshold {self.confidence_threshold:.3f}")

        if warnings:
            if coarse == "success_like" or len(warnings) >= 2:
                final_label["coarse_label"] = "ambiguous"
                final_label["detail_label"] = "downgraded_by_consistency_checker"
                final_label["confidence"] = min(confidence, 0.49)
                use_for_tage_pair = False

        final_label["use_for_tage_pair"] = use_for_tage_pair and final_label.get("coarse_label") != "ambiguous"
        final_label["consistency_warnings"] = warnings
        return final_label

    def _num(self, value: Any, default: float | None = 0.0) -> float | None:
        try:
            return float(value)
        except Exception:
            return default
