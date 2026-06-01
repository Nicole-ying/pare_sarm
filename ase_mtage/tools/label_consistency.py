"""Hard physical-constraint checks for ASE-MTAGE trajectory labels.

Enforces observable constraints: success_like cannot have high final speed,
early_failure cannot have long episode length, etc.
"""

from __future__ import annotations

from typing import Any


class LabelConsistencyChecker:
    """Validate labels against hard evidence constraints. No confidence logic."""

    def check(self, *, coarse_label: str, episode: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        length = int(episode.get("length", 0) or 0)
        max_steps = int(episode.get("max_episode_steps", 1000) or 1000)
        terminated = bool(episode.get("terminated", False))
        final_speed = self._num(features.get("final_speed"), default=None)
        final_angle = self._num(features.get("final_angle_abs"), default=None)
        improvement = self._num(features.get("distance_improvement", features.get("progress_improvement")), default=None)

        if coarse_label == "success_like":
            if final_speed is not None and final_speed > 0.85:
                warnings.append(f"success_like conflicts with high final_speed={final_speed:.3f}")
            if final_angle is not None and final_angle > 0.80:
                warnings.append(f"success_like conflicts with high final_angle_abs={final_angle:.3f}")
            if terminated and length < 0.25 * max_steps:
                warnings.append("success_like conflicts with early terminal episode")

        if coarse_label == "early_failure" and length > 0.75 * max_steps:
            warnings.append("early_failure conflicts with long episode length")

        if coarse_label == "low_progress_survival" and improvement is not None and improvement > 0.60:
            warnings.append(f"low_progress_survival conflicts with high improvement={improvement:.3f}")

        if warnings and (coarse_label == "success_like" or len(warnings) >= 2):
            return {"coarse_label": "ambiguous", "use_for_tage_pair": False, "consistency_warnings": warnings}

        return {"coarse_label": coarse_label, "use_for_tage_pair": coarse_label != "ambiguous", "consistency_warnings": warnings}

    def _num(self, value: Any, default: float | None = 0.0) -> float | None:
        try:
            return float(value)
        except Exception:
            return default
