"""Rollback manager for ASE-MTAGE Phase 6.

Rollback is a deterministic safety tool. It does not let the LLM decide hard
rollback conditions. The Analyzer/Reflector may explain rollback, but this tool
produces the actual rollback report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.memory.elite_archive import EliteArchive
from ase_mtage.utils.io import load_json, save_json


class RollbackManager:
    """Check whether the current selection should fall back to an elite parent."""

    def __init__(self, *, progress_drop_threshold: float = 0.30) -> None:
        self.progress_drop_threshold = float(progress_drop_threshold)

    def check(
        self,
        *,
        round_idx: int,
        current_selection_report: dict[str, Any],
        coverage_report: dict[str, Any],
        elite_archive: EliteArchive,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        best = elite_archive.best()  # success_like_count → partial_progress → tage_score
        selected_id = current_selection_report.get("selected_candidate")
        candidate_scores = current_selection_report.get("candidate_scores") or []
        selected = None
        for item in candidate_scores:
            if item.get("candidate_id") == selected_id:
                selected = item
                break

        selected_score = float((selected or {}).get("selection_score", (selected or {}).get("tage_score", 0.0)) or 0.0)
        best_score = float((best or {}).get("score", -1.0) if best else -1.0)
        best_training_return = (best or {}).get("training_return")
        no_selected = selected_id is None

        # Adaptive threshold: if the archive shows that TAGE scores are unreliable
        # (high training return with mediocre TAGE score), lower the bar for rollback.
        effective_threshold = self.progress_drop_threshold
        archive_data = elite_archive.read()
        all_entries = archive_data.get("elite_rewards") or []
        # Check for TAGE-training mismatch: any entry with great training but moderate TAGE?
        for entry in all_entries:
            tr = entry.get("training_return")
            sc = float(entry.get("score", 0) or 0)
            if tr is not None and tr > 100 and sc < 0.75:
                effective_threshold = min(effective_threshold, 0.20)
                break

        # Rollback if no candidate selected, or if score drops significantly
        score_drop = best is not None and selected_score + effective_threshold < best_score
        rollback_triggered = bool(best and (no_selected or score_drop))
        report = {
            "round": round_idx,
            "rollback_triggered": rollback_triggered,
            "current_reward_id": selected_id,
            "rollback_target_reward_id": best.get("reward_id") if best and rollback_triggered else None,
            "next_parent_reward_id": best.get("reward_id") if best and rollback_triggered else selected_id,
            "next_parent_reward_path": best.get("reward_path") if best and rollback_triggered else (selected or {}).get("reward_path"),
            "hard_conditions": {
                "selected_score": selected_score,
                "best_elite_score": best_score,
                "best_training_return": best_training_return,
                "score_drop_threshold": effective_threshold,
                "base_threshold": self.progress_drop_threshold,
                "score_drop_triggered": score_drop,
                "no_selected_candidate": no_selected,
            },
            "reason": self._reason(rollback_triggered, best, score_drop, no_selected, effective_threshold),
            "phase": "phase_6_rollback",
        }
        if output_path is not None:
            save_json(output_path, report)
        return report

    def _reason(self, rollback: bool, best: dict[str, Any] | None, score_drop: bool, no_selected: bool, effective_threshold: float) -> str:
        if not best:
            return "No elite reward exists yet; continue with current selected candidate."
        if no_selected:
            return "No selected candidate is available; rollback to elite reward."
        if rollback and score_drop:
            return f"Selected candidate score dropped below elite by threshold={effective_threshold:.2f}; rollback to best reward by success_like/training_return."
        return f"Rollback conditions not met (threshold={effective_threshold:.2f}); continue with selected candidate."
