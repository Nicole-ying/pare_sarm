"""Reflection / Memory Agent for ASE-MTAGE Phase 6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.memory.failure_repair_memory import FailureRepairMemory
from ase_mtage.utils.io import append_jsonl, ensure_dir, save_json, save_text


class ReflectionAgent:
    """Write round-level failure-repair lessons into memory."""

    def __init__(self, *, output_dir: str | Path, failure_memory_path: str | Path, archival_lessons_path: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.failure_memory = FailureRepairMemory(failure_memory_path)
        self.archival_lessons_path = Path(archival_lessons_path)

    def run(
        self,
        *,
        round_idx: int,
        analyzer_report: dict[str, Any] | None,
        selection_report: dict[str, Any] | None,
        coverage_report: dict[str, Any] | None,
        rollback_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        selected_id = (selection_report or {}).get("selected_candidate")
        mutation_family = None
        for item in (selection_report or {}).get("candidate_scores", []) or []:
            if item.get("candidate_id") == selected_id:
                mutation_family = item.get("mutation_family")
                break
        memory_interp = (analyzer_report or {}).get("memory_interpretation") or {}
        label_counts = memory_interp.get("label_counts") or (coverage_report or {}).get("label_counts") or {}
        coverage_type = (coverage_report or {}).get("coverage_type") or memory_interp.get("coverage_type")
        reflection = {
            "round": round_idx,
            "parent_reward_id": (analyzer_report or {}).get("parent_reward_id"),
            "selected_candidate_id": selected_id,
            "mutation_family": mutation_family,
            "observed_outcome": {
                "coarse_result": self._coarse_result(coverage_type, label_counts),
                "main_failure_remaining": self._main_failure(label_counts),
                "main_success_signal": self._main_success_signal(label_counts),
            },
            "failure_repair_outcome": {
                "failure_before": ", ".join((memory_interp.get("main_known_failures") or [])) or "unknown_or_insufficient_memory",
                "repair_attempt": "; ".join(((analyzer_report or {}).get("mutation_intent") or {}).get("required_changes", [])),
                "outcome_after": f"selection={selected_id}, coverage_type={coverage_type}",
            },
            "lesson": (analyzer_report or {}).get("self_evaluation_lesson") or "No analyzer lesson available.",
            "future_guidance": ((analyzer_report or {}).get("mutation_intent") or {}).get("required_changes", []),
            "archive_update": {
                "rollback_triggered": (rollback_report or {}).get("rollback_triggered", False),
                "next_parent_reward_id": (rollback_report or {}).get("next_parent_reward_id"),
                "reason": (rollback_report or {}).get("reason"),
            },
            "agent_mode": "phase_6_deterministic_reflector",
        }
        save_text(self.output_dir / "prompt.txt", "Phase 6 deterministic ReflectionAgent; no LLM prompt was sent.\n")
        save_text(self.output_dir / "response.txt", "Deterministic reflection.json generated from round artifacts.\n")
        save_json(self.output_dir / "reflection.json", reflection)
        self.failure_memory.add(reflection)
        append_jsonl(self.archival_lessons_path, {"round": round_idx, "lesson": reflection["lesson"], "future_guidance": reflection["future_guidance"]})
        return reflection

    def _coarse_result(self, coverage_type: str | None, label_counts: dict[str, Any]) -> str:
        if label_counts.get("success_like", 0):
            return "success_like_memory_available"
        if label_counts.get("partial_progress", 0):
            return "partial_progress_available"
        if label_counts.get("early_failure", 0) or label_counts.get("low_progress_survival", 0):
            return "failure_memory_available"
        return coverage_type or "unknown"

    def _main_failure(self, label_counts: dict[str, Any]) -> str:
        failures = {k: int(label_counts.get(k, 0) or 0) for k in ["early_failure", "low_progress_survival"]}
        if not any(failures.values()):
            return "none_observed_or_unknown"
        return max(failures, key=failures.get)

    def _main_success_signal(self, label_counts: dict[str, Any]) -> str:
        if label_counts.get("success_like", 0):
            return "success_like trajectories exist"
        if label_counts.get("partial_progress", 0):
            return "partial_progress trajectories exist"
        return "no clear positive trajectory signal"
