"""Candidate selector for ASE-MTAGE Phase 5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.utils.io import load_json, save_json


class CandidateSelector:
    """Select top-1 candidate from validator and Memory-TAGE reports."""

    def select(
        self,
        *,
        round_idx: int,
        candidate_records: list[dict[str, Any]],
        coverage_report: dict[str, Any],
        output_path: str | Path | None = None,
        selection_mode: str = "memory_tage",
    ) -> dict[str, Any]:
        scored: list[dict[str, Any]] = []
        for record in candidate_records:
            tage_report_path = record.get("tage_report_path")
            validator_report_path = record.get("validator_report_path")
            tage = load_json(tage_report_path, default={}) if tage_report_path else {}
            validator = load_json(validator_report_path, default={}) if validator_report_path else {}
            valid = bool(validator.get("valid", record.get("valid", False)))
            tage_score = float(tage.get("tage_score", 0.0) or 0.0)
            selection_score = tage_score if valid else -1.0
            scored.append(
                {
                    "candidate_id": record.get("candidate_id"),
                    "mutation_family": record.get("mutation_family"),
                    "valid": valid,
                    "tage_score": tage_score,
                    "failure_avoidance": (tage.get("failure_avoidance") or {}).get("normalized_score"),
                    "preference_consistency": (tage.get("preference_consistency") or {}).get("score"),
                    "novelty": (tage.get("candidate_redundancy") or {}).get("novelty_score"),
                    "selection_score": selection_score,
                    "reward_path": record.get("reward_path"),
                    "candidate_dir": record.get("candidate_dir"),
                    "tage_report_path": tage_report_path,
                    "validator_report_path": validator_report_path,
                    "decision": "pending",
                }
            )
        valid_scored = [s for s in scored if s["valid"]]
        selected = max(valid_scored, key=lambda x: x["selection_score"]) if valid_scored else None
        for item in scored:
            item["decision"] = "selected_for_long_training" if selected and item["candidate_id"] == selected["candidate_id"] else "not_selected"
        report = {
            "round": round_idx,
            "selected_candidate": selected["candidate_id"] if selected else None,
            "selection_mode": selection_mode,
            "memory_coverage_type": coverage_report.get("coverage_type", "unknown"),
            "candidate_scores": scored,
            "reason": "Selected valid candidate with highest Memory-TAGE score." if selected else "No valid candidate available.",
            "phase": "phase_5_selector",
        }
        if output_path is not None:
            save_json(output_path, report)
        return report
