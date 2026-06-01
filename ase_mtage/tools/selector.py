"""Error-aware candidate selector for ASE-MTAGE.

Respects memory coverage decision level when selecting candidates.
"""

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
        decision_level = str(coverage_report.get("decision_level", "no_decision"))
        scored: list[dict[str, Any]] = []
        for record in candidate_records:
            tage_report_path = record.get("tage_report_path")
            validator_report_path = record.get("validator_report_path")
            tage = load_json(tage_report_path, default={}) if tage_report_path else {}
            validator = load_json(validator_report_path, default={}) if validator_report_path else {}
            valid = bool(validator.get("valid", record.get("valid", False)))
            tage_score = float(tage.get("tage_score", 0.0) or 0.0)
            failure_avoidance = self._float_or_default((tage.get("failure_avoidance") or {}).get("normalized_score"), 0.0)
            preference = self._float_or_default((tage.get("preference_consistency") or {}).get("score"), 0.0)
            novelty = self._float_or_default((tage.get("candidate_redundancy") or {}).get("novelty_score"), 0.0)
            static_score = self._float_or_default(record.get("selection_static_score"), 0.0)
            selection_score = self._selection_score(
                valid=valid,
                decision_level=decision_level,
                tage_score=tage_score,
                failure_avoidance=failure_avoidance,
                preference=preference,
                novelty=novelty,
                static_score=static_score,
            )
            scored.append({
                "candidate_id": record.get("candidate_id"),
                "mutation_family": record.get("mutation_family"),
                "valid": valid,
                "decision_level": decision_level,
                "tage_score": tage_score,
                "failure_avoidance": failure_avoidance,
                "preference_consistency": preference,
                "novelty": novelty,
                "static_score": static_score,
                "selection_score": selection_score,
                "recommended_use": tage.get("recommended_use"),
                "allowed_decision": tage.get("allowed_decision", coverage_report.get("allowed_decision")),
                "reward_path": record.get("reward_path"),
                "candidate_dir": record.get("candidate_dir"),
                "tage_report_path": tage_report_path,
                "validator_report_path": validator_report_path,
                "decision": "pending",
            })
        valid_scored = [s for s in scored if s["valid"]]
        selected = max(valid_scored, key=lambda x: x["selection_score"]) if valid_scored else None
        for item in scored:
            item["decision"] = "selected_for_long_training" if selected and item["candidate_id"] == selected["candidate_id"] else "not_selected"
        report = {
            "round": round_idx,
            "selected_candidate": selected["candidate_id"] if selected else None,
            "selection_mode": selection_mode,
            "memory_coverage_type": coverage_report.get("coverage_type", "unknown"),
            "decision_level": decision_level,
            "allowed_decision": coverage_report.get("allowed_decision"),
            "candidate_scores": scored,
            "reason": self._reason(selected, decision_level),
            "phase": "error_aware_selector",
        }
        if output_path is not None:
            save_json(output_path, report)
        return report

    def _selection_score(self, *, valid: bool, decision_level: str, tage_score: float, failure_avoidance: float, preference: float, novelty: float, static_score: float) -> float:
        if not valid:
            return -1.0
        # static_score is a hardcoded family prior useful only for Round 0
        # bootstrap. From Round 1 onward, trajectory-derived signals
        # (failure_avoidance, novelty, preference) replace it entirely.
        if decision_level == "no_decision":
            return 0.60 * failure_avoidance + 0.40 * novelty
        if decision_level == "failure_filter_only":
            return 0.70 * failure_avoidance + 0.30 * novelty
        if decision_level == "weak_pairwise_selection":
            return 0.35 * failure_avoidance + 0.35 * preference + 0.30 * novelty
        if decision_level == "strong_pairwise_selection":
            return 0.75 * tage_score + 0.25 * novelty
        return tage_score

    def _reason(self, selected: dict[str, Any] | None, decision_level: str) -> str:
        if not selected:
            return "No valid candidate available."
        return {
            "no_decision": "Memory coverage has no selection authority; selected conservative valid candidate using static/novelty prior.",
            "failure_filter_only": "Selected valid candidate that best avoids known failure modes.",
            "weak_pairwise_selection": "Selected valid candidate using weak preference evidence.",
            "strong_pairwise_selection": "Selected valid candidate using full Memory-TAGE ranking.",
        }.get(decision_level, "Selected valid candidate using available evidence.")

    def _float_or_default(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default
