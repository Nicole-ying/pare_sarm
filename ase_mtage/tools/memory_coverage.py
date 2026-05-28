"""Memory coverage analysis for ASE-MTAGE Phase 5.

The coverage analyzer summarizes the current trajectory memory and decides which
kind of Memory-TAGE evaluation is allowed. It is deliberately conservative: if
memory only contains a single failure type, it does not invent low/mid/high
progress groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.utils.io import ensure_dir, load_jsonl, save_json


FAILURE_LABELS = {"early_failure", "low_progress_survival"}
PARTIAL_LABELS = {"partial_progress"}
SUCCESS_LABELS = {"success_like"}
AMBIGUOUS_LABELS = {"ambiguous"}


class MemoryCoverageAnalyzer:
    """Analyze trajectory-memory coverage and allowed preference relations."""

    def __init__(
        self,
        *,
        min_trajectories: int = 10,
        min_high_confidence_trajectories: int = 8,
        min_count_per_label: int = 2,
        confidence_threshold: float = 0.70,
    ) -> None:
        self.min_trajectories = min_trajectories
        self.min_high_confidence_trajectories = min_high_confidence_trajectories
        self.min_count_per_label = min_count_per_label
        self.confidence_threshold = confidence_threshold

    def analyze_file(self, *, memory_cards_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
        cards = load_jsonl(memory_cards_path)
        report = self.analyze_cards(cards)
        if output_path is not None:
            save_json(output_path, report)
        return report

    def analyze_cards(self, cards: list[dict[str, Any]]) -> dict[str, Any]:
        label_counts: dict[str, int] = {}
        high_conf_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        usable_for_tage = 0

        for card in cards:
            final_label = dict(card.get("final_label") or {})
            label = str(final_label.get("coarse_label", "ambiguous"))
            confidence = float(final_label.get("confidence", 0.0) or 0.0)
            use_for_tage = bool(card.get("use_for_tage_pair", False))
            role = str(card.get("allowed_preference_role", "none"))

            label_counts[label] = label_counts.get(label, 0) + 1
            if confidence >= self.confidence_threshold and label not in AMBIGUOUS_LABELS:
                high_conf_counts[label] = high_conf_counts.get(label, 0) + 1
            if use_for_tage:
                usable_for_tage += 1
                role_counts[role] = role_counts.get(role, 0) + 1

        nontrivial_labels = {
            label for label, count in high_conf_counts.items() if count >= self.min_count_per_label
        }
        has_failure = bool(nontrivial_labels & FAILURE_LABELS)
        has_partial = bool(nontrivial_labels & PARTIAL_LABELS)
        has_success = bool(nontrivial_labels & SUCCESS_LABELS)

        if len(cards) < self.min_trajectories or usable_for_tage < self.min_high_confidence_trajectories:
            coverage_type = "empty_or_too_small"
        elif len(nontrivial_labels) == 1 and has_failure:
            coverage_type = "single_failure_mode"
        elif len(nontrivial_labels) >= 2 and has_failure and not has_partial and not has_success:
            coverage_type = "multiple_failure_modes"
        elif has_failure and has_partial and not has_success:
            coverage_type = "failure_plus_partial_progress"
        elif has_failure and (has_partial or has_success):
            coverage_type = "balanced"
        elif has_success or has_partial:
            coverage_type = "partial_or_success_only"
        else:
            coverage_type = "ambiguous"

        allowed_relations = self._allowed_relations(coverage_type, nontrivial_labels)
        can_build_pairs = bool(allowed_relations)
        report = {
            "num_trajectories": len(cards),
            "num_high_confidence": sum(high_conf_counts.values()),
            "num_use_for_tage_pair": usable_for_tage,
            "label_counts": label_counts,
            "high_confidence_label_counts": high_conf_counts,
            "preference_role_counts": role_counts,
            "nontrivial_labels": sorted(nontrivial_labels),
            "coverage_type": coverage_type,
            "can_build_preference_pairs": can_build_pairs,
            "allowed_preference_relations": allowed_relations,
            "forbidden_assumptions": self._forbidden_assumptions(coverage_type, nontrivial_labels),
            "suggested_search_mode": self._suggested_search_mode(coverage_type),
            "phase": "phase_5_memory_coverage",
        }
        return report

    def _allowed_relations(self, coverage_type: str, labels: set[str]) -> list[list[str]]:
        relations: list[list[str]] = []
        if coverage_type in {"failure_plus_partial_progress", "balanced"}:
            if "partial_progress" in labels:
                if "early_failure" in labels:
                    relations.append(["partial_progress", "early_failure"])
                if "low_progress_survival" in labels:
                    relations.append(["partial_progress", "low_progress_survival"])
            if "success_like" in labels:
                if "partial_progress" in labels:
                    relations.append(["success_like", "partial_progress"])
                if "early_failure" in labels:
                    relations.append(["success_like", "early_failure"])
                if "low_progress_survival" in labels:
                    relations.append(["success_like", "low_progress_survival"])
        elif coverage_type == "partial_or_success_only":
            if "success_like" in labels and "partial_progress" in labels:
                relations.append(["success_like", "partial_progress"])
        return relations

    def _forbidden_assumptions(self, coverage_type: str, labels: set[str]) -> list[str]:
        warnings: list[str] = []
        if "success_like" not in labels:
            warnings.append("Do not construct success_like preference pairs because no high-confidence success_like trajectory exists.")
        if coverage_type in {"empty_or_too_small", "single_failure_mode", "multiple_failure_modes"}:
            warnings.append("Do not divide memory into low/mid/high by quantile; memory coverage is not balanced enough.")
        if coverage_type == "failure_plus_partial_progress":
            warnings.append("Do not treat partial_progress as success_like; only weak preference pairs are allowed.")
        if coverage_type == "ambiguous":
            warnings.append("Memory labels are not reliable enough for strong Memory-TAGE selection.")
        return warnings

    def _suggested_search_mode(self, coverage_type: str) -> str:
        mapping = {
            "empty_or_too_small": "static_structure_and_exploration",
            "single_failure_mode": "avoid_single_known_failure",
            "multiple_failure_modes": "failure_contrast_and_novelty",
            "failure_plus_partial_progress": "escape_known_failures_and_improve_partial_progress",
            "balanced": "full_memory_tage_ranking",
            "partial_or_success_only": "preserve_progress_and_explore_failures",
            "ambiguous": "conservative_selection_due_to_uncertain_memory",
        }
        return mapping.get(coverage_type, "conservative_selection")
