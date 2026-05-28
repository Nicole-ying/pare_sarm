"""Error-aware memory coverage analysis for ASE-MTAGE.

The coverage analyzer does not claim that trajectory labels are ground truth. It
estimates how much decision authority Memory-TAGE is allowed to have from the
current memory. It uses dynamic label-count thresholds, simple label-margin
checks, and explicit decision levels so noisy progress proxies do not silently
become strong reward-selection signals.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any

from ase_mtage.utils.io import load_jsonl, save_json


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
        min_count_per_label: int = 3,
        confidence_threshold: float = 0.70,
        dynamic_label_ratio: float = 0.15,
        progress_margin_threshold: float = 0.05,
        stability_margin_threshold: float = -0.05,
    ) -> None:
        self.min_trajectories = int(min_trajectories)
        self.min_high_confidence_trajectories = int(min_high_confidence_trajectories)
        self.min_count_per_label = int(min_count_per_label)
        self.confidence_threshold = float(confidence_threshold)
        self.dynamic_label_ratio = float(dynamic_label_ratio)
        self.progress_margin_threshold = float(progress_margin_threshold)
        self.stability_margin_threshold = float(stability_margin_threshold)

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
        confidence_by_label: dict[str, list[float]] = {}
        usable_for_tage = 0
        high_conf_cards: list[dict[str, Any]] = []

        for card in cards:
            final_label = dict(card.get("final_label") or {})
            label = str(final_label.get("coarse_label", "ambiguous"))
            confidence = float(final_label.get("confidence", 0.0) or 0.0)
            use_for_tage = bool(card.get("use_for_tage_pair", False))
            role = str(card.get("allowed_preference_role", "none"))
            label_counts[label] = label_counts.get(label, 0) + 1
            confidence_by_label.setdefault(label, []).append(confidence)
            if confidence >= self.confidence_threshold and label not in AMBIGUOUS_LABELS and use_for_tage:
                high_conf_counts[label] = high_conf_counts.get(label, 0) + 1
                high_conf_cards.append(card)
            if use_for_tage:
                usable_for_tage += 1
                role_counts[role] = role_counts.get(role, 0) + 1

        num_high_conf = sum(high_conf_counts.values())
        dynamic_min = self._dynamic_min_count(num_high_conf)
        nontrivial_labels = {label for label, count in high_conf_counts.items() if count >= dynamic_min}
        has_failure = bool(nontrivial_labels & FAILURE_LABELS)
        has_partial = bool(nontrivial_labels & PARTIAL_LABELS)
        has_success = bool(nontrivial_labels & SUCCESS_LABELS)
        label_margin = self._label_margin(high_conf_cards)
        margin_ok = bool(label_margin.get("is_margin_sufficient", False))

        if len(cards) < self.min_trajectories or num_high_conf < self.min_high_confidence_trajectories:
            coverage_type = "empty_or_too_small"
        elif len(nontrivial_labels) == 1 and has_failure:
            coverage_type = "single_failure_mode"
        elif len(nontrivial_labels) >= 2 and has_failure and not has_partial and not has_success:
            coverage_type = "multiple_failure_modes"
        elif has_failure and has_partial and not has_success:
            coverage_type = "failure_plus_partial_progress" if margin_ok else "failure_plus_weak_or_noisy_partial"
        elif has_failure and (has_partial or has_success):
            coverage_type = "balanced" if (has_success or margin_ok) else "failure_plus_weak_or_noisy_partial"
        elif has_success or has_partial:
            coverage_type = "partial_or_success_only" if (has_success or margin_ok) else "ambiguous"
        else:
            coverage_type = "ambiguous"

        decision_level = self._decision_level(coverage_type)
        allowed_relations = self._allowed_relations(coverage_type, nontrivial_labels)
        report = {
            "num_trajectories": len(cards),
            "num_high_confidence": num_high_conf,
            "num_use_for_tage_pair": usable_for_tage,
            "label_counts": label_counts,
            "high_confidence_label_counts": high_conf_counts,
            "confidence_stats_by_label": self._confidence_stats(confidence_by_label),
            "preference_role_counts": role_counts,
            "dynamic_min_count_per_label": dynamic_min,
            "nontrivial_labels": sorted(nontrivial_labels),
            "label_margin": label_margin,
            "coverage_type": coverage_type,
            "decision_level": decision_level,
            "can_build_preference_pairs": bool(allowed_relations),
            "allowed_preference_relations": allowed_relations,
            "allowed_decision": self._allowed_decision(decision_level),
            "forbidden_assumptions": self._forbidden_assumptions(coverage_type, nontrivial_labels, margin_ok),
            "suggested_search_mode": self._suggested_search_mode(coverage_type),
            "phase": "error_aware_memory_coverage",
        }
        return report

    def _dynamic_min_count(self, num_high_conf: int) -> int:
        return max(self.min_count_per_label, int(math.ceil(self.dynamic_label_ratio * max(0, num_high_conf))))

    def _label_margin(self, cards: list[dict[str, Any]]) -> dict[str, Any]:
        failure_cards = [c for c in cards if ((c.get("final_label") or {}).get("coarse_label") in FAILURE_LABELS)]
        partial_cards = [c for c in cards if ((c.get("final_label") or {}).get("coarse_label") == "partial_progress")]
        success_cards = [c for c in cards if ((c.get("final_label") or {}).get("coarse_label") == "success_like")]
        positive_cards = success_cards or partial_cards
        if not failure_cards or not positive_cards:
            return {"is_margin_sufficient": False, "reason": "Need both failure and positive/partial labels for margin check."}
        f_prog = self._feature_mean(failure_cards, ["progress_improvement", "distance_improvement", "estimated_forward_progress", "forward_displacement_proxy"])
        p_prog = self._feature_mean(positive_cards, ["progress_improvement", "distance_improvement", "estimated_forward_progress", "forward_displacement_proxy"])
        f_stab = self._stability_proxy(failure_cards)
        p_stab = self._stability_proxy(positive_cards)
        progress_gap = p_prog - f_prog
        stability_gap = p_stab - f_stab
        sufficient = progress_gap >= self.progress_margin_threshold and stability_gap >= self.stability_margin_threshold
        return {
            "is_margin_sufficient": bool(sufficient),
            "positive_label_group": "success_like" if success_cards else "partial_progress",
            "num_failure_cards": len(failure_cards),
            "num_positive_cards": len(positive_cards),
            "failure_progress_mean": f_prog,
            "positive_progress_mean": p_prog,
            "progress_gap": progress_gap,
            "failure_stability_proxy_mean": f_stab,
            "positive_stability_proxy_mean": p_stab,
            "stability_gap": stability_gap,
            "thresholds": {"progress_gap_min": self.progress_margin_threshold, "stability_gap_min": self.stability_margin_threshold},
        }

    def _feature_mean(self, cards: list[dict[str, Any]], keys: list[str]) -> float:
        values: list[float] = []
        for card in cards:
            features = dict(card.get("features") or {})
            for key in keys:
                if key in features:
                    try:
                        values.append(float(features[key]))
                        break
                    except Exception:
                        pass
        return mean(values) if values else 0.0

    def _stability_proxy(self, cards: list[dict[str, Any]]) -> float:
        # Larger is better. Convert common instability signals into negative costs.
        vals: list[float] = []
        for card in cards:
            f = dict(card.get("features") or {})
            speed = float(f.get("final_speed", abs(float(f.get("final_forward_velocity", 0.0))) + abs(float(f.get("vertical_velocity_abs", 0.0)))) or 0.0)
            angle = abs(float(f.get("final_angle_abs", 0.0) or 0.0))
            contact = float(f.get("contact_ratio_last20", f.get("contact_ratio", 0.0)) or 0.0)
            vals.append(contact - speed - angle)
        return mean(vals) if vals else 0.0

    def _confidence_stats(self, by_label: dict[str, list[float]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for label, values in by_label.items():
            out[label] = {"count": len(values), "mean": mean(values) if values else 0.0, "min": min(values) if values else 0.0, "max": max(values) if values else 0.0}
        return out

    def _decision_level(self, coverage_type: str) -> str:
        if coverage_type in {"empty_or_too_small", "ambiguous", "partial_or_success_only"}:
            return "no_decision"
        if coverage_type in {"single_failure_mode", "multiple_failure_modes", "failure_plus_weak_or_noisy_partial"}:
            return "failure_filter_only"
        if coverage_type == "failure_plus_partial_progress":
            return "weak_pairwise_selection"
        if coverage_type == "balanced":
            return "strong_pairwise_selection"
        return "no_decision"

    def _allowed_decision(self, decision_level: str) -> str:
        return {
            "no_decision": "Do not use Memory-TAGE as a strong selector; prefer conservative/elite-preserving mutation.",
            "failure_filter_only": "Use Memory-TAGE only to avoid known failure modes; do not claim progress preference.",
            "weak_pairwise_selection": "Use weak preference pairs to select a candidate for long training with low-confidence warning.",
            "strong_pairwise_selection": "Use full preference-aware Memory-TAGE ranking.",
        }.get(decision_level, "Conservative selection only.")

    def _allowed_relations(self, coverage_type: str, labels: set[str]) -> list[list[str]]:
        relations: list[list[str]] = []
        if coverage_type in {"failure_plus_partial_progress", "balanced"}:
            if "partial_progress" in labels:
                for low in ["early_failure", "low_progress_survival"]:
                    if low in labels:
                        relations.append(["partial_progress", low])
            if "success_like" in labels:
                if "partial_progress" in labels:
                    relations.append(["success_like", "partial_progress"])
                for low in ["early_failure", "low_progress_survival"]:
                    if low in labels:
                        relations.append(["success_like", low])
        return relations

    def _forbidden_assumptions(self, coverage_type: str, labels: set[str], margin_ok: bool) -> list[str]:
        warnings: list[str] = []
        if "success_like" not in labels:
            warnings.append("Do not construct success_like preference pairs because no high-confidence success_like trajectory exists.")
        if coverage_type in {"empty_or_too_small", "single_failure_mode", "multiple_failure_modes", "failure_plus_weak_or_noisy_partial"}:
            warnings.append("Do not divide memory into low/mid/high by quantile; memory coverage is not balanced enough.")
        if coverage_type == "failure_plus_partial_progress":
            warnings.append("Do not treat partial_progress as success_like; only weak preference pairs are allowed.")
        if not margin_ok:
            warnings.append("Do not treat positive/partial labels as reliable progress evidence because label-margin check is insufficient.")
        if coverage_type == "ambiguous":
            warnings.append("Memory labels are not reliable enough for strong Memory-TAGE selection.")
        return warnings

    def _suggested_search_mode(self, coverage_type: str) -> str:
        return {
            "empty_or_too_small": "static_structure_and_exploration",
            "single_failure_mode": "avoid_single_known_failure",
            "multiple_failure_modes": "failure_contrast_and_novelty",
            "failure_plus_weak_or_noisy_partial": "failure_avoidance_until_partial_margin_improves",
            "failure_plus_partial_progress": "escape_known_failures_and_improve_partial_progress",
            "balanced": "full_memory_tage_ranking",
            "partial_or_success_only": "preserve_progress_and_explore_failures",
            "ambiguous": "conservative_selection_due_to_uncertain_memory",
        }.get(coverage_type, "conservative_selection")
