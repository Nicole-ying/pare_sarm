"""Error-aware Memory-TAGE evaluator for ASE-MTAGE.

Memory-TAGE is not a ground-truth fitness function. It is a conservative offline
filter over remembered trajectories. Its report therefore contains score,
confidence, decision authority, allowed decision, and forbidden assumptions.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Callable

from ase_mtage.utils.io import load_json, load_jsonl, save_json


RewardFn = Callable[[Any, Any, Any, bool, bool, dict[str, Any]], tuple[float, dict[str, float]]]
FAILURE_LABELS = {"early_failure", "low_progress_survival"}


class MemoryTAGEEvaluator:
    """Evaluate one reward candidate on trajectory memory."""

    def evaluate_candidate(
        self,
        *,
        candidate_id: str,
        reward_path: str | Path,
        memory_cards_path: str | Path,
        coverage_report: dict[str, Any],
        output_path: str | Path | None = None,
        other_reward_vectors: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        reward_path = Path(reward_path)
        cards = load_jsonl(memory_cards_path)
        reward_fn = self._load_reward_fn(reward_path)
        scored_cards = self._score_cards(cards=cards, reward_fn=reward_fn)
        preference_report = self._preference_consistency(scored_cards, coverage_report)
        failure_report = self._failure_avoidance(scored_cards)
        component_report = self._component_alignment(scored_cards, coverage_report)
        reward_vector = [float(x["candidate_reward_total"]) for x in scored_cards]
        novelty_report = self._novelty(candidate_id, reward_vector, other_reward_vectors or {})

        decision_level = str(coverage_report.get("decision_level", "no_decision"))
        coverage_type = str(coverage_report.get("coverage_type", "ambiguous"))
        tage_score, score_mode = self._score_by_decision_level(
            decision_level=decision_level,
            preference_score=float(preference_report.get("score", 0.0) or 0.0),
            failure_score=float(failure_report.get("normalized_score", 0.0) or 0.0),
            component_score=float(component_report.get("mean_component_consistency", 0.0) or 0.0),
            novelty_score=float(novelty_report.get("novelty_score", 0.0) or 0.0),
        )
        tage_confidence = self._confidence(
            decision_level=decision_level,
            coverage_report=coverage_report,
            preference_report=preference_report,
            scored_cards=scored_cards,
        )
        recommended_use = self._recommended_use(tage_score, tage_confidence, decision_level)
        report = {
            "candidate_id": candidate_id,
            "reward_path": str(reward_path),
            "memory_coverage_type": coverage_type,
            "decision_level": decision_level,
            "score_mode": score_mode,
            "preference_consistency": preference_report,
            "failure_avoidance": failure_report,
            "component_alignment": component_report,
            "candidate_redundancy": novelty_report,
            "tage_score": float(tage_score),
            "tage_confidence": float(tage_confidence),
            "allowed_decision": coverage_report.get("allowed_decision", self._allowed_decision(decision_level)),
            "forbidden_assumptions": list(coverage_report.get("forbidden_assumptions") or []),
            "num_scored_trajectories": len(scored_cards),
            "warnings": list(coverage_report.get("forbidden_assumptions") or []),
            "recommended_use": recommended_use,
            "phase": "error_aware_memory_tage",
        }
        if output_path is not None:
            save_json(output_path, report)
        return report

    def _score_by_decision_level(self, *, decision_level: str, preference_score: float, failure_score: float, component_score: float, novelty_score: float) -> tuple[float, str]:
        if decision_level == "strong_pairwise_selection":
            return (0.45 * preference_score + 0.20 * component_score + 0.20 * failure_score + 0.15 * novelty_score, "strong_pairwise_weighted")
        if decision_level == "weak_pairwise_selection":
            return (0.30 * preference_score + 0.20 * component_score + 0.35 * failure_score + 0.15 * novelty_score, "weak_pairwise_failure_aware")
        if decision_level == "failure_filter_only":
            return (0.70 * failure_score + 0.20 * novelty_score + 0.10 * max(component_score, 0.0), "failure_filter_only")
        # no_decision: do not pretend TAGE can choose a better reward; use only weak safety priors.
        return (0.50 * failure_score + 0.30 * novelty_score + 0.20 * 0.50, "no_decision_safety_prior")

    def _confidence(self, *, decision_level: str, coverage_report: dict[str, Any], preference_report: dict[str, Any], scored_cards: list[dict[str, Any]]) -> float:
        num_high = float(coverage_report.get("num_high_confidence", 0) or 0)
        min_high = max(float(coverage_report.get("dynamic_min_count_per_label", 3) or 3), 1.0)
        memory_factor = min(1.0, num_high / max(2.0 * min_high, 1.0))
        pair_count = float(preference_report.get("num_pairs", 0) or 0)
        pair_factor = min(1.0, pair_count / 10.0)
        margin_ok = bool((coverage_report.get("label_margin") or {}).get("is_margin_sufficient", False))
        margin_factor = 1.0 if margin_ok else 0.45
        level_factor = {
            "no_decision": 0.20,
            "failure_filter_only": 0.40,
            "weak_pairwise_selection": 0.65,
            "strong_pairwise_selection": 0.90,
        }.get(decision_level, 0.20)
        if decision_level in {"no_decision", "failure_filter_only"}:
            pair_factor = max(pair_factor, 0.5)
        runtime_bad = any(float(s.get("candidate_reward_total", 0.0) or 0.0) < -1e5 for s in scored_cards)
        runtime_factor = 0.2 if runtime_bad else 1.0
        return max(0.0, min(1.0, level_factor * (0.45 * memory_factor + 0.35 * pair_factor + 0.20 * margin_factor) * runtime_factor))

    def _recommended_use(self, score: float, confidence: float, decision_level: str) -> str:
        if decision_level == "no_decision":
            return "do_not_promote_by_tage"
        if decision_level == "failure_filter_only":
            return "use_only_as_failure_filter" if score >= 0.45 else "do_not_promote"
        if confidence < 0.35:
            return "low_confidence_do_not_promote"
        return "promote_candidate_for_long_training" if score >= 0.50 else "do_not_promote"

    def _allowed_decision(self, decision_level: str) -> str:
        return {
            "no_decision": "Do not use Memory-TAGE as a strong selector.",
            "failure_filter_only": "Use only to avoid known failure modes.",
            "weak_pairwise_selection": "Use weak preference pairs with caution.",
            "strong_pairwise_selection": "Use full preference-aware Memory-TAGE ranking.",
        }.get(decision_level, "Conservative selection only.")

    def _score_cards(self, *, cards: list[dict[str, Any]], reward_fn: RewardFn) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for card in cards:
            traj_path = card.get("trajectory_path")
            if not traj_path:
                continue
            path = Path(str(traj_path))
            if not path.exists():
                continue
            trajectory = load_json(path)
            total = 0.0
            component_totals: dict[str, float] = {}
            for step in trajectory.get("steps", []):
                try:
                    reward_value, components = reward_fn(step.get("obs"), step.get("action"), step.get("next_obs"), bool(step.get("terminated", False)), bool(step.get("truncated", False)), dict(step.get("info") or {}))
                    total += self._finite(float(reward_value))
                    for name, value in dict(components).items():
                        component_totals[str(name)] = component_totals.get(str(name), 0.0) + self._finite(float(value))
                except Exception:
                    total -= 1e6
            label = (card.get("final_label") or {}).get("coarse_label", "ambiguous")
            scored.append({"trajectory_id": card.get("trajectory_id"), "label": label, "use_for_tage_pair": bool(card.get("use_for_tage_pair", False)), "candidate_reward_total": total, "component_totals": component_totals})
        return scored

    def _preference_consistency(self, scored: list[dict[str, Any]], coverage_report: dict[str, Any]) -> dict[str, Any]:
        allowed = list(coverage_report.get("allowed_preference_relations") or [])
        pairs = []
        for high_label, low_label in allowed:
            high_cards = [s for s in scored if s["use_for_tage_pair"] and s["label"] == high_label]
            low_cards = [s for s in scored if s["use_for_tage_pair"] and s["label"] == low_label]
            for h in high_cards:
                for l in low_cards:
                    pairs.append((h, l, high_label, low_label))
        if not pairs:
            return {"num_pairs": 0, "num_satisfied": 0, "score": 0.0, "relations": allowed}
        satisfied = 0
        relation_counts: dict[str, dict[str, int]] = {}
        for high, low, high_label, low_label in pairs:
            key = f"{high_label}>{low_label}"
            relation_counts.setdefault(key, {"pairs": 0, "satisfied": 0})
            relation_counts[key]["pairs"] += 1
            if float(high["candidate_reward_total"]) > float(low["candidate_reward_total"]):
                satisfied += 1
                relation_counts[key]["satisfied"] += 1
        return {"num_pairs": len(pairs), "num_satisfied": satisfied, "score": satisfied / len(pairs), "relations": allowed, "relation_counts": relation_counts}

    def _failure_avoidance(self, scored: list[dict[str, Any]]) -> dict[str, Any]:
        if not scored:
            return {"known_failure_labels": sorted(FAILURE_LABELS), "mean_reward_on_failure": 0.0, "normalized_score": 0.0}
        rewards = [float(s["candidate_reward_total"]) for s in scored]
        min_r, max_r = min(rewards), max(rewards)
        denom = max(max_r - min_r, 1e-9)
        failure_cards = [s for s in scored if s["label"] in FAILURE_LABELS]
        if not failure_cards:
            return {"known_failure_labels": sorted(FAILURE_LABELS), "mean_reward_on_failure": 0.0, "normalized_score": 0.50}
        normalized_failure_rewards = [(float(s["candidate_reward_total"]) - min_r) / denom for s in failure_cards]
        mean_norm_failure = sum(normalized_failure_rewards) / len(normalized_failure_rewards)
        mean_raw_failure = sum(float(s["candidate_reward_total"]) for s in failure_cards) / len(failure_cards)
        return {"known_failure_labels": sorted(FAILURE_LABELS), "num_failure_trajectories": len(failure_cards), "mean_reward_on_failure": mean_raw_failure, "normalized_score": 1.0 - mean_norm_failure}

    def _component_alignment(self, scored: list[dict[str, Any]], coverage_report: dict[str, Any]) -> dict[str, Any]:
        allowed = list(coverage_report.get("allowed_preference_relations") or [])
        component_names = sorted({name for s in scored for name in dict(s.get("component_totals") or {}).keys()})
        component_reports: dict[str, Any] = {}
        scores: list[float] = []
        for name in component_names:
            pairs = []
            for high_label, low_label in allowed:
                high_cards = [s for s in scored if s["use_for_tage_pair"] and s["label"] == high_label]
                low_cards = [s for s in scored if s["use_for_tage_pair"] and s["label"] == low_label]
                for h in high_cards:
                    for l in low_cards:
                        pairs.append((h, l))
            satisfied = 0
            for h, l in pairs:
                if float(h.get("component_totals", {}).get(name, 0.0)) > float(l.get("component_totals", {}).get(name, 0.0)):
                    satisfied += 1
            consistency = satisfied / len(pairs) if pairs else 0.0
            component_reports[name] = {"pair_consistency": consistency, "num_pairs": len(pairs), "num_satisfied": satisfied, "diagnosis": self._component_diagnosis(name, consistency)}
            scores.append(consistency)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {"mean_component_consistency": mean_score, "components": component_reports}

    def _component_diagnosis(self, name: str, consistency: float) -> str:
        if consistency >= 0.70:
            return "component generally favors higher-quality remembered trajectories"
        if consistency <= 0.35:
            return "component may over-reward known low-quality trajectories"
        return "component evidence is mixed"

    def _novelty(self, candidate_id: str, reward_vector: list[float], other_vectors: dict[str, list[float]]) -> dict[str, Any]:
        if not other_vectors:
            return {"max_reward_vector_corr_with_other_candidates": 0.0, "novelty_score": 0.50}
        max_corr = 0.0
        for _, other_vec in other_vectors.items():
            max_corr = max(max_corr, abs(self._pearson(reward_vector, other_vec)))
        return {"max_reward_vector_corr_with_other_candidates": max_corr, "novelty_score": 1.0 - max_corr}

    def _pearson(self, a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        if n < 2:
            return 0.0
        x, y = a[:n], b[:n]
        mx, my = sum(x) / n, sum(y) / n
        num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        denx = math.sqrt(sum((v - mx) ** 2 for v in x))
        deny = math.sqrt(sum((v - my) ** 2 for v in y))
        if denx <= 1e-12 or deny <= 1e-12:
            return 0.0
        return num / (denx * deny)

    def _load_reward_fn(self, reward_path: Path) -> RewardFn:
        module_name = f"ase_mtage_tage_reward_{abs(hash(str(reward_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, reward_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load reward module from {reward_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "compute_reward", None)
        if fn is None:
            raise AttributeError(f"{reward_path} does not define compute_reward")
        return fn

    def _finite(self, value: float) -> float:
        return value if math.isfinite(value) else 0.0
