"""Memory-TAGE evaluator for ASE-MTAGE Phase 5.

Memory-TAGE scores candidate reward functions offline on historical trajectories.
It does not train a policy. It asks whether a candidate reward ranks higher-
quality remembered trajectories above known lower-quality trajectories and
whether it avoids assigning high reward to known failures.
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
    """Evaluate one or more reward candidates on trajectory memory."""

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

        coverage_type = str(coverage_report.get("coverage_type", "ambiguous"))
        if coverage_type in {"balanced", "failure_plus_partial_progress", "partial_or_success_only"}:
            tage_score = (
                0.45 * preference_report["score"]
                + 0.20 * component_report["mean_component_consistency"]
                + 0.20 * failure_report["normalized_score"]
                + 0.15 * novelty_report["novelty_score"]
            )
            score_mode = "preference_consistency_weighted"
        else:
            # Low-coverage memory: do not pretend ranking is reliable.
            static_structure_score = 0.50
            tage_score = (
                0.35 * failure_report["normalized_score"]
                + 0.30 * static_structure_score
                + 0.20 * novelty_report["novelty_score"]
                + 0.15 * 0.50
            )
            score_mode = "low_coverage_failure_avoidance_weighted"

        report = {
            "candidate_id": candidate_id,
            "reward_path": str(reward_path),
            "memory_coverage_type": coverage_type,
            "score_mode": score_mode,
            "preference_consistency": preference_report,
            "failure_avoidance": failure_report,
            "component_alignment": component_report,
            "candidate_redundancy": novelty_report,
            "tage_score": float(tage_score),
            "num_scored_trajectories": len(scored_cards),
            "warnings": list(coverage_report.get("forbidden_assumptions") or []),
            "recommended_use": "promote_candidate" if tage_score >= 0.50 else "do_not_promote",
            "phase": "phase_5_memory_tage",
        }
        if output_path is not None:
            save_json(output_path, report)
        return report

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
                obs = step.get("obs")
                action = step.get("action")
                next_obs = step.get("next_obs")
                terminated = bool(step.get("terminated", False))
                truncated = bool(step.get("truncated", False))
                info = dict(step.get("info") or {})
                try:
                    reward_value, components = reward_fn(obs, action, next_obs, terminated, truncated, info)
                    reward_value = self._finite(float(reward_value))
                    total += reward_value
                    for name, value in dict(components).items():
                        component_totals[str(name)] = component_totals.get(str(name), 0.0) + self._finite(float(value))
                except Exception:
                    # Bad runtime on a remembered trajectory should strongly hurt the candidate.
                    total -= 1e6
            label = (card.get("final_label") or {}).get("coarse_label", "ambiguous")
            scored.append(
                {
                    "trajectory_id": card.get("trajectory_id"),
                    "label": label,
                    "use_for_tage_pair": bool(card.get("use_for_tage_pair", False)),
                    "candidate_reward_total": total,
                    "component_totals": component_totals,
                }
            )
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
        return {
            "num_pairs": len(pairs),
            "num_satisfied": satisfied,
            "score": satisfied / len(pairs),
            "relations": allowed,
            "relation_counts": relation_counts,
        }

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
        return {
            "known_failure_labels": sorted(FAILURE_LABELS),
            "num_failure_trajectories": len(failure_cards),
            "mean_reward_on_failure": mean_raw_failure,
            "normalized_score": 1.0 - mean_norm_failure,
        }

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
            if not pairs:
                consistency = 0.0
                satisfied = 0
            else:
                satisfied = 0
                for h, l in pairs:
                    if float(h.get("component_totals", {}).get(name, 0.0)) > float(l.get("component_totals", {}).get(name, 0.0)):
                        satisfied += 1
                consistency = satisfied / len(pairs)
            component_reports[name] = {
                "pair_consistency": consistency,
                "num_pairs": len(pairs),
                "num_satisfied": satisfied,
                "diagnosis": self._component_diagnosis(name, consistency),
            }
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
        for other_id, other_vec in other_vectors.items():
            corr = abs(self._pearson(reward_vector, other_vec))
            max_corr = max(max_corr, corr)
        return {"max_reward_vector_corr_with_other_candidates": max_corr, "novelty_score": 1.0 - max_corr}

    def _pearson(self, a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        if n < 2:
            return 0.0
        x = a[:n]
        y = b[:n]
        mx = sum(x) / n
        my = sum(y) / n
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
