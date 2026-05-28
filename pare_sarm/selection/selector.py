"""Budgeted candidate promotion for PARE-SARM.

This replaces hard ``best_health`` gates.  Short-training diagnostics are noisy;
they should rank candidates and allocate budget, not permanently block long
training.  The selector always promotes at least one valid candidate per round.
"""

from __future__ import annotations

from typing import Any


def compute_promotion_score(candidate: dict[str, Any], config: dict[str, Any] | None = None) -> float:
    cfg = (config or {}).get("selection", {}) if isinstance(config, dict) else {}
    weights = {
        "behavior": float(cfg.get("w_behavior", 0.30)),
        "component": float(cfg.get("w_component", 0.20)),
        "progress": float(cfg.get("w_progress", 0.20)),
        "short_return": float(cfg.get("w_short_return", 0.15)),
        "novelty": float(cfg.get("w_novelty", 0.15)),
    }

    behavior = float(candidate.get("behavior_report", {}).get("behavior_quality", 0.35))
    health = float(candidate.get("health", {}).get("overall_health", 0.0)) / 100.0
    progress = _progress_alignment(candidate.get("health", {}))
    short_return = _short_return_trend(candidate.get("proxy_result", {}).get("eval_history", []))
    novelty = float(candidate.get("diversity_bonus", candidate.get("novelty_score", 0.5)))

    score = (
        weights["behavior"] * behavior
        + weights["component"] * health
        + weights["progress"] * progress
        + weights["short_return"] * short_return
        + weights["novelty"] * novelty
    )
    return round(max(0.0, min(1.0, score)), 4)


def select_for_long_training(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    behavior_memory=None,
) -> list[dict[str, Any]]:
    """Return candidates selected for long training.

    Rules:
    1. Never compare against historical best_health as a hard gate.
    2. Always promote the top current-round valid candidate.
    3. If budget allows, promote a second candidate that is diverse or progress-gated.
    4. If behavior memory detects oscillation, force a progress_gated candidate into
       the selected set when available.
    """
    valid = [c for c in candidates if _is_valid(c)]
    if not valid:
        return []

    for c in valid:
        c["promotion_score"] = compute_promotion_score(c, config)

    ranked = sorted(valid, key=lambda x: x.get("promotion_score", 0.0), reverse=True)
    slots = int((config.get("selection", {}) if isinstance(config, dict) else {}).get("long_train_slots_per_round", 1))
    slots = max(1, slots)

    selected: list[dict[str, Any]] = [ranked[0]]

    oscillating = False
    if behavior_memory is not None and hasattr(behavior_memory, "detect_patterns"):
        try:
            oscillating = bool(behavior_memory.detect_patterns().get("oscillation"))
        except Exception:
            oscillating = False

    if oscillating:
        pg = _find_mutation(ranked, "progress_gated")
        if pg is not None and pg not in selected:
            if len(selected) < slots:
                selected.append(pg)
            else:
                selected[-1] = pg

    if len(selected) < slots and len(ranked) > 1:
        remaining = [c for c in ranked if c not in selected]
        remaining.sort(key=lambda c: (c.get("diversity_bonus", 0.5), c.get("promotion_score", 0.0)), reverse=True)
        selected.append(remaining[0])

    for c in ranked:
        c["selected_for_long_train"] = c in selected
        c["selection_reason"] = (
            "budgeted_promotion: selected by current-round promotion score; "
            "historical best_health is not used as a gate"
            if c in selected else
            "not selected: lower current-round promotion score or budget exhausted"
        )
    return selected


def _is_valid(c: dict[str, Any]) -> bool:
    if not c.get("parse_ok", True):
        return False
    if not c.get("code"):
        return False
    h = c.get("health", {})
    return h.get("overall_health", 0) >= 0


def _find_mutation(candidates: list[dict[str, Any]], mutation_type: str) -> dict[str, Any] | None:
    for c in candidates:
        if c.get("mutation_type") == mutation_type or c.get("style") == mutation_type:
            return c
    return None


def _progress_alignment(health: dict[str, Any]) -> float:
    comps = health.get("components", []) or []
    if not comps:
        return 0.5
    vals = []
    for c in comps:
        try:
            vals.append(max(0.0, float(c.get("progress_corr", 0.0))))
        except (TypeError, ValueError):
            pass
    return max(0.0, min(1.0, sum(vals) / max(len(vals), 1))) if vals else 0.5


def _short_return_trend(eval_history: list[dict[str, Any]]) -> float:
    vals = []
    for row in eval_history or []:
        for key in ("mean_reward", "mean_return", "reward", "ep_rew_mean"):
            if key in row:
                try:
                    vals.append(float(row[key]))
                    break
                except (TypeError, ValueError):
                    pass
    if len(vals) < 2:
        return 0.5
    delta = vals[-1] - vals[0]
    return max(0.0, min(1.0, 0.5 + delta / (2.0 * (abs(vals[0]) + abs(vals[-1]) + 1.0))))
