"""Behavior-aware PARE-SARM pipeline adapter.

This module keeps the original pipeline structure but fixes the critical search
bug: short-training health is never used as a hard gate for full training.  It
also records short-training behavior reports and uses budgeted candidate
promotion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pare_sarm.pipeline import Pipeline as BasePipeline
from pare_sarm.utils import ensure_dir, save_json, load_json
from pare_sarm.agents.analyzer import run_analyzer
from pare_sarm.agents.mutator import generate_mutation_candidates
from pare_sarm.reward.validator import validate_reward_quality, print_quality_report
from pare_sarm.reward.diversity import check_behavioral_diversity
from pare_sarm.training.proxy_trainer import run_proxy_training
from pare_sarm.diagnosis.behavior_classifier import (
    build_behavior_report_from_dir,
    summarize_behavior_report,
)
from pare_sarm.selection.selector import compute_promotion_score, select_for_long_training


class Pipeline(BasePipeline):
    """Drop-in replacement for the original Pipeline.

    Main changes:
    - every round long-trains at least one valid candidate;
    - each candidate gets a short_behavior_report.json;
    - selection uses behavior_quality plus component health;
    - round summaries explicitly record that historical best_health is not a gate.
    """

    def _proxy_train_all(self, candidates: list[dict], rdir: Path, round_num: int) -> list[dict]:
        steps = self.config.get("proxy_timesteps", 10000)
        trained: list[dict] = []
        for i, c in enumerate(candidates):
            c["idx"] = c.get("idx", i)
            style = c.get("style", c.get("mutation_type", "?"))
            print(f"  Candidate {i+1}/{len(candidates)} ({style})...")
            cand_dir = ensure_dir(rdir / f"candidate_{i}")
            self._save_reward(c["code"], cand_dir)

            if self.dry_run:
                c["proxy_result"] = {"success": True, "eval_history": []}
                c["health"] = {"overall_health": 50.0, "verdict": "dry-run", "components": [], "summary": "dry-run"}
                c["behavior_report"] = {"behavior_mode": "moderate", "behavior_quality": 0.5, "reason": "dry-run", "evidence": [], "metrics": {}}
            else:
                result = run_proxy_training(
                    self.env_dir, c["code"], cand_dir, self.config,
                    env_id_suffix=f"r{round_num}-c{i}",
                    total_timesteps=steps,
                    seed=self.config.get("seed", 42),
                    progress_fn_code=self.progress_fn_code,
                )
                c["proxy_result"] = result
                c["health"] = result.get("health", {})
                c["behavior_report"] = build_behavior_report_from_dir(
                    cand_dir,
                    env_name=self.env_dir.name,
                    max_episode_steps=self.config.get("max_episode_steps", 1000),
                )

            c["promotion_score"] = compute_promotion_score(c, self.config)
            c["selected_for_long_train"] = False
            self._save_candidate_summary(cand_dir, c, round_num)
            h = c.get("health", {})
            b = c.get("behavior_report", {})
            print(
                f"    Health: {h.get('overall_health', 0):.0f}/100 | "
                f"Behavior: {b.get('behavior_mode', '?')} q={b.get('behavior_quality', 0):.2f} | "
                f"Promotion: {c.get('promotion_score', 0):.3f}"
            )
            trained.append(c)
        return trained

    def _score_and_select(self, candidates: list[dict]) -> dict | None:
        valid = [c for c in candidates if c.get("code") and c.get("health", {}).get("overall_health", 0) >= 0]
        if not valid:
            return None
        valid = check_behavioral_diversity(valid)
        selected = select_for_long_training(valid, self.config, behavior_memory=self.memory.behavior)
        if not selected:
            return None
        winner = selected[0]
        self._print_candidate_ranking(valid, winner)
        return winner

    def _run_iteration_round(self, round_num: int) -> dict:
        rdir = ensure_dir(self.exp_dir / f"round{round_num}")
        prev_dir = self.exp_dir / f"round{round_num - 1}"

        prev_health = self._compute_health_for_round(prev_dir)
        prev_code = self._read_reward_code(prev_dir)

        print(f"\n>>> Phase 1: Analyzer (diagnosing round {round_num - 1})")
        history = self.memory.episodic.get_history_text()
        if self.dry_run:
            analysis = _dry_analysis_v2()
        else:
            behavior = self._build_behavior_summary(prev_dir)
            analysis = run_analyzer(
                prev_health, prev_health.get("component_stats", []), prev_code,
                self.env_perception_result["task_manifest"], round_num, history,
                self.api_key, self.model, temperature=0.4,
                output_dir=ensure_dir(rdir / "analyzer"),
                behavior_summary=behavior,
            )
        print(f"  Diagnosis: {analysis.get('diagnosis', 'N/A')[:150]}")
        print(f"  Failure mode: {analysis.get('failure_mode', '?')} | recommended mutation: {analysis.get('recommended_mutation', '?')}")

        action = analysis.get("pipeline_action", "continue")
        if action == "stop":
            self._copy_prev_reward(prev_dir, rdir)
            return {"success": True, "round": round_num, "stop": True, "reason": "converged"}

        print("\n>>> Phase 2: Candidate Generation / Mutation")
        memory_ctx = self._get_memory_context_for_mutation_v2(analysis)
        if action == "regenerate":
            candidates = self._generate_initial_candidates(
                self._load_exploration_text(), memory_context=memory_ctx,
                output_subdir=f"round{round_num}",
            )
        else:
            if self.dry_run:
                candidates = _dry_mutations_v2(prev_code)
            else:
                raw = generate_mutation_candidates(
                    analysis, prev_code,
                    self.env_perception_result["task_manifest"],
                    self.env_perception_result["reward_signature"],
                    self.progress_fn_code,
                    self.api_key, self.model,
                    memory_context=memory_ctx,
                    output_dir=rdir,
                )
                candidates = []
                for c in raw:
                    if c.get("parse_ok") and c.get("code"):
                        warnings = validate_reward_quality(c["code"], self.env_perception_result["task_manifest"])
                        print_quality_report(warnings, f"{c.get('mutation_type', 'candidate')} #{c.get('idx', '?')}")
                        c["warnings"] = warnings
                        candidates.append(c)

        if not candidates:
            print("  Candidate generation failed. Keeping previous reward.")
            self._copy_prev_reward(prev_dir, rdir)
            return {"success": True, "round": round_num, "stop": True, "reason": "candidate generation failed"}

        print("\n>>> Phase 3: Proxy Training")
        candidates = self._proxy_train_all(candidates, rdir, round_num)

        print("\n>>> Phase 4: Behavior-aware Selection")
        winner = self._score_and_select(candidates)
        if winner is None:
            self._copy_prev_reward(prev_dir, rdir)
            return {"success": True, "round": round_num, "stop": True, "reason": "all proxy failed"}

        print("\n>>> Phase 5: Full Training (budgeted promotion; no best_health gate)")
        self._save_reward(winner["code"], rdir)
        ft_result = self._run_full_training_for_round(rdir, winner, round_num)
        full_behavior = self._save_full_behavior_report(rdir, winner)

        self._store_round_memory_v2(round_num, winner, analysis, full_behavior)
        w_health = winner.get("health", {}).get("overall_health", 0)
        self.best_health = max(self.best_health, w_health)

        return {
            "success": True,
            "round": round_num,
            "winner_health": winner.get("health", {}),
            "winner_score": winner.get("promotion_score"),
            "winner_behavior": full_behavior or winner.get("behavior_report", {}),
            "selected_for_long_training": [winner.get("idx", 0)],
            "long_training_executed": True,
            "analysis": analysis,
            "full_training_result": ft_result,
        }

    def _save_full_behavior_report(self, rdir: Path, winner: dict) -> dict:
        report = build_behavior_report_from_dir(
            rdir / "full_training",
            env_name=self.env_dir.name,
            max_episode_steps=self.config.get("max_episode_steps", 1000),
        )
        save_json(rdir / "full_training" / "long_behavior_report.json", report)
        return report

    def _build_current_behavior(self, round_dir: Path) -> str:
        for p in [round_dir / "full_training" / "long_behavior_report.json", round_dir / "behavior_report.json"]:
            if p.exists():
                return summarize_behavior_report(load_json(p))
        report = build_behavior_report_from_dir(
            round_dir / "full_training",
            env_name=self.env_dir.name,
            max_episode_steps=self.config.get("max_episode_steps", 1000),
        )
        if report.get("evidence") or report.get("metrics"):
            return summarize_behavior_report(report)
        return ""

    def _get_memory_context_for_mutation_v2(self, analysis: dict | None = None) -> str:
        parts: list[str] = []
        behavior_table = self.memory.behavior.format_history_table()
        if behavior_table and "*(no behavior history)*" not in behavior_table:
            parts.append(behavior_table)
        all_rounds = self.memory.episodic.get_all_rounds()
        if all_rounds:
            parts.append("## Previous Failure-Repair-Outcome Records")
            for r in sorted(all_rounds.keys()):
                d = all_rounds[r]
                parts.append(
                    f"Round {r}: mode={d.get('behavior_mode','?')}; mutation={d.get('mutation_type','?')}; "
                    f"root={d.get('root_cause','?')}; lesson={d.get('lesson','')[:160]}"
                )
        query = ""
        if analysis:
            query = " ".join([analysis.get("failure_mode", ""), analysis.get("root_cause_type", ""), analysis.get("diagnosis", "")])
        archival = self.memory.archival.search(query, max_results=3) if query else []
        if archival:
            parts.append("## Relevant Archival Design Principles")
            for p in archival:
                parts.append(f"- {p}")
        patterns = self.memory.behavior.detect_patterns()
        if patterns.get("suggestion"):
            parts.append("## Triggered Memory Guidance")
            parts.append(patterns["suggestion"])
            self._log_tool_use("memory_context", "behavior_pattern_detected", patterns, used=True)
        return "\n".join(parts) if parts else ""

    def _store_round_memory_v2(self, round_num: int, winner: dict, analysis: dict, full_behavior: dict | None = None):
        wh = winner.get("health", {}).get("overall_health", 0)
        behavior = full_behavior or winner.get("behavior_report", {}) or {}
        mode = behavior.get("behavior_mode", "")
        quality = float(behavior.get("behavior_quality", 0) or 0)
        mutation_type = winner.get("mutation_type", winner.get("style", ""))
        diagnosis = analysis.get("diagnosis", winner.get("health", {}).get("summary", ""))
        lesson = _make_lesson(mode, mutation_type, analysis, behavior)
        self.memory.episodic.store(round_num, {
            "summary": f"Round {round_num}: mode={mode}, q={quality:.2f}, health={wh:.0f}, mutation={mutation_type}",
            "reward_fn_source": winner.get("code", ""),
            "health_score": wh,
            "behavior_mode": mode,
            "behavior_quality": quality,
            "mutation_type": mutation_type,
            "diagnosis": diagnosis,
            "root_cause": analysis.get("root_cause_type", ""),
            "repair_action": mutation_type,
            "outcome": mode,
            "lesson": lesson,
        })
        # Keep compatibility with existing BehaviorMemory implementation.
        pattern = mode if mode != "early_crash" else "crashing"
        metrics = behavior.get("metrics", {}) if isinstance(behavior, dict) else {}
        self.memory.behavior.record(
            round_num,
            final_length=metrics.get("final_length", 0),
            max_length=metrics.get("max_length", 0),
            max_episode_steps=self.config.get("max_episode_steps", 1000),
            length_trend=[],
            health_score=wh,
            pattern=pattern,
            diagnosis=f"{diagnosis[:160]} | behavior={mode} q={quality:.2f} | lesson={lesson[:160]}",
            repair_strategy=mutation_type,
        )
        patterns = self.memory.behavior.detect_patterns()
        if patterns.get("oscillation") or analysis.get("escalation_level") in ("structural", "rewrite"):
            self.memory.archival.add(
                f"Round {round_num}: failure={mode}; mutation={mutation_type}; root={analysis.get('root_cause_type','')}; lesson={lesson}",
                round_num,
                importance=2.0 if patterns.get("oscillation") else 1.5,
            )
        self.memory.save()

    def _save_candidate_summary(self, cand_dir: Path, c: dict, round_num: int):
        save_json(cand_dir / "candidate_summary.json", {
            "round": round_num,
            "candidate_idx": c.get("idx"),
            "mutation_type": c.get("mutation_type", c.get("style")),
            "validation_passed": bool(c.get("code")),
            "health": c.get("health", {}),
            "behavior_report": c.get("behavior_report", {}),
            "promotion_score": c.get("promotion_score"),
            "selected_for_long_train": c.get("selected_for_long_train", False),
            "selection_reason": c.get("selection_reason", ""),
            "warnings": c.get("warnings", []),
        })
        if c.get("behavior_report"):
            save_json(cand_dir / "short_behavior_report.json", c["behavior_report"])
        if c.get("health"):
            save_json(cand_dir / "component_health.json", c["health"])

    def _print_candidate_ranking(self, candidates: list[dict], winner: dict):
        ranked = sorted(candidates, key=lambda c: c.get("promotion_score", 0), reverse=True)
        for i, c in enumerate(ranked):
            h = c.get("health", {})
            b = c.get("behavior_report", {})
            marker = " *** WINNER" if c is winner else ""
            print(
                f"  {i+1}. {c.get('style', c.get('mutation_type','?'))}: "
                f"score={c.get('promotion_score', 0):.3f} health={h.get('overall_health', 0):.0f} "
                f"behavior={b.get('behavior_mode','?')} q={b.get('behavior_quality',0):.2f}{marker}"
            )

    def _save_round_state(self, round_num: int, result: dict):
        if self.dry_run:
            return
        rdir = ensure_dir(self.exp_dir / f"round{round_num}")
        save_json(rdir / "round_summary.json", {
            "round": round_num,
            "success": result.get("success", False),
            "winner_health": result.get("winner_health", {}),
            "winner_score": result.get("winner_score"),
            "winner_behavior": result.get("winner_behavior", {}),
            "selected_for_long_training": result.get("selected_for_long_training", []),
            "long_training_executed": result.get("long_training_executed", False),
            "historical_best_health_used_as_gate": False,
            "stopped": result.get("stop", False),
            "reason": result.get("reason", ""),
        })
        self._save_experiment_state(round_num, "ROUND_SUMMARY_SAVED")

    def _save_experiment_state(self, current_round: int, last_completed_node: str):
        if self.dry_run:
            return
        save_json(self.exp_dir / "experiment_state.json", {
            "env_id": self.env_dir.name,
            "current_round": current_round,
            "last_completed_node": last_completed_node,
            "best_health_diagnostic_only": self.best_health,
            "historical_best_health_used_as_gate": False,
            "force_long_train_each_round": True,
        })

    def _log_tool_use(self, agent: str, trigger: str, result: Any, used: bool):
        path = self.exp_dir / "memory" / "tool_use_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "agent": agent,
                "trigger": trigger,
                "used_in_decision": used,
                "result_summary": result,
            }, ensure_ascii=False) + "\n")

    def _build_summary(self) -> dict:
        return {
            "success": any(r.get("success") for r in self.round_results),
            "exp_dir": str(self.exp_dir),
            "env": self.env_dir.name,
            "n_rounds": len(self.round_results),
            "rounds": [
                {
                    "round": r.get("round"),
                    "health": r.get("winner_health", {}).get("overall_health"),
                    "score": r.get("winner_score"),
                    "behavior": r.get("winner_behavior", {}).get("behavior_mode"),
                    "long_training_executed": r.get("long_training_executed"),
                }
                for r in self.round_results
            ],
        }


def _make_lesson(mode: str, mutation_type: str, analysis: dict, behavior: dict) -> str:
    if mode == "hovering":
        return "High component health can be false-positive hovering; gate dense positive rewards and avoid per-step reward farming."
    if mode == "early_crash":
        return "Penalties may dominate or the last repair overcorrected; avoid simply increasing negative terms."
    if mode == "approach_but_unstable":
        return "Approach signal works but final stabilization or terminal landing signal is insufficient."
    if mode == "landing_progress":
        return "This mutation produced useful landing-progress behavior; preserve its core structure."
    return analysis.get("diagnosis", behavior.get("reason", ""))[:300]


def _dry_analysis_v2() -> dict:
    return {
        "diagnosis": "Components appear functional; test structural candidates.",
        "failure_mode": "moderate",
        "root_cause_type": "unclear",
        "recommended_mutation": "component_edit",
        "forbidden_mutation_types": [],
        "escalation_level": "coefficient",
        "component_verdicts": [],
        "pipeline_action": "continue",
    }


def _dry_mutations_v2(prev_code: str) -> list[dict]:
    return [
        {"idx": 0, "code": prev_code, "parse_ok": True, "style": "direct_fix", "mutation_type": "direct_fix"},
        {"idx": 1, "code": prev_code.replace("-abs(theta)", "-abs(theta) * 2.0"), "parse_ok": True, "style": "component_edit", "mutation_type": "component_edit"},
        {"idx": 2, "code": prev_code, "parse_ok": True, "style": "progress_gated", "mutation_type": "progress_gated"},
    ]
