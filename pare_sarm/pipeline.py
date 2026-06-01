"""PARE-SARM Pipeline: Progress-Aligned Reward Evolution with Structure-Aware Reward Mutation.

Main orchestrator integrating: memory system, tool registry, 4 agents,
progress diagnosis, training, and diversity checking.

Flow:
  Round 0: EnvPerception → Generate K=3 → Proxy train → Health score
           → Diversity check → Select winner → Full train → Memory save

  Round N: Load previous → Health score → Analyzer (diagnosis)
           → Mutator (3 candidates) → Proxy train → Health + Diversity
           → Select winner → Full train → Memory save
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

from pare_sarm.utils import ensure_dir, save_json, load_json, read_all_jsonl
from pare_sarm.llm import call_llm

from pare_sarm.memory import MemorySystem
from pare_sarm.tools import ToolRegistry

from pare_sarm.agents.env_perception import run_env_perception
from pare_sarm.agents.generator import generate_k_candidates
from pare_sarm.agents.analyzer import run_analyzer
from pare_sarm.agents.mutator import generate_mutation_candidates

from pare_sarm.diagnosis.component_stats import collect_component_stats
from pare_sarm.diagnosis.health_score import (
    compute_health_scores, compute_progress_correlations,
)
from pare_sarm.diagnosis.progress_proxy import execute_progress_proxy

from pare_sarm.reward.validator import validate_reward_quality, print_quality_report
from pare_sarm.reward.diversity import check_behavioral_diversity

from pare_sarm.training.proxy_trainer import run_proxy_training
from pare_sarm.training.full_trainer import run_full_training

_framework_dir = Path(__file__).resolve().parent


class Pipeline:
    """PARE-SARM reward evolution pipeline."""

    def __init__(
        self,
        env_dir: Path,
        exploration_path: Path,
        config: dict,
        api_key: str,
        model: str = "deepseek-reasoner",
        temperature: float = 0.6,
        dry_run: bool = False,
        resume_from: Path | None = None,
    ):
        self.env_dir = Path(env_dir).resolve()
        self.exploration_path = Path(exploration_path).resolve() if exploration_path.exists() else None
        self.config = config
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run

        # Experiment directory
        if resume_from:
            self.exp_dir = Path(resume_from).resolve()
        else:
            self.exp_dir = self._make_exp_dir()

        # Memory system
        self.memory = MemorySystem(self.exp_dir)

        # Tool registry for agents
        self.tools = ToolRegistry(exp_dir=self.exp_dir, memory_system=self.memory)

        # Cached state
        self.env_perception_result: dict | None = None
        self.round_results: list[dict] = []
        self.best_health = 0.0
        self.progress_fn_code: str | None = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════════

    def run(self, n_rounds: int = 3) -> dict:
        """Run the full PARE-SARM pipeline for n_rounds TOTAL (including Round 0).

        n_rounds=3 → Round 0, Round 1, Round 2.
        If resume_from was set in __init__, skips Round 0 and starts from Round 1.
        """
        start_round = self._detect_start_round()

        print(f"\n{'='*60}")
        print(f"  PARE-SARM Pipeline")
        print(f"  Env: {self.env_dir.name}  |  Rounds: {n_rounds} (0..{n_rounds-1})")
        if start_round > 0:
            print(f"  RESUMING from round {start_round}")
        print(f"  Model: {self.model}  |  K={self.config.get('k_candidates', 3)}")
        print(f"  Output: {self.exp_dir}")
        print(f"{'='*60}")

        self._save_config()

        # Round 0
        if start_round == 0:
            r0 = self._run_round0()
            self.round_results.append(r0)
            self._save_round_state(0, r0)
            if not r0["success"]:
                print("Round 0 failed. Aborting.")
                return self._build_summary()
        else:
            self._load_round0_state()
            self.round_results.append({"success": True, "round": 0, "resumed": True})

        # Rounds 1..N-1 (n_rounds total means n_rounds-1 iteration rounds)
        for r in range(max(1, start_round), n_rounds):
            print(f"\n{'#'*60}\n  ROUND {r}\n{'#'*60}")
            result = self._run_iteration_round(r)
            self.round_results.append(result)
            self._save_round_state(r, result)
            if result.get("stop"):
                print(f"Pipeline stopping at round {r}: {result.get('reason', '')}")
                break

        # Final eval + save status
        self._run_final_eval()
        self._save_status()
        self.memory.save()
        return self._build_summary()

    # ═══════════════════════════════════════════════════════════════════════════
    # Round 0
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_round0(self) -> dict:
        """EnvPerception → Generate K→ Proxy → Health → Diversity → Select → Full train."""
        rdir = ensure_dir(self.exp_dir / "round0")

        # 1. EnvPerception
        print("\n>>> Phase 1: Environment Perception")
        if self.dry_run:
            self.env_perception_result = _dry_env_perception()
        else:
            self.env_perception_result = run_env_perception(
                self.env_dir, self.exploration_path, self.api_key,
                self.model, temperature=0.3,
                output_dir=ensure_dir(rdir / "env_perception"),
            )

        ep = self.env_perception_result
        self.progress_fn_code = ep["progress_fn_code"]

        # Save to memory + disk
        self.memory.core.add_fact("task_manifest", ep["task_manifest"])
        self.memory.core.add_fact("reward_signature", ep["reward_signature"])
        self.memory.core.add_fact("progress_fn_code", ep["progress_fn_code"] or "")
        save_json(rdir / "env_perception.json", ep)
        (rdir / "task_manifest.md").write_text(ep["task_manifest"], encoding="utf-8")
        if ep["progress_fn_code"]:
            (rdir / "progress_fn.py").write_text(ep["progress_fn_code"] + "\n", encoding="utf-8")

        print(f"  Signature: {ep['reward_signature']}, Obs dim: {ep['obs_dim']}")

        # 2. Generate K candidates
        print(f"\n>>> Phase 2: Generate K={self.config.get('k_candidates', 3)} Candidates")
        exploration_text = self._load_exploration_text()
        candidates = self._generate_initial_candidates(exploration_text)
        if not candidates:
            return {"success": False, "error": "No valid candidates generated"}

        # 3. Proxy train
        print(f"\n>>> Phase 3: Proxy Training ({self.config.get('proxy_timesteps', 10000)} steps each)")
        candidates = self._proxy_train_all(candidates, rdir, 0)

        # 4. Health + Diversity + Select
        print(f"\n>>> Phase 4: Health Scoring + Diversity + Selection")
        winner = self._score_and_select(candidates)
        if winner is None:
            return {"success": False, "error": "All proxy trainings failed"}

        # 5. Full training
        print(f"\n>>> Phase 5: Full Training ({self.config.get('full_timesteps', 100000)} steps)")
        self._save_reward(winner["code"], rdir)
        ft_result = self._run_full_training_for_round(rdir, winner, 0)

        # 6. Store to episodic memory + behavior memory
        wh = winner["health"].get("overall_health", 0)
        self.memory.episodic.store(0, {
            "summary": f"Round 0: winner health={wh:.0f}",
            "reward_fn_source": winner["code"],
            "health_score": wh,
            "diagnosis": "Initial generation — no diagnosis yet",
        })
        self._record_behavior(0, winner, "Initial generation")
        # Sliding window: best recent health, not all-time
        if wh > self.best_health:
            self.best_health = wh
        # Also decay best if it's from many rounds ago
        if len(self.round_results) >= 5:
            recent_max = max(
                (r.get("winner_health", {}).get("overall_health", 0)
                 for r in self.round_results[-5:] if r.get("winner_health")),
                default=0
            )
            if recent_max < self.best_health - 10:
                self.best_health = recent_max  # old best is stale

        return {"success": True, "round": 0, "winner_health": winner["health"]}

    # ═══════════════════════════════════════════════════════════════════════════
    # Iteration Round (1..N)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_iteration_round(self, round_num: int) -> dict:
        """Analyze → Mutate → Proxy → Health → Diversity → Select → Full train."""
        rdir = ensure_dir(self.exp_dir / f"round{round_num}")
        prev_dir = self.exp_dir / f"round{round_num - 1}"

        # 1. Health score on previous round
        prev_health = self._compute_health_for_round(prev_dir)
        prev_code = self._read_reward_code(prev_dir)

        # 2. Analyzer
        print(f"\n>>> Phase 1: Analyzer (diagnosing round {round_num - 1})")
        history = self.memory.episodic.get_history_text()

        if self.dry_run:
            analysis = _dry_analysis()
        else:
            behavior = self._build_behavior_summary(prev_dir)
            analysis = run_analyzer(
                prev_health, prev_health.get("component_stats", []),
                prev_code, self.env_perception_result["task_manifest"],
                round_num, history,
                self.api_key, self.model, temperature=0.4,
                output_dir=ensure_dir(rdir / "analyzer"),
                behavior_summary=behavior,
            )

        print(f"  Diagnosis: {analysis.get('diagnosis', 'N/A')[:150]}")
        print(f"  Escalation: {analysis.get('escalation_level', '?')}")
        action = analysis.get("pipeline_action", "continue")
        print(f"  Action: {action}")

        if action == "stop":
            self._copy_prev_reward(prev_dir, rdir)
            return {"success": True, "round": round_num, "stop": True, "reason": "converged"}

        if action == "regenerate":
            # Analyzer says the current reward is fundamentally broken.
            # Don't mutate — generate entirely new rewards from scratch.
            print(f"\n>>> Phase 2: REGENERATE (starting fresh, informed by behavior history)")
            exploration_text = self._load_exploration_text()
            memory_ctx = self._get_memory_context_for_mutation()
            candidates = self._generate_initial_candidates(
                exploration_text, memory_context=memory_ctx,
                output_subdir=f"round{round_num}",
            )
            if not candidates:
                print("  Regeneration failed. Keeping previous reward.")
                self._copy_prev_reward(prev_dir, rdir)
                return {"success": True, "round": round_num, "stop": True, "reason": "regeneration failed"}
        else:
            # 3. Mutator → 3 candidates
            print(f"\n>>> Phase 2: Mutation (3 structural candidates)")
            memory_ctx = self._get_memory_context_for_mutation()

            if self.dry_run:
                candidates = _dry_mutations(prev_code)
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
                        print_quality_report(warnings, f"Agent run {c.get('idx', '?')} (temp={c.get('temperature', '?')})")
                        c["warnings"] = warnings
                        candidates.append(c)

        if not candidates:
            print("  All mutations failed. Keeping previous round's reward.")
            self._copy_prev_reward(prev_dir, rdir)
            return {"success": True, "round": round_num, "stop": True, "reason": "mutation failed"}

        # 4. Proxy train
        print(f"\n>>> Phase 3: Proxy Training")
        candidates = self._proxy_train_all(candidates, rdir, round_num)

        # 5. Health + Diversity + Select
        print(f"\n>>> Phase 4: Health + Diversity + Selection")
        winner = self._score_and_select(candidates)
        if winner is None:
            self._copy_prev_reward(prev_dir, rdir)
            return {"success": True, "round": round_num, "stop": True, "reason": "all proxy failed"}

        # Full training decision: use a SLIDING WINDOW of recent health
        w_health = winner["health"].get("overall_health", 0)
        # Compare against average of last 2 rounds, not all-time best
        recent_healths = [
            r.get("winner_health", {}).get("overall_health", 0)
            for r in self.round_results[-3:] if r.get("winner_health")
        ]
        recent_avg = sum(recent_healths) / len(recent_healths) if recent_healths else self.best_health

        # Also check: did the previous round have full training? If not, this round
        # represents a new attempt — always give it full training to see real behavior.
        prev_had_full = (prev_dir / "full_training" / "model.zip").exists()

        skip = (
            w_health < recent_avg - 15  # significantly worse than recent average
            and prev_had_full            # and we already have behavioral data
            and w_health < 35            # and very poor absolute health
        )

        if skip:
            print(f"  Winner health ({w_health:.0f}) << recent avg ({recent_avg:.0f}) — skipping full training")
            self._save_reward(winner["code"], rdir)
        else:
            # Always run full training — need real behavior data to learn
            print(f"\n>>> Phase 5: Full Training")
            self._save_reward(winner["code"], rdir)
            self._run_full_training_for_round(rdir, winner, round_num)

        # 7. Store to memory
        self.memory.episodic.store(round_num, {
            "summary": f"Round {round_num}: winner health={w_health:.0f}, {analysis.get('diagnosis', '')[:200]}",
            "reward_fn_source": winner["code"],
            "health_score": w_health,
            "diagnosis": analysis.get("diagnosis", ""),
        })
        self._record_behavior(round_num, winner,
                              f"Agent repair: {analysis.get('diagnosis', '')[:100]}")

        # 8. Archival: store learned patterns
        if analysis.get("escalation_level") in ("structural", "rewrite"):
            self.memory.archival.add(
                f"Round {round_num}: {analysis.get('diagnosis', '')[:300]}",
                round_num, importance=2.0
            )

        if w_health > self.best_health:
            self.best_health = w_health
        # Decay stale best
        if len(self.round_results) >= 5:
            recent_max = max(
                (r.get("winner_health", {}).get("overall_health", 0)
                 for r in self.round_results[-5:] if r.get("winner_health")),
                default=0
            )
            if recent_max < self.best_health - 10:
                self.best_health = recent_max

        return {"success": True, "round": round_num, "winner_health": winner["health"]}

    # ═══════════════════════════════════════════════════════════════════════════
    # Resume support
    # ═══════════════════════════════════════════════════════════════════════════

    def _detect_start_round(self) -> int:
        """Find the first incomplete round. Returns 0 if starting fresh.

        A round is "complete" if it has round_summary.json.
        If a round directory exists but has no summary, redo that round.
        """
        for rdir in sorted(self.exp_dir.glob("round*")):
            try:
                r = int(rdir.name.replace("round", ""))
            except ValueError:
                continue
            if not (rdir / "round_summary.json").exists():
                # This round is incomplete — restart from here
                # Clean up partial data so the round restarts fresh
                import shutil
                for sub in list(rdir.glob("candidate_*")) + list(rdir.glob("mutation_*")):
                    if sub.is_dir():
                        shutil.rmtree(sub)
                    else:
                        sub.unlink()
                for f in list(rdir.glob("analyzer/*")):
                    f.unlink()
                print(f"  Cleaning incomplete round {r}, restarting from here")
                return r
        # All existing rounds are complete, start next one
        max_complete = -1
        for rdir in sorted(self.exp_dir.glob("round*")):
            if (rdir / "round_summary.json").exists():
                try:
                    r = int(rdir.name.replace("round", ""))
                    max_complete = max(max_complete, r)
                except ValueError:
                    continue
        return max_complete + 1 if max_complete >= 0 else 0

    def _load_round0_state(self):
        """Restore env_perception_result and progress_fn from disk."""
        ep_file = self.exp_dir / "round0" / "env_perception.json"
        if ep_file.exists():
            self.env_perception_result = load_json(ep_file)
        else:
            # Reconstruct minimal state from files
            self.env_perception_result = {
                "task_manifest": (self.exp_dir / "round0" / "task_manifest.md").read_text("utf-8"),
                "progress_fn_code": (self.exp_dir / "round0" / "progress_fn.py").read_text("utf-8"),
                "reward_signature": self.memory.core.get_fact("reward_signature") or "state, action, terminated",
                "obs_dim": 8,
                "max_episode_steps": self.config.get("max_episode_steps", 1000),
            }

        self.progress_fn_code = self.env_perception_result.get("progress_fn_code")
        # Ensure core memory is populated (may have been lost if experiment was killed)
        ep = self.env_perception_result
        self.memory.core.add_fact("task_manifest", ep.get("task_manifest", ""))
        self.memory.core.add_fact("reward_signature", ep.get("reward_signature", ""))
        self.memory.core.add_fact("progress_fn_code", ep.get("progress_fn_code", "") or "")
        self.memory.save()
        print(f"  Loaded Round 0 state: signature={self.env_perception_result.get('reward_signature', '?')}")

        # Restore best health from previous rounds
        for rdir in sorted(self.exp_dir.glob("round*")):
            summary_file = rdir / "round_summary.json"
            if summary_file.exists():
                s = load_json(summary_file)
                wh = s.get("winner_health", {}).get("overall_health", 0)
                if wh > self.best_health:
                    self.best_health = wh

        print(f"  Best health so far: {self.best_health:.0f}/100")

    # ═══════════════════════════════════════════════════════════════════════════
    # Helpers — Generation, Training, Scoring
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_initial_candidates(self, exploration_text: str,
                                       memory_context: str = "",
                                       output_subdir: str = "round0") -> list[dict]:
        """Generate K initial reward candidates.

        When memory_context is provided (regenerate mode), the Generator
        learns from past failures to avoid repeating the same mistakes.
        """
        k = self.config.get("k_candidates", 3)
        if self.dry_run:
            return [_dry_candidate(i) for i in range(k)]

        raw = generate_k_candidates(
            self.env_perception_result["task_manifest"],
            self.env_perception_result["progress_fn_code"] or "",
            self.env_perception_result["reward_signature"],
            exploration_text,
            self.api_key, k=k, model=self.model,
            base_temperature=self.temperature,
            memory_context=memory_context,
            output_dir=self.exp_dir / output_subdir,
        )
        valid = []
        for c in raw:
            if c.get("parse_ok") and c.get("code"):
                warnings = validate_reward_quality(c["code"], self.env_perception_result["task_manifest"])
                print_quality_report(warnings, f"Candidate {c['idx']}")
                c["warnings"] = warnings
                c["style"] = ["conservative", "balanced", "bold"][c["idx"]] if c["idx"] < 3 else "initial"
                valid.append(c)
        return valid

    def _proxy_train_all(self, candidates: list[dict], rdir: Path, round_num: int) -> list[dict]:
        """Run proxy training on all candidates."""
        steps = self.config.get("proxy_timesteps", 10000)
        trained = []
        for i, c in enumerate(candidates):
            style = c.get("style", c.get("mutation_type", "?"))
            print(f"  Candidate {i+1}/{len(candidates)} ({style})...")
            cand_dir = ensure_dir(rdir / f"candidate_{i}")
            self._save_reward(c["code"], cand_dir)

            if self.dry_run:
                c["proxy_result"] = {"success": True}
                c["health"] = {"overall_health": 50.0, "verdict": "dry-run", "components": [],
                               "summary": "dry-run", "component_stats": []}
                trained.append(c)
                continue

            result = run_proxy_training(
                self.env_dir, c["code"], cand_dir, self.config,
                env_id_suffix=f"r{round_num}-c{i}",
                total_timesteps=steps,
                seed=self.config.get("seed", 42),
                progress_fn_code=self.progress_fn_code,
            )
            c["proxy_result"] = result
            c["health"] = result.get("health", {})
            trained.append(c)
            h = c.get("health", {})
            print(f"    Health: {h.get('overall_health', 0):.0f}/100 ({h.get('verdict', '?')})")
        return trained

    def _score_and_select(self, candidates: list[dict]) -> dict | None:
        """Score candidates per GPT §7 formula, check diversity, select winner.

        Score = 0.35 * norm_return + 0.25 * health + 0.20 * progress_align
              + 0.10 * episode_quality + 0.10 * diversity_bonus
        """
        valid = [c for c in candidates if c.get("health", {}).get("overall_health", 0) >= 0]
        if not valid:
            return None

        # Diversity check
        valid = check_behavioral_diversity(valid)

        # Compute weighted scores (§7)
        from pare_sarm.schemas import compute_candidate_score
        for c in valid:
            c["_score"] = compute_candidate_score(
                c.get("health", {}),
                c.get("proxy_result", {}).get("eval_history", []),
                max_episode_steps=self.config.get("max_episode_steps", 1000),
            )

        # Sort by weighted score
        valid.sort(key=lambda c: c["_score"], reverse=True)
        winner = valid[0]
        for i, c in enumerate(valid):
            h = c["health"]
            marker = " *** WINNER" if c is winner else ""
            style = c.get("style", c.get("mutation_type", "?"))
            print(f"  {i+1}. {style}: score={c['_score']:.3f} health={h.get('overall_health', 0):.0f} ({h.get('verdict', '?')}){marker}")
        return winner

    def _compute_health_for_round(self, round_dir: Path) -> dict:
        """Compute health scores from trajectory logs."""
        if self.dry_run:
            return {"overall_health": 50.0, "components": [], "summary": "dry-run", "component_stats": []}

        traj_dir = round_dir / "full_training" / "trajectory_logs"
        if not traj_dir.exists():
            traj_dir = round_dir / "trajectory_logs"
            if not traj_dir.exists():
                return {"overall_health": 0.0, "components": [], "summary": "no data", "component_stats": []}

        stats = collect_component_stats(traj_dir)
        records = read_all_jsonl(traj_dir)
        max_eps = self.config.get("max_episode_steps", 500)

        progress_vals = execute_progress_proxy(self.progress_fn_code, records, max_eps)
        failure_flags = [1.0 if r.get("length", max_eps) < max_eps * 0.3 else 0.0 for r in records]

        stats = compute_progress_correlations(stats, records, progress_vals, max_eps)
        health = compute_health_scores(stats, progress_vals, failure_flags, max_eps)
        health["component_stats"] = stats
        return health

    def _run_full_training_for_round(self, rdir: Path, winner: dict, round_num: int) -> dict:
        """Run full training on winner."""
        ft_dir = ensure_dir(rdir / "full_training")
        full_steps = self.config.get("full_timesteps", 100000)
        if self.dry_run:
            return {"success": True}

        warmstart = None
        proxy_model = rdir / f"candidate_{winner.get('idx', 0)}" / "model.zip"
        if proxy_model.exists():
            warmstart = proxy_model

        return run_full_training(
            self.env_dir, winner["code"], ft_dir, self.config,
            env_id=f"{self.env_dir.name}-r{round_num}",
            warmstart_model_path=warmstart,
            total_timesteps=full_steps,
            seed=self.config.get("seed", 42),
            progress_fn_code=self.progress_fn_code,
        )

    def _record_behavior(self, round_num: int, winner: dict, repair_strategy: str):
        """Record agent behavior for this round into BehaviorMemory."""
        try:
            rdir = self.exp_dir / f"round{round_num}"
            eval_csv = rdir / "full_training" / "evaluations" / "history.csv"
            if not eval_csv.exists():
                return

            import csv
            rows = list(csv.DictReader(eval_csv.open("r")))
            if not rows:
                return

            lengths = []
            for row in rows:
                try:
                    lengths.append((int(row.get("timesteps", 0)), float(row.get("mean_length", 0))))
                except (ValueError, TypeError):
                    continue

            if not lengths:
                return

            final_len = lengths[-1][1]
            max_len = max(l[1] for l in lengths)
            max_eps = self.config.get("max_episode_steps", 1000)

            pattern = self.memory.behavior.classify_behavior(
                final_len, max_eps, [l[1] for l in lengths]
            )

            self.memory.behavior.record(
                round_num,
                final_length=final_len,
                max_length=max_len,
                max_episode_steps=max_eps,
                length_trend=[{"step": l[0], "len": l[1]} for l in lengths],
                health_score=winner.get("health", {}).get("overall_health", 0),
                pattern=pattern,
                diagnosis=winner.get("health", {}).get("summary", ""),
                repair_strategy=repair_strategy,
            )
        except Exception as e:
            print(f"  [WARN] Failed to record behavior: {e}")

    def _build_behavior_summary(self, round_dir: Path) -> str:
        """Summarize what the agent ACTUALLY does (not just reward stats).

        Now includes CROSS-ROUND behavior history from BehaviorMemory,
        enabling oscillation detection and trend analysis.
        """
        # 1. Current round behavior
        current = self._build_current_behavior(round_dir)

        # 2. Cross-round history from BehaviorMemory
        history = self.memory.behavior.format_history_table()

        # 3. Combine
        parts = []
        if current:
            parts.append(f"## Current Round Behavior\n{current}")
        if history and "*(no behavior history)*" not in history:
            parts.append(history)
        return "\n\n".join(parts)

    def _build_current_behavior(self, round_dir: Path) -> str:
        """Summarize what the agent ACTUALLY does (not just reward stats).

        Extracts from eval history and per-step logs:
        - Episode length trend (is it learning? hovering? crashing early?)
        - Termination pattern (crash vs timeout)
        - Observed behavioral clues
        """
        eval_csv = round_dir / "full_training" / "evaluations" / "history.csv"
        if not eval_csv.exists():
            eval_csv = round_dir / "evaluations" / "history.csv"
        if not eval_csv.exists():
            # Check candidate dirs
            for cdir in sorted(round_dir.glob("candidate_*")):
                ecsv = cdir / "evaluations" / "history.csv"
                if ecsv.exists():
                    eval_csv = ecsv
                    break

        if not eval_csv.exists():
            return ""

        import csv
        rows = list(csv.DictReader(eval_csv.open("r")))
        if not rows:
            return ""

        # Extract episode length data
        lengths = []
        for r in rows:
            try:
                lengths.append(float(r.get("mean_length", 0)))
            except (ValueError, TypeError):
                continue

        if not lengths:
            return ""

        first_len = lengths[0]
        last_len = lengths[-1]
        max_len = max(lengths)
        max_eps = self.config.get("max_episode_steps", 1000)
        trend = "improving" if last_len > first_len * 1.1 else (
            "declining" if last_len < first_len * 0.9 else "stable"
        )

        # Determine behavioral pattern
        if last_len > max_eps * 0.8:
            behavior = ("The agent survives near the maximum episode length. "
                        "If the task is NOT being completed (e.g., hovering without landing), "
                        "this indicates REWARD HACKING: the agent found a way to collect "
                        "per-step rewards without completing the task. Check which per-step "
                        "components are large enough to incentivize stalling.")
        elif last_len < max_eps * 0.2:
            behavior = ("The agent terminates very early. This suggests negative rewards dominate — "
                        "the agent learns that dying quickly minimizes punishment. "
                        "Check if penalty components overwhelm any positive signals.")
        else:
            behavior = (f"The agent survives a moderate duration ({last_len:.0f} steps). "
                        "Check whether it is making genuine progress or oscillating without direction.")

        return f"""Episode length trend: {trend} ({first_len:.0f} → {last_len:.0f} steps, max={max_len:.0f})
Max episode steps: {max_eps}
Pattern: {behavior}"""

    def _get_memory_context_for_mutation(self) -> str:
        """Build rich memory context for the Mutator agent.

        Includes:
        - Cross-round history (what was diagnosed and tried each round)
        - Archived patterns relevant to the current diagnosis
        - Previous reward code snippets and their outcomes
        """
        parts = []

        # 1. Behavior history with oscillation detection (MOST IMPORTANT)
        behavior_table = self.memory.behavior.format_history_table()
        if behavior_table and "*(no behavior history)*" not in behavior_table:
            parts.append(behavior_table)

        # 2. Cross-round episodic history
        all_rounds = self.memory.episodic.get_all_rounds()
        if all_rounds:
            parts.append("## Previous Diagnosis Summaries")
            for r in sorted(all_rounds.keys()):
                data = all_rounds[r]
                health = data.get("health_score", "?")
                summary = data.get("summary", "")[:200]
                parts.append(f"  Round {r} (health={health}): {summary}")

        # 3. Relevant archival patterns
        last_round = max(all_rounds.keys()) if all_rounds else None
        if last_round is not None:
            data = all_rounds[last_round]
            diagnosis = data.get("diagnosis", "")
            archival = self.memory.archival.search(diagnosis, max_results=3)
            if archival:
                parts.append("\n## Cross-Experiment Design Principles")
                for p in archival:
                    parts.append(f"  - {p}")

        # 3. Detect repeated patterns
        if len(all_rounds) >= 2:
            diagnoses = [all_rounds[r].get("diagnosis", "") for r in sorted(all_rounds.keys())[-3:]]
            # Simple check: do any key phrases repeat?
            import re
            for d in diagnoses:
                keywords = set(re.findall(r'\b(misalign|dominat|inactive|weak|structural|coefficient)\w*', d.lower()))
            parts.append("\n## Hint")
            parts.append("  If the same root cause has persisted across multiple rounds, consider a FUNDAMENTAL")
            parts.append("  redesign rather than incremental tuning. Split conflated concepts into independent components.")

        return "\n".join(parts) if parts else ""

    # ═══════════════════════════════════════════════════════════════════════════
    # File I/O Helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_exp_dir(self) -> Path:
        from pare_sarm.utils import experiment_name
        steps = self.config.get("full_timesteps", 100000)
        name = experiment_name(self.env_dir.name.lower(), steps)
        return _framework_dir.parent / "outputs" / name

    def _save_config(self):
        from pare_sarm.utils import save_yaml
        save_yaml(self.exp_dir / "config.yaml", self.config)

    def _save_round_state(self, round_num: int, result: dict):
        if self.dry_run:
            return  # Don't persist state in dry-run mode
        rdir = ensure_dir(self.exp_dir / f"round{round_num}")
        save_json(rdir / "round_summary.json", {
            "round": round_num,
            "success": result.get("success", False),
            "winner_health": result.get("winner_health", {}),
            "stopped": result.get("stop", False),
        })

    def _save_status(self):
        (self.exp_dir / "STATUS").write_text(f"COMPLETED (PARE-SARM v0.1)\n")

    def _save_reward(self, code: str, dir_path: Path):
        ensure_dir(dir_path)
        cleaned = re.sub(r'^"""LLM[- ].*?"""', '', code.strip(), flags=re.DOTALL)
        cleaned = re.sub(r'^import\s+(math|numpy).*?\n', '', cleaned, flags=re.MULTILINE)
        (dir_path / "reward_fn_source.py").write_text(
            f'"""LLM-generated reward function.\n"""\n\nimport math\nimport numpy as np\n\n{cleaned}\n',
            encoding="utf-8")

    def _read_reward_code(self, round_dir: Path) -> str:
        for p in [round_dir / "reward_fn_source.py",
                   round_dir / "full_training" / "reward_fn_source.py"]:
            if p.exists():
                return p.read_text("utf-8")
        return ""

    def _copy_prev_reward(self, prev_dir: Path, round_dir: Path):
        src = prev_dir / "reward_fn_source.py"
        if src.exists():
            ensure_dir(round_dir)
            (round_dir / "reward_fn_source.py").write_bytes(src.read_bytes())

    def _load_exploration_text(self) -> str:
        if self.exploration_path and self.exploration_path.exists():
            return self.exploration_path.read_text("utf-8")[:8000]
        return "(no exploration data)"

    def _run_final_eval(self):
        print(f"\n>>> Final Evaluation")
        try:
            from pare_sarm.eval.final_eval import evaluate_all_rounds
            official_env = _infer_official_env(self.env_dir.name)
            results = evaluate_all_rounds(self.exp_dir, official_env, n_episodes=50)
            save_json(self.exp_dir / "final_eval.json", results)
        except Exception as e:
            print(f"  Skipped: {e}")

    def _build_summary(self) -> dict:
        return {
            "success": any(r.get("success") for r in self.round_results),
            "exp_dir": str(self.exp_dir),
            "env": self.env_dir.name,
            "n_rounds": len(self.round_results),
            "rounds": [
                {"round": r.get("round"), "health": r.get("winner_health", {}).get("overall_health")}
                for r in self.round_results
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Dry-run helpers
# ═══════════════════════════════════════════════════════════════════════════

def _dry_env_perception():
    return {
        "task_manifest": "# CartPole Task\nBalance a pole on a cart.",
        "progress_fn_code": "def progress_fn(obs):\n    return -abs(obs[2]) - 0.1 * abs(obs[0])",
        "reward_signature": "compute_reward(state, action, terminated)",
        "obs_dim": 4,
        "max_episode_steps": 500,
    }

def _dry_candidate(idx: int) -> dict:
    styles = ["conservative", "balanced", "bold"]
    return {
        "idx": idx,
        "code": (
            "def compute_reward(state, action, terminated):\n"
            "    x, x_dot, theta, theta_dot = state\n"
            "    balance = -abs(theta)\n"
            "    centering = -abs(x)\n"
            "    components = {'balance': balance, 'centering': centering}\n"
            "    total = balance + centering\n"
            "    components['_outcome'] = -1.0 if terminated else 0.0\n"
            "    return float(total), components\n"
        ),
        "parse_ok": True,
        "temperature": 0.6,
        "style": styles[idx % 3],
    }

def _dry_analysis():
    return {
        "diagnosis": "Components appear functional. Balance term is active but could be strengthened.",
        "escalation_level": "coefficient",
        "component_verdicts": [
            {"component": "balance", "verdict": "strengthen", "reason": "Active but weak alignment"},
            {"component": "centering", "verdict": "keep", "reason": "Healthy"},
        ],
        "pipeline_action": "continue",
    }

def _dry_mutations(prev_code: str) -> list[dict]:
    return [
        {"idx": 0, "code": prev_code, "parse_ok": True, "style": "agent-t0.3", "temperature": 0.3},
        {"idx": 1, "code": prev_code.replace("-abs(theta)", "-abs(theta) * 2.0"), "parse_ok": True, "style": "agent-t0.5", "temperature": 0.5},
        {"idx": 2, "code": prev_code, "parse_ok": True, "style": "agent-t0.7", "temperature": 0.7},
    ]

def _infer_official_env(env_name: str) -> str:
    mapping = {
        "CartPole-v1": "CartPole-v1",
        "LunarLander-v2": "LunarLander-v2",
        "LunarLander-v3": "LunarLander-v3",
        "BipedalWalker-v3": "BipedalWalker-v3",
        "HalfCheetah-v4": "HalfCheetah-v4",
    }
    return mapping.get(env_name, env_name)
