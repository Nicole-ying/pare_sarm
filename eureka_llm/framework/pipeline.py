"""
pipeline.py — Evolutionary multi-agent reward design pipeline.

Key design (aligned with Eureka paper + CARD + Auto MC-Reward):

Round 0:  EnvPerception → K=3 parallel candidates → 200K proxy train each
         → select best via component stats → 1M full train on winner

Round N:  Perception → Analyzer (gets component stats from prev round)
         → Generator: K=3 candidates → TPE pre-filter (skip bad ones)
         → 200K proxy each survivor → component stats ranking → pick best
         → 1M full train on winner → Reflection → Memory

This solves: single-point failure (K>1), training waste (proxy first),
and missing feedback (component stats replace metrics_fn).
"""

from __future__ import annotations

import json, os, re, subprocess, sys, time, csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

_framework_dir = Path(__file__).resolve().parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from communication.message_pool import MessagePool
from communication.schemas import AgentMessage
from memory.memory_system import MemorySystem
from memory.context import build_memory_context, inject_memory_into_prompt
from agents.v2_agents import (
    EnvPerceptionAgent, PerceptionAgent, AnalyzerAgent,
    GeneratorAgent, ReflectionAgent, EvaluatorAgent,
)
from llm_call import call_llm, extract_reward_fn
from template_engine import build_round0_prompt

BEIJING = timezone(timedelta(hours=8))

# ── Constants ────────────────────────────────────────────────────────────────

K_CANDIDATES = 3          # Reward candidates per round
PROXY_TIMESTEPS = 100_000 # Short proxy evaluation per candidate
FULL_TIMESTEPS = 1_000_000


# ═══════════════════════════════════════════════════════════════════════════════
# Component Health Scoring (PARE: Progress-Aligned Reward Evolution)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_component_health(
    proxy_dir: Path = None,
    aggregated_stats: list[dict] = None,
    eval_history: list[dict] = None,
    max_episode_steps: int = 1000,
) -> dict:
    """Score a reward candidate using PARE diagnosis.

    Computes per-component progress alignment and failure conflict
    from trajectory logs, then produces a health score in [0, 100].

    Weights: activation 25% + balance 25% + progress_alignment 40% - failure_conflict 10%
    """
    # Try PARE diagnosis first (uses trajectory logs for progress correlation)
    if proxy_dir is not None:
        from progress_diagnosis import compute_progress_diagnosis
        diag = compute_progress_diagnosis(
            proxy_dir / "trajectory_logs",
            progress_fn_code=None,
            max_episode_steps=max_episode_steps,
        )
        comps = diag.get("components", [])
        if comps:
            return {
                "score": diag["health_score"],
                "activation": sum(1 for c in comps if c.get("active")) / max(len(comps), 1),
                "balance": max(0, 1 - max((c.get("share", 0) for c in comps), default=0)),
                "progress_alignment": sum(max(0, c.get("progress_corr", 0)) for c in comps) / max(len(comps), 1),
                "failure_conflict": sum(max(0, c.get("failure_corr", 0)) for c in comps) / max(len(comps), 1),
                "verdict": "good" if diag["health_score"] >= 60 else ("ok" if diag["health_score"] >= 35 else "poor"),
                "n_components": len(comps),
                "n_active": sum(1 for c in comps if c.get("active")),
                "diagnosis_summary": diag.get("summary", ""),
            }

    # Fallback: use aggregated stats + eval history
    if aggregated_stats:
        abs_means = [abs(s["mean"]) for s in aggregated_stats]
        active = sum(1 for m in abs_means if m > 0.01)
        activation = active / max(len(abs_means), 1)
        total = sum(abs_means) if abs_means else 0
        balance = max(0.0, 1.0 - max(abs_means) / total) if total > 1e-9 else 0.0
        lengths = [float(r.get("mean_length", 0)) for r in (eval_history or []) if r.get("mean_length")]
        len_q = 0.5
        if lengths:
            frac = lengths[-1] / max(max_episode_steps, 1)
            len_q = 0.8 if 0.15 < frac < 0.70 else (0.3 if frac > 0.90 else 0.4)
        score = 100 * (0.3 * activation + 0.3 * balance + 0.4 * len_q)
        return {
            "score": round(score, 1),
            "activation": round(activation, 3),
            "balance": round(balance, 3),
            "length_quality": round(len_q, 3),
            "verdict": "good" if score >= 60 else ("ok" if score >= 35 else "poor"),
            "n_components": len(abs_means),
            "n_active": active,
        }

    return {"score": 0.0, "verdict": "no_data"}


def select_best_candidate(candidates: list[dict], exp_dir: Path = None,
                           round_num: int = None) -> dict:
    """Select the best candidate using PARE diagnosis when possible."""
    for c in candidates:
        proxy = c.get("proxy_results", {})
        # Build proxy_dir path for PARE diagnosis
        proxy_dir = None
        if exp_dir and round_num is not None:
            proxy_dir = exp_dir / f"round{round_num}" / f"candidate_{c['idx']}"
        c["health"] = compute_component_health(
            proxy_dir=proxy_dir,
            aggregated_stats=proxy.get("component_history", []),
            eval_history=proxy.get("eval_history", []),
        )
    candidates.sort(key=lambda c: c["health"]["score"], reverse=True)
    return candidates[0]


# ═══════════════════════════════════════════════════════════════════════════════
# TPE Pre-Filter (CARD-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

def tpe_prefilter(
    candidate_code: str,
    previous_code: str,
    env_dir: Path = None,
    env_id: str = "",
    n_episodes: int = 5,
) -> bool:
    """Pre-filter: check if the candidate is meaningfully different from previous.

    Filters out candidates that:
    - Are syntactically identical to the previous code
    - Have no compute_reward function
    - Are suspiciously short (< 200 chars of actual logic)

    Returns True if candidate passes (worth training), False to filter out.

    Full TPE (trajectory preference evaluation) requires running the old policy
    with the new reward on a few episodes, then comparing reward rankings.
    For efficiency, we start with static checks and can add rollout-based TPE later.
    """
    if not previous_code:
        return True

    # Parse out just the function body (no imports, no docstrings)
    import re as _re

    def _extract_body(code: str) -> str:
        # Remove imports, docstrings, comments
        code = _re.sub(r'^"""[\s\S]*?"""', '', code.strip())
        code = _re.sub(r'^import\s+.*?\n', '', code, flags=_re.MULTILINE)
        code = _re.sub(r'^from\s+.*?\n', '', code, flags=_re.MULTILINE)
        code = _re.sub(r'#.*$', '', code, flags=_re.MULTILINE)
        code = _re.sub(r'\n\s*\n', '\n', code)
        return code.strip()

    cand_body = _extract_body(candidate_code)
    prev_body = _extract_body(previous_code)

    # Identical code → skip
    if cand_body == prev_body:
        return False

    # No compute_reward → skip
    if "def compute_reward" not in candidate_code:
        return False

    # Suspiciously short (< 200 chars of logic) → skip
    if len(cand_body) < 200:
        return False

    # Component structure check: extract component names from the dict
    cand_comps = set(_re.findall(r'"(\w+)"\s*:', candidate_code))
    prev_comps = set(_re.findall(r'"(\w+)"\s*:', previous_code))

    # If the component structure is identical AND the code is nearly identical
    # (just coefficient changes), it's borderline — keep it but flag
    if cand_comps == prev_comps and _levenshtein_ratio(cand_body, prev_body) > 0.85:
        # Very similar — still try (coefficient tuning can matter)
        return True

    return True


def _levenshtein_ratio(a: str, b: str) -> float:
    """Quick similarity ratio. 1.0 = identical, 0.0 = completely different."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    longer = max(len(a), len(b))
    # Simple character-level Jaccard for speed (not true Levenshtein, but fast)
    set_a, set_b = set(a), set(b)
    return len(set_a & set_b) / len(set_a | set_b)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class RewardDesignPipeline:
    """Evolutionary multi-agent reward design pipeline.

    Usage:
        p = RewardDesignPipeline(env_dir=..., exploration_path=..., config=..., api_key=...)
        p.run(n_rounds=5)
    """

    def __init__(self, env_dir, exploration_path, config, api_key,
                 model="deepseek-reasoner", temperature=0.6, dry_run=False,
                 k_candidates=K_CANDIDATES):
        self.env_dir = Path(env_dir).resolve()
        self.exploration_path = Path(exploration_path).resolve()
        self.config = config
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.k = k_candidates

        self.exp_dir: Path = None
        self._setup_experiment_dir()

        self.pool = MessagePool()
        self.memory = MemorySystem(self.exp_dir)

        template_dir = _framework_dir.parent / "templates"
        self.env_agent = EnvPerceptionAgent(api_key, model, pool=self.pool, memory=self.memory)
        self.perception_agent = PerceptionAgent(api_key, model, pool=self.pool, memory=self.memory, template_dir=template_dir)
        self.analyzer_agent = AnalyzerAgent(api_key, model, pool=self.pool, memory=self.memory)
        self.generator_agent = GeneratorAgent(api_key, model, pool=self.pool, memory=self.memory)
        self.evaluator_agent = EvaluatorAgent(api_key, model, pool=self.pool, memory=self.memory)
        self.reflector_agent = ReflectionAgent(api_key, model, pool=self.pool, memory=self.memory)

        self._register_subscriptions()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, n_rounds: int = 5) -> dict:
        print(f"\n{'='*60}\n  Reward Design Pipeline v2\n  Env: {self.env_dir.name}\n  Rounds: 0..{n_rounds}  |  K={self.k} candidates  |  Proxy={PROXY_TIMESTEPS//1000}K  |  Full={FULL_TIMESTEPS//1000}K\n{'='*60}")

        self._save_config()

        # ── Round 0 ──
        r0 = self._run_round0()
        if not r0["success"]:
            print("Round 0 failed. Aborting."); return r0

        # ── Rounds 1..N ──
        for r in range(1, n_rounds + 1):
            print(f"\n{'#'*60}\n  ROUND {r}\n{'#'*60}")
            result = self._run_iteration_round(r)
            if not result.get("success"):
                print(f"Round {r} failed. Stopping."); break

        self._save_status(n_rounds)
        return {"success": True, "exp_dir": self.exp_dir, "messages": self.pool.message_count}

    # ── Round 0: EnvPerception → Generate K candidates → Proxy → Select → Full ──

    def _run_round0(self) -> dict:
        print("\n>>> Env Perception")
        env_desc = self._load_env_description(self.env_dir.name)
        manifest = self.env_agent.run_discovery(self.env_dir, env_desc, self.exploration_path, memory_system=self.memory)
        self.memory.core.add_key_fact(f"signature: {self._extract_signature(manifest)}")
        self.memory.save()

        print(f"\n>>> Generating K={self.k} candidates")
        candidates = self._generate_candidates(manifest, round_num=0)

        print(f"\n>>> Proxy training ({PROXY_TIMESTEPS//1000}K each)")
        for i, c in enumerate(candidates):
            print(f"  Candidate {i+1}/{self.k}...")
            c["proxy_results"] = self._proxy_train(c["code"], round_num=0, candidate_idx=i)

        print(f"\n>>> Selecting best via component health")
        winner = select_best_candidate(candidates, self.exp_dir, 0)
        for i, c in enumerate(candidates):
            h = c.get("health", {})
            marker = " ★ WINNER" if c is winner else ""
            print(f"  Candidate {i+1}: score={h.get('score', 0):.0f} active={h.get('n_active', 0)}/{h.get('n_components', 0)} {h.get('verdict', '?')}{marker}")

        if winner["health"]["verdict"] == "poor":
            print("  All candidates poor — regenerating...")
            candidates2 = self._generate_candidates(manifest, round_num=0, extra_context="Previous candidates all scored poorly. Generate substantially different approaches.")
            for i, c in enumerate(candidates2):
                c["idx"] = i + self.k
                c["proxy_results"] = self._proxy_train(c["code"], round_num=0, candidate_idx=c["idx"])
            winner = select_best_candidate(candidates + candidates2, self.exp_dir, 0)

        print(f"\n>>> Full training ({FULL_TIMESTEPS//1000}K) on winner")
        self._save_reward(winner["code"], self.exp_dir / "round0", 0)
        self._full_train(round_num=0, warmstart_candidate_idx=winner.get("idx"))
        self._run_perception_on_round(0)

        self.memory.episodic.store_round(0, {
            "summary": f"K={self.k} candidates, winner_score={winner['health']['score']}",
            "reward_fn_source": winner["code"],
        })

        return {"success": True, "winner_health": winner["health"]}

    # ── Iteration Round ────────────────────────────────────────────────────────

    def _run_iteration_round(self, round_num: int) -> dict:
        prev = round_num - 1
        prev_dir = self.exp_dir / f"round{prev}"

        # 1. Perception on previous round
        print(f"\n>>> Perception (round {prev})")
        self._run_perception_on_round(prev)

        # 2. Analyze — single analyst or dual-analyst debate
        use_debate = self.config.get("use_debate", False)
        if use_debate:
            print(f"\n>>> Debate: Exploration + Exploitation analysts (round {prev} → {round_num})")
            from debate import run_explore_exploit_debate
            analysis = run_explore_exploit_debate(
                self.exp_dir / f"round{prev}", round_num, self.memory,
                self.api_key, self.model,
            )
        else:
            print(f"\n>>> Analyzer (round {prev} → {round_num})")
            analysis = self._run_analyzer_with_history(prev, round_num)

        proposal = analysis.get("proposal", {})
        changed = proposal.get("changed_count", 0) > 0
        print(f"  Diagnosis: {proposal.get('diagnosis', 'N/A')[:120]}")
        print(f"  Escalation: {proposal.get('escalation_level', 'coefficient')}")
        if analysis.get("_debate_meta"):
            print(f"  Debate winner: {analysis['_debate_meta']['winner']}")

        # Fallback: if analysis failed, retry once with different temperature
        if not changed and proposal.get("analysis_status") != "ok":
            print("  Analyzer produced no valid proposal — retrying with backup prompt...")
            analysis = self._run_analyzer_with_history(prev, round_num, is_retry=True)
            proposal = analysis.get("proposal", {})
            changed = proposal.get("changed_count", 0) > 0

        if not changed:
            print("  No changes proposed after retry. Skipping full training for this round.")
            self._copy_prev_reward(prev_dir, self.exp_dir / f"round{round_num}")
            return {"success": True, "skipped": True, "reason": "no changes"}

        # 3. Generate K diverse candidates
        print(f"\n>>> Generating K={self.k} candidates")
        candidates = self._generate_candidates_from_proposal(proposal, round_num)

        # 4. TPE Pre-filter
        prev_code = (self.exp_dir / f"round{prev}" / "reward_fn_source.py").read_text("utf-8") if (self.exp_dir / f"round{prev}" / "reward_fn_source.py").exists() else ""
        survivors = []
        for c in candidates:
            if tpe_prefilter(c["code"], prev_code):
                survivors.append(c)
        if not survivors:
            survivors = candidates

        # 5. Proxy train survivors
        print(f"\n>>> Proxy training {len(survivors)} survivors ({PROXY_TIMESTEPS//1000}K each)")
        for i, c in enumerate(survivors):
            c["proxy_results"] = self._proxy_train(c["code"], round_num=round_num, candidate_idx=i)

        # 6. Select best + early stop check
        print(f"\n>>> Selecting best via component health")
        winner = select_best_candidate(survivors, self.exp_dir, round_num)
        for i, c in enumerate(survivors):
            h = c.get("health", {})
            marker = " ★ WINNER" if c is winner else ""
            print(f"  Candidate {i+1}: score={h.get('score', 0):.0f} {h.get('verdict', '?')}{marker}")

        # Early stop: if winner score is worse than previous round, skip full training
        prev_health = self._get_previous_health(prev)
        if prev_health and winner["health"]["score"] < prev_health["score"] - 10:
            print(f"  Winner score ({winner['health']['score']:.0f}) < previous ({prev_health['score']:.0f}) — skipping full training")
            # Save winner code so next round can read it
            self._save_reward(winner["code"], self.exp_dir / f"round{round_num}", round_num)
            return {"success": True, "skipped": True, "reason": "regression", "winner_health": winner["health"]}

        # 7. Full train winner (continues from proxy checkpoint)
        print(f"\n>>> Full training ({FULL_TIMESTEPS//1000}K) on winner")
        self._save_reward(winner["code"], self.exp_dir / f"round{round_num}", round_num)
        self._full_train(round_num=round_num, warmstart_candidate_idx=winner.get("idx"))

        # 8. Perception on this round
        self._run_perception_on_round(round_num)

        # 9. Reflection
        self._run_reflection(round_num)

        self.memory.episodic.store_round(round_num, {
            "summary": f"score={winner['health']['score']}, diagnosis={proposal.get('diagnosis', '')[:100]}",
            "reward_fn_source": winner["code"],
            "proposal": proposal,
            "perception_report": self._read_file(self.exp_dir / f"round{round_num}" / "perception_report.md"),
            "reflection": self._read_file(self.exp_dir / f"round{round_num}" / "reflection.md"),
        })
        self.memory.save()

        return {"success": True, "winner_health": winner["health"]}

    # ── Candidate Generation ───────────────────────────────────────────────────

    def _generate_candidates(self, manifest: str, round_num: int, extra_context: str = "") -> list[dict]:
        """Generate K diverse reward candidates for round 0."""
        candidates = []
        template_path = _framework_dir.parent / "templates" / "round0_prompt.txt"

        for i in range(self.k):
            # Vary temperature for diversity
            temp = self.temperature + (i - 1) * 0.2  # e.g., 0.4, 0.6, 0.8
            temp = max(0.2, min(1.0, temp))

            prompt = build_round0_prompt(self.env_dir, template_path, self.exploration_path, task_manifest=manifest)
            prompt = inject_memory_into_prompt(prompt, self.memory, query="reward function design", max_tokens=500)

            if i > 0:
                prompt += f"\n\n## Diversity Instruction\nGenerate a DIFFERENT approach from typical designs. Vary the component structure, scaling, or shaping strategy."

            if extra_context:
                prompt += f"\n\n## Context from Previous Attempt\n{extra_context}"

            if self.dry_run:
                candidates.append({"idx": i, "code": f"# dry-run candidate {i}\ndef compute_reward(): return 0.0, {{}}", "temperature": temp})
                continue

            code, _ = self._call_llm_for_code(prompt, temp, f"Candidate{i}")
            if code:
                candidates.append({"idx": i, "code": code, "temperature": temp})
                print(f"  Candidate {i+1} generated ({len(code)} chars, temp={temp})")

        if not candidates:
            raise RuntimeError("Failed to generate any valid reward candidates")
        return candidates

    def _generate_candidates_from_proposal(self, proposal: dict, round_num: int) -> list[dict]:
        """Generate K diverse candidates from the SAME diagnosis but with
        DIFFERENT RL perspectives, producing genuinely diverse implementations.

        Candidate A (exploration): Focus on escaping local optima, adding shaping
            gradients, and encouraging discovery of new behaviors.
        Candidate B (exploitation): Focus on component balance, reward hacking
            prevention, and efficient task completion.
        Candidate C (balanced): Apply the diagnosis as stated, balancing both.

        Different temperatures and system prompts ensure prompt-level diversity,
        not just cosmetic variation.
        """
        candidates = []
        prev_round_dir = self.exp_dir / f"round{round_num - 1}"

        perspectives = [
            ("exploration", 0.5,
             "You are an exploration-focused reward designer. Your priority is helping "
             "the agent ESCAPE local optima and DISCOVER new behaviors. Interpret the "
             "diagnosis through this lens: which changes will expand the agent's "
             "behavioral repertoire? Add shaping terms, increase reward density in "
             "dead zones, create escape routes from local optima. Prefer ADDING new "
             "components over adjusting existing ones."),
            ("exploitation", 0.3,
             "You are an exploitation-focused reward designer. Your priority is "
             "REFINING the agent's existing skills toward mastery. Interpret the "
             "diagnosis through this lens: which changes will improve component "
             "balance, close reward-hacking loopholes, and align the reward more "
             "tightly with task success? Prefer RESCALING and REMOVING over adding."),
            ("balanced", 0.4,
             "Apply the diagnosis as stated, balancing exploration and exploitation. "
             "Make the changes that best address the root cause identified in the "
             "diagnosis, without biasing toward either expansion or refinement."),
        ]

        for i, (persp_name, temp, persp_prompt) in enumerate(perspectives[:self.k]):
            # Build a perspective-specific proposal by prepending the lens to diagnosis
            styled = dict(proposal)
            styled["diagnosis"] = (
                persp_prompt + "\n\n" +
                "=== Diagnosis to implement ===\n" +
                proposal.get("diagnosis", "")
            )
            code = self.generator_agent.run_with_proposal(
                styled, prev_round_dir / "reward_fn_source.py",
                prev_round_dir, memory_system=self.memory,
            )
            if code:
                candidates.append({"idx": i, "code": code, "style": persp_name})
                print(f"  Candidate {i+1} ({persp_name}, temp={temp}): {len(code)} chars")

        if not candidates:
            prev_code = (prev_round_dir / "reward_fn_source.py").read_text("utf-8")
            candidates.append({"idx": 0, "code": prev_code, "style": "fallback"})
        return candidates

    # ── Proxy Training ─────────────────────────────────────────────────────────

    def _build_training_config(self, timesteps: int, extra_seed: int = 0) -> dict:
        """Build a complete training config with sensible defaults for missing PPO params.

        Prevents KeyError crashes when user config lacks parameters like gae_lambda.
        """
        ppo_defaults = {
            "policy": "MlpPolicy", "learning_rate": 3e-4, "n_steps": 1024,
            "batch_size": 64, "n_epochs": 4, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,
            "vf_coef": 0.5, "max_grad_norm": 0.5,
        }
        user_ppo = self.config.get("ppo", {})
        full_ppo = {**ppo_defaults, **user_ppo}

        cfg = {
            "total_timesteps": timesteps,
            "n_envs": self.config.get("n_envs", 16),
            "seed": self.config.get("seed", 42) + extra_seed,
            "device": self.config.get("device", "cpu"),
            "normalize": False,
            "ppo": full_ppo,
            "evaluation": {"freq": timesteps // 2, "episodes": 5, "deterministic": True},
            "checkpoint": {"freq": timesteps * 10},
        }
        # Only pass gif_steps for full training (not proxy), so candidates don't waste time on GIFs
        if timesteps > PROXY_TIMESTEPS:
            for k in ("gif_steps", "gif_fps", "gif_max_steps"):
                if k in self.config:
                    cfg[k] = self.config[k]
        if "max_episode_steps" in self.config:
            cfg["max_episode_steps"] = self.config["max_episode_steps"]
        return cfg

    @staticmethod
    def _run_subprocess_live(cmd: list) -> subprocess.CompletedProcess:
        """Run a subprocess, streaming stdout/stderr in real time.

        Unlike capture_output=True, this lets SB3 training progress bars
        flow through to experiment.log immediately.
        """
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        stdout_lines = []
        for line in iter(proc.stdout.readline, ""):
            print(line, end="")  # goes through TeeWrite to experiment.log
            stdout_lines.append(line)
        proc.wait()
        return subprocess.CompletedProcess(cmd, proc.returncode,
                                           stdout="".join(stdout_lines), stderr="")

    def _proxy_train(self, code: str, round_num: int, candidate_idx: int) -> dict:
        """Short training (PROXY_TIMESTEPS) to evaluate a reward candidate."""
        if self.dry_run:
            return {"component_history": [], "eval_history": [], "health": {"score": 50}}

        proxy_dir = self.exp_dir / f"round{round_num}" / f"candidate_{candidate_idx}"
        proxy_dir.mkdir(parents=True, exist_ok=True)

        reward_path = proxy_dir / "reward_fn_source.py"
        cleaned = re.sub(r'^"""LLM[- ].*?"""', '', code.strip(), flags=re.DOTALL)
        cleaned = re.sub(r'^import\s+(math|numpy).*?\n', '', cleaned, flags=re.MULTILINE)
        reward_path.write_text(
            f'"""Proxy reward candidate {candidate_idx}."""\n\nimport math\nimport numpy as np\n\n{cleaned}\n',
            encoding="utf-8"
        )

        # Build proxy config with PPO defaults
        round_config = self._build_training_config(PROXY_TIMESTEPS, extra_seed=candidate_idx)
        config_path = proxy_dir / "config.yaml"
        dump_fn = yaml.safe_dump if yaml else lambda d, **kw: json.dumps(d, indent=2)
        config_path.write_text(dump_fn(round_config, sort_keys=False), encoding="utf-8")

        train_script = _framework_dir / "train.py"
        env_id = f"{self.env_dir.name}-round{round_num}-c{candidate_idx}"
        cmd = [
            sys.executable, str(train_script),
            "--env-dir", str(self.env_dir), "--env-id", env_id,
            "--config", str(config_path), "--run-dir", str(proxy_dir),
            "--reward-source", str(reward_path),
        ]
        if self.config.get("max_episode_steps"):
            cmd += ["--max-episode-steps", str(self.config["max_episode_steps"])]

        print(f"    Proxy training ({PROXY_TIMESTEPS} steps)...")
        result = self._run_subprocess_live(cmd)
        success = result.returncode == 0

        # Collect component history from trajectory logs
        component_history = self._collect_component_stats(proxy_dir)
        eval_history = self._read_eval_csv(proxy_dir / "evaluations" / "history.csv")

        return {
            "success": success,
            "component_history": component_history,
            "eval_history": eval_history,
        }

    # ── Full Training ──────────────────────────────────────────────────────────

    def _full_train(self, round_num: int, warmstart_candidate_idx: int = None):
        """Full training. If warmstart_candidate_idx is provided, continues from
        that candidate's proxy checkpoint instead of starting from scratch."""
        if self.dry_run:
            return

        round_dir = self.exp_dir / f"round{round_num}"
        reward_path = round_dir / "reward_fn_source.py"

        # Determine remaining steps and warmstart path
        warmstart_path = None
        if warmstart_candidate_idx is not None:
            proxy_model = (self.exp_dir / f"round{round_num}" /
                          f"candidate_{warmstart_candidate_idx}" / "model.zip")
            if proxy_model.exists():
                warmstart_path = proxy_model
                remaining = FULL_TIMESTEPS - PROXY_TIMESTEPS
                print(f"  Continuing from proxy checkpoint ({PROXY_TIMESTEPS//1000}K + {remaining//1000}K = {FULL_TIMESTEPS//1000}K total)")
            else:
                remaining = FULL_TIMESTEPS
        else:
            remaining = FULL_TIMESTEPS

        full_config = self._build_training_config(remaining)
        full_config["n_envs"] = self.config.get("n_envs", 16)
        full_config["normalize"] = self.config.get("normalize", False)
        for k in ("checkpoint", "gif_fps", "gif_max_steps"):
            if k in self.config:
                full_config[k] = self.config[k]

        # Adjust gif_steps for warmstart: SB3 resets the step counter in learn(),
        # so absolute gif steps need to be shifted relative to the remaining training.
        if warmstart_path and "gif_steps" in self.config:
            warmstart_offset = FULL_TIMESTEPS - remaining
            adjusted = [s - warmstart_offset for s in self.config["gif_steps"] if s > warmstart_offset]
            if adjusted:
                full_config["gif_steps"] = adjusted
                print(f"  Adjusted gif_steps for warmstart: {self.config['gif_steps']} → {adjusted}")

        config_path = round_dir / "config.yaml"
        dump_fn = yaml.safe_dump if yaml else lambda d, **kw: json.dumps(d, indent=2)
        config_path.write_text(dump_fn(full_config, sort_keys=False), encoding="utf-8")

        cmd = [
            sys.executable, str(_framework_dir / "train.py"),
            "--env-dir", str(self.env_dir),
            "--env-id", f"{self.env_dir.name}-round{round_num}",
            "--config", str(config_path), "--run-dir", str(round_dir),
            "--reward-source", str(reward_path),
        ]
        if warmstart_path:
            cmd += ["--warmstart", str(warmstart_path)]
        if self.config.get("max_episode_steps"):
            cmd += ["--max-episode-steps", str(self.config["max_episode_steps"])]

        print(f"  Training {remaining//1000}K steps...")
        t0 = time.perf_counter()
        result = self._run_subprocess_live(cmd)
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-300:]}")
            raise RuntimeError(f"Training failed at round {round_num}")

        print(f"  Done ({elapsed / 60:.1f} min)")

    # ── Agent Wrappers ─────────────────────────────────────────────────────────

    def _run_perception_on_round(self, round_num: int):
        """Run perception agent and ensure report is saved."""
        round_dir = self.exp_dir / f"round{round_num}"
        if not round_dir.exists():
            return
        try:
            report = self.perception_agent.run_on_round(round_dir, round_num)
            # Save explicitly if agent didn't
            report_path = round_dir / "perception_report.md"
            if report and not report_path.exists():
                report_path.write_text(report, encoding="utf-8")
                print(f"  Perception report saved")
        except Exception as e:
            print(f"  Perception failed: {e}")

    def _run_analyzer_with_history(self, prev_round: int, current_round: int,
                                     is_retry: bool = False) -> dict:
        """Run Analyzer with cross-round history + component stats injected.

        This is the key feedback mechanism: the Analyzer sees what was diagnosed,
        tried, and what happened in ALL previous rounds. This enables escalation
        when coefficient changes fail repeatedly.
        """
        prev_dir = self.exp_dir / f"round{prev_round}"
        if not prev_dir.exists():
            return {"proposal": {"changed_count": 0, "analysis_status": "failed"}}

        # Collect component stats
        comp_stats = self._collect_component_stats(prev_dir)
        stats_text = self._format_component_stats_for_prompt(comp_stats)
        (prev_dir / "component_stats.md").write_text(stats_text, encoding="utf-8")

        # Build cross-round history
        history_text = self._build_cross_round_history(current_round)
        if history_text:
            (prev_dir / "cross_round_history.md").write_text(history_text, encoding="utf-8")

        from agents.analyzer_agent import run_analyzer_agent
        temp = 0.5 if is_retry else 0.4  # higher temp on retry for diversity
        result = run_analyzer_agent(prev_dir, current_round, self.memory,
                                     self.api_key, self.model, temperature=temp)

        # Inject history + stats into saved prompt for audit
        analysis_file = prev_dir / "analyzer_prompt.txt"
        if analysis_file.exists():
            prompt = analysis_file.read_text("utf-8")
            additions = []
            if history_text and "## Cross-Round History" not in prompt:
                additions.append(f"## Cross-Round History\n{history_text}")
            if "## Component Training Statistics" not in prompt:
                additions.append(f"## Component Training Statistics\n{stats_text}")
            if additions:
                prompt = prompt.replace("## Current Reward Code",
                                        "\n\n".join(additions) + "\n\n## Current Reward Code")
                analysis_file.write_text(prompt)
                print(f"  Injected cross-round history + component stats into analyzer prompt")

        return result

    def _build_cross_round_history(self, current_round: int) -> str:
        """Build a summary of what was tried in each previous round and what happened.

        Format: "Round N: diagnosed X, changed Y → result was Z (len=..., score=...)"
        This gives the Analyzer the full arc of the experiment so it can avoid
        repeating failed approaches.
        """
        parts = []
        for r in range(current_round):
            rdir = self.exp_dir / f"round{r}"
            if not rdir.exists():
                continue

            # Get analyzer proposal
            prop = {}
            prop_file = rdir / "analyzer_proposal.json"
            if prop_file.exists():
                try:
                    prop = json.loads(prop_file.read_text("utf-8"))
                except Exception:
                    pass

            # Get eval
            eval_len = "?"
            eval_file = rdir / "evaluations" / "history.csv"
            if eval_file.exists():
                try:
                    last = list(csv.DictReader(eval_file.open("r")))[-1]
                    eval_len = last.get("mean_length", "?")
                except Exception:
                    pass

            # Get component health
            health_score = "?"
            cs_file = rdir / "component_stats.md"
            if cs_file.exists():
                try:
                    lines = [l for l in cs_file.read_text("utf-8").split("\n") if l.startswith("|")][2:-2]
                    active = sum(1 for l in lines if "active" in l)
                    health_score = f"active={active}/{len(lines)}" if lines else "?"
                except Exception:
                    pass

            diagnosis = prop.get("diagnosis", "?")[:150]
            changed = prop.get("changed_count", 0)

            parts.append(
                f"**Round {r}:** Diagnosed: {diagnosis}\n"
                f"  Proposed {changed} change(s). "
                f"Result: eval_length={eval_len}, components={health_score}"
            )

        if not parts:
            return ""

        repeated_diagnoses = self._detect_repeated_diagnoses(current_round)
        header = "What was diagnosed, tried, and what happened in each previous round:"
        if repeated_diagnoses:
            header += f"\n\n**WARNING: The same diagnosis has appeared in {repeated_diagnoses} consecutive rounds without resolving the issue. You MUST propose a STRUCTURAL change, not another coefficient tweak.**"

        return header + "\n\n" + "\n\n".join(parts)

    def _detect_repeated_diagnoses(self, current_round: int) -> int:
        """Count how many consecutive previous rounds had the same core diagnosis."""
        diagnoses = []
        for r in range(current_round):
            prop_file = self.exp_dir / f"round{r}" / "analyzer_proposal.json"
            if prop_file.exists():
                try:
                    d = json.loads(prop_file.read_text("utf-8")).get("diagnosis", "")
                    # Extract key phrases
                    keywords = set(re.findall(r'\b(per-step|terminal|dominat|coefficient|structur|hover|surviv)\w*', d.lower()))
                    diagnoses.append(frozenset(keywords))
                except Exception:
                    pass

        if len(diagnoses) < 2:
            return 0

        # Count from the end how many are similar to the last one
        count = 1
        for i in range(len(diagnoses) - 2, -1, -1):
            sim = len(diagnoses[i] & diagnoses[-1]) / max(len(diagnoses[i] | diagnoses[-1]), 1)
            if sim > 0.3:
                count += 1
            else:
                break
        return count if count >= 2 else 0

    def _get_previous_health(self, round_num: int) -> dict | None:
        """Get the health score from a previous round's component stats."""
        rdir = self.exp_dir / f"round{round_num}"
        stats = self._collect_component_stats(rdir)
        evals = self._read_eval_csv(rdir / "evaluations" / "history.csv")
        if stats or evals:
            return compute_component_health(stats, evals)
        return None

    def _run_analyzer(self, prev_round: int, current_round: int) -> dict:
        """Backward-compat wrapper."""
        return self._run_analyzer_with_history(prev_round, current_round)

    def _run_reflection(self, round_num: int):
        """Run reflection agent and consolidate to memory."""
        round_dir = self.exp_dir / f"round{round_num}"
        if not round_dir.exists():
            return
        try:
            self.reflector_agent.run_on_round(round_dir, round_num, memory_system=self.memory)
        except Exception as e:
            print(f"  Reflection LLM call failed: {e}")
            # Write a basic reflection from component stats
            self._write_fallback_reflection(round_dir, round_num)

        reflection_path = round_dir / "reflection.md"
        if reflection_path.exists():
            try:
                n = self.memory.consolidate_to_archival(
                    reflection_path.read_text("utf-8"), round_num,
                    env_description=self._load_env_description(self.env_dir.name),
                )
                if n: print(f"  +{n} archival patterns")
            except Exception as e:
                print(f"  Archival consolidation failed: {e}")

    def _write_fallback_reflection(self, round_dir: Path, round_num: int):
        """Write a basic reflection from available data when LLM fails."""
        comp_stats = self._collect_component_stats(round_dir)
        evals = self._read_eval_csv(round_dir / "evaluations" / "history.csv")
        health = compute_component_health(comp_stats, evals)

        reflection = f"""## Round {round_num} Reflection (auto-generated)

### What We Learned
Component health score: {health['score']:.0f}/100 ({health['verdict']}).
{health['n_active']}/{health['n_components']} components active.
Episode length quality: {health['length_quality']:.2f} (1.0=best).
Outcome signal: {health['outcome']:.2f} (1.0=all success).

### Abstract Principle
Reward component balance and activation are key indicators of training health.

### For Next Round
- [ ] Action: Review component activation — dead components need structural changes
"""
        (round_dir / "reflection.md").write_text(reflection, encoding="utf-8")
        print(f"  Fallback reflection written")

    # ── Component Stats ────────────────────────────────────────────────────────

    def _collect_component_stats(self, round_dir: Path) -> list[dict]:
        """Collect per-component statistics from trajectory JSONL files."""
        traj_dir = round_dir / "trajectory_logs"
        if not traj_dir.exists():
            return []

        all_comps: dict[str, list[float]] = {}
        for f in sorted(traj_dir.glob("*.jsonl")):
            for line in f.read_text("utf-8").strip().split("\n"):
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    for comp, val in (record.get("component_means", {})).items():
                        all_comps.setdefault(comp, []).append(float(val))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        result = []
        for comp, vals in all_comps.items():
            arr = vals
            n = len(arr)
            if n == 0: continue
            mean = sum(arr) / n
            std = (sum((x - mean)**2 for x in arr) / n) ** 0.5 if n > 1 else 0.0
            result.append({
                "component": comp,
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(min(arr), 6),
                "max": round(max(arr), 6),
                "n": n,
            })

        return result

    def _format_component_stats_for_prompt(self, stats: list[dict]) -> str:
        """Format component statistics as markdown table (Eureka-style reflection)."""
        if not stats:
            return "*(no component statistics available)*"
        lines = [
            "| Component | Mean | Std | Min | Max | N | Status |",
            "|-----------|------|-----|-----|-----|---|--------|",
        ]
        for s in stats:
            status = "active" if abs(s["mean"]) > 0.01 else ("dead" if abs(s["mean"]) < 1e-6 else "weak")
            lines.append(
                f"| {s['component']} | {s['mean']:.4f} | {s['std']:.4f} | "
                f"{s['min']:.4f} | {s['max']:.4f} | {s['n']} | {status} |"
            )
        lines.append("")
        lines.append("**Key:** active = |mean| > 0.01, weak = in between, dead = |mean| ≈ 0")
        lines.append("Dead components indicate the reward signal is not reaching the agent — consider removing or rescaling them.")
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _setup_experiment_dir(self):
        env_name = self.env_dir.name.lower()
        ts = datetime.now(BEIJING).strftime("%y%m%d%H%M")
        steps = self.config.get("total_timesteps", FULL_TIMESTEPS)
        self.exp_dir = _framework_dir.parent / "runs" / f"{env_name}_{ts}_{steps}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def _register_subscriptions(self):
        for agent, types in [
            ("generator", ["task_manifest", "evaluation_report", "reflection_report"]),
            ("analyzer", ["perception_report", "reward_code", "training_result"]),
            ("evaluator", ["reward_code", "training_result"]),
            ("reflector", ["evaluation_report", "generator_proposal"]),
        ]:
            self.pool.subscribe(agent, types)

    def _save_config(self):
        cfg = {k: v for k, v in self.config.items() if k != "llm_api_key"}
        dump_fn = yaml.safe_dump if yaml else lambda d, **kw: json.dumps(d, indent=2)
        (self.exp_dir / "config.yaml").write_text(dump_fn(cfg, sort_keys=False), encoding="utf-8")

    def _save_status(self, n_rounds):
        (self.exp_dir / "STATUS").write_text(
            f"COMPLETED (v2, {n_rounds} rounds, {self.pool.message_count} msgs, {self.memory.archival.pattern_count} archival)\n")

    def _copy_prev_reward(self, prev_dir: Path, round_dir: Path):
        """Copy previous round's reward code to current round."""
        src = prev_dir / "reward_fn_source.py"
        if src.exists():
            round_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(str(src), str(round_dir / "reward_fn_source.py"))

    def _save_reward(self, code: str, round_dir: Path, round_num: int):
        round_dir.mkdir(parents=True, exist_ok=True)
        cleaned = re.sub(r'^"""LLM[- ].*?"""', '', code.strip(), flags=re.DOTALL)
        cleaned = re.sub(r'^import\s+(math|numpy).*?\n', '', cleaned, flags=re.MULTILINE)
        (round_dir / "reward_fn_source.py").write_text(
            f'"""LLM-generated reward (round {round_num}).\n"""\n\nimport math\nimport numpy as np\n\n{cleaned}\n',
            encoding="utf-8")

    def _load_env_description(self, name: str) -> str:
        p = _framework_dir.parent / "env_descriptions" / f"{name}.md"
        return p.read_text("utf-8").strip() if p.exists() else ""

    def _extract_signature(self, manifest: str) -> str:
        m = re.search(r'compute_reward\s*\(([^)]+)\)', manifest)
        return m.group(1).strip() if m else "state, action, terminated"

    def _call_llm_for_code(self, prompt, temp, label=""):
        for attempt in range(1, 4):
            try:
                response = call_llm(prompt, self.api_key, self.model, temp)
                code = extract_reward_fn(response)
                if "def compute_reward" in code:
                    compile(code, "<gen>", "exec")
                    return code, response
            except Exception as e:
                if attempt == 3: print(f"  {label} FAILED: {e}")
        return None, None

    def _read_eval_csv(self, path: Path) -> list[dict]:
        if not path.exists(): return []
        with path.open("r") as f:
            return list(csv.DictReader(f))

    def _read_file(self, path: Path) -> str:
        return path.read_text("utf-8") if path.exists() else ""
