"""Reward candidate generator for ASE-MTAGE.

The Mutator supports two modes:
1. deterministic templates, used when llm.enabled=false;
2. LLM-backed mutation, used when a client is passed in.

Both modes write the same candidate artifacts:
- reward_fn_source.py
- mutation_metadata.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase_mtage.llm_client import LLMClient, extract_python_code, load_prompt
from ase_mtage.utils.io import ensure_dir, save_json, save_text


MUTATION_FAMILIES = ["local_repair", "component_recomposition", "progress_conditioned"]


@dataclass(slots=True)
class CandidateArtifact:
    candidate_id: str
    candidate_dir: Path
    reward_path: Path
    metadata_path: Path
    mutation_family: str


class MutatorAgent:
    """Generate K reward candidates and save candidate directories."""

    def __init__(self, candidates_root: str | Path, *, llm_client: LLMClient | None = None, temperature: float = 0.6) -> None:
        self.candidates_root = ensure_dir(candidates_root)
        self.llm_client = llm_client
        self.temperature = temperature

    def generate_initial_candidates(self, *, env_manifest: dict[str, Any], k_candidates: int = 3, round_idx: int = 0) -> list[CandidateArtifact]:
        return self.generate_candidates(
            env_manifest=env_manifest,
            k_candidates=k_candidates,
            round_idx=round_idx,
            analyzer_report=None,
            parent_reward_code=None,
        )

    def generate_candidates(
        self,
        *,
        env_manifest: dict[str, Any],
        k_candidates: int = 3,
        round_idx: int = 0,
        analyzer_report: dict[str, Any] | None = None,
        parent_reward_code: str | None = None,
        task_manifest: str | None = None,
        coverage_report: dict[str, Any] | None = None,
    ) -> list[CandidateArtifact]:
        artifacts: list[CandidateArtifact] = []
        for idx in range(k_candidates):
            family = MUTATION_FAMILIES[idx % len(MUTATION_FAMILIES)]
            candidate_id = f"round{round_idx}_candidate{idx}_{family}"
            candidate_dir = ensure_dir(self.candidates_root / f"candidate_{idx}_{family}")
            if self.llm_client is not None:
                reward_source = self._generate_llm_reward(
                    env_manifest=env_manifest,
                    family=family,
                    analyzer_report=analyzer_report,
                    parent_reward_code=parent_reward_code,
                    task_manifest=task_manifest,
                    coverage_report=coverage_report,
                    candidate_dir=candidate_dir,
                )
                generation_mode = "llm_mutator"
            else:
                reward_source = self._build_reward_source(env_manifest=env_manifest, family=family)
                generation_mode = "deterministic_template"
            reward_path = save_text(candidate_dir / "reward_fn_source.py", reward_source)
            metadata = {
                "candidate_id": candidate_id,
                "round": round_idx,
                "mutation_family": family,
                "generation_mode": generation_mode,
                "parent_reward_id": (analyzer_report or {}).get("parent_reward_id"),
                "intended_fix": self._family_intended_fix(family),
                "expected_effect": self._family_expected_effect(family),
                "risk": self._family_risk(family),
                "official_reward_used": False,
            }
            metadata_path = save_json(candidate_dir / "mutation_metadata.json", metadata)
            artifacts.append(CandidateArtifact(candidate_id, candidate_dir, reward_path, metadata_path, family))
        return artifacts

    def _generate_llm_reward(
        self,
        *,
        env_manifest: dict[str, Any],
        family: str,
        analyzer_report: dict[str, Any] | None,
        parent_reward_code: str | None,
        task_manifest: str | None,
        coverage_report: dict[str, Any] | None,
        candidate_dir: Path,
    ) -> str:
        template = load_prompt("mutator.md")
        input_artifacts = {
            "task_manifest": task_manifest or "",
            "env_manifest": env_manifest,
            "parent_reward_code_optional": parent_reward_code or "",
            "analyzer_self_evaluation": analyzer_report or {},
            "memory_coverage_report": coverage_report or {},
            "mutation_family": family,
        }
        user_prompt = template.replace("{input_artifacts}", json.dumps(input_artifacts, ensure_ascii=False, indent=2))
        save_text(candidate_dir / "llm_prompt.txt", user_prompt)
        resp = self.llm_client.chat(
            system_prompt="You are the ASE-MTAGE Mutator Agent. Output only safe Python reward code.",
            user_prompt=user_prompt,
            temperature=self.temperature,
        )
        save_text(candidate_dir / "llm_response.txt", resp.content)
        save_json(candidate_dir / "llm_raw_response.json", resp.raw)
        return extract_python_code(resp.content)

    def _build_reward_source(self, *, env_manifest: dict[str, Any], family: str) -> str:
        env_name = str(env_manifest.get("env_name", "UnknownEnv"))
        if "lunarlander" in env_name.lower():
            body = self._lunarlander_reward_body(family)
        elif "cartpole" in env_name.lower():
            body = self._cartpole_reward_body(family)
        else:
            body = self._generic_reward_body(family)
        return f'''"""ASE-MTAGE deterministic reward candidate.

Environment: {env_name}
Mutation family: {family}
Official environment reward is not used.
"""

import math


def _safe_float(x, default=0.0):
    try:
        value = float(x)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value


def compute_reward(obs, action, next_obs, terminated, truncated, info):
{body}
'''

    def _lunarlander_reward_body(self, family: str) -> str:
        if family == "local_repair":
            return '''    x = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    left_contact = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_contact = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0
    center_progress = -abs(x)
    controlled_descent = -abs(vy) - 0.5 * abs(vx)
    attitude = -abs(angle)
    contact = 0.25 * (left_contact + right_contact)
    terminal_penalty = -1.0 if terminated and y < 0.15 and abs(vy) > 0.6 else 0.0
    components = {"center_progress": center_progress, "controlled_descent": controlled_descent, "attitude": attitude, "contact": contact, "terminal_penalty": terminal_penalty}
    total_reward = center_progress + 0.8 * controlled_descent + 0.4 * attitude + contact + terminal_penalty
    return float(total_reward), components
'''
        if family == "component_recomposition":
            return '''    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    prev_distance = math.sqrt(x0 * x0 + y0 * y0)
    next_distance = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_distance - next_distance
    speed_control = -(vx * vx + vy * vy)
    angle_control = -abs(angle)
    terminal_failure = -2.0 if terminated and (abs(x1) > 0.25 or abs(vy) > 0.7) else 0.0
    components = {"progress_delta": progress_delta, "speed_control": speed_control, "angle_control": angle_control, "terminal_failure": terminal_failure}
    total_reward = 4.0 * progress_delta + 0.5 * speed_control + 0.3 * angle_control + terminal_failure
    return float(total_reward), components
'''
        return '''    x = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    left_contact = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_contact = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0
    distance = math.sqrt(x * x + y * y)
    far_stage = 1.0 if distance > 0.7 else 0.0
    near_stage = 1.0 - far_stage
    approach = -abs(x) - 0.25 * max(y, 0.0)
    near_stability = -abs(vx) - abs(vy) - abs(angle)
    stable_contact = near_stage * 0.5 * (left_contact + right_contact) * (1.0 if abs(vy) < 0.4 else 0.0)
    timeout_penalty = -0.2 if truncated and distance > 0.4 else 0.0
    terminal_penalty = -1.5 if terminated and (abs(vy) > 0.8 or abs(angle) > 0.6) else 0.0
    components = {"far_stage_approach": far_stage * approach, "near_stage_stability": near_stage * near_stability, "stable_contact": stable_contact, "timeout_penalty": timeout_penalty, "terminal_penalty": terminal_penalty}
    total_reward = 1.5 * components["far_stage_approach"] + 1.2 * components["near_stage_stability"] + stable_contact + timeout_penalty + terminal_penalty
    return float(total_reward), components
'''

    def _cartpole_reward_body(self, family: str) -> str:
        if family == "local_repair":
            return '''    x = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    theta = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    center = -abs(x)
    balance = -abs(theta)
    failure = -1.0 if terminated else 0.0
    components = {"center": center, "balance": balance, "failure": failure}
    total_reward = 0.3 * center + balance + failure
    return float(total_reward), components
'''
        if family == "component_recomposition":
            return '''    theta0 = _safe_float(obs[2]) if len(obs) > 2 else 0.0
    theta1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    angle_improvement = abs(theta0) - abs(theta1)
    center_control = -abs(x1)
    failure = -2.0 if terminated else 0.0
    components = {"angle_improvement": angle_improvement, "center_control": center_control, "failure": failure}
    total_reward = 2.0 * angle_improvement + 0.2 * center_control + failure
    return float(total_reward), components
'''
        return '''    x = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    theta = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    stable_stage = 1.0 if abs(theta) < 0.08 else 0.0
    recovery_stage = 1.0 - stable_stage
    recovery = -abs(theta) - 0.1 * abs(x)
    maintenance = stable_stage * (0.2 - abs(theta))
    failure = -2.0 if terminated else 0.0
    components = {"recovery": recovery_stage * recovery, "maintenance": maintenance, "failure": failure}
    total_reward = components["recovery"] + components["maintenance"] + failure
    return float(total_reward), components
'''

    def _generic_reward_body(self, family: str) -> str:
        return '''    terminated_penalty = -1.0 if terminated else 0.0
    truncated_penalty = -0.1 if truncated else 0.0
    action_magnitude = 0.0
    try:
        if hasattr(action, "__iter__"):
            action_magnitude = sum(abs(_safe_float(a)) for a in action)
        else:
            action_magnitude = abs(_safe_float(action))
    except Exception:
        action_magnitude = 0.0
    action_smoothness = -0.01 * action_magnitude
    components = {"termination_penalty": terminated_penalty, "truncation_penalty": truncated_penalty, "action_smoothness": action_smoothness}
    total_reward = termination_penalty + truncated_penalty + action_smoothness
    return float(total_reward), components
'''

    def _family_intended_fix(self, family: str) -> list[str]:
        if family == "local_repair":
            return ["make a conservative reward with clear components", "avoid official reward leakage"]
        if family == "component_recomposition":
            return ["use delta/progress-style components", "avoid pure survival reward"]
        return ["condition reward by progress or stability stage", "reduce low-progress reward hacking"]

    def _family_expected_effect(self, family: str) -> str:
        if family == "local_repair":
            return "Provide a simple, interpretable baseline candidate."
        if family == "component_recomposition":
            return "Separate progress, stability, and terminal terms."
        return "Handle stage-dependent task conflicts and discourage low-progress survival."

    def _family_risk(self, family: str) -> str:
        if family == "local_repair":
            return "May be too weak or too dense for difficult tasks."
        if family == "component_recomposition":
            return "Delta terms may be noisy if the progress proxy is imperfect."
        return "Stage gates may be imperfect before trajectory memory is available."
