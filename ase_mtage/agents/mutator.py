"""Reward candidate generator for ASE-MTAGE.

The Mutator supports:
1. LLM-backed mutation with Analyzer evidence;
2. analyzer-aware deterministic fallback.

The fallback is intentionally not a fixed template anymore. It reads
`analyzer_report`, `coverage_report`, and `round_idx` to produce different reward
code across rounds, so no-LLM runs still exercise the self-evolution protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase_mtage.llm_client import LLMClient, extract_python_code, load_prompt
from ase_mtage.utils.io import ensure_dir, save_json, save_text


MUTATION_FAMILIES = ["local_repair", "component_recomposition", "progress_conditioned"]

FAMILY_INSTRUCTIONS: dict[str, str] = {
    "local_repair": """### `local_repair` — Conservative Mutation

You are generating a **conservative repair** candidate. Your task:

1. **Keep the parent reward structure mostly intact.** Do NOT rewrite the whole reward from scratch.
2. **Identify and adjust one or two problematic components.** Look at the Analyzer report for which components are flagged. Gate them (add conditions so they don't fire in wrong situations) or reduce their coefficients.
3. **Preserve useful components** that the Analyzer identified as working well.
4. **Avoid broad structural rewrites** — no new component decomposition, no stage gates, no progress conditioning (those are done by other mutation families).
5. If the Analyzer says a component over-rewards known failures, add an `if` condition to suppress it in those scenarios.

Your output should look similar to the parent code but with 1-2 targeted fixes. Think "precision surgery" not "rebuild." The temperature for this call is lower, so prefer safe, minimal edits.""",

    "component_recomposition": """### `component_recomposition` — Medium Structural Mutation

You are generating a **component recomposition** candidate. Your task:

1. **Remove or replace misleading components** identified by the Analyzer as rewarding failure trajectories.
2. **Add missing components** — if the parent code lacks progress, stability, or terminal penalty terms, add them.
3. **Turn absolute rewards into delta/progress rewards** when appropriate. For example, change an absolute position reward into a "distance decreased from previous step" delta.
4. **Separate into clear categories:** progress terms, stability/safety terms, terminal bonuses/penalties, and penalty terms. Each category should be a distinct named component.
5. **Avoid component interactions that reward known failures** — if two components together can create a reward-hacking loophole, the recomposition must close it.

Your output should have a clearly different component structure from the parent. Don't just rename components — change what they measure and how they combine. Think "refactor with purpose." The temperature for this call is medium, encouraging moderate structural innovation.""",

    "progress_conditioned": """### `progress_conditioned` — Large Structural Mutation

You are generating a **progress-conditioned** candidate. Your task:

1. **Use stage gates or progress-conditioned terms.** Define at least two stages (e.g., "far from target" vs "near target", or "unstable" vs "stable") and use different reward logic per stage.
2. **Make early-stage reward different from late-stage reward.** Early stages should reward approach/progress; late stages should reward precision/stability.
3. **Require stability near terminal regions.** When the agent is close to the goal, reward gentle, controlled behavior — not just being close.
4. **Reduce or eliminate rewards for low-progress survival.** If the agent is alive but not making progress toward the goal, it should receive little or no positive reward.
5. **Avoid giving dense positive reward when task progress is absent.** Use sparse or gated bonuses instead of continuous survival rewards.

Your output can depart significantly from the parent code's structure. You may introduce new stage-detection logic using `if/else` based on observable features. Think "redesign for the long horizon." The temperature for this call is higher, encouraging exploration of novel structures.""",
}



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
        return self.generate_candidates(env_manifest=env_manifest, k_candidates=k_candidates, round_idx=round_idx)

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
        reflection_guidance: list[str] | None = None,
    ) -> list[CandidateArtifact]:
        artifacts: list[CandidateArtifact] = []
        ordered_families = self._family_order(analyzer_report, k_candidates)
        for idx in range(k_candidates):
            family = ordered_families[idx % len(ordered_families)]
            candidate_id = f"round{round_idx}_candidate{idx}_{family}"
            candidate_dir = ensure_dir(self.candidates_root / f"candidate_{idx}_{family}")
            if self.llm_client is not None:
                reward_source = self._generate_llm_reward(
                    env_manifest=env_manifest,
                    family=family,
                    idx=idx,
                    analyzer_report=analyzer_report,
                    parent_reward_code=parent_reward_code,
                    task_manifest=task_manifest,
                    coverage_report=coverage_report,
                    candidate_dir=candidate_dir,
                    reflection_guidance=reflection_guidance,
                )
                generation_mode = "llm_mutator"
            else:
                reward_source = self._build_reward_source(
                    env_manifest=env_manifest,
                    family=family,
                    analyzer_report=analyzer_report,
                    coverage_report=coverage_report,
                    round_idx=round_idx,
                    candidate_idx=idx,
                )
                generation_mode = "deterministic_analyzer_aware"
            reward_path = save_text(candidate_dir / "reward_fn_source.py", reward_source)
            metadata = {
                "candidate_id": candidate_id,
                "round": round_idx,
                "mutation_family": family,
                "generation_mode": generation_mode,
                "parent_reward_id": (analyzer_report or {}).get("parent_reward_id"),
                "mutation_intent_used": (analyzer_report or {}).get("mutation_intent", {}),
                "coverage_type_used": (coverage_report or {}).get("coverage_type"),
                "intended_fix": self._family_intended_fix(family),
                "expected_effect": self._family_expected_effect(family),
                "risk": self._family_risk(family),
                "official_reward_used": False,
            }
            metadata_path = save_json(candidate_dir / "mutation_metadata.json", metadata)
            artifacts.append(CandidateArtifact(candidate_id, candidate_dir, reward_path, metadata_path, family))
        return artifacts

    def _family_order(self, analyzer_report: dict[str, Any] | None, k: int) -> list[str]:
        intent = (analyzer_report or {}).get("mutation_intent") or {}
        primary = intent.get("primary_family")
        secondary = intent.get("secondary_family")
        order: list[str] = []
        for fam in [primary, secondary, *MUTATION_FAMILIES]:
            if fam in MUTATION_FAMILIES and fam not in order:
                order.append(fam)
        return order or MUTATION_FAMILIES

    def _generate_llm_reward(self, *, env_manifest: dict[str, Any], family: str, idx: int, analyzer_report: dict[str, Any] | None, parent_reward_code: str | None, task_manifest: str | None, coverage_report: dict[str, Any] | None, candidate_dir: Path, reflection_guidance: list[str] | None = None) -> str:
        template = load_prompt("mutator.md")
        family_section = FAMILY_INSTRUCTIONS.get(family, FAMILY_INSTRUCTIONS["local_repair"])
        template = template.replace("{mutation_family_section}", family_section)
        input_artifacts = {
            "task_manifest": task_manifest or "",
            "env_manifest": env_manifest,
            "parent_reward_code_optional": parent_reward_code or "",
            "analyzer_self_evaluation": analyzer_report or {},
            "memory_coverage_report": coverage_report or {},
            "mutation_family": family,
            "reflection_guidance": reflection_guidance or [],
        }
        user_prompt = template.replace("{input_artifacts}", json.dumps(input_artifacts, ensure_ascii=False, indent=2))
        save_text(candidate_dir / "llm_prompt.txt", user_prompt)

        # Lower temperature range: 0.50 → 0.74 instead of 0.60 → 0.90
        per_index_temperature = self.temperature + idx * 0.12

        code = self._call_and_extract(user_prompt, per_index_temperature, candidate_dir)
        # Retry once if code fails basic syntax check
        if not self._is_valid_python(code):
            save_text(candidate_dir / "llm_prompt_retry.txt", user_prompt)
            retry_hint = "\n\nYOUR PREVIOUS OUTPUT HAD A SYNTAX ERROR (unbalanced parentheses, missing return, or truncated code). Output COMPLETE, syntactically valid Python code. Ensure all parentheses are balanced and the function returns (float, dict)."
            code = self._call_and_extract(user_prompt + retry_hint, max(0.2, per_index_temperature - 0.2), candidate_dir, suffix="_retry")
            save_text(candidate_dir / "llm_response_retry.txt", code)
        return code

    def _call_and_extract(self, user_prompt: str, temperature: float, candidate_dir: Path, suffix: str = "") -> str:
        resp = self.llm_client.chat(system_prompt="You are the ASE-MTAGE Mutator Agent. Output only safe, COMPLETE Python code.", user_prompt=user_prompt, temperature=temperature, agent_name="mutator")
        save_text(candidate_dir / f"llm_response{suffix}.txt", resp.content)
        save_json(candidate_dir / f"llm_raw_response{suffix}.json", resp.raw)
        return extract_python_code(resp.content)

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        import ast
        try:
            ast.parse(code)
            return "def compute_reward(" in code
        except SyntaxError:
            return False

    def _context(self, analyzer_report: dict[str, Any] | None, coverage_report: dict[str, Any] | None, round_idx: int, candidate_idx: int) -> dict[str, Any]:
        intent = (analyzer_report or {}).get("mutation_intent") or {}
        memory = (analyzer_report or {}).get("memory_interpretation") or {}
        known_failures = set(memory.get("main_known_failures") or [])
        preserve = set(intent.get("preserve_components") or [])
        remove = set(intent.get("remove_or_gate_components") or [])
        coverage_type = str((coverage_report or {}).get("coverage_type") or memory.get("coverage_type") or "unknown")
        failure_pressure = 1.0 + 0.20 * round_idx + (0.15 if "early_failure" in known_failures else 0.0)
        survival_gate = 0.0 if ("low_progress_survival" in known_failures or coverage_type in {"single_failure_mode", "multiple_failure_modes"}) else 0.15
        progress_gain = 1.0 + 0.10 * round_idx + (0.20 if coverage_type in {"failure_plus_partial_progress", "balanced"} else 0.0)
        stability_gain = 1.0 + 0.12 * round_idx + (0.20 if remove else 0.0)
        novelty = 1.0 + 0.05 * candidate_idx
        return {
            "known_failures": sorted(known_failures),
            "preserve_components": sorted(preserve),
            "remove_or_gate_components": sorted(remove),
            "coverage_type": coverage_type,
            "failure_pressure": round(failure_pressure * novelty, 4),
            "survival_gate": round(survival_gate, 4),
            "progress_gain": round(progress_gain * novelty, 4),
            "stability_gain": round(stability_gain, 4),
        }

    def _build_reward_source(self, *, env_manifest: dict[str, Any], family: str, analyzer_report: dict[str, Any] | None, coverage_report: dict[str, Any] | None, round_idx: int, candidate_idx: int) -> str:
        env_name = str(env_manifest.get("env_name", "UnknownEnv"))
        ctx = self._context(analyzer_report, coverage_report, round_idx, candidate_idx)
        if "lunarlander" in env_name.lower():
            body = self._lunarlander_reward_body(family, ctx)
        elif "cartpole" in env_name.lower():
            body = self._cartpole_reward_body(family, ctx)
        elif "bipedalwalker" in env_name.lower():
            body = self._bipedal_reward_body(family, ctx)
        else:
            body = self._generic_reward_body(family, ctx)
        return f'''"""ASE-MTAGE analyzer-aware deterministic reward candidate.

Environment: {env_name}
Mutation family: {family}
Context: {json.dumps(ctx, ensure_ascii=False)}
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


def _at(values, idx, default=0.0):
    try:
        return _safe_float(values[idx], default)
    except Exception:
        return default


def compute_reward(obs, action, next_obs, terminated, truncated, info):
{body}
'''

    def _lunarlander_reward_body(self, family: str, ctx: dict[str, Any]) -> str:
        pg, sg, fp, alive = ctx["progress_gain"], ctx["stability_gain"], ctx["failure_pressure"], ctx["survival_gate"]
        if family == "local_repair":
            return f'''    x, y, vx, vy, angle = _at(next_obs, 0), _at(next_obs, 1), _at(next_obs, 2), _at(next_obs, 3), _at(next_obs, 4)
    left_contact, right_contact = _at(next_obs, 6), _at(next_obs, 7)
    center_progress = -abs(x)
    controlled_descent = -abs(vy) - 0.5 * abs(vx)
    attitude = -abs(angle)
    gated_contact = 0.25 * (left_contact + right_contact) if abs(vy) < 0.45 and abs(angle) < 0.35 else 0.0
    low_progress_survival_gate = {alive} if (not terminated and not truncated and abs(x) < 0.5 and y > 0.05) else 0.0
    terminal_penalty = -{fp} if terminated and (y < 0.15 or abs(vy) > 0.65 or abs(angle) > 0.55) else 0.0
    components = {{"center_progress": center_progress, "controlled_descent": controlled_descent, "attitude": attitude, "gated_contact": gated_contact, "low_progress_survival_gate": low_progress_survival_gate, "terminal_penalty": terminal_penalty}}
    total_reward = {pg} * center_progress + {sg} * (0.8 * controlled_descent + 0.4 * attitude) + gated_contact + low_progress_survival_gate + terminal_penalty
    return float(total_reward), components
'''
        if family == "component_recomposition":
            return f'''    x0, y0 = _at(obs, 0), _at(obs, 1)
    x1, y1, vx, vy, angle = _at(next_obs, 0), _at(next_obs, 1), _at(next_obs, 2), _at(next_obs, 3), _at(next_obs, 4)
    prev_distance = math.sqrt(x0 * x0 + y0 * y0)
    next_distance = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_distance - next_distance
    speed_control = -(vx * vx + vy * vy)
    angle_control = -abs(angle)
    no_progress_penalty = -0.25 if progress_delta <= 0.0 and next_distance > 0.35 else 0.0
    terminal_failure = -{2.0 * fp} if terminated and (abs(x1) > 0.25 or abs(vy) > 0.7 or abs(angle) > 0.55) else 0.0
    components = {{"progress_delta": progress_delta, "speed_control": speed_control, "angle_control": angle_control, "no_progress_penalty": no_progress_penalty, "terminal_failure": terminal_failure}}
    total_reward = {4.0 * pg} * progress_delta + {0.5 * sg} * speed_control + {0.3 * sg} * angle_control + no_progress_penalty + terminal_failure
    return float(total_reward), components
'''
        return f'''    x, y, vx, vy, angle = _at(next_obs, 0), _at(next_obs, 1), _at(next_obs, 2), _at(next_obs, 3), _at(next_obs, 4)
    left_contact, right_contact = _at(next_obs, 6), _at(next_obs, 7)
    distance = math.sqrt(x * x + y * y)
    far_stage = 1.0 if distance > 0.7 else 0.0
    near_stage = 1.0 - far_stage
    approach = -abs(x) - 0.25 * max(y, 0.0)
    near_stability = -abs(vx) - abs(vy) - abs(angle)
    stable_contact = near_stage * 0.5 * (left_contact + right_contact) if abs(vy) < 0.4 and abs(angle) < 0.45 else 0.0
    timeout_penalty = -{0.5 * fp} if truncated and distance > 0.4 else 0.0
    terminal_penalty = -{1.5 * fp} if terminated and (abs(vy) > 0.8 or abs(angle) > 0.6) else 0.0
    components = {{"far_stage_approach": far_stage * approach, "near_stage_stability": near_stage * near_stability, "stable_contact": stable_contact, "timeout_penalty": timeout_penalty, "terminal_penalty": terminal_penalty}}
    total_reward = {1.5 * pg} * components["far_stage_approach"] + {1.2 * sg} * components["near_stage_stability"] + stable_contact + timeout_penalty + terminal_penalty
    return float(total_reward), components
'''

    def _cartpole_reward_body(self, family: str, ctx: dict[str, Any]) -> str:
        pg, sg, fp = ctx["progress_gain"], ctx["stability_gain"], ctx["failure_pressure"]
        if family == "component_recomposition":
            return f'''    theta0, theta1 = _at(obs, 2), _at(next_obs, 2)
    x1 = _at(next_obs, 0)
    angle_improvement = abs(theta0) - abs(theta1)
    center_control = -abs(x1)
    failure = -{2.0 * fp} if terminated else 0.0
    components = {{"angle_improvement": angle_improvement, "center_control": center_control, "failure": failure}}
    total_reward = {2.0 * pg} * angle_improvement + {0.2 * sg} * center_control + failure
    return float(total_reward), components
'''
        if family == "progress_conditioned":
            return f'''    x, theta = _at(next_obs, 0), _at(next_obs, 2)
    stable_stage = 1.0 if abs(theta) < 0.08 else 0.0
    recovery_stage = 1.0 - stable_stage
    recovery = -abs(theta) - 0.1 * abs(x)
    maintenance = stable_stage * (0.2 - abs(theta))
    failure = -{2.0 * fp} if terminated else 0.0
    components = {{"recovery": recovery_stage * recovery, "maintenance": maintenance, "failure": failure}}
    total_reward = {pg} * components["recovery"] + {sg} * components["maintenance"] + failure
    return float(total_reward), components
'''
        return f'''    x, theta = _at(next_obs, 0), _at(next_obs, 2)
    center = -abs(x)
    balance = -abs(theta)
    failure = -{fp} if terminated else 0.0
    components = {{"center": center, "balance": balance, "failure": failure}}
    total_reward = {0.3 * sg} * center + {sg} * balance + failure
    return float(total_reward), components
'''

    def _bipedal_reward_body(self, family: str, ctx: dict[str, Any]) -> str:
        pg, sg, fp, alive = ctx["progress_gain"], ctx["stability_gain"], ctx["failure_pressure"], ctx["survival_gate"]
        return f'''    hull_angle = _at(next_obs, 0)
    hull_ang_vel = _at(next_obs, 1)
    hull_vx = _at(next_obs, 2)
    hull_vy = _at(next_obs, 3)
    hip1, knee1, contact1 = _at(next_obs, 4), _at(next_obs, 6), _at(next_obs, 8)
    hip2, knee2, contact2 = _at(next_obs, 9), _at(next_obs, 11), _at(next_obs, 13)
    forward_velocity = hull_vx
    stability = -(abs(hull_angle) + 0.5 * abs(hull_ang_vel) + 0.2 * abs(hull_vy))
    gait_activity = abs(hip1) + abs(knee1) + abs(hip2) + abs(knee2)
    contact_balance = 0.1 * (contact1 + contact2)
    stalled_penalty = -0.4 if forward_velocity < 0.05 and not terminated else 0.0
    fall_penalty = -{2.0 * fp} if terminated and abs(hull_angle) > 0.7 else 0.0
    gated_alive = {alive} if forward_velocity > 0.05 and abs(hull_angle) < 0.7 else 0.0
    components = {{"forward_velocity": forward_velocity, "stability": stability, "gait_activity": gait_activity, "contact_balance": contact_balance, "stalled_penalty": stalled_penalty, "fall_penalty": fall_penalty, "gated_alive": gated_alive}}
    total_reward = {pg} * forward_velocity + {sg} * stability + 0.02 * gait_activity + contact_balance + stalled_penalty + fall_penalty + gated_alive
    return float(total_reward), components
'''

    def _generic_reward_body(self, family: str, ctx: dict[str, Any]) -> str:
        fp = ctx["failure_pressure"]
        return f'''    terminated_penalty = -{fp} if terminated else 0.0
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
    components = {{"termination_penalty": terminated_penalty, "truncation_penalty": truncated_penalty, "action_smoothness": action_smoothness}}
    total_reward = termination_penalty + truncated_penalty + action_smoothness
    return float(total_reward), components
'''

    def _family_intended_fix(self, family: str) -> list[str]:
        if family == "local_repair":
            return ["make a conservative reward with clear components", "use analyzer context to gate known failure patterns"]
        if family == "component_recomposition":
            return ["use delta/progress-style components", "avoid pure survival reward", "recompose components flagged by analyzer"]
        return ["condition reward by progress or stability stage", "reduce low-progress reward hacking"]

    def _family_expected_effect(self, family: str) -> str:
        if family == "local_repair":
            return "Provide a simple, interpretable baseline candidate that reacts to known failures."
        if family == "component_recomposition":
            return "Separate progress, stability, and terminal terms based on analyzer intent."
        return "Handle stage-dependent task conflicts and discourage low-progress survival."

    def _family_risk(self, family: str) -> str:
        if family == "local_repair":
            return "May be too weak for difficult tasks."
        if family == "component_recomposition":
            return "Delta terms may be noisy if the progress proxy is imperfect."
        return "Stage gates may be imperfect before trajectory memory is available."
