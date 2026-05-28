"""Environment perception agent for ASE-MTAGE Phase 2.

This phase implements a deterministic, reward-safe placeholder that writes the
Core Memory artifacts expected by later agents. A real LLM-backed perception
agent can replace `build_env_manifest` later, but the output schema should remain
stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase_mtage.utils.io import ensure_dir, save_json, save_text


COARSE_OUTCOME_LABELS = [
    "early_failure",
    "low_progress_survival",
    "partial_progress",
    "success_like",
    "ambiguous",
]


@dataclass(slots=True)
class EnvPerceptionArtifacts:
    task_manifest_path: Path
    env_manifest_path: Path
    outcome_label_schema_path: Path
    feature_schema_path: Path


class EnvPerceptionAgent:
    """Create sanitized task/env manifests without exposing official reward code."""

    def __init__(self, output_core_dir: str | Path) -> None:
        self.output_core_dir = ensure_dir(output_core_dir)

    def run(self, *, env_id: str, task_description: str | None = None) -> EnvPerceptionArtifacts:
        """Write phase-2 core memory files for an environment."""
        task_goal = task_description or self._default_task_goal(env_id)
        manifest = self.build_env_manifest(env_id=env_id, task_goal=task_goal)

        task_manifest_text = self._build_task_manifest_md(manifest)
        task_manifest_path = save_text(self.output_core_dir / "task_manifest.md", task_manifest_text)
        env_manifest_path = save_json(self.output_core_dir / "env_manifest.json", manifest)
        outcome_label_schema_path = save_json(
            self.output_core_dir / "outcome_label_schema.json",
            {
                "coarse_outcome_labels": COARSE_OUTCOME_LABELS,
                "label_definitions": {
                    "early_failure": "Episode ends early with little useful progress.",
                    "low_progress_survival": "Episode lasts but shows little task progress.",
                    "partial_progress": "Trajectory shows real progress but does not reach stable success.",
                    "success_like": "Trajectory strongly matches the task goal according to observable evidence.",
                    "ambiguous": "Evidence is insufficient or conflicting.",
                },
                "official_reward_visible": False,
            },
        )
        feature_schema_path = save_json(
            self.output_core_dir / "feature_schema.json",
            {
                "trajectory_features_to_extract": manifest["trajectory_features_to_extract"],
                "notes": "Phase 2 generic feature schema; environment adapters can specialize this later.",
            },
        )
        return EnvPerceptionArtifacts(
            task_manifest_path=task_manifest_path,
            env_manifest_path=env_manifest_path,
            outcome_label_schema_path=outcome_label_schema_path,
            feature_schema_path=feature_schema_path,
        )

    def build_env_manifest(self, *, env_id: str, task_goal: str) -> dict[str, Any]:
        """Return a sanitized env manifest.

        The manifest intentionally contains no official reward formula. It only
        records task-level semantics and observable trajectory features.
        """
        obs_schema = self._generic_observation_schema(env_id)
        action_schema = self._generic_action_schema(env_id)
        return {
            "env_name": env_id,
            "task_goal": task_goal,
            "official_reward_visible": False,
            "observation_schema": obs_schema,
            "action_schema": action_schema,
            "termination_signals": [
                "terminated indicates an environment terminal event; classify using observable final-state evidence.",
                "truncated indicates a time-limit or max-episode-step cutoff.",
            ],
            "available_info_keys": [],
            "trajectory_features_to_extract": [
                "episode_length",
                "max_episode_steps",
                "terminated",
                "truncated",
                "initial_progress_proxy",
                "final_progress_proxy",
                "progress_improvement",
                "final_state_summary",
                "action_usage_summary",
                "reward_component_totals",
            ],
            "coarse_outcome_labels": COARSE_OUTCOME_LABELS,
            "labeling_cautions": [
                "Do not use or infer the official environment reward.",
                "Do not treat long episode length alone as success_like.",
                "Do not treat high progress with unstable terminal evidence as success_like.",
                "Use ambiguous when observable evidence is insufficient or conflicting.",
            ],
        }

    def _default_task_goal(self, env_id: str) -> str:
        name = env_id.lower()
        if "lunarlander" in name:
            return "land safely on the landing pad using observable state evidence, without using the official reward"
        if "bipedalwalker" in name:
            return "move forward with a stable gait while avoiding falls, without using the official reward"
        if "cartpole" in name:
            return "keep the pole balanced for as long as possible, without using the official reward"
        return "complete the task using observable state and outcome evidence, without using the official reward"

    def _generic_observation_schema(self, env_id: str) -> list[dict[str, Any]]:
        name = env_id.lower()
        if "lunarlander" in name:
            names = [
                "x_position",
                "y_position",
                "x_velocity",
                "y_velocity",
                "angle",
                "angular_velocity",
                "left_leg_contact",
                "right_leg_contact",
            ]
            return [
                {"index": i, "name": n, "meaning": n.replace("_", " "), "used_for_labeling": True}
                for i, n in enumerate(names)
            ]
        if "cartpole" in name:
            names = ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]
            return [
                {"index": i, "name": n, "meaning": n.replace("_", " "), "used_for_labeling": True}
                for i, n in enumerate(names)
            ]
        return [
            {
                "index": "unknown",
                "name": "observation_vector",
                "meaning": "environment observation vector; specialize via env adapter in later phases",
                "used_for_labeling": True,
            }
        ]

    def _generic_action_schema(self, env_id: str) -> list[dict[str, Any]]:
        name = env_id.lower()
        if "lunarlander" in name:
            meanings = ["do nothing", "fire left engine", "fire main engine", "fire right engine"]
            return [{"id": i, "meaning": m} for i, m in enumerate(meanings)]
        if "cartpole" in name:
            return [{"id": 0, "meaning": "push cart left"}, {"id": 1, "meaning": "push cart right"}]
        return [{"id": "unknown", "meaning": "environment action; specialize via env adapter in later phases"}]

    def _build_task_manifest_md(self, manifest: dict[str, Any]) -> str:
        labels = "\n".join(f"- `{x}`" for x in manifest["coarse_outcome_labels"])
        features = "\n".join(f"- `{x}`" for x in manifest["trajectory_features_to_extract"])
        cautions = "\n".join(f"- {x}" for x in manifest["labeling_cautions"])
        return f"""# Task Manifest

## Environment

`{manifest['env_name']}`

## Task Goal

{manifest['task_goal']}

## Reward Leakage Policy

Official environment reward is not visible and must not be used.

## Coarse Outcome Labels

{labels}

## Trajectory Features To Extract

{features}

## Labeling Cautions

{cautions}
"""
