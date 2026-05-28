"""Evidence Card Builder for ASE-MTAGE Phase 4.

This tool converts raw evaluation trajectory logs and component logs from Phase 3
into compact trajectory evidence cards. Cards are then finalized by the guarded
Trajectory Judge Agent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ase_mtage.agents.trajectory_judge import TrajectoryJudgeAgent
from ase_mtage.tools.rule_labeler import RuleLabeler
from ase_mtage.utils.io import ensure_dir, load_json, save_json


class EvidenceCardBuilder:
    """Build trajectory evidence cards from raw trajectory logs."""

    def __init__(self, *, env_id: str, confidence_threshold: float = 0.70) -> None:
        self.env_id = env_id
        self.rule_labeler = RuleLabeler()
        self.judge = TrajectoryJudgeAgent(confidence_threshold=confidence_threshold, llm_enabled=False)

    def build_from_training_dir(
        self,
        *,
        full_training_dir: str | Path,
        round_dir: str | Path,
        memory_dir: str | Path,
        source_round: int,
        source_reward_id: str | None,
    ) -> dict[str, Any]:
        """Build cards for all raw trajectories in a full_training directory."""
        full_training_dir = Path(full_training_dir)
        round_dir = Path(round_dir)
        memory_dir = ensure_dir(memory_dir)
        trajectory_dir = full_training_dir / "trajectory_logs"
        component_dir = full_training_dir / "component_logs"
        cards: list[dict[str, Any]] = []
        judgments: list[dict[str, Any]] = []

        for traj_path in sorted(trajectory_dir.glob("*.json")):
            trajectory = load_json(traj_path)
            trajectory_id = str(trajectory.get("trajectory_id", traj_path.stem))
            comp_path = component_dir / f"{trajectory_id}_components.json"
            component_record = load_json(comp_path, default={})
            card = self.build_card(
                trajectory=trajectory,
                component_record=component_record,
                source_round=source_round,
                source_reward_id=source_reward_id,
                trajectory_path=traj_path,
                component_path=comp_path if comp_path.exists() else None,
            )
            judgment = self.judge.judge(card).to_dict()
            card["llm_label"] = None
            card["final_label"] = judgment["final_label"]
            card["use_for_tage_pair"] = judgment["use_for_tage_pair"]
            card["allowed_preference_role"] = judgment["allowed_preference_role"]
            cards.append(card)
            judgments.append(judgment)

        round_cards_path = round_dir / "trajectory_cards.jsonl"
        round_judgment_path = round_dir / "trajectory_judgment.jsonl"
        memory_cards_path = memory_dir / "trajectory_cards.jsonl"
        self._write_jsonl(round_cards_path, cards)
        self._write_jsonl(round_judgment_path, judgments)
        self._append_jsonl(memory_cards_path, cards)

        summary = self._summarize(cards, judgments)
        save_json(round_dir / "trajectory_judgment_summary.json", summary)
        return {
            "num_cards": len(cards),
            "round_cards_path": str(round_cards_path),
            "round_judgment_path": str(round_judgment_path),
            "memory_cards_path": str(memory_cards_path),
            "summary": summary,
        }

    def build_card(
        self,
        *,
        trajectory: dict[str, Any],
        component_record: dict[str, Any],
        source_round: int,
        source_reward_id: str | None,
        trajectory_path: Path,
        component_path: Path | None,
    ) -> dict[str, Any]:
        steps = list(trajectory.get("steps") or [])
        episode_length = int(trajectory.get("episode_length", len(steps)) or 0)
        max_steps = self._infer_max_episode_steps(episode_length, steps)
        final_step = steps[-1] if steps else {}
        first_obs = steps[0].get("obs") if steps else None
        final_obs = trajectory.get("final_obs", final_step.get("next_obs"))
        terminated = bool(final_step.get("terminated", False)) if final_step else False
        truncated = bool(final_step.get("truncated", False)) if final_step else False
        features = self._extract_features(first_obs=first_obs, final_obs=final_obs, steps=steps, episode_length=episode_length)
        episode = {
            "length": episode_length,
            "max_episode_steps": max_steps,
            "terminated": terminated,
            "truncated": truncated,
            "terminal_event_from_env": "unknown",
        }
        rule_label = self.rule_labeler.label(env_id=self.env_id, episode=episode, features=features).to_dict()
        component_totals = dict(component_record.get("component_totals") or trajectory.get("component_totals") or {})
        return {
            "trajectory_id": str(trajectory.get("trajectory_id", trajectory_path.stem)),
            "source_round": source_round,
            "source_reward_id": source_reward_id,
            "source_reward_path": trajectory.get("reward_path"),
            "policy_checkpoint": str(Path(trajectory_path).parent.parent / "model_final.zip"),
            "trajectory_path": str(trajectory_path),
            "component_path": str(component_path) if component_path else None,
            "episode": episode,
            "features": features,
            "reward_component_totals": component_totals,
            "candidate_return": trajectory.get("candidate_return"),
            "env_return_recorded_for_debug_only": trajectory.get("env_return_recorded_for_debug_only"),
            "rule_label": rule_label,
            "llm_label": None,
            "final_label": None,
            "use_for_tage_pair": None,
        }

    def _extract_features(self, *, first_obs: Any, final_obs: Any, steps: list[dict[str, Any]], episode_length: int) -> dict[str, Any]:
        env = self.env_id.lower()
        if "lunarlander" in env:
            return self._extract_lunarlander_features(first_obs, final_obs, steps)
        if "cartpole" in env:
            return self._extract_cartpole_features(final_obs, steps)
        if "bipedalwalker" in env:
            return self._extract_bipedalwalker_features(first_obs, final_obs, steps)
        return self._extract_generic_features(first_obs, final_obs, steps, episode_length)

    def _extract_lunarlander_features(self, first_obs: Any, final_obs: Any, steps: list[dict[str, Any]]) -> dict[str, Any]:
        first = self._list(first_obs)
        final = self._list(final_obs)
        all_next = [self._list(s.get("next_obs")) for s in steps]
        initial_distance = self._dist2(first, 0, 1)
        final_distance = self._dist2(final, 0, 1)
        min_distance = min([self._dist2(o, 0, 1) for o in all_next] or [final_distance])
        vx = self._at(final, 2)
        vy = self._at(final, 3)
        final_speed = math.sqrt(vx * vx + vy * vy)
        final_angle = abs(self._at(final, 4))
        last20 = all_next[-20:] if all_next else []
        if last20:
            contacts = [self._at(o, 6) + self._at(o, 7) for o in last20]
            contact_ratio = sum(1.0 for c in contacts if c > 0.5) / len(last20)
        else:
            contact_ratio = 0.0
        return {
            "initial_distance_to_target": initial_distance,
            "min_distance_to_target": min_distance,
            "final_distance_to_target": final_distance,
            "distance_improvement": initial_distance - final_distance,
            "final_speed": final_speed,
            "final_vertical_speed_abs": abs(vy),
            "final_horizontal_speed_abs": abs(vx),
            "final_angle_abs": final_angle,
            "contact_ratio_last20": contact_ratio,
            "progress_improvement": initial_distance - final_distance,
        }

    def _extract_cartpole_features(self, final_obs: Any, steps: list[dict[str, Any]]) -> dict[str, Any]:
        final = self._list(final_obs)
        angles = [abs(self._at(self._list(s.get("next_obs")), 2)) for s in steps]
        return {
            "final_position_abs": abs(self._at(final, 0)),
            "final_angle_abs": abs(self._at(final, 2)),
            "max_angle_abs": max(angles) if angles else abs(self._at(final, 2)),
            "progress_improvement": float(len(steps)),
        }

    def _extract_bipedalwalker_features(self, first_obs: Any, final_obs: Any, steps: list[dict[str, Any]]) -> dict[str, Any]:
        # BipedalWalker observations do not directly expose x position in the standard observation.
        # Keep conservative generic stability features until an env adapter is added.
        final = self._list(final_obs)
        return {
            "forward_displacement": 0.0,
            "final_height": self._at(final, 0),
            "final_angle_abs": abs(self._at(final, 1)),
            "progress_improvement": 0.0,
        }

    def _extract_generic_features(self, first_obs: Any, final_obs: Any, steps: list[dict[str, Any]], episode_length: int) -> dict[str, Any]:
        return {
            "episode_length": episode_length,
            "progress_improvement": 0.0,
            "final_state_summary": self._list(final_obs),
        }

    def _infer_max_episode_steps(self, episode_length: int, steps: list[dict[str, Any]]) -> int:
        # Gym wrappers do not always expose max_episode_steps in saved logs; use conservative defaults.
        if episode_length >= 1000:
            return episode_length
        if "cartpole" in self.env_id.lower():
            return 500
        if "bipedalwalker" in self.env_id.lower():
            return 1600
        return 1000

    def _summarize(self, cards: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        usable = 0
        for card in cards:
            label = ((card.get("final_label") or {}).get("coarse_label")) or "ambiguous"
            counts[label] = counts.get(label, 0) + 1
            if card.get("use_for_tage_pair"):
                usable += 1
        return {
            "num_trajectories": len(cards),
            "num_judgments": len(judgments),
            "num_use_for_tage_pair": usable,
            "label_counts": counts,
            "judge_mode": "phase_4_guarded_rule_first",
        }

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(self._dumps(row) + "\n")

    def _append_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(self._dumps(row) + "\n")

    def _dumps(self, row: dict[str, Any]) -> str:
        import json
        return json.dumps(row, ensure_ascii=False)

    def _list(self, obs: Any) -> list[Any]:
        if obs is None:
            return []
        if isinstance(obs, list):
            return obs
        if isinstance(obs, tuple):
            return list(obs)
        try:
            import numpy as np  # type: ignore
            if isinstance(obs, np.ndarray):
                return obs.tolist()
        except Exception:
            pass
        return [obs]

    def _at(self, values: list[Any], idx: int, default: float = 0.0) -> float:
        try:
            return float(values[idx])
        except Exception:
            return default

    def _dist2(self, values: list[Any], i: int, j: int) -> float:
        x = self._at(values, i)
        y = self._at(values, j)
        return math.sqrt(x * x + y * y)
