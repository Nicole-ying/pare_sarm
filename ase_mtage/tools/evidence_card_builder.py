"""Build trajectory evidence cards for ASE-MTAGE."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ase_mtage.agents.trajectory_judge import TrajectoryJudgeAgent
from ase_mtage.llm_client import LLMClient
from ase_mtage.utils.io import ensure_dir, load_json, load_jsonl, save_json


class EvidenceCardBuilder:
    def __init__(
        self,
        *,
        env_id: str,
        llm_client: LLMClient | None = None,
        judge_temperature: float = 0.2,
        task_manifest: str | None = None,
        env_manifest: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        fallback_on_error: bool = True,
        batch_size: int = 10,
    ) -> None:
        self.env_id = env_id
        self.judge = TrajectoryJudgeAgent(
            llm_client=llm_client,
            temperature=judge_temperature,
            output_dir=output_dir,
            task_manifest=task_manifest,
            env_manifest=env_manifest,
            fallback_on_error=fallback_on_error,
            batch_size=batch_size,
        )

    def build_from_training_dir(self, *, full_training_dir: str | Path, round_dir: str | Path, memory_dir: str | Path, source_round: int, source_reward_id: str | None) -> dict[str, Any]:
        from ase_mtage.utils.logger import get_logger

        log = get_logger()
        full_training_dir = Path(full_training_dir)
        round_dir = Path(round_dir)
        memory_dir = ensure_dir(memory_dir)
        trajectory_dir = full_training_dir / "trajectory_logs"
        component_dir = full_training_dir / "component_logs"
        traj_files = sorted(trajectory_dir.glob("*.json"))
        total_traj = len(traj_files)

        existing_cards = load_jsonl(round_dir / "trajectory_cards.jsonl")
        existing_ids = {c.get("trajectory_id") for c in existing_cards}
        existing_judgments = load_jsonl(round_dir / "trajectory_judgment.jsonl")

        cards: list[dict[str, Any]] = list(existing_cards)
        judgments: list[dict[str, Any]] = list(existing_judgments)

        # Build cards for new trajectories
        new_cards: list[dict[str, Any]] = []
        for traj_path in traj_files:
            trajectory = load_json(traj_path)
            trajectory_id = str(trajectory.get("trajectory_id", traj_path.stem))
            if trajectory_id in existing_ids:
                continue
            comp_path = component_dir / f"{trajectory_id}_components.json"
            component_record = load_json(comp_path, default={})
            card = self._build_card(
                trajectory=trajectory,
                component_record=component_record,
                source_round=source_round,
                source_reward_id=source_reward_id,
                trajectory_path=traj_path,
                component_path=comp_path if comp_path.exists() else None,
            )
            new_cards.append(card)
            cards.append(card)

        # Batch judge all new cards
        if new_cards:
            log.info(f"Judging {len(new_cards)} new trajectories in batches of {self.judge.batch_size}...")
            batch_judgments = self.judge.judge_batch(new_cards)
            for card, judgment in zip(new_cards, batch_judgments):
                jdict = self.judge.judgment_to_dict(judgment)
                card["coarse_label"] = jdict["coarse_label"]
                card["use_for_tage_pair"] = jdict["use_for_tage_pair"]
                card["allowed_preference_role"] = jdict["allowed_preference_role"]
                # Update existing judgment if present, otherwise append
                existing_idx = next((idx for idx, ej in enumerate(judgments) if ej.get("trajectory_id") == judgment.trajectory_id), None)
                if existing_idx is not None:
                    judgments[existing_idx] = jdict
                else:
                    judgments.append(jdict)

            # Write after each batch completes
            self._write_jsonl(round_dir / "trajectory_cards.jsonl", cards)
            self._write_jsonl(round_dir / "trajectory_judgment.jsonl", judgments)

        round_cards_path = round_dir / "trajectory_cards.jsonl"
        round_judgment_path = round_dir / "trajectory_judgment.jsonl"
        memory_cards_path = memory_dir / "trajectory_cards.jsonl"
        self._write_jsonl(round_cards_path, cards)
        self._write_jsonl(round_judgment_path, judgments)

        # Only non-ambiguous, tage-usable cards enter memory
        memory_cards = [c for c in cards if c.get("use_for_tage_pair") and c.get("coarse_label") != "ambiguous"]
        self._append_jsonl(memory_cards_path, memory_cards)

        summary = self._summarize(cards)
        save_json(round_dir / "trajectory_judgment_summary.json", summary)

        log.info(f"Round {source_round}: {len(cards)} trajectory cards built | labels={summary.get('label_counts', {})}")
        return {
            "num_cards": len(cards),
            "round_cards_path": str(round_cards_path),
            "round_judgment_path": str(round_judgment_path),
            "memory_cards_path": str(memory_cards_path),
            "summary": summary,
        }

    def _build_card(self, *, trajectory: dict[str, Any], component_record: dict[str, Any], source_round: int, source_reward_id: str | None, trajectory_path: Path, component_path: Path | None) -> dict[str, Any]:
        steps = list(trajectory.get("steps") or [])
        episode_length = int(trajectory.get("episode_length", len(steps)) or 0)
        final_step = steps[-1] if steps else {}
        first_obs = steps[0].get("obs") if steps else None
        final_obs = trajectory.get("final_obs", final_step.get("next_obs"))
        episode = {
            "length": episode_length,
            "max_episode_steps": self._infer_max_episode_steps(episode_length),
            "terminated": bool(final_step.get("terminated", False)) if final_step else False,
            "truncated": bool(final_step.get("truncated", False)) if final_step else False,
        }
        features = self._extract_features(first_obs=first_obs, final_obs=final_obs, steps=steps, episode_length=episode_length)
        return {
            "trajectory_id": str(trajectory.get("trajectory_id", trajectory_path.stem)),
            "source_round": source_round,
            "source_reward_id": source_reward_id,
            "source_reward_path": trajectory.get("reward_path"),
            "trajectory_path": str(trajectory_path),
            "component_path": str(component_path) if component_path else None,
            "episode": episode,
            "features": features,
            "reward_component_totals": dict(component_record.get("component_totals") or trajectory.get("component_totals") or {}),
            "candidate_return": trajectory.get("candidate_return"),
            "env_return_recorded_for_debug_only": trajectory.get("env_return_recorded_for_debug_only"),
            "coarse_label": None,
            "use_for_tage_pair": None,
            "allowed_preference_role": None,
        }

    def _extract_features(self, *, first_obs: Any, final_obs: Any, steps: list[dict[str, Any]], episode_length: int) -> dict[str, Any]:
        env = self.env_id.lower()
        if "lunarlander" in env:
            return self._extract_lunarlander(first_obs, final_obs, steps)
        if "cartpole" in env:
            final = self._list(final_obs)
            angles = [abs(self._at(self._list(s.get("next_obs")), 2)) for s in steps]
            return {"final_position_abs": abs(self._at(final, 0)), "final_angle_abs": abs(self._at(final, 2)), "max_angle_abs": max(angles) if angles else abs(self._at(final, 2)), "progress_improvement": float(len(steps))}
        if "bipedalwalker" in env:
            return self._extract_bipedal(final_obs, steps)
        return {"episode_length": episode_length, "progress_improvement": 0.0, "final_state_summary": self._list(final_obs)}

    def _extract_lunarlander(self, first_obs: Any, final_obs: Any, steps: list[dict[str, Any]]) -> dict[str, Any]:
        first, final = self._list(first_obs), self._list(final_obs)
        all_next = [self._list(s.get("next_obs")) for s in steps]
        initial_distance = self._dist2(first, 0, 1)
        final_distance = self._dist2(final, 0, 1)
        min_distance = min([self._dist2(o, 0, 1) for o in all_next] or [final_distance])
        vx, vy = self._at(final, 2), self._at(final, 3)
        last20 = all_next[-20:] if all_next else []
        contact_ratio = 0.0
        if last20:
            contact_ratio = sum(1.0 for o in last20 if self._at(o, 6) + self._at(o, 7) > 0.5) / len(last20)
        return {"initial_distance_to_target": initial_distance, "min_distance_to_target": min_distance, "final_distance_to_target": final_distance, "distance_improvement": initial_distance - final_distance, "final_speed": math.sqrt(vx * vx + vy * vy), "final_vertical_speed_abs": abs(vy), "final_horizontal_speed_abs": abs(vx), "final_angle_abs": abs(self._at(final, 4)), "contact_ratio_last20": contact_ratio, "progress_improvement": initial_distance - final_distance}

    def _extract_bipedal(self, final_obs: Any, steps: list[dict[str, Any]]) -> dict[str, Any]:
        final = self._list(final_obs)
        all_next = [self._list(s.get("next_obs")) for s in steps]
        hull_angle, hull_ang_vel = self._at(final, 0), self._at(final, 1)
        hull_vx, hull_vy = self._at(final, 2), self._at(final, 3)
        hip1, knee1, hip2, knee2 = self._at(final, 4), self._at(final, 6), self._at(final, 9), self._at(final, 11)
        contact_ratio = sum(1.0 for o in all_next if max(self._at(o, 8), self._at(o, 13)) > 0.5) / max(1, len(all_next))
        mean_vx = sum(self._at(o, 2) for o in all_next) / max(1, len(all_next))
        estimated_progress = mean_vx * max(1, len(all_next))
        q = max(1, len(all_next) // 4)
        early_vx = sum(self._at(o, 2) for o in all_next[:q]) / max(1, len(all_next[:q])) if all_next else 0.0
        late_vx = sum(self._at(o, 2) for o in all_next[-q:]) / max(1, len(all_next[-q:])) if all_next else 0.0
        return {"forward_displacement_proxy": estimated_progress, "estimated_forward_progress": estimated_progress, "mean_forward_velocity": mean_vx, "final_forward_velocity": hull_vx, "velocity_improvement": late_vx - early_vx, "final_angle_abs": abs(hull_angle), "final_angular_velocity_abs": abs(hull_ang_vel), "vertical_velocity_abs": abs(hull_vy), "contact_ratio": contact_ratio, "gait_activity": abs(hip1) + abs(knee1) + abs(hip2) + abs(knee2), "stability_penalty": abs(hull_angle) + 0.5 * abs(hull_ang_vel) + 0.2 * abs(hull_vy), "progress_improvement": estimated_progress}

    def _infer_max_episode_steps(self, episode_length: int) -> int:
        if episode_length >= 1000:
            return episode_length
        if "cartpole" in self.env_id.lower():
            return 500
        if "bipedalwalker" in self.env_id.lower():
            return 1600
        return 1000

    def _summarize(self, cards: list[dict[str, Any]]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        usable = 0
        for card in cards:
            label = card.get("coarse_label") or "ambiguous"
            counts[label] = counts.get(label, 0) + 1
            if card.get("use_for_tage_pair"):
                usable += 1
        return {"num_trajectories": len(cards), "num_use_for_tage_pair": usable, "label_counts": counts}

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _list(self, obs: Any) -> list[Any]:
        if obs is None:
            return []
        if isinstance(obs, list):
            return obs
        if isinstance(obs, tuple):
            return list(obs)
        try:
            import numpy as np
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
        x, y = self._at(values, i), self._at(values, j)
        return math.sqrt(x * x + y * y)
