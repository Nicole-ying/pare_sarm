"""Trajectory collection for ASE-MTAGE Phase 3.

The collector evaluates a trained policy under a candidate reward function and
saves raw trajectory logs, reward component logs, and an evaluation summary. The
recorded trajectories are later converted into Evidence Cards in Phase 4.
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from ase_mtage.utils.io import ensure_dir, save_json


RewardFn = Callable[[Any, Any, Any, bool, bool, dict[str, Any]], tuple[float, dict[str, float]]]


@dataclass(slots=True)
class TrajectoryCollectionResult:
    num_episodes: int
    trajectory_dir: Path
    component_log_dir: Path
    eval_summary_path: Path
    mean_candidate_return: float
    mean_episode_length: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_episodes": self.num_episodes,
            "trajectory_dir": str(self.trajectory_dir),
            "component_log_dir": str(self.component_log_dir),
            "eval_summary_path": str(self.eval_summary_path),
            "mean_candidate_return": self.mean_candidate_return,
            "mean_episode_length": self.mean_episode_length,
        }


class TrajectoryCollector:
    """Collect evaluation trajectories and candidate reward component logs."""

    def __init__(self, *, env_id: str, reward_path: str | Path, output_dir: str | Path, seed: int = 42) -> None:
        self.env_id = env_id
        self.reward_path = Path(reward_path)
        self.output_dir = ensure_dir(output_dir)
        self.seed = seed
        self.reward_fn = self._load_reward_fn(self.reward_path)

    def collect(self, *, model: Any | None, num_episodes: int, step_tag: str = "final") -> TrajectoryCollectionResult:
        """Collect trajectories using a model or random actions when model is None."""
        trajectory_dir = ensure_dir(self.output_dir / "trajectory_logs")
        component_log_dir = ensure_dir(self.output_dir / "component_logs")
        env = self._make_env()

        episode_summaries: list[dict[str, Any]] = []
        all_candidate_returns: list[float] = []
        all_lengths: list[int] = []

        for ep_idx in range(num_episodes):
            reset_result = env.reset(seed=self.seed + ep_idx)
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            done = False
            ep_len = 0
            candidate_return = 0.0
            env_return = 0.0
            states: list[Any] = []
            actions: list[Any] = []
            steps: list[dict[str, Any]] = []
            component_totals: dict[str, float] = {}

            while not done:
                action = self._predict_action(model, env, obs)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, env_reward, terminated, truncated, info = step_result
                else:  # old gym API
                    next_obs, env_reward, done_old, info = step_result
                    terminated = bool(done_old)
                    truncated = False
                info = dict(info or {})
                done = bool(terminated or truncated)

                cand_reward, components = self._safe_reward(obs, action, next_obs, bool(terminated), bool(truncated), info)
                candidate_return += cand_reward
                env_return += float(env_reward)
                for name, value in components.items():
                    component_totals[name] = component_totals.get(name, 0.0) + float(value)

                states.append(self._to_jsonable(obs))
                actions.append(self._to_jsonable(action))
                steps.append(
                    {
                        "t": ep_len,
                        "obs": self._to_jsonable(obs),
                        "action": self._to_jsonable(action),
                        "next_obs": self._to_jsonable(next_obs),
                        "env_reward_recorded_for_debug_only": float(env_reward),
                        "candidate_reward": cand_reward,
                        "components": components,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": self._to_jsonable(info),
                    }
                )
                obs = next_obs
                ep_len += 1

            trajectory_id = f"{step_tag}_ep{ep_idx:03d}"
            traj_path = trajectory_dir / f"{trajectory_id}.json"
            comp_path = component_log_dir / f"{trajectory_id}_components.json"
            final_obs = steps[-1]["next_obs"] if steps else self._to_jsonable(obs)
            traj_record = {
                "trajectory_id": trajectory_id,
                "env_id": self.env_id,
                "reward_path": str(self.reward_path),
                "step_tag": step_tag,
                "episode_length": ep_len,
                "candidate_return": candidate_return,
                "env_return_recorded_for_debug_only": env_return,
                "states": states,
                "actions": actions,
                "final_obs": final_obs,
                "steps": steps,
            }
            save_json(traj_path, traj_record)
            save_json(
                comp_path,
                {
                    "trajectory_id": trajectory_id,
                    "component_totals": component_totals,
                    "candidate_return": candidate_return,
                    "episode_length": ep_len,
                },
            )
            all_candidate_returns.append(candidate_return)
            all_lengths.append(ep_len)
            episode_summaries.append(
                {
                    "trajectory_id": trajectory_id,
                    "trajectory_path": str(traj_path),
                    "component_path": str(comp_path),
                    "episode_length": ep_len,
                    "candidate_return": candidate_return,
                    "env_return_recorded_for_debug_only": env_return,
                    "component_totals": component_totals,
                    "final_obs": final_obs,
                }
            )

        try:
            env.close()
        except Exception:
            pass

        mean_return = sum(all_candidate_returns) / max(1, len(all_candidate_returns))
        mean_len = sum(all_lengths) / max(1, len(all_lengths))
        eval_summary = {
            "env_id": self.env_id,
            "reward_path": str(self.reward_path),
            "num_episodes": num_episodes,
            "step_tag": step_tag,
            "mean_candidate_return": mean_return,
            "mean_episode_length": mean_len,
            "episodes": episode_summaries,
            "note": "env_return is recorded only for debugging and must not be used as official reward feedback.",
        }
        eval_summary_path = save_json(self.output_dir / "eval_summary.json", eval_summary)
        return TrajectoryCollectionResult(
            num_episodes=num_episodes,
            trajectory_dir=trajectory_dir,
            component_log_dir=component_log_dir,
            eval_summary_path=eval_summary_path,
            mean_candidate_return=mean_return,
            mean_episode_length=mean_len,
        )

    def _make_env(self) -> Any:
        try:
            import gymnasium as gym  # type: ignore
        except Exception:
            try:
                import gym  # type: ignore
            except Exception as exc:
                raise RuntimeError("Install gymnasium or gym to run Phase 3 long training.") from exc
        return gym.make(self.env_id)

    def _predict_action(self, model: Any | None, env: Any, obs: Any) -> Any:
        if model is None:
            return env.action_space.sample()
        action, _ = model.predict(obs, deterministic=True)
        return action

    def _safe_reward(self, obs: Any, action: Any, next_obs: Any, terminated: bool, truncated: bool, info: dict[str, Any]) -> tuple[float, dict[str, float]]:
        total, components = self.reward_fn(obs, action, next_obs, terminated, truncated, info)
        total_f = float(total)
        if not math.isfinite(total_f):
            total_f = 0.0
        clean_components: dict[str, float] = {}
        for key, value in dict(components).items():
            try:
                v = float(value)
            except Exception:
                v = 0.0
            if not math.isfinite(v):
                v = 0.0
            clean_components[str(key)] = v
        return total_f, clean_components

    def _load_reward_fn(self, reward_path: Path) -> RewardFn:
        module_name = f"ase_mtage_runtime_reward_{abs(hash(str(reward_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, reward_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load reward module from {reward_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "compute_reward", None)
        if fn is None:
            raise AttributeError(f"{reward_path} does not define compute_reward")
        return fn

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        try:
            import numpy as np  # type: ignore
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return float(value)
        except Exception:
            return str(value)
