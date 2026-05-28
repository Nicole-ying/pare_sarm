"""Long training runner for ASE-MTAGE Phase 3.

Phase 3 introduces a minimal but functional long-training path. It wraps a Gym /
Gymnasium environment so Stable-Baselines3 trains on the selected candidate
reward, then evaluates the trained policy and records trajectories plus reward
component logs.

If Stable-Baselines3 is not installed or training fails, the runner records a
clear failure report instead of silently pretending success.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ase_mtage.training.trajectory_collector import TrajectoryCollector
from ase_mtage.utils.io import ensure_dir, save_json, save_text


RewardFn = Callable[[Any, Any, Any, bool, bool, dict[str, Any]], tuple[float, dict[str, float]]]


@dataclass(slots=True)
class LongTrainingResult:
    success: bool
    selected_candidate_id: str
    reward_path: Path
    output_dir: Path
    model_path: Path | None
    eval_summary_path: Path | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "selected_candidate_id": self.selected_candidate_id,
            "reward_path": str(self.reward_path),
            "output_dir": str(self.output_dir),
            "model_path": str(self.model_path) if self.model_path else None,
            "eval_summary_path": str(self.eval_summary_path) if self.eval_summary_path else None,
            "error": self.error,
        }


class CandidateRewardWrapper:
    """Gym wrapper that replaces env reward with candidate reward.

    The official reward is not used for learning. It is still stored in the info
    dict as debug-only information because Gym step returns it, but the training
    algorithm receives only the candidate reward.
    """

    def __init__(self, env: Any, reward_fn: RewardFn) -> None:
        self.env = env
        self.reward_fn = reward_fn
        self._last_obs = None
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        self.spec = getattr(env, "spec", None)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, tuple):
            obs, info = result
            self._last_obs = obs
            return obs, info
        self._last_obs = result
        return result

    def step(self, action: Any) -> Any:
        step_result = self.env.step(action)
        if len(step_result) == 5:
            next_obs, env_reward, terminated, truncated, info = step_result
            info = dict(info or {})
            cand_reward, components = self._safe_reward(self._last_obs, action, next_obs, bool(terminated), bool(truncated), info)
            info["candidate_reward"] = cand_reward
            info["candidate_components"] = components
            info["env_reward_recorded_for_debug_only"] = float(env_reward)
            self._last_obs = next_obs
            return next_obs, cand_reward, terminated, truncated, info
        next_obs, env_reward, done, info = step_result
        info = dict(info or {})
        cand_reward, components = self._safe_reward(self._last_obs, action, next_obs, bool(done), False, info)
        info["candidate_reward"] = cand_reward
        info["candidate_components"] = components
        info["env_reward_recorded_for_debug_only"] = float(env_reward)
        self._last_obs = next_obs
        return next_obs, cand_reward, done, info

    def close(self) -> None:
        return self.env.close()

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def seed(self, seed: int | None = None) -> Any:
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

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


class LongTrainer:
    """Train one selected reward candidate and collect evaluation trajectories."""

    def __init__(
        self,
        *,
        env_id: str,
        reward_path: str | Path,
        output_dir: str | Path,
        selected_candidate_id: str,
        seed: int = 42,
        full_timesteps: int = 100_000,
        final_eval_episodes: int = 20,
        algorithm: str = "PPO",
    ) -> None:
        self.env_id = env_id
        self.reward_path = Path(reward_path)
        self.output_dir = ensure_dir(output_dir)
        self.selected_candidate_id = selected_candidate_id
        self.seed = seed
        self.full_timesteps = int(full_timesteps)
        self.final_eval_episodes = int(final_eval_episodes)
        self.algorithm = algorithm.upper()
        self.reward_fn = self._load_reward_fn(self.reward_path)

    def run(self) -> LongTrainingResult:
        """Run long training and trajectory collection."""
        save_json(
            self.output_dir / "training_config.json",
            {
                "env_id": self.env_id,
                "reward_path": str(self.reward_path),
                "selected_candidate_id": self.selected_candidate_id,
                "seed": self.seed,
                "full_timesteps": self.full_timesteps,
                "final_eval_episodes": self.final_eval_episodes,
                "algorithm": self.algorithm,
                "official_reward_used_for_training": False,
            },
        )

        try:
            model = self._train_model()
            model_path = self.output_dir / "model_final.zip"
            model.save(str(model_path))
            collector = TrajectoryCollector(
                env_id=self.env_id,
                reward_path=self.reward_path,
                output_dir=self.output_dir,
                seed=self.seed + 1000,
            )
            collection = collector.collect(model=model, num_episodes=self.final_eval_episodes, step_tag="final")
            result = LongTrainingResult(
                success=True,
                selected_candidate_id=self.selected_candidate_id,
                reward_path=self.reward_path,
                output_dir=self.output_dir,
                model_path=model_path,
                eval_summary_path=collection.eval_summary_path,
            )
            save_json(self.output_dir / "long_training_result.json", result.to_dict())
            return result
        except Exception as exc:
            error = str(exc)
            save_text(self.output_dir / "ERROR.txt", error + "\n")
            result = LongTrainingResult(
                success=False,
                selected_candidate_id=self.selected_candidate_id,
                reward_path=self.reward_path,
                output_dir=self.output_dir,
                model_path=None,
                eval_summary_path=None,
                error=error,
            )
            save_json(self.output_dir / "long_training_result.json", result.to_dict())
            return result

    def _train_model(self) -> Any:
        try:
            import gymnasium as gym  # type: ignore
        except Exception:
            try:
                import gym  # type: ignore
            except Exception as exc:
                raise RuntimeError("Install gymnasium or gym to run Phase 3 long training.") from exc

        try:
            from stable_baselines3 import PPO, A2C, SAC, TD3, DQN  # type: ignore
            from stable_baselines3.common.monitor import Monitor  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install stable-baselines3 to run Phase 3 long training.") from exc

        base_env = gym.make(self.env_id)
        wrapped_env = CandidateRewardWrapper(base_env, self.reward_fn)
        monitor_path = str(self.output_dir / "monitor.csv")
        env = Monitor(wrapped_env, filename=monitor_path)
        try:
            env.reset(seed=self.seed)
        except TypeError:
            env.seed(self.seed)

        algo_map = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3, "DQN": DQN}
        algo_cls = algo_map.get(self.algorithm)
        if algo_cls is None:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}. Use one of {sorted(algo_map)}")

        # Conservative defaults for smoke tests. Users can extend config later.
        model = algo_cls("MlpPolicy", env, verbose=0, seed=self.seed)
        model.learn(total_timesteps=self.full_timesteps)
        return model

    def _load_reward_fn(self, reward_path: Path) -> RewardFn:
        module_name = f"ase_mtage_train_reward_{abs(hash(str(reward_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, reward_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load reward module from {reward_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "compute_reward", None)
        if fn is None:
            raise AttributeError(f"{reward_path} does not define compute_reward")
        return fn
