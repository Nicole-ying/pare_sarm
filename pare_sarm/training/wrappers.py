"""Environment wrappers: episode tracking, component logging, pickle safety,
and per-step trajectory logging with progress values for PARE diagnosis.
"""

import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import gymnasium as gym


class EpisodeInfoWrapper(gym.Wrapper):
    """Captures end-of-episode state into the info dict before auto-reset."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["_episode_terminated"] = terminated
        info["_episode_truncated"] = truncated
        if terminated or truncated:
            info["_episode_end"] = True
        return obs, reward, terminated, truncated, info


class ComponentTrackerWrapper(gym.Wrapper):
    """Accumulates reward components per episode and writes summaries to JSONL.

    Reads from info["reward_components"] set by the env.
    """

    def __init__(self, env, log_path: Path):
        super().__init__(env)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._step_count = 0
        self._components: dict[str, list] = {}
        self._env_metrics: dict[str, list] = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        for name, value in info.get("reward_components", {}).items():
            self._components.setdefault(name, []).append(float(value))

        for name, value in info.get("env_metrics", {}).items():
            self._env_metrics.setdefault(name, []).append(float(value))

        if terminated or truncated:
            self._save_episode()
            self._episode += 1
            self._reset_accumulators()

        return obs, reward, terminated, truncated, info

    def _save_episode(self):
        record = {
            "episode": self._episode,
            "length": self._step_count,
            "component_means": {
                k: round(float(np.mean(v)), 6) for k, v in self._components.items()
            },
            "component_stds": {
                k: round(float(np.std(v)), 6) for k, v in self._components.items()
            },
        }
        if self._env_metrics:
            record["env_metrics_means"] = {
                k: round(float(np.mean(v)), 6) for k, v in self._env_metrics.items()
            }
            record["env_metrics_stds"] = {
                k: round(float(np.std(v)), 6) for k, v in self._env_metrics.items()
            }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class StepLoggerWrapper(gym.Wrapper):
    """Logs PER-STEP data for PARE progress-aligned component diagnosis.

    This is the key wrapper that enables the core innovation:
      P_i = corr(component_i_step_values, progress_delta_step_values)

    For each env step, records:
      - episode, step index within episode
      - action taken
      - done flag (terminated or truncated)
      - total_reward (scalar returned by compute_reward)
      - components (dict from info["reward_components"])
      - progress (progress_fn(prev_obs))
      - next_progress (progress_fn(obs))
      - progress_delta (next_progress - progress)
      - failure flag (episode ended in termination, not truncation)

    The progress_fn is the LLM-generated task progress proxy.
    It must accept a numpy observation array and return a float.
    """

    def __init__(self, env, log_path: Path, progress_fn: Callable | None = None):
        super().__init__(env)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._progress_fn = progress_fn
        self._episode = 0
        self._step = 0
        self._prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = np.asarray(obs, dtype=np.float32)
        self._episode += 1
        self._step = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute progress from observation using LLM-generated progress_fn
        progress = None
        next_progress = None
        progress_delta = None
        if self._progress_fn is not None and self._prev_obs is not None:
            try:
                progress = float(self._progress_fn(self._prev_obs))
                next_progress = float(self._progress_fn(np.asarray(obs, dtype=np.float32)))
                progress_delta = next_progress - progress
            except Exception:
                pass

        # Extract action as a simple value
        try:
            action_val = int(action)
        except (TypeError, ValueError):
            action_val = float(action) if hasattr(action, '__float__') else str(action)

        # Extract components from info dict
        components = {}
        for k, v in info.get("reward_components", {}).items():
            try:
                components[k] = float(v)
            except (TypeError, ValueError):
                components[k] = 0.0

        record = {
            "episode": self._episode,
            "step": self._step,
            "action": action_val,
            "done": bool(terminated or truncated),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "total_reward": float(reward),
            "components": components,
            "progress": progress,
            "next_progress": next_progress,
            "progress_delta": progress_delta,
        }

        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self._prev_obs = np.asarray(obs, dtype=np.float32)
        self._step += 1
        return obs, reward, terminated, truncated, info


class PickleSafeInfoWrapper(gym.Wrapper):
    """Strip non-picklable items from info dict for SubprocVecEnv safety."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = _sanitize_pickle(info)
        return obs, reward, terminated, truncated, info


def _sanitize_pickle(obj, depth=0):
    """Recursively strip non-picklable values from info structures."""
    if depth > 20:
        return str(obj)

    try:
        pickle.dumps(obj)
        return obj
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _sanitize_pickle(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq_type = type(obj)
        cleaned = [_sanitize_pickle(v, depth + 1) for v in obj]
        return seq_type(cleaned)

    return repr(obj)
