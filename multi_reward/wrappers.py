"""
Wrappers for multi_reward framework.

Adapted from eureka_llm/framework/wrappers.py.

EpisodeInfoWrapper — captures end-of-episode state for generic completion detection
ComponentTrackerWrapper — logs reward components per episode to JSONL (CARD data)
PickleSafeInfoWrapper — strips non-picklable items from info dict (SubprocVecEnv safety)
MetricsTrackingWrapper — calls metrics_fn on each step and stores results

metrics_fn is NO LONGER LLM-generated. It is a fixed algorithmic function
that extracts behavior indicators from env state. This eliminates the
unreliable LLM-generated metrics problem.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import gymnasium as gym


class EpisodeInfoWrapper(gym.Wrapper):
    """Captures end-of-episode state into info dict before auto-reset.

    Adds:
        info["_episode_terminated"] = terminated (on EVERY step)
        info["_episode_end"] = True (on terminal steps)
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["_episode_terminated"] = terminated
        info["_episode_truncated"] = truncated
        if terminated or truncated:
            info["_episode_end"] = True
        return obs, reward, terminated, truncated, info


class MetricsTrackingWrapper(gym.Wrapper):
    """Calls the ALGORITHMIC metrics_fn on each step.

    metrics_fn is injected onto the env class by inject_and_register() in train.py.
    Unlike eureka_llm, metrics_fn here is NOT LLM-generated — it's a fixed,
    algorithmic function that extracts behavior indicators from env state.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        metrics_fn = getattr(self.env.unwrapped, "metrics_fn", None)
        if metrics_fn is not None:
            try:
                metrics = metrics_fn(self.env.unwrapped, action, obs)
                if isinstance(metrics, dict):
                    info["env_metrics"] = metrics
            except Exception as e:
                info.setdefault("_metrics_fn_errors", []).append(str(e))
        return obs, reward, terminated, truncated, info


class ComponentTrackerWrapper(gym.Wrapper):
    """Accumulates reward components per episode and writes per-episode summaries
    to a JSONL log file.

    Reads info["reward_components"] set by the env's step() return.
    Also reads info["env_metrics"] if available (set by MetricsTrackingWrapper).
    Tracks termination reason and agent state for behavior analysis.
    """

    def __init__(self, env, log_path: Path):
        super().__init__(env)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._step_count = 0
        self._components: dict[str, list] = defaultdict(list)
        self._env_metrics: dict[str, list] = defaultdict(list)
        self._metrics_errors: list[str] = []
        self._obs_snapshots: list = []  # obs snapshots for state tracking
        self._termination_reason: str = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        for name, value in info.get("reward_components", {}).items():
            self._components[name].append(float(value))

        for name, value in info.get("env_metrics", {}).items():
            self._env_metrics[name].append(float(value))

        for err in info.get("_metrics_fn_errors", []):
            self._metrics_errors.append(str(err))

        # Sample obs snapshot every 10 steps for state tracking
        if self._step_count % 10 == 0:
            try:
                obs_arr = np.asarray(obs).flatten().tolist()
                self._obs_snapshots.append(obs_arr[:16])  # first 16 dims max
            except Exception:
                pass

        # Track termination reason
        if terminated:
            self._termination_reason = "terminated"
        elif truncated:
            self._termination_reason = "truncated"

        if terminated or truncated:
            self._save_episode()
            self._episode += 1
            self._reset_accumulators()

        return obs, reward, terminated, truncated, info

    def _save_episode(self):
        comp_keys = list(self._components.keys())
        record = {
            "episode": self._episode,
            "length": self._step_count,
            "termination_reason": self._termination_reason or "unknown",
            "component_means": {
                k: round(float(np.mean(v)), 6)
                for k, v in self._components.items()
            },
            "component_stds": {
                k: round(float(np.std(v)), 6)
                for k, v in self._components.items()
            },
        }
        if self._env_metrics:
            record["env_metrics_means"] = {
                k: round(float(np.mean(v)), 6)
                for k, v in self._env_metrics.items()
            }
            record["env_metrics_stds"] = {
                k: round(float(np.std(v)), 6)
                for k, v in self._env_metrics.items()
            }
        if self._obs_snapshots:
            obs_arr = np.array(self._obs_snapshots)
            n_dims = obs_arr.shape[1] if obs_arr.ndim > 1 else len(obs_arr[0]) if obs_arr.size else 0
            record["obs_stats"] = {
                "mean": [round(float(x), 6) for x in np.mean(obs_arr, axis=0)],
                "std": [round(float(x), 6) for x in np.std(obs_arr, axis=0)],
                "min": [round(float(x), 6) for x in np.min(obs_arr, axis=0)],
                "max": [round(float(x), 6) for x in np.max(obs_arr, axis=0)],
                "n_dims": n_dims,
                "n_snapshots": len(self._obs_snapshots),
            }
        if self._metrics_errors:
            record["metrics_fn_errors"] = self._metrics_errors[:5]

        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class PickleSafeInfoWrapper(gym.Wrapper):
    """Strip non-picklable items from the info dict.

    LLM-generated reward functions may store simulator objects as instance
    attributes captured in info. SubprocVecEnv workers send info through a
    pipe requiring pickle — non-picklable values cause crashes.

    This wrapper sanitises the info dict so training survives bad data.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = _sanitize_pickle(info)
        return obs, reward, terminated, truncated, info


def _sanitize_pickle(obj, depth: int = 0):
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
