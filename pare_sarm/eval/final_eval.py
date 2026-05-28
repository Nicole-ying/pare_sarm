"""Final evaluation: evaluate trained policies on the OFFICIAL environment reward.

This is the gold-standard metric — it uses the built-in env reward, not the
LLM-generated reward, to measure actual task performance.
"""

import json
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


def evaluate_model(
    model_path: Path,
    official_env_id: str,
    n_episodes: int = 100,
    seed: int = 42,
    max_episode_steps: int = None,
) -> dict:
    """Evaluate a trained PPO model on the OFFICIAL environment reward.

    Returns dict with: reward_mean, reward_std, completion_rate, mean_length.
    """
    if not model_path.exists():
        return {"error": f"Model not found: {model_path}"}

    env = gym.make(official_env_id, max_episode_steps=max_episode_steps)
    model = PPO.load(model_path)

    returns = []
    lengths = []
    completions = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_length)
        if not terminated:
            completions += 1  # truncated at max steps = survived

    env.close()

    return {
        "reward_mean": round(float(np.mean(returns)), 2),
        "reward_std": round(float(np.std(returns)), 2),
        "reward_min": round(float(np.min(returns)), 2),
        "reward_max": round(float(np.max(returns)), 2),
        "mean_length": round(float(np.mean(lengths)), 1),
        "completion_rate": round(completions / n_episodes, 3),
        "n_episodes": n_episodes,
    }


def evaluate_all_rounds(
    exp_dir: Path,
    official_env_id: str,
    n_episodes: int = 100,
) -> dict:
    """Evaluate all rounds in an experiment. Returns {round_name: metrics}."""
    results = {}
    for rdir in sorted(exp_dir.glob("round*")):
        round_name = rdir.name
        model_path = rdir / "full_training" / "model.zip"
        if not model_path.exists():
            model_path = rdir / "model.zip"
        if model_path.exists():
            print(f"  Evaluating {round_name}...")
            metrics = evaluate_model(model_path, official_env_id, n_episodes=n_episodes)
            results[round_name] = metrics
            print(f"    Reward: {metrics.get('reward_mean', 'N/A')} +/- {metrics.get('reward_std', 'N/A')}")
        else:
            results[round_name] = {"error": "no model found"}

    return results
