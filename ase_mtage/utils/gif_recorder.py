"""Record evaluation rollout GIFs for ASE-MTAGE training visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ase_mtage.utils.io import ensure_dir


def record_eval_gif(
    *,
    env_id: str,
    model: Any,
    output_path: str | Path,
    num_episodes: int = 3,
    fps: int = 30,
    seed: int = 42,
    max_steps: int = 1000,
) -> list[str]:
    """Record a few evaluation episodes as GIFs using the trained model.

    Returns list of saved GIF paths.
    """
    import imageio.v2 as imageio

    try:
        import gymnasium as gym
    except Exception:
        import gym  # type: ignore

    output_dir = ensure_dir(Path(output_path))
    saved: list[str] = []

    for ep_idx in range(num_episodes):
        env = gym.make(env_id, render_mode="rgb_array")
        frames: list[np.ndarray] = []
        try:
            obs = env.reset(seed=seed + ep_idx * 100)
        except TypeError:
            obs = env.reset()
            env.seed(seed + ep_idx * 100)

        if isinstance(obs, tuple):
            obs = obs[0]

        done = False
        step = 0
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, _, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                obs, _, done, _ = step_result
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            step += 1

        env.close()

        if frames:
            gif_path = output_dir / f"eval_ep{ep_idx:02d}.gif"
            imageio.mimsave(str(gif_path), frames, duration=1000 / fps, loop=0)
            saved.append(str(gif_path))

    return saved
