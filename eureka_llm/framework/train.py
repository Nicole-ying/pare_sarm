"""
train.py — PPO training with class-injection reward registration.

Injects compute_reward from reward_source into the env class, registers the
env with gym (in-process), and trains.  Everything in one process so that
SubprocVecEnv (fork) children inherit the registered env.

Usage:
    python train.py --env-dir envs/BipedalWalker-v3/ --env-id BipedalWalker-v3-round0 \\
        --reward-source runs/round0/reward_fn_source.py --config configs/ppo_5M.yaml --run-dir runs/round0
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import perf_counter

# Headless rendering: use EGL for MuJoCo, dummy SDL driver for PyGame envs.
# Must be set before any MuJoCo/gymnasium imports that trigger GL init.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import yaml
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from wrappers import EpisodeInfoWrapper, ComponentTrackerWrapper, PickleSafeInfoWrapper

BEIJING = timezone(timedelta(hours=8))


# ─────────────────────────────────────────────────────────────────────────────
# Reward injection (in-process, before fork)
# ─────────────────────────────────────────────────────────────────────────────

def inject_and_register(env_dir: Path, reward_source: Path, register_as: str,
                         max_episode_steps: int = None):
    """Load env class, inject compute_reward, register with gym."""
    env_py = env_dir / "env.py"
    if not env_py.exists():
        raise FileNotFoundError(f"{env_py} not found")

    spec = importlib.util.spec_from_file_location("_env_module", str(env_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    env_class = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, gym.Env) and obj is not gym.Env:
            env_class = obj
            break
    if env_class is None:
        raise ValueError(f"No gym.Env subclass found in {env_py}")

    reward_code = reward_source.read_text(encoding="utf-8")
    compiled = compile(reward_code, str(reward_source), "exec")
    scope = {"np": np, "math": math}
    for _name in dir(mod):
        _val = getattr(mod, _name)
        if _name.isupper() and isinstance(_val, (int, float)):
            scope[_name] = _val
            setattr(env_class, _name, _val)
    exec(compiled, scope)

    if "compute_reward" not in scope:
        raise ValueError(f"reward_source must define 'compute_reward'")

    setattr(env_class, "compute_reward", staticmethod(scope["compute_reward"]))

    gym.envs.registration.registry.pop(register_as, None)
    register_kwargs = {"id": register_as, "entry_point": lambda **kwargs: env_class(**kwargs)}
    if max_episode_steps is not None:
        register_kwargs["max_episode_steps"] = max_episode_steps
    gym.register(**register_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Env factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(env_id: str, monitor_path: Path, traj_path: Path, seed: int = 0):
    """Env factory with wrappers for training."""
    def _init():
        env = gym.make(env_id)
        env = EpisodeInfoWrapper(env)
        env = ComponentTrackerWrapper(env, traj_path)
        env = PickleSafeInfoWrapper(env)
        env = Monitor(env, filename=str(monitor_path))
        env.reset(seed=seed)
        return env
    return _init


def make_eval_env(env_id: str, seed: int = None):
    """Plain env for behavior evaluation (no component tracking)."""
    def _init():
        env = gym.make(env_id)
        env = EpisodeInfoWrapper(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Behavior evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_behavior_eval(env_id, model, vecnormalize_path, episodes, seed=None) -> dict:
    """Run behavior evaluation — tracks episode lengths.

    Per-component statistics are captured by ComponentTrackerWrapper during
    training and aggregated by pipeline._collect_component_stats().
    """
    base_env = DummyVecEnv([make_eval_env(env_id, seed=seed)])
    if vecnormalize_path:
        env = VecNormalize.load(str(vecnormalize_path), base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env

    lengths = []
    current_length = 0

    obs = env.reset()
    while len(lengths) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        current_length += 1

        if dones[0]:
            lengths.append(current_length)
            current_length = 0
            obs = env.reset()

    env.close()
    n = len(lengths)
    return {
        "episodes": n,
        "mean_length": round(float(np.mean(lengths)), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GIF recording
# ─────────────────────────────────────────────────────────────────────────────

def record_gif(env_id, model, vecnormalize_path, output_path, max_steps=2000, fps=30):
    """Record one deterministic rollout as a GIF."""
    def _make():
        return gym.make(env_id, render_mode="rgb_array")

    render_vec = DummyVecEnv([_make])
    if vecnormalize_path:
        render_vec = VecNormalize.load(str(vecnormalize_path), render_vec)
        render_vec.training = False
        render_vec.norm_reward = False

    try:
        raw_env = render_vec.venv.envs[0].unwrapped
    except AttributeError:
        raw_env = render_vec.envs[0].unwrapped
    frames = []
    obs = render_vec.reset()
    frames.append(raw_env.render())

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = render_vec.step(action)
        frames.append(raw_env.render())
        if dones[0]:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, duration=1000 / fps)
    render_vec.close()


# ─────────────────────────────────────────────────────────────────────────────
# Training callback
# ─────────────────────────────────────────────────────────────────────────────

class TrainCallback(BaseCallback):
    """Checkpoint + evaluation + GIF callback."""

    def __init__(self, cfg, run_dir, env_id, seed=None):
        super().__init__()
        self.cfg = cfg
        self.run_dir = run_dir
        self.env_id = env_id
        self.seed = seed
        checkpoint_cfg = cfg.get("checkpoint", {}) or {}
        eval_cfg = cfg.get("evaluation", {}) or {}
        self.checkpoint_freq = checkpoint_cfg.get("freq", cfg.get("total_timesteps", 1_000_000) * 10)
        self.eval_freq = eval_cfg.get("freq", cfg.get("total_timesteps", 200_000) // 2)
        self.eval_episodes = eval_cfg.get("episodes", 5)
        self.next_checkpoint = self.checkpoint_freq
        self.next_eval = self.eval_freq
        self._history_path = run_dir / "evaluations" / "history.csv"
        self.gif_steps = sorted(cfg.get("gif_steps", []))
        self.next_gif = self.gif_steps[0] if self.gif_steps else None

    def _init_callback(self):
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._history_path.exists():
            with self._history_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timesteps", "mean_length"])

    def _save_checkpoint(self, timesteps):
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        self.model.save(ckpt_dir / f"model_{timesteps:07d}")
        vn_path = ckpt_dir / f"vecnormalize_{timesteps:07d}.pkl"
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(str(vn_path))
        return vn_path

    def _run_evaluation(self, timesteps, vn_path):
        metrics = run_behavior_eval(
            self.env_id, self.model, vn_path, self.eval_episodes,
            seed=self.seed,
        )
        metrics["timesteps"] = timesteps
        step_dir = self.run_dir / "evaluations" / f"step_{timesteps:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        with self._history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timesteps, metrics.get("mean_length", "")])

        # Record policy entropy (training dynamics signal)
        try:
            vec_env = self.model.get_env()
            obs = vec_env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                entropy = dist.entropy().mean().item()
            entropy_path = self.run_dir / "entropy_history.jsonl"
            with open(entropy_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestep": timesteps, "entropy": round(entropy, 6)}) + "\n")
        except Exception:
            pass  # non-critical — entropy is auxiliary

        print(f"  [eval t={timesteps}] len={metrics.get('mean_length', 0):.0f}")

    def _on_step(self):
        while self.num_timesteps >= self.next_checkpoint:
            vn = self._save_checkpoint(self.num_timesteps)
            self.next_checkpoint += self.checkpoint_freq

        if self.num_timesteps >= self.next_eval:
            vn_path = self._get_latest_vn()
            self._run_evaluation(self.num_timesteps, vn_path)
            self.next_eval += self.eval_freq

        if self.next_gif is not None and self.num_timesteps >= self.next_gif:
            vn_path = self._get_latest_vn()
            gif_path = self.run_dir / "gifs" / f"rollout_{self.num_timesteps:07d}.gif"
            print(f"  [gif t={self.num_timesteps}] -> {gif_path.name}")
            try:
                record_gif(self.env_id, self.model, vn_path, gif_path,
                           max_steps=self.cfg.get("gif_max_steps", 2000),
                           fps=self.cfg.get("gif_fps", 30))
            except Exception as e:
                print(f"  [gif t={self.num_timesteps}] SKIPPED (render not available: {e})")
            self.gif_steps.pop(0)
            self.next_gif = self.gif_steps[0] if self.gif_steps else None
        return True

    def _get_latest_vn(self):
        """Find the most recent vecnormalize checkpoint, or fall back to current model."""
        ckpt_dir = self.run_dir / "checkpoints"
        if ckpt_dir.exists():
            vn_files = sorted(ckpt_dir.glob("vecnormalize_*.pkl"))
            if vn_files:
                return vn_files[-1]
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vn_path = self.run_dir / "vecnormalize_latest.pkl"
            vec_env.save(str(vn_path))
            return vn_path
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", required=True,
                        help="Env directory (contains env.py) — used for class injection")
    parser.add_argument("--env-id", required=True,
                        help="Registered env ID (will be registered in-process)")
    parser.add_argument("--config", default="configs/ppo_5M.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reward-source", required=True,
                        help="Path to reward_fn_source.py (injected into env class)")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--warmstart", default=None,
                        help="Path to a model.zip checkpoint to continue training from")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    monitor_dir = run_dir / "train_monitor"
    monitor_dir.mkdir(exist_ok=True)
    traj_dir = run_dir / "trajectory_logs"
    traj_dir.mkdir(exist_ok=True)

    # Save config
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Copy reward source for reproducibility
    reward_source_path = Path(args.reward_source)
    (run_dir / "reward_fn_source.py").write_bytes(reward_source_path.read_bytes())

    # ── Inject reward into env class & register ──
    print(f"Injecting reward from {reward_source_path} into {args.env_dir} ...")
    inject_and_register(
        Path(args.env_dir), reward_source_path, args.env_id,
        max_episode_steps=args.max_episode_steps,
    )
    print(f"  Registered: {args.env_id}")

    # ── Build env(s) ──
    n_envs = cfg.get("n_envs", 8)
    env_id = args.env_id
    seed = cfg.get("seed", None)

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)
    # Use DummyVecEnv by default: SubprocVecEnv+fork corrupts C++ physics
    # engines (Box2D, MuJoCo) in child processes. DummyVecEnv runs envs in
    # the main process, avoiding fork-related crashes and silent data loss.
    env = DummyVecEnv([
        make_env(env_id, monitor_dir / f"{i}.monitor.csv", traj_dir / f"{i}.trajectory.jsonl",
                 seed=(seed or 0) + i)
        for i in range(n_envs)
    ])

    if cfg.get("normalize", True):
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # ── Model (warmstart or fresh) ──
    warmstart_path = args.warmstart
    warmstart_steps = 0
    if warmstart_path and Path(warmstart_path).exists():
        print(f"Loading warmstart model from {warmstart_path} ...")
        model = PPO.load(warmstart_path, env=env, device=cfg.get("device", "cpu"))
        warmstart_steps = model.num_timesteps
        print(f"  Warmstart loaded ({warmstart_steps} prior steps) — continuing training")
    else:
        ppo = cfg["ppo"]
        model_kwargs = dict(
            policy=ppo["policy"], env=env,
            device=cfg.get("device", "cpu"),
            learning_rate=ppo["learning_rate"],
            n_steps=ppo["n_steps"],
            batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"],
            gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"],
            clip_range=ppo["clip_range"],
            ent_coef=ppo["ent_coef"],
            vf_coef=ppo["vf_coef"],
            max_grad_norm=ppo["max_grad_norm"],
            verbose=1,
        )
        if seed is not None:
            model_kwargs["seed"] = seed
        model = PPO(**model_kwargs)

    print(f"Run dir  : {run_dir}")
    print(f"Env ID   : {env_id}")
    print(f"Steps    : {cfg['total_timesteps']:,}")

    t0 = perf_counter()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=TrainCallback(cfg, run_dir, env_id, seed=seed),
    )
    # SB3 resets num_timesteps in learn(). Restore total for correct checkpointing.
    if warmstart_steps:
        model.num_timesteps = warmstart_steps + model.num_timesteps
    elapsed = perf_counter() - t0

    model.save(run_dir / "model")
    vec_env = model.get_vec_normalize_env()
    if vec_env is not None:
        vec_env.save(str(run_dir / "vecnormalize.pkl"))

    run_info = {
        "env_id": env_id,
        "total_timesteps": cfg["total_timesteps"],
        "n_envs": n_envs,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_minutes": round(elapsed / 60, 2),
        "timestamp": datetime.now(BEIJING).strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"\nDone. Elapsed: {elapsed/60:.1f} min → {run_dir}")
