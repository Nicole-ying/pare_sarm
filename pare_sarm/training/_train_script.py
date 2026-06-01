"""Standalone PPO training script (subprocess entry point).

Usage:
    python _train_script.py --env-dir <path> --env-id <id> --config <yaml> \
        --run-dir <path> --reward-source <path> \
        [--progress-source <path>] [--warmstart <model.zip>]
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import yaml
import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Import wrappers from the same package
_framework_dir = Path(__file__).resolve().parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from wrappers import (EpisodeInfoWrapper, ComponentTrackerWrapper,
                         PickleSafeInfoWrapper, StepLoggerWrapper)


# ── Reward injection ─────────────────────────────────────────────────────────

def inject_and_register(env_dir: Path, reward_source: Path, register_as: str,
                         max_episode_steps: int = None,
                         progress_source: Path = None):
    """Load env class, inject compute_reward + progress_fn, register with gym.

    Returns the progress_fn callable (or None if no progress source).
    """
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

    # Shared scope: make np, math, and env constants available to both
    # progress_fn and compute_reward
    scope = {"np": np, "math": math}
    for _name in dir(mod):
        _val = getattr(mod, _name)
        if _name.isupper() and isinstance(_val, (int, float)):
            scope[_name] = _val
            setattr(env_class, _name, _val)

    # ── Inject progress_fn FIRST (so compute_reward can call it) ──
    progress_fn = None
    if progress_source and progress_source.exists():
        progress_code = progress_source.read_text(encoding="utf-8")
        exec(compile(progress_code, str(progress_source), "exec"), scope)
        if "progress_fn" in scope:
            progress_fn = scope["progress_fn"]
            setattr(env_class, "progress_fn", staticmethod(progress_fn))
            print(f"  Progress function injected from {progress_source}")
        else:
            print(f"  WARNING: progress_source does not define progress_fn")

    # ── Inject compute_reward (now sees progress_fn in scope) ──
    reward_code = reward_source.read_text(encoding="utf-8")
    exec(compile(reward_code, str(reward_source), "exec"), scope)

    if "compute_reward" not in scope:
        raise ValueError("reward_source must define 'compute_reward'")

    setattr(env_class, "compute_reward", staticmethod(scope["compute_reward"]))

    gym.envs.registration.registry.pop(register_as, None)
    register_kwargs = {"id": register_as, "entry_point": lambda **kwargs: env_class(**kwargs)}
    if max_episode_steps is not None:
        register_kwargs["max_episode_steps"] = max_episode_steps
    gym.register(**register_kwargs)

    return progress_fn


# ── Env factories ────────────────────────────────────────────────────────────

def make_env(env_id: str, monitor_path: Path, traj_path: Path, seed: int = 0,
              step_log_path: Path = None, progress_fn=None):
    def _init():
        env = gym.make(env_id)
        env = EpisodeInfoWrapper(env)
        env = ComponentTrackerWrapper(env, traj_path)
        if step_log_path is not None:
            env = StepLoggerWrapper(env, step_log_path, progress_fn=progress_fn)
        env = PickleSafeInfoWrapper(env)
        env = Monitor(env, filename=str(monitor_path))
        env.reset(seed=seed)
        return env
    return _init


def make_eval_env(env_id: str, seed: int = None):
    def _init():
        env = gym.make(env_id)
        env = EpisodeInfoWrapper(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


# ── Eval + GIF ───────────────────────────────────────────────────────────────

def run_behavior_eval(env_id, model, episodes, seed=None) -> dict:
    base_env = DummyVecEnv([make_eval_env(env_id, seed=seed)])
    lengths = []
    obs = base_env.reset()
    current_length = 0
    while len(lengths) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = base_env.step(action)
        current_length += 1
        if dones[0]:
            lengths.append(current_length)
            current_length = 0
            obs = base_env.reset()
    base_env.close()
    n = len(lengths)
    return {"episodes": n, "mean_length": round(float(np.mean(lengths)), 2)}


def record_gif(env_id, model, output_path, max_steps=2000, fps=30):
    def _make():
        return gym.make(env_id, render_mode="rgb_array")
    render_vec = DummyVecEnv([_make])
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


# ── Callback ─────────────────────────────────────────────────────────────────

class TrainCallback(BaseCallback):
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

    def _run_evaluation(self, timesteps):
        metrics = run_behavior_eval(self.env_id, self.model, self.eval_episodes, seed=self.seed)
        metrics["timesteps"] = timesteps
        step_dir = self.run_dir / "evaluations" / f"step_{timesteps:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with self._history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timesteps, metrics.get("mean_length", "")])
        print(f"  [eval t={timesteps}] len={metrics.get('mean_length', 0):.0f}")

    def _on_step(self):
        while self.num_timesteps >= self.next_checkpoint:
            self._save_checkpoint(self.num_timesteps)
            self.next_checkpoint += self.checkpoint_freq

        if self.num_timesteps >= self.next_eval:
            self._run_evaluation(self.num_timesteps)
            self.next_eval += self.eval_freq

        if self.next_gif is not None and self.num_timesteps >= self.next_gif:
            gif_path = self.run_dir / "gifs" / f"rollout_{self.num_timesteps:07d}.gif"
            print(f"  [gif t={self.num_timesteps}] -> {gif_path.name}")
            try:
                record_gif(self.env_id, self.model, gif_path,
                           max_steps=self.cfg.get("gif_max_steps", 2000),
                           fps=self.cfg.get("gif_fps", 30))
            except Exception as e:
                print(f"  [gif] SKIPPED ({e})")
            self.gif_steps.pop(0)
            self.next_gif = self.gif_steps[0] if self.gif_steps else None
        return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reward-source", required=True)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--warmstart", default=None)
    parser.add_argument("--progress-source", default=None,
                        help="Path to progress_fn.py (injected alongside reward for per-step logging)")
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
    step_log_dir = run_dir / "step_logs"
    step_log_dir.mkdir(exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    reward_source_path = Path(args.reward_source)
    (run_dir / "reward_fn_source.py").write_bytes(reward_source_path.read_bytes())

    # Inject reward + progress_fn
    print(f"Injecting reward from {reward_source_path} into {args.env_dir} ...")
    progress_source = Path(args.progress_source) if args.progress_source else None
    progress_fn = inject_and_register(
        Path(args.env_dir), reward_source_path, args.env_id,
        max_episode_steps=args.max_episode_steps,
        progress_source=progress_source,
    )
    print(f"  Registered: {args.env_id}")

    n_envs = cfg.get("n_envs", 8)
    seed = cfg.get("seed", None)

    if seed is not None:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(1)

    step_log_path = step_log_dir / "steps.jsonl"
    env = DummyVecEnv([
        make_env(args.env_id, monitor_dir / f"{i}.monitor.csv",
                 traj_dir / f"{i}.trajectory.jsonl", seed=(seed or 0) + i,
                 step_log_path=step_log_path, progress_fn=progress_fn)
        for i in range(n_envs)
    ])

    if cfg.get("normalize", True):
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    warmstart_path = args.warmstart
    warmstart_steps = 0
    if warmstart_path and Path(warmstart_path).exists():
        print(f"Loading warmstart model from {warmstart_path} ...")
        model = PPO.load(warmstart_path, env=env, device=cfg.get("device", "cpu"))
        warmstart_steps = model.num_timesteps
        print(f"  Warmstart loaded ({warmstart_steps} prior steps)")
    else:
        ppo_cfg = cfg["ppo"]
        model_kwargs = dict(
            policy=ppo_cfg.get("policy", "MlpPolicy"), env=env,
            device=cfg.get("device", "cpu"),
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            verbose=1,
        )
        if seed is not None:
            model_kwargs["seed"] = seed
        model = PPO(**model_kwargs)

    print(f"Run dir  : {run_dir}")
    print(f"Env ID   : {args.env_id}")
    print(f"Steps    : {cfg['total_timesteps']:,}")

    t0 = perf_counter()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=TrainCallback(cfg, run_dir, args.env_id, seed=seed),
    )
    if warmstart_steps:
        model.num_timesteps = warmstart_steps + model.num_timesteps
    elapsed = perf_counter() - t0

    model.save(run_dir / "model")
    vec_env = model.get_vec_normalize_env()
    if vec_env is not None:
        vec_env.save(str(run_dir / "vecnormalize.pkl"))

    run_info = {
        "env_id": args.env_id,
        "total_timesteps": cfg["total_timesteps"],
        "elapsed_minutes": round(elapsed / 60, 2),
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"\nDone. Elapsed: {elapsed/60:.1f} min → {run_dir}")


if __name__ == "__main__":
    main()
