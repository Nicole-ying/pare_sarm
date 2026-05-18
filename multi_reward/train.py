"""
train.py — PPO training with reward injection for multi_reward framework.

Adapted from eureka_llm/framework/train.py.

Key differences from eureka_llm:
- metrics_fn is ALGORITHMIC (not LLM-generated). Fixed function that extracts
  behavior indicators from env state.
- Records richer trajectory data for EvidenceAnalyzer.
- Records episode-level termination breakdowns.

Usage:
    python train.py --env-dir envs/BipedalWalker-v3/ --env-id BipedalWalker-v3-round0 \\
        --reward-source runs/.../reward_fn_source.py --config configs/ppo_5M.yaml \\
        --run-dir runs/.../round0
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import yaml
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Import wrappers from multi_reward
_wd = Path(__file__).resolve().parent
if str(_wd) not in sys.path:
    sys.path.insert(0, str(_wd))
from wrappers import EpisodeInfoWrapper, MetricsTrackingWrapper, ComponentTrackerWrapper, PickleSafeInfoWrapper

BEIJING = timezone(timedelta(hours=8))


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithmic metrics_fn — replaces LLM-generated metrics_fn from eureka_llm
# ═══════════════════════════════════════════════════════════════════════════════

def make_algorithmic_metrics_fn(env_dir: Path):
    """Create the algorithmic metrics_fn.

    Records only action_magnitude — the one universally meaningful metric.
    All observation-level analysis (which dims matter, task progress per
    EnvInterpreter's critical_variables) is done by EvidenceAnalyzer using
    per-episode obs_stats from trajectory data.

    This is FIXED infrastructure — no LLM output, no env-specific heuristics.
    """
    def metrics_fn(env, action, obs):
        try:
            return {"action_magnitude": float(np.mean(np.abs(np.asarray(action).flatten())))}
        except Exception:
            return {"action_magnitude": 0.0}

    return metrics_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Reward injection (in-process, before fork)
# ═══════════════════════════════════════════════════════════════════════════════

def inject_and_register(env_dir: Path, reward_source: Path, register_as: str,
                         max_episode_steps: int = None):
    """Load env class, inject compute_reward AND algorithmic metrics_fn, register.

    Must be called BEFORE env creation (before SubprocVecEnv fork).
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

    # Inject compute_reward from LLM-generated source
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
        raise ValueError("reward_source must define 'compute_reward'")
    setattr(env_class, "compute_reward", scope["compute_reward"])

    # Inject ALGORITHMIC metrics_fn (not LLM-generated!)
    metrics_fn = make_algorithmic_metrics_fn(env_dir)
    setattr(env_class, "metrics_fn", metrics_fn)

    # Register with gym
    gym.envs.registration.registry.pop(register_as, None)
    register_kwargs = {"id": register_as, "entry_point": lambda **kwargs: env_class(**kwargs)}
    if max_episode_steps is not None:
        register_kwargs["max_episode_steps"] = max_episode_steps
    gym.register(**register_kwargs)

    return metrics_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Env factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(env_id: str, monitor_path: Path, traj_path: Path, seed: int = 0):
    """Env factory with wrappers for training."""
    def _init():
        env = gym.make(env_id)
        env = EpisodeInfoWrapper(env)
        env = MetricsTrackingWrapper(env)
        env = ComponentTrackerWrapper(env, traj_path)
        env = PickleSafeInfoWrapper(env)
        env = Monitor(env, filename=str(monitor_path))
        env.reset(seed=seed)
        return env
    return _init


def make_eval_env(env_id: str, seed: int = None):
    """Plain env for behavior evaluation."""
    def _init():
        env = gym.make(env_id)
        env = EpisodeInfoWrapper(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


# ═══════════════════════════════════════════════════════════════════════════════
# Behavior evaluation (algorithmic — NO LLM metrics_fn dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def run_behavior_eval(env_id, model, vecnormalize_path, episodes,
                      metrics_fn=None, seed=None) -> dict:
    """Run behavior-metric evaluation.

    Uses the ALGORITHMIC metrics_fn (fixed infrastructure, not LLM-generated).
    Also records per-episode termination reasons.
    """
    base_env = DummyVecEnv([make_eval_env(env_id, seed=seed)])
    if vecnormalize_path:
        env = VecNormalize.load(str(vecnormalize_path), base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env

    lengths = []
    term_reasons = {"terminated": 0, "truncated": 0, "unknown": 0}
    step_metrics: dict[str, list] = defaultdict(list)
    metrics_errors: list[str] = []
    current_length = 0

    obs = env.reset()
    while len(lengths) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        current_length += 1

        if metrics_fn is not None:
            try:
                raw_env = env.envs[0].unwrapped if hasattr(env, "envs") else env.venv.envs[0].unwrapped
                m = metrics_fn(raw_env, action[0] if isinstance(action, np.ndarray) else action, obs[0])
                if isinstance(m, dict):
                    for name, value in m.items():
                        step_metrics[name].append(float(value))
            except Exception as e:
                if len(metrics_errors) < 10:
                    metrics_errors.append(str(e))

        if dones[0]:
            lengths.append(current_length)
            # Track termination reason
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            if isinstance(info, dict):
                if info.get("_episode_terminated"):
                    term_reasons["terminated"] += 1
                elif info.get("_episode_truncated"):
                    term_reasons["truncated"] += 1
                else:
                    term_reasons["unknown"] += 1
            current_length = 0
            obs = env.reset()

    env.close()
    n = len(lengths)
    result = {
        "episodes": n,
        "mean_length": round(float(np.mean(lengths)), 2),
        "std_length": round(float(np.std(lengths)), 2),
        "termination_breakdown": term_reasons,
    }
    if step_metrics:
        result["env_metrics"] = {
            k: {"mean": round(float(np.mean(v)), 6), "std": round(float(np.std(v)), 6)}
            for k, v in step_metrics.items()
        }
    if metrics_errors:
        result["metrics_fn_errors"] = metrics_errors
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Training callback
# ═══════════════════════════════════════════════════════════════════════════════

class TrainCallback(BaseCallback):
    """Checkpoint + evaluation + entropy tracking + final GIF callback."""

    def __init__(self, cfg, run_dir, env_id, metrics_fn=None, seed=None):
        super().__init__()
        self.cfg = cfg
        self.run_dir = run_dir
        self.env_id = env_id
        self.metrics_fn = metrics_fn
        self.seed = seed
        self.checkpoint_freq = cfg["checkpoint"]["freq"]
        self.eval_freq = cfg["evaluation"]["freq"]
        self.eval_episodes = cfg["evaluation"]["episodes"]
        self.total_timesteps = cfg["total_timesteps"]
        self.next_checkpoint = self.checkpoint_freq
        self.next_eval = self.eval_freq
        self._history_path = run_dir / "evaluations" / "history.csv"
        self._gif_recorded = False

    def _init_callback(self):
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._history_path.exists():
            with self._history_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timesteps", "mean_length", "std_length",
                    "termination_breakdown", "env_metrics",
                ])

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
            metrics_fn=self.metrics_fn, seed=self.seed,
        )
        metrics["timesteps"] = timesteps
        step_dir = self.run_dir / "evaluations" / f"step_{timesteps:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "summary.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        with self._history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                timesteps,
                metrics.get("mean_length", ""),
                metrics.get("std_length", ""),
                json.dumps(metrics.get("termination_breakdown", {})),
                json.dumps(metrics.get("env_metrics", {})),
            ])

        # Record policy entropy
        try:
            vec_env = self.model.get_env()
            obs = vec_env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                entropy = dist.entropy().mean().item()
            entropy_path = self.run_dir / "entropy_history.jsonl"
            with open(entropy_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"timestep": timesteps, "entropy": round(entropy, 6)}
                ) + "\n")
        except Exception:
            pass

        env_str = " | ".join(
            f"{k}={v['mean']:.3f}"
            for k, v in metrics.get("env_metrics", {}).items()
        )
        print(
            f"  [eval t={timesteps}] "
            f"len={metrics.get('mean_length', 0):.0f}"
            + (f" | {env_str}" if env_str else "")
        )

    def _on_step(self):
        while self.num_timesteps >= self.next_checkpoint:
            self._save_checkpoint(self.num_timesteps)
            self.next_checkpoint += self.checkpoint_freq

        if self.num_timesteps >= self.next_eval:
            vn_path = self._get_latest_vn()
            self._run_evaluation(self.num_timesteps, vn_path)
            # Record GIF only at the final evaluation
            if self.num_timesteps >= self.total_timesteps and not self._gif_recorded:
                self._record_gif()
                self._gif_recorded = True
            self.next_eval += self.eval_freq
        return True

    def _record_gif(self):
        """Record one GIF of the agent's behavior using the current model."""
        try:
            import imageio
            env = gym.make(self.env_id, render_mode="rgb_array")
            model = self.model
            obs, _ = env.reset(seed=self.seed or 42)
            frames = []
            total_reward = 0.0
            for _ in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                if terminated or truncated:
                    break
            env.close()
            gif_path = self.run_dir / "final_behavior.gif"
            imageio.mimsave(gif_path, frames, fps=30)
            print(f"  [gif] Recorded {len(frames)} frames, total_reward={total_reward:.1f} -> {gif_path}")
        except ImportError:
            print("  [gif] imageio not installed, skipping GIF")
        except Exception as e:
            print(f"  [gif] Failed: {e}")

    def _get_latest_vn(self):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", required=True,
                        help="Env directory (contains env.py)")
    parser.add_argument("--env-id", required=True,
                        help="Registered env ID")
    parser.add_argument("--config", default="configs/ppo_5M.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reward-source", required=True,
                        help="Path to reward_fn_source.py")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="Episode time limit")
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
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    # Copy reward source for reproducibility (skip if same file)
    reward_source_path = Path(args.reward_source).resolve()
    dest_path = (run_dir / "reward_fn_source.py").resolve()
    if reward_source_path != dest_path:
        import shutil
        shutil.copy2(reward_source_path, dest_path)

    # Inject reward + algorithmic metrics_fn
    print(f"Injecting reward from {reward_source_path} into {args.env_dir} ...")
    metrics_fn = inject_and_register(
        Path(args.env_dir), reward_source_path, args.env_id,
        max_episode_steps=args.max_episode_steps,
    )
    print(f"  Registered: {args.env_id}")

    # Build env(s)
    n_envs = cfg.get("n_envs", 8)
    env_id = args.env_id
    seed = cfg.get("seed", None)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)
    # Use DummyVecEnv on Windows (fork unavailable), SubprocVecEnv on Unix
    _make = lambda i: make_env(env_id, monitor_dir / f"{i}.monitor.csv",
                                traj_dir / f"{i}.trajectory.jsonl",
                                seed=(seed or 0) + i)
    if os.name == "nt":
        env = DummyVecEnv([_make(i) for i in range(n_envs)])
    else:
        env = SubprocVecEnv([_make(i) for i in range(n_envs)], start_method="fork")

    if cfg.get("normalize", True):
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Model
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
        callback=TrainCallback(cfg, run_dir, env_id, metrics_fn, seed=seed),
    )
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
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Elapsed: {elapsed/60:.1f} min -> {run_dir}")
