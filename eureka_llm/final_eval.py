"""
final_eval.py — Official Gymnasium evaluation for trained reward models.

Evaluates each round's trained policy on the OFFICIAL environment (with the
env's built-in reward function, NOT the LLM-generated reward). This produces
reward-agnostic comparison: how well does each round's policy perform on the
true task reward?

Usage:
    # Evaluate all rounds in an experiment
    python final_eval.py --run-dir runs/lunarlander-v2_2605051027_1000000

    # Evaluate specific rounds
    python final_eval.py --run-dir runs/lunarlander-v2_2605051027_1000000 --rounds 0 3 5

    # Specify official env name explicitly
    python final_eval.py --run-dir runs/... --official-env LunarLander-v2 --episodes 200
"""

import argparse
import json
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parent


def _find_round_dirs(run_dir: Path) -> list[Path]:
    """Find all roundN subdirectories with a trained model."""
    rounds = []
    for d in sorted(run_dir.iterdir()):
        if d.name.startswith("round") and (d / "model.zip").exists():
            rounds.append(d)
    return rounds


def _infer_official_env(env_id: str) -> str:
    """Map from the experiment env_id to the official Gymnasium env name."""
    # Strip round suffix
    base = re.sub(r"-round\d+$", "", env_id)
    # Known mappings
    mapping = {
        "LunarLander-v2": "LunarLander-v2",
        "LunarLanderContinuous-v2": "LunarLander-v2",
        "BipedalWalker-v3": "BipedalWalker-v3",
        "CartPole-v1": "CartPole-v1",
        "MountainCarContinuous-v0": "MountainCarContinuous-v0",
        "HalfCheetah-v4": "HalfCheetah-v4",
    }
    return mapping.get(base, base)


def evaluate_round(round_dir: Path, official_env_id: str,
                   episodes: int = 100) -> dict:
    """Evaluate one round's model on the official environment.

    Uses the OFFICIAL reward function (gym.make), not the LLM-generated one.
    """
    # Build env without any custom reward wrappers
    def _make_env():
        return gym.make(official_env_id)

    base_env = DummyVecEnv([_make_env])
    vn_path = round_dir / "vecnormalize.pkl"

    if vn_path.exists():
        env = VecNormalize.load(str(vn_path), base_env)
        env.training = False
        env.norm_reward = False  # raw official rewards
    else:
        env = base_env

    model = PPO.load(round_dir / "model")

    ep_rewards = []
    ep_lengths = []
    completed = 0
    fell = 0
    truncated = 0

    obs = env.reset()
    cur_reward = 0.0
    cur_length = 0

    while len(ep_rewards) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        cur_reward += float(reward[0])
        cur_length += 1

        if dones[0]:
            ep_rewards.append(cur_reward)
            ep_lengths.append(cur_length)
            info = infos[0]

            # Generic outcome classification (no env-specific thresholds)
            comps = info.get("reward_components", {})
            outcome = comps.get("_outcome", None) if isinstance(comps, dict) else None

            if outcome == -1.0:
                fell += 1
            elif outcome == 1.0:
                completed += 1
            elif outcome == 0.0:
                truncated += 1
            elif info.get("_episode_terminated", False):
                # Terminated by env (not timeout) — infer from reward sign
                if cur_reward > 0:
                    completed += 1
                else:
                    fell += 1
            elif info.get("_episode_truncated", False):
                truncated += 1
            else:
                # Should not reach here with proper wrappers
                truncated += 1

            cur_reward = 0.0
            cur_length = 0
            obs = env.reset()

    env.close()
    n = len(ep_rewards)

    return {
        "round": round_dir.name,
        "episodes": n,
        "reward_mean": round(float(np.mean(ep_rewards)), 4),
        "reward_std": round(float(np.std(ep_rewards)), 4),
        "reward_min": round(float(np.min(ep_rewards)), 4),
        "reward_max": round(float(np.max(ep_rewards)), 4),
        "reward_median": round(float(np.median(ep_rewards)), 4),
        "completion_rate": round(completed / n, 4) if n > 0 else 0.0,
        "fall_rate": round(fell / n, 4) if n > 0 else 0.0,
        "truncation_rate": round(truncated / n, 4) if n > 0 else 0.0,
        "mean_length": round(float(np.mean(ep_lengths)), 2) if ep_lengths else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on the official Gymnasium reward."
    )
    parser.add_argument("--run-dir", required=True,
                        help="Experiment run directory (e.g. runs/lunarlander-v2_2605051027_1000000)")
    parser.add_argument("--official-env", default=None,
                        help="Official Gymnasium env ID (e.g. LunarLander-v2). "
                             "If omitted, inferred from the experiment config.")
    parser.add_argument("--rounds", nargs="+", type=int, default=None,
                        help="Specific rounds to evaluate (default: all rounds with trained models)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes per round (default: 100)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    # Load experiment config
    exp_config = run_dir / "config.yaml"
    if not exp_config.exists():
        raise SystemExit(f"No config.yaml found in {run_dir}")
    cfg = yaml.safe_load(exp_config.read_text("utf-8"))
    env_id = cfg.get("env_id", "")
    official_env_id = args.official_env or _infer_official_env(env_id)
    print(f"Experiment  : {run_dir.name}")
    print(f"Official env: {official_env_id}")
    print(f"Episodes    : {args.episodes}")
    print()

    # Find round directories
    all_rounds = _find_round_dirs(run_dir)
    if not all_rounds:
        raise SystemExit(f"No trained models found in {run_dir}/roundN/")

    if args.rounds:
        round_dirs = [run_dir / f"round{r}" for r in args.rounds]
        round_dirs = [d for d in round_dirs if d in all_rounds]
    else:
        round_dirs = all_rounds

    print(f"Evaluating {len(round_dirs)} round(s): {', '.join(d.name for d in round_dirs)}")
    print()

    results = {}
    for rd in round_dirs:
        print(f"  Evaluating {rd.name} ... ", end="", flush=True)
        result = evaluate_round(rd, official_env_id, args.episodes)
        results[rd.name] = result
        print(f"reward={result['reward_mean']:.2f}±{result['reward_std']:.2f}  "
              f"completion={result['completion_rate']:.0%}  "
              f"fall={result['fall_rate']:.0%}  "
              f"length={result['mean_length']:.0f}")

    # Summary table
    print()
    print("=" * 70)
    print(f"{'Round':<10} {'Reward':<16} {'Completion':<12} {'Fall':<8} {'Length':<8}")
    print("-" * 70)
    for rd in round_dirs:
        r = results[rd.name]
        reward_str = f"{r['reward_mean']:.2f} ± {r['reward_std']:.2f}"
        print(f"{r['round']:<10} {reward_str:<16} "
              f"{r['completion_rate']:.0%}         "
              f"{r['fall_rate']:.0%}       "
              f"{r['mean_length']:<8.0f}")
    print("=" * 70)

    # Save results
    output_path = run_dir / "final_eval.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
