#!/usr/bin/env python3
"""
final_eval.py — Evaluate trained models in official environment.

Loads each round's model and runs N episodes in the ORIGINAL env
(official reward function, no LLM injection). Reports mean official reward.

This is used ONLY for paper reporting — never during iteration.

Usage:
  python final_eval.py --experiment-dir ../runs/lunarlander-v2_xxx/ \\
      --env-id LunarLander-v2 --episodes 100
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_wd = Path(__file__).resolve().parent
if str(_wd) not in sys.path:
    sys.path.insert(0, str(_wd))


def load_model_and_vecnormalize(round_dir: Path) -> tuple:
    """Load PPO model and VecNormalize (if exists) from a round directory."""
    model_path = round_dir / "model.zip"
    if not model_path.exists():
        # Try checkpoint
        ckpt_dir = round_dir / "checkpoints"
        if ckpt_dir.exists():
            models = sorted(ckpt_dir.glob("model_*.zip"))
            model_path = models[-1] if models else None
        if not model_path or not model_path.exists():
            return None, None

    vn_path = round_dir / "vecnormalize.pkl"
    if not vn_path.exists():
        # Try checkpoints
        ckpt_dir = round_dir / "checkpoints"
        if ckpt_dir.exists():
            vns = sorted(ckpt_dir.glob("vecnormalize_*.pkl"))
            vn_path = vns[-1] if vns else None

    model = PPO.load(model_path)
    return model, vn_path


def evaluate_model(model_path: Path, vn_path: Path, env_id: str,
                    n_episodes: int = 100, deterministic: bool = True,
                    seed: int = None, render: bool = False) -> dict:
    """Evaluate a trained model in the OFFICIAL environment.

    Uses the original env WITHOUT any LLM reward injection.
    Records official env rewards.
    """
    # Create env with OFFICIAL reward (no injection)
    env = gym.make(env_id, render_mode="human" if render else None)

    model = PPO.load(model_path)
    print(f"  Model: {model_path.name}")

    episode_rewards = []
    episode_lengths = []
    term_reasons = {"terminated": 0, "truncated": 0}

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep if seed else None)
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            steps += 1
            done = terminated or truncated
            if done:
                term_reasons["terminated" if terminated else "truncated"] += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)

        if (ep + 1) % 20 == 0:
            recent_mean = np.mean(episode_rewards[-20:])
            print(f"    ep {ep+1}/{n_episodes} | recent20_mean={recent_mean:.1f}")

    env.close()

    rewards = np.array(episode_rewards)
    return {
        "n_episodes": n_episodes,
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "median_reward": round(float(np.median(rewards)), 2),
        "min_reward": round(float(np.min(rewards)), 2),
        "max_reward": round(float(np.max(rewards)), 2),
        "mean_length": round(float(np.mean(episode_lengths)), 1),
        "termination_breakdown": term_reasons,
        "reward_distribution": {
            "q10": round(float(np.percentile(rewards, 10)), 2),
            "q25": round(float(np.percentile(rewards, 25)), 2),
            "q50": round(float(np.percentile(rewards, 50)), 2),
            "q75": round(float(np.percentile(rewards, 75)), 2),
            "q90": round(float(np.percentile(rewards, 90)), 2),
        },
        "positive_rate": round(float(np.mean(rewards > 0)), 3),
        "success_rate": round(float(np.mean(rewards > 200)), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models in official env")
    parser.add_argument("--experiment-dir", required=True,
                        help="Path to experiment directory")
    parser.add_argument("--env-id", required=True,
                        help="Official gym env ID (e.g. LunarLander-v2)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes per model")
    parser.add_argument("--rounds", nargs="+", type=int,
                        help="Specific rounds to evaluate (default: all)")
    parser.add_argument("--render", action="store_true",
                        help="Render the best model")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    if not exp_dir.exists():
        print(f"ERROR: {exp_dir} not found")
        sys.exit(1)

    # Find all round directories
    round_dirs = sorted(
        exp_dir.glob("round*"),
        key=lambda p: int(p.name.replace("round", ""))
    )

    if args.rounds:
        round_dirs = [d for d in round_dirs
                      if int(d.name.replace("round", "")) in args.rounds]

    if not round_dirs:
        print(f"ERROR: no round directories found in {exp_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION — Official {args.env_id}")
    print(f"  Experiment: {exp_dir.name}")
    print(f"  Rounds: {[d.name for d in round_dirs]}")
    print(f"  Episodes per model: {args.episodes}")
    print(f"{'='*60}\n")

    results = {}
    for rd in round_dirs:
        round_num = int(rd.name.replace("round", ""))
        print(f"--- Round {round_num} ---")

        model, vn = load_model_and_vecnormalize(rd)
        if model is None:
            print(f"  SKIP: no model found in {rd}")
            results[round_num] = {"error": "no model"}
            continue

        result = evaluate_model(
            rd / "model.zip", vn, args.env_id,
            n_episodes=args.episodes,
            seed=42,
            render=args.render and (round_num == len(round_dirs) - 1),
        )
        results[round_num] = result

        print(f"  Mean official reward: {result['mean_reward']:.1f} "
              f"± {result['std_reward']:.1f}")
        print(f"  Success rate (>200): {result['success_rate']:.1%}")
        print(f"  Positive rate: {result['positive_rate']:.1%}")
        print(f"  Mean length: {result['mean_length']:.0f}")
        print()

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Round':<8} {'Mean Reward':>12} {'Std':>8} {'Success%':>10} {'Mean Len':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    for r, res in sorted(results.items()):
        if "error" in res:
            print(f"  round{r:<4} ERROR: {res['error']}")
        else:
            print(f"  round{r:<4} {res['mean_reward']:>12.1f} "
                  f"{res['std_reward']:>8.1f} {res['success_rate']:>9.1%} "
                  f"{res['mean_length']:>10.0f}")

    # Save results
    out_path = exp_dir / "final_eval_results.json"
    save_json = lambda p, d: p.write_text(
        json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_json(out_path, {
        "experiment": exp_dir.name,
        "env_id": args.env_id,
        "episodes_per_model": args.episodes,
        "results": {str(k): v for k, v in results.items()},
    })
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
