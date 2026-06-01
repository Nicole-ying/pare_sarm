#!/usr/bin/env python3
"""PARE-SARM CLI entry point.

Usage:
    # Dry-run (test orchestration without LLM/training):
    python main.py --env-dir envs/CartPole-v1 --dry-run --n-rounds 2

    # Real run with CartPole:
    python main.py --env-dir envs/CartPole-v1 --config configs/cartpole.yaml --n-rounds 3

    # Resume:
    python main.py --resume outputs/cartpole-v1_XXXXXX_100000
"""

import argparse
import os
import sys
from pathlib import Path

_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from pare_sarm.utils import load_yaml, die


def main():
    parser = argparse.ArgumentParser(
        description="PARE-SARM: Progress-Aligned Reward Evolution with Structure-Aware Reward Mutation"
    )
    parser.add_argument("--env-dir", default=None,
                        help="Environment directory. Auto-detected when --resume is used.")
    parser.add_argument("--exploration", default=None,
                        help="Path to exploration JSON. Auto-detected from env-dir if not specified.")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file. Auto-detected when --resume is used.")
    parser.add_argument("--api-key", default=None,
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--model", default="deepseek-reasoner",
                        help="LLM model name. Default: deepseek-reasoner")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Base LLM temperature. Default: 0.6")
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="Number of iteration rounds (including round 0). Default: 3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without LLM calls or training (test orchestration)")
    parser.add_argument("--resume", default=None,
                        help="Resume from an existing experiment directory")

    args = parser.parse_args()

    resume_dir = Path(args.resume) if args.resume else None

    # ── Determine config ──
    if resume_dir and resume_dir.exists():
        config = load_yaml(resume_dir / "config.yaml")
    elif args.config:
        config = load_yaml(Path(args.config))
    else:
        config = load_yaml(Path("configs/cartpole.yaml"))

    # ── Determine env dir ──
    if args.env_dir:
        env_dir = Path(args.env_dir)
    elif resume_dir:
        env_name = config.get("env_id", "")
        for cand in [Path(f"envs/{env_name}"),
                      Path(f"/home/utseus22/eure/eureka_llm/envs/{env_name}")]:
            if cand.exists():
                env_dir = cand
                break
        else:
            die(f"Cannot auto-detect env dir for {env_name}. Pass --env-dir.")
    else:
        env_dir = Path("envs/CartPole-v1")

    if not env_dir.exists():
        die(f"Environment directory not found: {env_dir}")

    # ── API key ──
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or config.get("llm_api_key")
    if not api_key and not args.dry_run:
        die("Set DEEPSEEK_API_KEY, pass --api-key, or set llm_api_key in config")

    # ── Exploration data ──
    if args.exploration:
        exploration_path = Path(args.exploration)
    else:
        exploration_path = env_dir / "exploration.json"
    if not exploration_path.exists():
        print(f"WARNING: No exploration data at {exploration_path}")

    # ── Run ──
    from pare_sarm.pipeline import Pipeline

    pipeline = Pipeline(
        env_dir=env_dir,
        exploration_path=exploration_path,
        config=config,
        api_key=api_key or "dry-run",
        model=args.model,
        temperature=args.temperature,
        dry_run=args.dry_run,
        resume_from=resume_dir,
    )

    results = pipeline.run(n_rounds=args.n_rounds)

    if results.get("success"):
        print(f"\n{'='*60}")
        print(f"  Pipeline complete!")
        print(f"  Rounds: {results['n_rounds']}")
        print(f"  Output: {results['exp_dir']}")
        print(f"{'='*60}")
    else:
        print(f"\nPipeline completed with issues: {results.get('error', 'check logs')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
