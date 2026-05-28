#!/usr/bin/env python3
"""Run the behavior-aware PARE-SARM pipeline.

This entry point uses pare_sarm.pipeline_v2.Pipeline, which disables historical
best-health hard gates and promotes at least one candidate to full training each
round.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pare_sarm.utils import load_yaml, die
from pare_sarm.pipeline_v2 import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run behavior-aware PARE-SARM")
    parser.add_argument("--env-dir", default=None)
    parser.add_argument("--exploration", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    resume_dir = Path(args.resume) if args.resume else None

    if resume_dir and resume_dir.exists():
        config = load_yaml(resume_dir / "config.yaml")
    elif args.config:
        config = load_yaml(Path(args.config))
    else:
        config = load_yaml(Path("configs/cartpole.yaml"))

    if args.env_dir:
        env_dir = Path(args.env_dir)
    elif resume_dir:
        env_name = config.get("env_id", "")
        candidates = [Path(f"envs/{env_name}"), Path(f"/home/utseus22/eure/eureka_llm/envs/{env_name}")]
        env_dir = next((p for p in candidates if p.exists()), None)
        if env_dir is None:
            die(f"Cannot auto-detect env dir for {env_name}. Pass --env-dir.")
    else:
        env_dir = Path("envs/CartPole-v1")

    if not env_dir.exists():
        die(f"Environment directory not found: {env_dir}")

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or config.get("llm_api_key")
    if not api_key and not args.dry_run:
        die("Set API key by --api-key, environment variable, or config llm_api_key")

    exploration_path = Path(args.exploration) if args.exploration else env_dir / "exploration.json"
    if not exploration_path.exists():
        print(f"WARNING: No exploration data at {exploration_path}")

    pipe = Pipeline(
        env_dir=env_dir,
        exploration_path=exploration_path,
        config=config,
        api_key=api_key or "dry-run",
        model=args.model,
        temperature=args.temperature,
        dry_run=args.dry_run,
        resume_from=resume_dir,
    )
    result = pipe.run(n_rounds=args.n_rounds)
    print(result)


if __name__ == "__main__":
    main()
