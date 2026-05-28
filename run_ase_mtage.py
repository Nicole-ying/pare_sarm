#!/usr/bin/env python3
"""ASE-MTAGE Phase 1 CLI entry point.

Phase 1 verifies the new project skeleton only:
- create experiment directory;
- read config;
- run empty rounds;
- save experiment_state.json.

Example:
    python run_ase_mtage.py --config configs/ase_mtage_cartpole.json --n-rounds 2
    python run_ase_mtage.py --env-id LunarLander-v2 --n-rounds 1 --experiment-name smoke_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase_mtage.pipeline import run_phase1
from ase_mtage.utils.io import load_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ASE-MTAGE Phase 1 skeleton")
    parser.add_argument("--config", default=None, help="Path to JSON/YAML config. Optional in Phase 1.")
    parser.add_argument("--output-root", default=None, help="Output root directory. Default: outputs")
    parser.add_argument("--experiment-name", default=None, help="Fixed experiment directory name under output root")
    parser.add_argument("--env-id", default=None, help="Override training.env_id for Phase 1 smoke tests")
    parser.add_argument("--n-rounds", type=int, default=None, help="Number of empty rounds to create")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    # Lightweight CLI override for the most common smoke-test setting.
    config_path = Path(args.config) if args.config else None
    raw_config = load_config(config_path)
    if args.env_id:
        raw_config.setdefault("training", {})["env_id"] = args.env_id
    if args.experiment_name:
        raw_config["experiment_name"] = args.experiment_name

    # If CLI overrides were used, pass the in-memory config through a temporary
    # run by calling the pipeline class indirectly through run_phase1 would lose
    # them. Therefore save-free overrides are handled by writing them into a
    # small local closure here.
    if raw_config:
        from ase_mtage.pipeline import ASEMTAGEPipeline

        pipeline = ASEMTAGEPipeline(raw_config, config_path=config_path, output_root=args.output_root, dry_run=True)
        result = pipeline.run(n_rounds=args.n_rounds)
    else:
        result = run_phase1(
            config_path=config_path,
            output_root=args.output_root,
            n_rounds=args.n_rounds,
            experiment_name=args.experiment_name,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nASE-MTAGE Phase 1 output: {result['exp_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
