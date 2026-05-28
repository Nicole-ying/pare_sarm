#!/usr/bin/env python3
"""ASE-MTAGE CLI entry point.

Examples:
    python run_ase_mtage.py --config configs/ase_mtage_cartpole_smoke.json
    python run_ase_mtage.py --env-id CartPole-v1 --n-rounds 3 --experiment-name smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase_mtage.pipeline import ASEMTAGEPipeline
from ase_mtage.utils.io import load_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ASE-MTAGE")
    parser.add_argument("--config", default=None, help="Path to JSON/YAML config")
    parser.add_argument("--output-root", default=None, help="Output root directory")
    parser.add_argument("--experiment-name", default=None, help="Fixed experiment directory name under output root")
    parser.add_argument("--env-id", default=None, help="Override training.env_id")
    parser.add_argument("--n-rounds", type=int, default=None, help="Override method.max_rounds for this run")
    parser.add_argument("--dry-run", action="store_true", help="Create artifacts without long training. Useful for fast file-flow checks.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config_path = Path(args.config) if args.config else None
    raw_config = load_config(config_path)

    if args.env_id:
        raw_config.setdefault("training", {})["env_id"] = args.env_id
    if args.experiment_name:
        raw_config["experiment_name"] = args.experiment_name
    if args.output_root:
        raw_config["output_root"] = args.output_root

    pipeline = ASEMTAGEPipeline(raw_config, config_path=config_path, output_root=args.output_root, dry_run=args.dry_run)
    result = pipeline.run(n_rounds=args.n_rounds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nASE-MTAGE output: {result['exp_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
