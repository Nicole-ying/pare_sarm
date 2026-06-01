"""Full training on the winning reward function."""

import re
import subprocess
import sys
from pathlib import Path

from pare_sarm.utils import save_yaml, ensure_dir


def run_full_training(
    env_dir: Path,
    reward_code: str,
    output_dir: Path,
    config: dict,
    env_id: str,
    warmstart_model_path: Path | None = None,
    total_timesteps: int = 1_000_000,
    n_envs: int = 16,
    seed: int = 42,
    record_gifs: bool = True,
    progress_fn_code: str | None = None,
) -> dict:
    """Run full PPO training on the winning reward function.

    Returns dict with: success, model_path, vecnormalize_path, elapsed_minutes.
    """
    ensure_dir(output_dir)

    # Build full training config
    full_config = _build_full_config(config, total_timesteps, seed, record_gifs)

    # Write reward code
    reward_path = output_dir / "reward_fn_source.py"
    cleaned = re.sub(r'^"""LLM[- ].*?"""', '', reward_code.strip(), flags=re.DOTALL)
    cleaned = re.sub(r'^import\s+(math|numpy).*?\n', '', cleaned, flags=re.MULTILINE)
    reward_path.write_text(
        f'"""LLM-generated reward function.\n"""\n\nimport math\nimport numpy as np\n\n{cleaned}\n',
        encoding="utf-8"
    )

    # Write config
    config_path = output_dir / "config.yaml"
    save_yaml(config_path, full_config)

    # Build command
    train_script = Path(__file__).resolve().parent / "_train_script.py"
    cmd = [
        sys.executable, str(train_script),
        "--env-dir", str(env_dir),
        "--env-id", env_id,
        "--config", str(config_path),
        "--run-dir", str(output_dir),
        "--reward-source", str(reward_path),
    ]
    if warmstart_model_path and warmstart_model_path.exists():
        cmd += ["--warmstart", str(warmstart_model_path)]
    if progress_fn_code and "def progress_fn" in progress_fn_code:
        progress_path = output_dir / "progress_fn.py"
        progress_path.write_text(progress_fn_code + "\n", encoding="utf-8")
        cmd += ["--progress-source", str(progress_path)]
    max_eps = config.get("max_episode_steps")
    if max_eps:
        cmd += ["--max-episode-steps", str(max_eps)]

    print(f"  Full training ({total_timesteps} steps)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Full training FAILED: {result.stderr[-500:]}")
        return {"success": False, "error": result.stderr[-1000:]}

    model_path = output_dir / "model.zip"
    vn_path = output_dir / "vecnormalize.pkl"

    # Read elapsed time from run_info
    import json
    run_info_path = output_dir / "run_info.json"
    elapsed = 0.0
    if run_info_path.exists():
        run_info = json.loads(run_info_path.read_text())
        elapsed = run_info.get("elapsed_minutes", 0.0)

    print(f"  Full training done ({elapsed:.1f} min)")

    return {
        "success": True,
        "model_path": model_path if model_path.exists() else None,
        "vecnormalize_path": vn_path if vn_path.exists() else None,
        "elapsed_minutes": elapsed,
    }


def _build_full_config(
    base_config: dict,
    timesteps: int,
    seed: int,
    record_gifs: bool,
) -> dict:
    """Build a config dict for full training."""
    ppo_cfg = base_config.get("ppo", {})
    cfg = {
        "total_timesteps": timesteps,
        "n_envs": base_config.get("n_envs_full", 16),
        "seed": seed,
        "device": base_config.get("device", "cpu"),
        "normalize": base_config.get("normalize", False),
        "ppo": ppo_cfg,
        "evaluation": {
            "freq": base_config.get("eval_freq", 200_000),
            "episodes": base_config.get("eval_episodes", 10),
        },
        "checkpoint": {"freq": base_config.get("checkpoint_freq", 200_000)},
    }
    if record_gifs:
        cfg["gif_steps"] = base_config.get("gif_steps", [])
        cfg["gif_fps"] = base_config.get("gif_fps", 30)
        cfg["gif_max_steps"] = base_config.get("gif_max_steps", 2000)
    if "max_episode_steps" in base_config:
        cfg["max_episode_steps"] = base_config["max_episode_steps"]
    return cfg
