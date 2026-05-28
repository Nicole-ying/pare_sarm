"""Elite reward archive for ASE-MTAGE Phase 6."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ase_mtage.utils.io import ensure_dir, load_json, save_json


class EliteArchive:
    """Maintain a simple archive of best/usable reward checkpoints."""

    def __init__(self, archive_path: str | Path, rewards_dir: str | Path | None = None) -> None:
        self.archive_path = Path(archive_path)
        self.rewards_dir = ensure_dir(rewards_dir or self.archive_path.parent / "elite_rewards")
        if not self.archive_path.exists():
            save_json(self.archive_path, {"elite_rewards": [], "best_reward_id": None})

    def read(self) -> dict[str, Any]:
        return load_json(self.archive_path, default={"elite_rewards": [], "best_reward_id": None})

    def add_or_update(
        self,
        *,
        reward_id: str,
        reward_path: str | Path,
        score: float,
        round_idx: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        archive = self.read()
        rewards = list(archive.get("elite_rewards") or [])
        copied_path = self.rewards_dir / f"{reward_id}.py"
        try:
            shutil.copy2(Path(reward_path), copied_path)
        except Exception:
            copied_path = Path(reward_path)

        entry = {
            "reward_id": reward_id,
            "reward_path": str(copied_path),
            "source_reward_path": str(reward_path),
            "score": float(score),
            "round": round_idx,
            "metadata": metadata or {},
        }
        rewards = [r for r in rewards if r.get("reward_id") != reward_id]
        rewards.append(entry)
        rewards.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
        archive["elite_rewards"] = rewards
        archive["best_reward_id"] = rewards[0]["reward_id"] if rewards else None
        save_json(self.archive_path, archive)
        return entry

    def best(self) -> dict[str, Any] | None:
        rewards = self.read().get("elite_rewards") or []
        return rewards[0] if rewards else None
