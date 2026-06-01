"""Elite reward archive for ASE-MTAGE Phase 6.

Tracks TAGE scores, training returns, and independent trajectory-label counts.
``best()`` selects the historically best reward by success_like trajectory count,
falling back through partial_progress count → training return → TAGE score.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

from ase_mtage.utils.io import ensure_dir, load_json, save_json

BestMode = Literal["success_like_count", "training_return", "tage_score"]


class EliteArchive:
    """Maintain an archive of reward checkpoints ranked by trajectory quality."""

    def __init__(self, archive_path: str | Path, rewards_dir: str | Path | None = None) -> None:
        self.archive_path = Path(archive_path)
        self.rewards_dir = ensure_dir(rewards_dir or self.archive_path.parent / "elite_rewards")
        if not self.archive_path.exists():
            save_json(self.archive_path, {"elite_rewards": [], "best_reward_id": None, "best_by_training_return_id": None, "best_by_success_like_id": None})

    def read(self) -> dict[str, Any]:
        return load_json(self.archive_path, default={
            "elite_rewards": [], "best_reward_id": None, "best_by_training_return_id": None, "best_by_success_like_id": None,
        })

    def add_or_update(
        self,
        *,
        reward_id: str,
        reward_path: str | Path,
        score: float,
        round_idx: int,
        metadata: dict[str, Any] | None = None,
        training_return: float | None = None,
        num_success_like: int = 0,
        num_partial_progress: int = 0,
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
            "training_return": float(training_return) if training_return is not None else None,
            "num_success_like": int(num_success_like),
            "num_partial_progress": int(num_partial_progress),
        }
        rewards = [r for r in rewards if r.get("reward_id") != reward_id]
        rewards.append(entry)

        archive["elite_rewards"] = rewards

        # Best by success_like_count (default mode): labels are the only
        # methodologically defensible ranking signal — they are based on
        # physical observables, not any reward function's values.
        sl_entries = [r for r in rewards if r.get("num_success_like", 0) > 0 or r.get("num_partial_progress", 0) > 0]
        if sl_entries:
            sl_entries.sort(key=lambda x: (
                int(x.get("num_success_like", 0) or 0),
                int(x.get("num_partial_progress", 0) or 0),
                float(x.get("score", -1e9) or -1e9),
            ), reverse=True)
            archive["best_reward_id"] = sl_entries[0]["reward_id"]
        else:
            rewards.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
            archive["best_reward_id"] = rewards[0]["reward_id"] if rewards else None

        # Track best by training return (informational only — training return
        # is self-referential and should not drive parent selection).
        tr_entries = [r for r in rewards if r.get("training_return") is not None]
        if tr_entries:
            tr_entries.sort(key=lambda x: float(x["training_return"]), reverse=True)
            archive["best_by_training_return_id"] = tr_entries[0]["reward_id"]
        else:
            archive["best_by_training_return_id"] = None

        archive["best_by_success_like_id"] = self._best_by_success_like(rewards)

        save_json(self.archive_path, archive)
        return entry

    def best(self, mode: BestMode = "success_like_count") -> dict[str, Any] | None:
        """Return the best reward entry.

        ``mode="success_like_count"`` (default): ranks by independent
        trajectory-judge labels — highest ``num_success_like``, tie-broken by
        ``num_partial_progress``, then TAGE score. Falls back to
        ``tage_score`` mode when no entry has any positive labels.

        ``mode="training_return"``: highest training return first (only use
        when you have verified the training return is not self-referential).

        ``mode="tage_score"``: highest TAGE score (legacy behaviour).
        """
        archive = self.read()
        rewards: list[dict[str, Any]] = list(archive.get("elite_rewards") or [])
        if not rewards:
            return None

        if mode == "success_like_count":
            has_labels = any(
                r.get("num_success_like", 0) > 0 or r.get("num_partial_progress", 0) > 0
                for r in rewards
            )
            if has_labels:
                rewards.sort(key=lambda x: (
                    int(x.get("num_success_like", 0) or 0),
                    int(x.get("num_partial_progress", 0) or 0),
                    float(x.get("score", -1e9) or -1e9),
                ), reverse=True)
                return rewards[0]
            # Fallback: no positive labels yet (bootstrap), use TAGE score
            return self.best(mode="tage_score")

        if mode == "training_return":
            entries_with_tr = [r for r in rewards if r.get("training_return") is not None]
            if entries_with_tr:
                entries_with_tr.sort(key=lambda x: (
                    float(x["training_return"]),
                    int(x.get("num_success_like", 0) or 0),
                    int(x.get("num_partial_progress", 0) or 0),
                ), reverse=True)
                return entries_with_tr[0]
            rewards.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
            return rewards[0]

        # mode == "tage_score"
        rewards.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
        return rewards[0]

    @staticmethod
    def _best_by_success_like(rewards: list[dict[str, Any]]) -> str | None:
        has_labels = any(
            r.get("num_success_like", 0) > 0 or r.get("num_partial_progress", 0) > 0
            for r in rewards
        )
        if not has_labels:
            return None
        sorted_rewards = sorted(rewards, key=lambda x: (
            int(x.get("num_success_like", 0) or 0),
            int(x.get("num_partial_progress", 0) or 0),
            float(x.get("score", -1e9) or -1e9),
        ), reverse=True)
        return sorted_rewards[0]["reward_id"] if sorted_rewards else None
