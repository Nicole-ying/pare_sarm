"""ASE-MTAGE Phase 1 pipeline skeleton.

This module implements only the bootstrapping skeleton required by Phase 1:
- create a reproducible experiment directory;
- save normalized config;
- create the initial memory folders and placeholder memory files;
- run empty rounds;
- save experiment_state.json and round_summary.json.

No LLM calls, reward generation, training, trajectory collection, or TAGE scoring
are performed in Phase 1. Later phases will replace the placeholder round steps
with the real ASE-MTAGE workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.schemas import ASEMTAGEConfig, ExperimentLayout, ExperimentState, RoundSummary
from ase_mtage.utils.io import ensure_dir, load_config, now_timestamp, save_json, save_text


class ASEMTAGEPipeline:
    """Minimal ASE-MTAGE pipeline used to validate project layout and resume state."""

    def __init__(
        self,
        config: ASEMTAGEConfig | dict[str, Any] | None = None,
        *,
        config_path: str | Path | None = None,
        output_root: str | Path | None = None,
        resume_from: str | Path | None = None,
        dry_run: bool = True,
    ) -> None:
        if isinstance(config, ASEMTAGEConfig):
            self.config = config
        else:
            raw = config if config is not None else load_config(config_path)
            self.config = ASEMTAGEConfig.from_dict(raw)

        if output_root is not None:
            self.config.output_root = str(output_root)

        self.config_path = Path(config_path) if config_path else None
        self.resume_from = Path(resume_from) if resume_from else None
        self.dry_run = dry_run

        self.exp_dir = self._resolve_exp_dir()
        self.layout = ExperimentLayout.from_exp_dir(self.exp_dir)
        self.state = ExperimentState(
            method=self.config.method.name,
            env_id=self.config.training.env_id,
            exp_dir=str(self.exp_dir),
            max_rounds=self.config.method.max_rounds,
            use_short_training=self.config.method.use_short_training,
            selected_long_train_per_round=self.config.method.selected_long_train_per_round,
            notes=[
                "Phase 1 skeleton only: no LLM calls, reward generation, training, or TAGE scoring.",
                "Historical best health is not used as a gate in ASE-MTAGE.",
            ],
        )

    def _resolve_exp_dir(self) -> Path:
        if self.resume_from is not None:
            return self.resume_from

        if self.config.experiment_name:
            name = self.config.experiment_name
        else:
            safe_env = self.config.training.env_id.lower().replace("/", "_").replace("-", "_")
            name = f"ase_mtage_{safe_env}_{now_timestamp()}"
        return Path(self.config.output_root) / name

    def setup_experiment(self) -> None:
        """Create experiment folders and initial config/memory artifacts."""
        ensure_dir(self.layout.exp_dir)
        ensure_dir(self.layout.memory_dir)
        ensure_dir(self.layout.core_memory_dir)
        ensure_dir(self.layout.raw_trajectories_dir)
        ensure_dir(self.layout.elite_rewards_dir)

        # Persist normalized config regardless of input config format.
        save_json(self.layout.exp_dir / "config.json", self.config.to_dict())

        if self.config_path and self.config_path.exists():
            save_text(
                self.layout.exp_dir / "config_source.txt",
                f"Original config path: {self.config_path}\n",
            )

        # Placeholder memory artifacts. Later phases will populate these files.
        save_text(
            self.layout.core_memory_dir / "task_manifest.md",
            "# Task Manifest\n\nPhase 1 placeholder. Env Perception Agent will fill this in Phase 2.\n",
        )
        save_json(
            self.layout.core_memory_dir / "env_manifest.json",
            {
                "env_name": self.config.training.env_id,
                "task_goal": "unknown_phase_1_placeholder",
                "official_reward_visible": False,
                "phase": "phase_1_placeholder",
            },
        )
        save_json(
            self.layout.core_memory_dir / "outcome_label_schema.json",
            {
                "coarse_outcome_labels": [
                    "early_failure",
                    "low_progress_survival",
                    "partial_progress",
                    "success_like",
                    "ambiguous",
                ],
                "phase": "phase_1_placeholder",
            },
        )
        save_json(
            self.layout.core_memory_dir / "feature_schema.json",
            {
                "trajectory_features_to_extract": [],
                "phase": "phase_1_placeholder",
            },
        )
        save_text(self.layout.memory_dir / "trajectory_cards.jsonl", "")
        save_text(self.layout.memory_dir / "failure_repair_memory.jsonl", "")
        save_text(self.layout.memory_dir / "archival_lessons.jsonl", "")
        save_json(
            self.layout.memory_dir / "elite_archive.json",
            {
                "elite_rewards": [],
                "phase": "phase_1_placeholder",
            },
        )
        save_json(
            self.layout.memory_dir / "coverage_report.json",
            {
                "num_trajectories": 0,
                "num_high_confidence": 0,
                "coverage_type": "empty_or_too_small",
                "can_build_preference_pairs": False,
                "phase": "phase_1_placeholder",
            },
        )
        self._save_state("INIT")

    def run(self, n_rounds: int | None = None) -> dict[str, Any]:
        """Run empty Phase 1 rounds and return a summary dict."""
        self.setup_experiment()
        total_rounds = n_rounds if n_rounds is not None else self.config.method.max_rounds
        if total_rounds < 0:
            raise ValueError("n_rounds must be non-negative")

        round_summaries: list[dict[str, Any]] = []
        for round_idx in range(total_rounds):
            summary = self.run_empty_round(round_idx)
            round_summaries.append(summary.to_dict())

        self.state.last_completed_node = "EXPERIMENT_COMPLETED"
        self._save_state("EXPERIMENT_COMPLETED")
        final_summary = {
            "success": True,
            "phase": "phase_1_skeleton",
            "method": self.config.method.name,
            "env_id": self.config.training.env_id,
            "exp_dir": str(self.layout.exp_dir),
            "rounds": round_summaries,
            "message": "Phase 1 completed: experiment directory, empty rounds, and experiment_state.json were created.",
        }
        save_json(self.layout.exp_dir / "summary.json", final_summary)
        return final_summary

    def run_empty_round(self, round_idx: int) -> RoundSummary:
        """Create a placeholder round directory and round_summary.json."""
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []

        save_text(
            round_dir / "README.md",
            (
                f"# ASE-MTAGE Round {round_idx}\n\n"
                "Phase 1 empty round. Later phases will add candidates, training, "
                "trajectory cards, TAGE reports, and reflection artifacts.\n"
            ),
        )
        artifacts.append("README.md")

        save_json(
            round_dir / "round_state.json",
            {
                "round": round_idx,
                "phase": "phase_1_empty_round",
                "llm_called": False,
                "long_training_executed": False,
                "short_training_executed": False,
                "memory_tage_executed": False,
            },
        )
        artifacts.append("round_state.json")

        summary = RoundSummary(
            round=round_idx,
            status="completed",
            phase="phase_1_empty_round",
            message="Empty round completed. No LLM or training executed in Phase 1.",
            round_dir=str(round_dir),
            artifacts_created=artifacts,
            long_training_executed=False,
            short_training_executed=False,
        )
        save_json(round_dir / "round_summary.json", summary.to_dict())

        self.state.current_round = round_idx
        self.state.completed_rounds.append(round_idx)
        self.state.last_completed_node = "ROUND_COMPLETED"
        self._save_state("ROUND_COMPLETED")
        return summary

    def _save_state(self, node: str) -> None:
        self.state.last_completed_node = node
        save_json(self.layout.exp_dir / "experiment_state.json", self.state.to_dict())


def run_phase1(
    *,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    n_rounds: int | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Convenience function used by the CLI and tests."""
    raw_config = load_config(config_path)
    if experiment_name:
        raw_config["experiment_name"] = experiment_name
    pipeline = ASEMTAGEPipeline(raw_config, config_path=config_path, output_root=output_root, dry_run=True)
    return pipeline.run(n_rounds=n_rounds)
