"""ASE-MTAGE Phase 2 pipeline skeleton.

This module now supports the first functional ASE-MTAGE artifacts:
- create a reproducible experiment directory;
- save normalized config;
- build Core Memory with Env Perception Agent;
- generate K=3 initial reward candidates for round 0;
- validate every candidate reward;
- run later rounds as empty placeholders until Phase 3+ are implemented;
- save experiment_state.json and round_summary.json.

No policy training, trajectory collection, Memory-TAGE scoring, or rollback is
performed in Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase_mtage.agents.env_perception import EnvPerceptionAgent
from ase_mtage.agents.mutator import MutatorAgent
from ase_mtage.schemas import ASEMTAGEConfig, ExperimentLayout, ExperimentState, RoundSummary
from ase_mtage.tools.reward_validator import RewardValidator
from ase_mtage.utils.io import ensure_dir, load_config, load_json, now_timestamp, save_json, save_text


class ASEMTAGEPipeline:
    """ASE-MTAGE pipeline skeleton through Phase 2."""

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
                "Phase 2 skeleton: Env Perception, reward candidate generation, and reward validation are enabled.",
                "No policy training, trajectory collection, Memory-TAGE scoring, or rollback is executed yet.",
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

        # Phase 2: Env Perception Agent creates real placeholder-safe Core Memory.
        env_agent = EnvPerceptionAgent(self.layout.core_memory_dir)
        env_agent.run(env_id=self.config.training.env_id)

        # Shared memory placeholders. Later phases will populate these files.
        save_text(self.layout.memory_dir / "trajectory_cards.jsonl", "")
        save_text(self.layout.memory_dir / "failure_repair_memory.jsonl", "")
        save_text(self.layout.memory_dir / "archival_lessons.jsonl", "")
        save_json(
            self.layout.memory_dir / "elite_archive.json",
            {
                "elite_rewards": [],
                "phase": "phase_2_placeholder",
            },
        )
        save_json(
            self.layout.memory_dir / "coverage_report.json",
            {
                "num_trajectories": 0,
                "num_high_confidence": 0,
                "coverage_type": "empty_or_too_small",
                "can_build_preference_pairs": False,
                "phase": "phase_2_placeholder",
            },
        )
        self._save_state("INIT")

    def run(self, n_rounds: int | None = None) -> dict[str, Any]:
        """Run Phase 2 rounds and return a summary dict."""
        self.setup_experiment()
        total_rounds = n_rounds if n_rounds is not None else self.config.method.max_rounds
        if total_rounds < 0:
            raise ValueError("n_rounds must be non-negative")

        round_summaries: list[dict[str, Any]] = []
        for round_idx in range(total_rounds):
            if round_idx == 0:
                summary = self.run_round0_candidate_generation(round_idx)
            else:
                summary = self.run_empty_round(round_idx)
            round_summaries.append(summary.to_dict())

        self.state.last_completed_node = "EXPERIMENT_COMPLETED"
        self._save_state("EXPERIMENT_COMPLETED")
        final_summary = {
            "success": True,
            "phase": "phase_2_reward_generation_and_validation",
            "method": self.config.method.name,
            "env_id": self.config.training.env_id,
            "exp_dir": str(self.layout.exp_dir),
            "rounds": round_summaries,
            "message": "Phase 2 completed: experiment directory, core memory, K reward candidates, validation reports, and experiment_state.json were created.",
        }
        save_json(self.layout.exp_dir / "summary.json", final_summary)
        return final_summary

    def run_round0_candidate_generation(self, round_idx: int) -> RoundSummary:
        """Generate and validate K initial reward candidates for round 0."""
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        candidates_root = ensure_dir(round_dir / "candidates")
        artifacts: list[str] = []

        env_manifest = load_json(self.layout.core_memory_dir / "env_manifest.json")
        mutator = MutatorAgent(candidates_root)
        candidates = mutator.generate_initial_candidates(
            env_manifest=env_manifest,
            k_candidates=self.config.method.k_candidates,
            round_idx=round_idx,
        )

        validator = RewardValidator()
        validation_results = []
        for cand in candidates:
            report_path = cand.candidate_dir / "validator_report.json"
            result = validator.validate_file(
                cand.reward_path,
                candidate_id=cand.candidate_id,
                report_path=report_path,
            )
            validation_results.append(result.to_dict())
            artifacts.append(str(cand.reward_path.relative_to(round_dir)))
            artifacts.append(str(cand.metadata_path.relative_to(round_dir)))
            artifacts.append(str(report_path.relative_to(round_dir)))

        valid_candidates = [r for r in validation_results if r.get("valid")]
        generation_report = {
            "round": round_idx,
            "phase": "phase_2_reward_generation_and_validation",
            "k_requested": self.config.method.k_candidates,
            "num_generated": len(candidates),
            "num_valid": len(valid_candidates),
            "llm_called": False,
            "generation_mode": "deterministic_templates",
            "candidates": validation_results,
            "next_phase_note": "Phase 3 will add Round0 selection and long training. Phase 2 only validates candidates.",
        }
        save_json(round_dir / "candidate_generation_report.json", generation_report)
        artifacts.append("candidate_generation_report.json")

        save_text(
            round_dir / "README.md",
            (
                f"# ASE-MTAGE Round {round_idx}\n\n"
                "Phase 2 generated and validated initial reward candidates. "
                "No long training is executed until Phase 3.\n"
            ),
        )
        artifacts.append("README.md")

        save_json(
            round_dir / "round_state.json",
            {
                "round": round_idx,
                "phase": "phase_2_reward_generation_and_validation",
                "llm_called": False,
                "reward_candidates_generated": True,
                "reward_candidates_validated": True,
                "num_candidates": len(candidates),
                "num_valid_candidates": len(valid_candidates),
                "long_training_executed": False,
                "short_training_executed": False,
                "memory_tage_executed": False,
            },
        )
        artifacts.append("round_state.json")

        summary = RoundSummary(
            round=round_idx,
            status="completed" if valid_candidates else "completed_with_no_valid_candidates",
            phase="phase_2_reward_generation_and_validation",
            message=(
                f"Generated {len(candidates)} candidates and validated {len(valid_candidates)}. "
                "No LLM or training executed in Phase 2."
            ),
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

    def run_empty_round(self, round_idx: int) -> RoundSummary:
        """Create a placeholder round directory for phases not implemented yet."""
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []

        save_text(
            round_dir / "README.md",
            (
                f"# ASE-MTAGE Round {round_idx}\n\n"
                "Placeholder round. Phase 3+ will add parent selection, Analyzer, "
                "Mutator children, Memory-TAGE, training, and reflection artifacts.\n"
            ),
        )
        artifacts.append("README.md")

        save_json(
            round_dir / "round_state.json",
            {
                "round": round_idx,
                "phase": "phase_2_placeholder_for_later_rounds",
                "llm_called": False,
                "reward_candidates_generated": False,
                "long_training_executed": False,
                "short_training_executed": False,
                "memory_tage_executed": False,
            },
        )
        artifacts.append("round_state.json")

        summary = RoundSummary(
            round=round_idx,
            status="completed",
            phase="phase_2_placeholder_for_later_rounds",
            message="Placeholder round completed. Later phases will implement the full cross-round workflow.",
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
    """Backward-compatible convenience function used by the Phase 1 CLI path."""
    raw_config = load_config(config_path)
    if experiment_name:
        raw_config["experiment_name"] = experiment_name
    pipeline = ASEMTAGEPipeline(raw_config, config_path=config_path, output_root=output_root, dry_run=True)
    return pipeline.run(n_rounds=n_rounds)
