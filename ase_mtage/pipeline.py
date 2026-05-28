"""ASE-MTAGE Phase 4 pipeline skeleton.

This module now supports:
- experiment directory and Core Memory creation;
- K=3 deterministic reward candidate generation for round 0;
- reward validation;
- simple Round-0 top-1 selection from valid candidates;
- long training of the selected reward using the candidate reward, not official reward;
- final evaluation trajectory and reward-component log collection;
- automatic Evidence Card construction and guarded trajectory labeling.

Memory-TAGE, Analyzer/Mutator cross-round evolution, reflection, and rollback are
introduced in later phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.agents.env_perception import EnvPerceptionAgent
from ase_mtage.agents.mutator import MutatorAgent
from ase_mtage.schemas import ASEMTAGEConfig, ExperimentLayout, ExperimentState, RoundSummary
from ase_mtage.tools.evidence_card_builder import EvidenceCardBuilder
from ase_mtage.tools.reward_validator import RewardValidator
from ase_mtage.training.long_trainer import LongTrainer
from ase_mtage.utils.io import ensure_dir, load_config, load_json, now_timestamp, save_json, save_text


class ASEMTAGEPipeline:
    """ASE-MTAGE pipeline skeleton through Phase 4."""

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
                "Phase 4: reward candidate generation, validation, Round-0 selection, long training, trajectory/component logging, evidence-card construction, and guarded trajectory labeling are enabled.",
                "Memory-TAGE, Analyzer/Mutator cross-round evolution, reflection, and rollback are not executed yet.",
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

        save_json(self.layout.exp_dir / "config.json", self.config.to_dict())
        if self.config_path and self.config_path.exists():
            save_text(self.layout.exp_dir / "config_source.txt", f"Original config path: {self.config_path}\n")

        env_agent = EnvPerceptionAgent(self.layout.core_memory_dir)
        env_agent.run(env_id=self.config.training.env_id)

        save_text(self.layout.memory_dir / "trajectory_cards.jsonl", "")
        save_text(self.layout.memory_dir / "failure_repair_memory.jsonl", "")
        save_text(self.layout.memory_dir / "archival_lessons.jsonl", "")
        save_json(self.layout.memory_dir / "elite_archive.json", {"elite_rewards": [], "phase": "phase_4_placeholder"})
        save_json(
            self.layout.memory_dir / "coverage_report.json",
            {
                "num_trajectories": 0,
                "num_high_confidence": 0,
                "coverage_type": "empty_or_too_small",
                "can_build_preference_pairs": False,
                "phase": "phase_4_placeholder",
            },
        )
        self._save_state("INIT")

    def run(self, n_rounds: int | None = None) -> dict[str, Any]:
        """Run Phase 4 pipeline and return a summary dict."""
        self.setup_experiment()
        total_rounds = n_rounds if n_rounds is not None else self.config.method.max_rounds
        if total_rounds < 0:
            raise ValueError("n_rounds must be non-negative")

        round_summaries: list[dict[str, Any]] = []
        for round_idx in range(total_rounds):
            if round_idx == 0:
                summary = self.run_round0_generation_selection_training_labeling(round_idx)
            else:
                summary = self.run_empty_round(round_idx)
            round_summaries.append(summary.to_dict())

        self.state.last_completed_node = "EXPERIMENT_COMPLETED"
        self._save_state("EXPERIMENT_COMPLETED")
        final_summary = {
            "success": True,
            "phase": "phase_4_evidence_cards_and_trajectory_labels",
            "method": self.config.method.name,
            "env_id": self.config.training.env_id,
            "exp_dir": str(self.layout.exp_dir),
            "rounds": round_summaries,
            "message": "Phase 4 completed: round0 candidates were generated, one valid candidate was selected, long training was attempted, and trajectory cards/final labels were saved when training succeeded.",
        }
        save_json(self.layout.exp_dir / "summary.json", final_summary)
        return final_summary

    def run_round0_generation_selection_training_labeling(self, round_idx: int) -> RoundSummary:
        """Generate, validate, select, long-train, and label Round-0 trajectories."""
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
        validation_results: list[dict[str, Any]] = []
        candidate_records: list[dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            report_path = cand.candidate_dir / "validator_report.json"
            result = validator.validate_file(cand.reward_path, candidate_id=cand.candidate_id, report_path=report_path)
            result_dict = result.to_dict()
            validation_results.append(result_dict)
            candidate_records.append(
                {
                    "index": idx,
                    "candidate_id": cand.candidate_id,
                    "mutation_family": cand.mutation_family,
                    "candidate_dir": str(cand.candidate_dir),
                    "reward_path": str(cand.reward_path),
                    "metadata_path": str(cand.metadata_path),
                    "validator_report_path": str(report_path),
                    "valid": result.valid,
                    "selection_static_score": self._round0_static_score(cand.mutation_family, result.valid),
                }
            )
            artifacts.extend(
                [
                    str(cand.reward_path.relative_to(round_dir)),
                    str(cand.metadata_path.relative_to(round_dir)),
                    str(report_path.relative_to(round_dir)),
                ]
            )

        valid_records = [r for r in candidate_records if r["valid"]]
        selected = max(valid_records, key=lambda r: r["selection_static_score"]) if valid_records else None
        selection_report = {
            "round": round_idx,
            "selection_mode": "phase_4_static_round0_selection",
            "memory_tage_used": False,
            "selected_candidate": selected["candidate_id"] if selected else None,
            "reason": "Phase 4 uses a simple static Round-0 selector. Memory-TAGE selection starts in Phase 5.",
            "candidate_scores": candidate_records,
        }
        save_json(round_dir / "selection_report.json", selection_report)
        artifacts.append("selection_report.json")

        generation_report = {
            "round": round_idx,
            "phase": "phase_4_generation_validation_selection_training_labeling",
            "k_requested": self.config.method.k_candidates,
            "num_generated": len(candidates),
            "num_valid": len(valid_records),
            "llm_called": False,
            "generation_mode": "deterministic_templates",
            "candidates": validation_results,
            "selected_candidate": selected["candidate_id"] if selected else None,
        }
        save_json(round_dir / "candidate_generation_report.json", generation_report)
        artifacts.append("candidate_generation_report.json")

        long_training_executed = False
        long_training_success = False
        evidence_cards_created = False
        num_trajectory_cards = 0
        if selected is not None:
            full_training_dir = ensure_dir(round_dir / "full_training")
            reward_path = Path(selected["reward_path"])
            trainer = LongTrainer(
                env_id=self.config.training.env_id,
                reward_path=reward_path,
                output_dir=full_training_dir,
                selected_candidate_id=selected["candidate_id"],
                seed=self.config.training.seed,
                full_timesteps=self.config.training.full_timesteps,
                final_eval_episodes=self.config.training.final_eval_episodes,
            )
            training_result = trainer.run()
            long_training_executed = True
            long_training_success = training_result.success
            artifacts.append("full_training/training_config.json")
            artifacts.append("full_training/long_training_result.json")
            if training_result.model_path:
                artifacts.append(str(training_result.model_path.relative_to(round_dir)))
            if training_result.eval_summary_path:
                artifacts.append(str(training_result.eval_summary_path.relative_to(round_dir)))
                artifacts.append("full_training/trajectory_logs/")
                artifacts.append("full_training/component_logs/")

            if training_result.success:
                builder = EvidenceCardBuilder(
                    env_id=self.config.training.env_id,
                    confidence_threshold=self.config.trajectory_memory.label_confidence_threshold,
                )
                card_result = builder.build_from_training_dir(
                    full_training_dir=full_training_dir,
                    round_dir=round_dir,
                    memory_dir=self.layout.memory_dir,
                    source_round=round_idx,
                    source_reward_id=selected["candidate_id"],
                )
                evidence_cards_created = True
                num_trajectory_cards = int(card_result.get("num_cards", 0))
                artifacts.extend(
                    [
                        "trajectory_cards.jsonl",
                        "trajectory_judgment.jsonl",
                        "trajectory_judgment_summary.json",
                        "memory/trajectory_cards.jsonl",
                    ]
                )
        else:
            save_json(
                round_dir / "full_training_skipped.json",
                {"skipped": True, "reason": "No valid Round-0 reward candidates were available for long training."},
            )
            artifacts.append("full_training_skipped.json")

        save_text(
            round_dir / "README.md",
            (
                f"# ASE-MTAGE Round {round_idx}\n\n"
                "Phase 4 generated and validated initial reward candidates, selected one candidate, "
                "attempted long training, and built guarded trajectory labels when trajectories were available.\n"
            ),
        )
        artifacts.append("README.md")

        save_json(
            round_dir / "round_state.json",
            {
                "round": round_idx,
                "phase": "phase_4_generation_validation_selection_training_labeling",
                "llm_called": False,
                "reward_candidates_generated": True,
                "reward_candidates_validated": True,
                "num_candidates": len(candidates),
                "num_valid_candidates": len(valid_records),
                "selected_candidate": selected["candidate_id"] if selected else None,
                "long_training_executed": long_training_executed,
                "long_training_success": long_training_success,
                "evidence_cards_created": evidence_cards_created,
                "num_trajectory_cards": num_trajectory_cards,
                "short_training_executed": False,
                "memory_tage_executed": False,
            },
        )
        artifacts.append("round_state.json")

        status = "completed"
        if not valid_records:
            status = "completed_with_no_valid_candidates"
        elif long_training_executed and not long_training_success:
            status = "completed_with_training_failure"
        elif long_training_success and not evidence_cards_created:
            status = "completed_without_evidence_cards"

        summary = RoundSummary(
            round=round_idx,
            status=status,
            phase="phase_4_generation_validation_selection_training_labeling",
            message=(
                f"Generated {len(candidates)} candidates, validated {len(valid_records)}, "
                f"selected {selected['candidate_id'] if selected else 'none'}, "
                f"long_training_executed={long_training_executed}, success={long_training_success}, "
                f"trajectory_cards={num_trajectory_cards}."
            ),
            round_dir=str(round_dir),
            artifacts_created=artifacts,
            long_training_executed=long_training_executed,
            short_training_executed=False,
        )
        save_json(round_dir / "round_summary.json", summary.to_dict())

        self.state.current_round = round_idx
        self.state.completed_rounds.append(round_idx)
        self.state.last_completed_node = "ROUND_COMPLETED"
        self._save_state("ROUND_COMPLETED")
        return summary

    def _round0_static_score(self, mutation_family: str, valid: bool) -> float:
        if not valid:
            return -1.0
        family_prior = {"progress_conditioned": 0.90, "component_recomposition": 0.80, "local_repair": 0.70}
        return family_prior.get(mutation_family, 0.50)

    def run_empty_round(self, round_idx: int) -> RoundSummary:
        """Create a placeholder round directory for phases not implemented yet."""
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []
        save_text(
            round_dir / "README.md",
            (
                f"# ASE-MTAGE Round {round_idx}\n\n"
                "Placeholder round. Phase 5+ will add Memory-TAGE, Analyzer, Mutator children, rollback, and reflection artifacts.\n"
            ),
        )
        artifacts.append("README.md")
        save_json(
            round_dir / "round_state.json",
            {
                "round": round_idx,
                "phase": "phase_4_placeholder_for_later_rounds",
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
            phase="phase_4_placeholder_for_later_rounds",
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
