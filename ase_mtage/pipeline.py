"""ASE-MTAGE Phase 6 pipeline skeleton.

Supported now:
- Round 0: candidate generation, validation, static top-1 selection, long training,
  trajectory/component logs, evidence cards, and guarded final labels;
- Round 1+: coverage analysis, Analyzer self-evaluation, child generation,
  offline Memory-TAGE, top-1 selection, elite archive update, rollback report,
  and Reflection/Failure-Repair memory writing.

The Phase 6 agents are deterministic schema-compatible implementations. They
create the same artifacts expected from future LLM-backed Analyzer/Reflector
agents without requiring LLM credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.agents.analyzer import AnalyzerAgent
from ase_mtage.agents.env_perception import EnvPerceptionAgent
from ase_mtage.agents.mutator import MutatorAgent
from ase_mtage.agents.reflector import ReflectionAgent
from ase_mtage.memory.elite_archive import EliteArchive
from ase_mtage.memory.failure_repair_memory import FailureRepairMemory
from ase_mtage.schemas import ASEMTAGEConfig, ExperimentLayout, ExperimentState, RoundSummary
from ase_mtage.tools.evidence_card_builder import EvidenceCardBuilder
from ase_mtage.tools.memory_coverage import MemoryCoverageAnalyzer
from ase_mtage.tools.mtage_evaluator import MemoryTAGEEvaluator
from ase_mtage.tools.reward_validator import RewardValidator
from ase_mtage.tools.rollback import RollbackManager
from ase_mtage.tools.selector import CandidateSelector
from ase_mtage.training.long_trainer import LongTrainer
from ase_mtage.utils.io import ensure_dir, load_config, load_json, now_timestamp, save_json, save_text


class ASEMTAGEPipeline:
    """ASE-MTAGE pipeline skeleton through Phase 6."""

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
                "Phase 6: Analyzer, Reflection, EliteArchive, and Rollback artifacts are enabled.",
                "LLM-backed mutation is not enabled yet; deterministic templates preserve artifact protocol.",
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
        ensure_dir(self.layout.exp_dir)
        ensure_dir(self.layout.memory_dir)
        ensure_dir(self.layout.core_memory_dir)
        ensure_dir(self.layout.raw_trajectories_dir)
        ensure_dir(self.layout.elite_rewards_dir)
        save_json(self.layout.exp_dir / "config.json", self.config.to_dict())
        if self.config_path and self.config_path.exists():
            save_text(self.layout.exp_dir / "config_source.txt", f"Original config path: {self.config_path}\n")

        EnvPerceptionAgent(self.layout.core_memory_dir).run(env_id=self.config.training.env_id)
        save_text(self.layout.memory_dir / "trajectory_cards.jsonl", "")
        save_text(self.layout.memory_dir / "failure_repair_memory.jsonl", "")
        save_text(self.layout.memory_dir / "archival_lessons.jsonl", "")
        EliteArchive(self.layout.memory_dir / "elite_archive.json", self.layout.elite_rewards_dir)
        save_json(
            self.layout.memory_dir / "coverage_report.json",
            {
                "num_trajectories": 0,
                "num_high_confidence": 0,
                "coverage_type": "empty_or_too_small",
                "can_build_preference_pairs": False,
                "phase": "phase_6_placeholder",
            },
        )
        self._save_state("INIT")

    def run(self, n_rounds: int | None = None) -> dict[str, Any]:
        self.setup_experiment()
        total_rounds = n_rounds if n_rounds is not None else self.config.method.max_rounds
        if total_rounds < 0:
            raise ValueError("n_rounds must be non-negative")

        summaries: list[dict[str, Any]] = []
        for round_idx in range(total_rounds):
            if round_idx == 0:
                summary = self.run_round0_generation_selection_training_labeling(round_idx)
            else:
                summary = self.run_self_evolution_round(round_idx)
            summaries.append(summary.to_dict())

        self.state.last_completed_node = "EXPERIMENT_COMPLETED"
        self._save_state("EXPERIMENT_COMPLETED")
        final_summary = {
            "success": True,
            "phase": "phase_6_analyzer_reflection_rollback",
            "method": self.config.method.name,
            "env_id": self.config.training.env_id,
            "exp_dir": str(self.layout.exp_dir),
            "rounds": summaries,
            "message": "Phase 6 completed: Analyzer/Reflection/Rollback memory artifacts are produced across rounds.",
        }
        save_json(self.layout.exp_dir / "summary.json", final_summary)
        return final_summary

    def _generate_and_validate_candidates(self, *, round_idx: int, round_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        candidates_root = ensure_dir(round_dir / "candidates")
        artifacts: list[str] = []
        env_manifest = load_json(self.layout.core_memory_dir / "env_manifest.json")
        candidates = MutatorAgent(candidates_root).generate_initial_candidates(
            env_manifest=env_manifest,
            k_candidates=self.config.method.k_candidates,
            round_idx=round_idx,
        )
        validator = RewardValidator()
        validation_results: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            report_path = cand.candidate_dir / "validator_report.json"
            result = validator.validate_file(cand.reward_path, candidate_id=cand.candidate_id, report_path=report_path)
            validation_results.append(result.to_dict())
            records.append(
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
            artifacts.extend([
                str(cand.reward_path.relative_to(round_dir)),
                str(cand.metadata_path.relative_to(round_dir)),
                str(report_path.relative_to(round_dir)),
            ])
        return records, validation_results, artifacts

    def run_round0_generation_selection_training_labeling(self, round_idx: int) -> RoundSummary:
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []
        records, validation_results, cand_artifacts = self._generate_and_validate_candidates(round_idx=round_idx, round_dir=round_dir)
        artifacts.extend(cand_artifacts)
        valid_records = [r for r in records if r["valid"]]
        selected = max(valid_records, key=lambda r: r["selection_static_score"]) if valid_records else None
        selection_report = {
            "round": round_idx,
            "selection_mode": "phase_6_static_round0_selection",
            "memory_tage_used": False,
            "selected_candidate": selected["candidate_id"] if selected else None,
            "reason": "Round 0 has no trajectory memory yet; use static selector.",
            "candidate_scores": records,
        }
        save_json(round_dir / "selection_report.json", selection_report)
        artifacts.append("selection_report.json")
        save_json(
            round_dir / "candidate_generation_report.json",
            {
                "round": round_idx,
                "phase": "phase_6_round0_generation_validation_selection_training_labeling",
                "k_requested": self.config.method.k_candidates,
                "num_generated": len(records),
                "num_valid": len(valid_records),
                "llm_called": False,
                "generation_mode": "deterministic_templates",
                "candidates": validation_results,
                "selected_candidate": selected["candidate_id"] if selected else None,
            },
        )
        artifacts.append("candidate_generation_report.json")

        long_training_executed = False
        long_training_success = False
        num_cards = 0
        if selected is not None:
            full_training_dir = ensure_dir(round_dir / "full_training")
            training_result = LongTrainer(
                env_id=self.config.training.env_id,
                reward_path=Path(selected["reward_path"]),
                output_dir=full_training_dir,
                selected_candidate_id=selected["candidate_id"],
                seed=self.config.training.seed,
                full_timesteps=self.config.training.full_timesteps,
                final_eval_episodes=self.config.training.final_eval_episodes,
            ).run()
            long_training_executed = True
            long_training_success = training_result.success
            artifacts.extend(["full_training/training_config.json", "full_training/long_training_result.json"])
            if training_result.model_path:
                artifacts.append(str(training_result.model_path.relative_to(round_dir)))
            if training_result.eval_summary_path:
                artifacts.append(str(training_result.eval_summary_path.relative_to(round_dir)))
                artifacts.extend(["full_training/trajectory_logs/", "full_training/component_logs/"])
            if training_result.success:
                card_result = EvidenceCardBuilder(
                    env_id=self.config.training.env_id,
                    confidence_threshold=self.config.trajectory_memory.label_confidence_threshold,
                ).build_from_training_dir(
                    full_training_dir=full_training_dir,
                    round_dir=round_dir,
                    memory_dir=self.layout.memory_dir,
                    source_round=round_idx,
                    source_reward_id=selected["candidate_id"],
                )
                num_cards = int(card_result.get("num_cards", 0))
                artifacts.extend(["trajectory_cards.jsonl", "trajectory_judgment.jsonl", "trajectory_judgment_summary.json", "memory/trajectory_cards.jsonl"])
                # Round 0 selected reward becomes initial elite using static score.
                EliteArchive(self.layout.memory_dir / "elite_archive.json", self.layout.elite_rewards_dir).add_or_update(
                    reward_id=selected["candidate_id"],
                    reward_path=selected["reward_path"],
                    score=float(selected.get("selection_static_score", 0.0)),
                    round_idx=round_idx,
                    metadata={"source": "round0_static_selection", "num_trajectory_cards": num_cards},
                )
                artifacts.append("memory/elite_archive.json")
        else:
            save_json(round_dir / "full_training_skipped.json", {"skipped": True, "reason": "No valid Round-0 reward candidates."})
            artifacts.append("full_training_skipped.json")

        # Round 0 also writes an initial reflection memory record.
        reflection_dir = ensure_dir(round_dir / "reflection")
        reflection = ReflectionAgent(
            output_dir=reflection_dir,
            failure_memory_path=self.layout.memory_dir / "failure_repair_memory.jsonl",
            archival_lessons_path=self.layout.memory_dir / "archival_lessons.jsonl",
        ).run(
            round_idx=round_idx,
            analyzer_report={"parent_reward_id": None, "self_evaluation_lesson": "Round 0 bootstrapped initial trajectory memory.", "mutation_intent": {"required_changes": []}},
            selection_report=selection_report,
            coverage_report={"coverage_type": "bootstrap", "label_counts": {}},
            rollback_report={"rollback_triggered": False, "next_parent_reward_id": selected["candidate_id"] if selected else None, "reason": "bootstrap"},
        )
        artifacts.extend(["reflection/reflection.json", "memory/failure_repair_memory.jsonl", "memory/archival_lessons.jsonl"])
        self._write_round_readme(round_dir, round_idx, "Phase 6 Round 0 built initial trajectory memory and initial reflection record.")
        artifacts.append("README.md")
        save_json(
            round_dir / "round_state.json",
            {
                "round": round_idx,
                "phase": "phase_6_round0_bootstrap",
                "selected_candidate": selected["candidate_id"] if selected else None,
                "long_training_executed": long_training_executed,
                "long_training_success": long_training_success,
                "num_trajectory_cards": num_cards,
                "reflection_written": True,
                "memory_tage_executed": False,
            },
        )
        artifacts.append("round_state.json")
        summary = RoundSummary(
            round=round_idx,
            status="completed" if (long_training_success or not selected) else "completed_with_training_failure",
            phase="phase_6_round0_bootstrap",
            message=f"Round 0 selected {selected['candidate_id'] if selected else 'none'}; trajectory_cards={num_cards}; reflection written.",
            round_dir=str(round_dir),
            artifacts_created=artifacts,
            long_training_executed=long_training_executed,
            short_training_executed=False,
        )
        save_json(round_dir / "round_summary.json", summary.to_dict())
        self._mark_round_completed(round_idx)
        return summary

    def run_self_evolution_round(self, round_idx: int) -> RoundSummary:
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []

        # 1) Coverage report from existing trajectory memory.
        coverage = MemoryCoverageAnalyzer(
            min_trajectories=self.config.trajectory_memory.min_trajectories,
            min_high_confidence_trajectories=self.config.trajectory_memory.min_high_confidence_trajectories,
            confidence_threshold=self.config.trajectory_memory.label_confidence_threshold,
        ).analyze_file(
            memory_cards_path=self.layout.memory_dir / "trajectory_cards.jsonl",
            output_path=round_dir / "coverage_report.json",
        )
        save_json(self.layout.memory_dir / "coverage_report.json", coverage)
        artifacts.extend(["coverage_report.json", "memory/coverage_report.json"])

        # 2) Analyzer self-evaluation.
        archive = EliteArchive(self.layout.memory_dir / "elite_archive.json", self.layout.elite_rewards_dir)
        best = archive.best()
        failure_memory = FailureRepairMemory(self.layout.memory_dir / "failure_repair_memory.jsonl")
        analyzer_dir = ensure_dir(round_dir / "analyzer")
        analyzer_report = AnalyzerAgent(analyzer_dir).run(
            round_idx=round_idx,
            parent_reward_id=(best or {}).get("reward_id"),
            coverage_report=coverage,
            previous_selection_report=None,
            failure_memory_records=failure_memory.read_recent(limit=5),
            elite_archive=archive.read(),
        )
        artifacts.extend(["analyzer/prompt.txt", "analyzer/response.txt", "analyzer/self_evaluation.json"])

        # 3) Generate and validate children. Current Mutator is deterministic but artifact-compatible.
        records, validation_results, cand_artifacts = self._generate_and_validate_candidates(round_idx=round_idx, round_dir=round_dir)
        artifacts.extend(cand_artifacts)
        save_json(
            round_dir / "candidate_generation_report.json",
            {
                "round": round_idx,
                "phase": "phase_6_analyzer_guided_candidate_generation",
                "k_requested": self.config.method.k_candidates,
                "num_generated": len(records),
                "num_valid": sum(1 for r in records if r["valid"]),
                "llm_called": False,
                "generation_mode": "deterministic_templates_with_analyzer_artifact",
                "analyzer_self_evaluation_path": str(analyzer_dir / "self_evaluation.json"),
                "candidates": validation_results,
            },
        )
        artifacts.append("candidate_generation_report.json")

        # 4) Memory-TAGE each child.
        evaluator = MemoryTAGEEvaluator()
        tage_reports: list[dict[str, Any]] = []
        for record in records:
            report_path = Path(record["candidate_dir"]) / "tage_report.json"
            if record["valid"]:
                report = evaluator.evaluate_candidate(
                    candidate_id=record["candidate_id"],
                    reward_path=record["reward_path"],
                    memory_cards_path=self.layout.memory_dir / "trajectory_cards.jsonl",
                    coverage_report=coverage,
                    output_path=report_path,
                    other_reward_vectors={},
                )
                record["tage_report_path"] = str(report_path)
                record["tage_score"] = report.get("tage_score")
                tage_reports.append(report)
            else:
                save_json(report_path, {"candidate_id": record["candidate_id"], "valid": False, "tage_score": -1.0, "phase": "phase_6_memory_tage"})
                record["tage_report_path"] = str(report_path)
                record["tage_score"] = -1.0
            artifacts.append(str(report_path.relative_to(round_dir)))
        save_json(round_dir / "tage_summary.json", {"round": round_idx, "coverage_report": coverage, "candidate_tage_reports": tage_reports})
        artifacts.append("tage_summary.json")

        # 5) Select top-1 child and update elite archive if useful.
        selection = CandidateSelector().select(
            round_idx=round_idx,
            candidate_records=records,
            coverage_report=coverage,
            output_path=round_dir / "selection_report.json",
            selection_mode="phase_6_memory_tage_offline_selection",
        )
        artifacts.append("selection_report.json")
        selected_record = next((r for r in selection.get("candidate_scores", []) if r.get("candidate_id") == selection.get("selected_candidate")), None)
        if selected_record and float(selected_record.get("selection_score", 0.0) or 0.0) >= 0.0:
            archive.add_or_update(
                reward_id=selected_record["candidate_id"],
                reward_path=selected_record["reward_path"],
                score=float(selected_record.get("selection_score", 0.0) or 0.0),
                round_idx=round_idx,
                metadata={"source": "phase6_memory_tage_selection", "coverage_type": coverage.get("coverage_type")},
            )
            artifacts.append("memory/elite_archive.json")

        # 6) Rollback check chooses next parent if current selection is unsafe.
        rollback = RollbackManager().check(
            round_idx=round_idx,
            current_selection_report=selection,
            coverage_report=coverage,
            elite_archive=archive,
            output_path=round_dir / "rollback_report.json",
        )
        artifacts.append("rollback_report.json")

        # 7) Reflection writes failure-repair memory and archival lessons.
        reflection_dir = ensure_dir(round_dir / "reflection")
        ReflectionAgent(
            output_dir=reflection_dir,
            failure_memory_path=self.layout.memory_dir / "failure_repair_memory.jsonl",
            archival_lessons_path=self.layout.memory_dir / "archival_lessons.jsonl",
        ).run(
            round_idx=round_idx,
            analyzer_report=analyzer_report,
            selection_report=selection,
            coverage_report=coverage,
            rollback_report=rollback,
        )
        artifacts.extend(["reflection/reflection.json", "memory/failure_repair_memory.jsonl", "memory/archival_lessons.jsonl"])

        self._write_round_readme(round_dir, round_idx, "Phase 6 ran Analyzer self-evaluation, Memory-TAGE selection, rollback check, and reflection memory writing. Later phases can add long training for selected children.")
        artifacts.append("README.md")
        save_json(
            round_dir / "round_state.json",
            {
                "round": round_idx,
                "phase": "phase_6_self_evolution_memory_tage_rollback_reflection",
                "llm_called": False,
                "memory_coverage_type": coverage.get("coverage_type"),
                "analyzer_executed": True,
                "memory_tage_executed": True,
                "selected_candidate": selection.get("selected_candidate"),
                "rollback_triggered": rollback.get("rollback_triggered"),
                "next_parent_reward_id": rollback.get("next_parent_reward_id"),
                "reflection_written": True,
                "long_training_executed": False,
                "short_training_executed": False,
            },
        )
        artifacts.append("round_state.json")
        summary = RoundSummary(
            round=round_idx,
            status="completed",
            phase="phase_6_self_evolution_memory_tage_rollback_reflection",
            message=f"Round {round_idx} selected {selection.get('selected_candidate')}; rollback={rollback.get('rollback_triggered')}; reflection written.",
            round_dir=str(round_dir),
            artifacts_created=artifacts,
            long_training_executed=False,
            short_training_executed=False,
        )
        save_json(round_dir / "round_summary.json", summary.to_dict())
        self._mark_round_completed(round_idx)
        return summary

    def _round0_static_score(self, mutation_family: str, valid: bool) -> float:
        if not valid:
            return -1.0
        family_prior = {"progress_conditioned": 0.90, "component_recomposition": 0.80, "local_repair": 0.70}
        return family_prior.get(mutation_family, 0.50)

    def _write_round_readme(self, round_dir: Path, round_idx: int, message: str) -> None:
        save_text(round_dir / "README.md", f"# ASE-MTAGE Round {round_idx}\n\n{message}\n")

    def _mark_round_completed(self, round_idx: int) -> None:
        self.state.current_round = round_idx
        self.state.completed_rounds.append(round_idx)
        self.state.last_completed_node = "ROUND_COMPLETED"
        self._save_state("ROUND_COMPLETED")

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
