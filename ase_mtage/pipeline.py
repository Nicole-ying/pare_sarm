"""ASE-MTAGE pipeline.

This file now implements the runnable closed-loop version of the framework:

Round 0:
- generate/validate K rewards;
- select one reward;
- long-train it;
- collect trajectories/components;
- build evidence cards and trajectory memory;
- update elite archive and reflection memory.

Round 1+:
- analyze memory coverage;
- generate/validate K children, optionally with LLM Mutator;
- run offline Memory-TAGE;
- select top-1;
- rollback check;
- long-train the chosen parent/child;
- collect new trajectories and append memory;
- update elite archive and reflection memory.

The code can run without LLM. When llm.enabled=true and DEEPSEEK_API_KEY is set,
the Mutator Agent uses DeepSeek/OpenAI-compatible chat completion to generate
reward code. Deterministic fallbacks remain available for stable debugging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase_mtage.agents.analyzer import AnalyzerAgent
from ase_mtage.agents.env_perception import EnvPerceptionAgent
from ase_mtage.agents.mutator import MutatorAgent
from ase_mtage.agents.reflector import ReflectionAgent
from ase_mtage.llm_client import LLMClient
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
    """Closed-loop ASE-MTAGE pipeline."""

    def __init__(
        self,
        config: ASEMTAGEConfig | dict[str, Any] | None = None,
        *,
        config_path: str | Path | None = None,
        output_root: str | Path | None = None,
        resume_from: str | Path | None = None,
        dry_run: bool = False,
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
        self.dry_run = bool(dry_run)
        self.exp_dir = self._resolve_exp_dir()
        self.layout = ExperimentLayout.from_exp_dir(self.exp_dir)
        self.llm_client = LLMClient.from_config(self.config.llm)
        self.state = ExperimentState(
            method=self.config.method.name,
            env_id=self.config.training.env_id,
            exp_dir=str(self.exp_dir),
            max_rounds=self.config.method.max_rounds,
            use_short_training=self.config.method.use_short_training,
            selected_long_train_per_round=self.config.method.selected_long_train_per_round,
            notes=[
                "Closed-loop ASE-MTAGE: later selected children are long-trained and appended to trajectory memory.",
                "If llm.enabled=true, Mutator uses DeepSeek/OpenAI-compatible API; otherwise deterministic templates are used.",
                "Historical best health is not used as a gate.",
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
        save_json(self.layout.memory_dir / "coverage_report.json", {"num_trajectories": 0, "coverage_type": "empty_or_too_small"})
        self._save_state("INIT")

    def run(self, n_rounds: int | None = None) -> dict[str, Any]:
        self.setup_experiment()
        total_rounds = n_rounds if n_rounds is not None else self.config.method.max_rounds
        if total_rounds < 0:
            raise ValueError("n_rounds must be non-negative")
        summaries: list[dict[str, Any]] = []
        for round_idx in range(total_rounds):
            if round_idx == 0:
                summary = self.run_round0(round_idx)
            else:
                summary = self.run_self_evolution_round(round_idx)
            summaries.append(summary.to_dict())
        self._save_state("EXPERIMENT_COMPLETED")
        final_summary = {
            "success": True,
            "phase": "closed_loop_ase_mtage",
            "method": self.config.method.name,
            "env_id": self.config.training.env_id,
            "llm_enabled": bool(self.llm_client),
            "exp_dir": str(self.layout.exp_dir),
            "rounds": summaries,
        }
        save_json(self.layout.exp_dir / "summary.json", final_summary)
        return final_summary

    def _generate_and_validate_candidates(
        self,
        *,
        round_idx: int,
        round_dir: Path,
        analyzer_report: dict[str, Any] | None = None,
        parent_reward_code: str | None = None,
        coverage_report: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        candidates_root = ensure_dir(round_dir / "candidates")
        artifacts: list[str] = []
        env_manifest = load_json(self.layout.core_memory_dir / "env_manifest.json")
        task_manifest_path = self.layout.core_memory_dir / "task_manifest.md"
        task_manifest = task_manifest_path.read_text(encoding="utf-8") if task_manifest_path.exists() else ""
        mutator = MutatorAgent(
            candidates_root,
            llm_client=self.llm_client,
            temperature=float(self.config.llm.temperature.get("mutator", 0.6)),
        )
        candidates = mutator.generate_candidates(
            env_manifest=env_manifest,
            k_candidates=self.config.method.k_candidates,
            round_idx=round_idx,
            analyzer_report=analyzer_report,
            parent_reward_code=parent_reward_code,
            task_manifest=task_manifest,
            coverage_report=coverage_report,
        )
        validator = RewardValidator()
        validation_results: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            report_path = cand.candidate_dir / "validator_report.json"
            result = validator.validate_file(cand.reward_path, candidate_id=cand.candidate_id, report_path=report_path)
            validation_results.append(result.to_dict())
            records.append({
                "index": idx,
                "candidate_id": cand.candidate_id,
                "mutation_family": cand.mutation_family,
                "candidate_dir": str(cand.candidate_dir),
                "reward_path": str(cand.reward_path),
                "metadata_path": str(cand.metadata_path),
                "validator_report_path": str(report_path),
                "valid": result.valid,
                "selection_static_score": self._round0_static_score(cand.mutation_family, result.valid),
            })
            artifacts.extend([str(cand.reward_path.relative_to(round_dir)), str(cand.metadata_path.relative_to(round_dir)), str(report_path.relative_to(round_dir))])
        return records, validation_results, artifacts

    def run_round0(self, round_idx: int) -> RoundSummary:
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []
        records, validation_results, cand_artifacts = self._generate_and_validate_candidates(round_idx=round_idx, round_dir=round_dir)
        artifacts.extend(cand_artifacts)
        valid_records = [r for r in records if r["valid"]]
        selected = max(valid_records, key=lambda r: r["selection_static_score"]) if valid_records else None
        selection = {
            "round": round_idx,
            "selection_mode": "round0_static_selection",
            "memory_tage_used": False,
            "selected_candidate": selected["candidate_id"] if selected else None,
            "candidate_scores": records,
            "reason": "Round 0 has no trajectory memory yet; use static selection.",
        }
        save_json(round_dir / "selection_report.json", selection)
        save_json(round_dir / "candidate_generation_report.json", {"round": round_idx, "candidates": validation_results, "selected_candidate": selection["selected_candidate"], "llm_called": bool(self.llm_client)})
        artifacts.extend(["selection_report.json", "candidate_generation_report.json"])
        training_result = self._train_selected_and_update_memory(round_idx=round_idx, round_dir=round_dir, selected_record=selected, artifacts=artifacts)
        self._write_reflection(round_idx=round_idx, round_dir=round_dir, analyzer_report={"parent_reward_id": None, "self_evaluation_lesson": "Round 0 bootstrapped initial trajectory memory.", "mutation_intent": {"required_changes": []}}, selection_report=selection, coverage_report={"coverage_type": "bootstrap", "label_counts": {}}, rollback_report={"rollback_triggered": False, "next_parent_reward_id": selection["selected_candidate"], "reason": "bootstrap"}, artifacts=artifacts)
        self._write_round_readme(round_dir, round_idx, "Round 0 generated, selected, long-trained one reward, and built initial trajectory memory.")
        artifacts.append("README.md")
        state = {"round": round_idx, "phase": "round0_bootstrap", "selected_candidate": selection["selected_candidate"], "long_training_executed": training_result["executed"], "long_training_success": training_result["success"], "num_trajectory_cards": training_result["num_cards"], "memory_tage_executed": False}
        save_json(round_dir / "round_state.json", state)
        artifacts.append("round_state.json")
        summary = RoundSummary(round=round_idx, status="completed" if training_result["success"] else "completed_with_training_failure", phase="round0_bootstrap", message=f"Round 0 selected {selection['selected_candidate']}; trajectory_cards={training_result['num_cards']}.", round_dir=str(round_dir), artifacts_created=artifacts, long_training_executed=training_result["executed"], short_training_executed=False)
        save_json(round_dir / "round_summary.json", summary.to_dict())
        self._mark_round_completed(round_idx)
        return summary

    def run_self_evolution_round(self, round_idx: int) -> RoundSummary:
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []
        coverage = self._analyze_coverage(round_dir, artifacts)
        archive = EliteArchive(self.layout.memory_dir / "elite_archive.json", self.layout.elite_rewards_dir)
        best = archive.best()
        analyzer_dir = ensure_dir(round_dir / "analyzer")
        analyzer_report = AnalyzerAgent(analyzer_dir).run(
            round_idx=round_idx,
            parent_reward_id=(best or {}).get("reward_id"),
            coverage_report=coverage,
            previous_selection_report=None,
            failure_memory_records=FailureRepairMemory(self.layout.memory_dir / "failure_repair_memory.jsonl").read_recent(limit=5),
            elite_archive=archive.read(),
        )
        artifacts.extend(["analyzer/prompt.txt", "analyzer/response.txt", "analyzer/self_evaluation.json"])
        parent_code = ""
        if best and best.get("reward_path") and Path(best["reward_path"]).exists():
            parent_code = Path(best["reward_path"]).read_text(encoding="utf-8")
        records, validation_results, cand_artifacts = self._generate_and_validate_candidates(round_idx=round_idx, round_dir=round_dir, analyzer_report=analyzer_report, parent_reward_code=parent_code, coverage_report=coverage)
        artifacts.extend(cand_artifacts)
        save_json(round_dir / "candidate_generation_report.json", {"round": round_idx, "analyzer_self_evaluation_path": str(analyzer_dir / "self_evaluation.json"), "candidates": validation_results, "llm_called": bool(self.llm_client)})
        artifacts.append("candidate_generation_report.json")
        tage_reports = self._run_tage(round_idx=round_idx, round_dir=round_dir, records=records, coverage=coverage, artifacts=artifacts)
        selection = CandidateSelector().select(round_idx=round_idx, candidate_records=records, coverage_report=coverage, output_path=round_dir / "selection_report.json", selection_mode="memory_tage_offline_selection")
        artifacts.append("selection_report.json")
        rollback = RollbackManager().check(round_idx=round_idx, current_selection_report=selection, coverage_report=coverage, elite_archive=archive, output_path=round_dir / "rollback_report.json")
        artifacts.append("rollback_report.json")
        selected = self._resolve_selected_record(selection, records, rollback)
        training_result = self._train_selected_and_update_memory(round_idx=round_idx, round_dir=round_dir, selected_record=selected, artifacts=artifacts)
        self._write_reflection(round_idx=round_idx, round_dir=round_dir, analyzer_report=analyzer_report, selection_report=selection, coverage_report=coverage, rollback_report=rollback, artifacts=artifacts)
        self._write_round_readme(round_dir, round_idx, "Self-evolution round: Analyzer -> Mutator -> Memory-TAGE -> Rollback -> Long training -> New trajectory memory -> Reflection.")
        artifacts.append("README.md")
        save_json(round_dir / "round_state.json", {"round": round_idx, "phase": "closed_loop_self_evolution", "llm_called": bool(self.llm_client), "memory_coverage_type": coverage.get("coverage_type"), "selected_candidate": selection.get("selected_candidate"), "rollback_triggered": rollback.get("rollback_triggered"), "next_parent_reward_id": rollback.get("next_parent_reward_id"), "long_training_executed": training_result["executed"], "long_training_success": training_result["success"], "num_trajectory_cards": training_result["num_cards"], "reflection_written": True})
        artifacts.append("round_state.json")
        summary = RoundSummary(round=round_idx, status="completed" if training_result["success"] else "completed_with_training_failure", phase="closed_loop_self_evolution", message=f"Round {round_idx} selected {selection.get('selected_candidate')}; rollback={rollback.get('rollback_triggered')}; new_cards={training_result['num_cards']}.", round_dir=str(round_dir), artifacts_created=artifacts, long_training_executed=training_result["executed"], short_training_executed=False)
        save_json(round_dir / "round_summary.json", summary.to_dict())
        self._mark_round_completed(round_idx)
        return summary

    def _analyze_coverage(self, round_dir: Path, artifacts: list[str]) -> dict[str, Any]:
        coverage = MemoryCoverageAnalyzer(
            min_trajectories=self.config.trajectory_memory.min_trajectories,
            min_high_confidence_trajectories=self.config.trajectory_memory.min_high_confidence_trajectories,
            confidence_threshold=self.config.trajectory_memory.label_confidence_threshold,
        ).analyze_file(memory_cards_path=self.layout.memory_dir / "trajectory_cards.jsonl", output_path=round_dir / "coverage_report.json")
        save_json(self.layout.memory_dir / "coverage_report.json", coverage)
        artifacts.extend(["coverage_report.json", "memory/coverage_report.json"])
        return coverage

    def _run_tage(self, *, round_idx: int, round_dir: Path, records: list[dict[str, Any]], coverage: dict[str, Any], artifacts: list[str]) -> list[dict[str, Any]]:
        evaluator = MemoryTAGEEvaluator()
        reports: list[dict[str, Any]] = []
        for record in records:
            report_path = Path(record["candidate_dir"]) / "tage_report.json"
            if record["valid"]:
                report = evaluator.evaluate_candidate(candidate_id=record["candidate_id"], reward_path=record["reward_path"], memory_cards_path=self.layout.memory_dir / "trajectory_cards.jsonl", coverage_report=coverage, output_path=report_path, other_reward_vectors={})
                record["tage_report_path"] = str(report_path)
                record["tage_score"] = report.get("tage_score")
                reports.append(report)
            else:
                save_json(report_path, {"candidate_id": record["candidate_id"], "valid": False, "tage_score": -1.0})
                record["tage_report_path"] = str(report_path)
                record["tage_score"] = -1.0
            artifacts.append(str(report_path.relative_to(round_dir)))
        save_json(round_dir / "tage_summary.json", {"round": round_idx, "coverage_report": coverage, "candidate_tage_reports": reports})
        artifacts.append("tage_summary.json")
        return reports

    def _resolve_selected_record(self, selection: dict[str, Any], records: list[dict[str, Any]], rollback: dict[str, Any]) -> dict[str, Any] | None:
        if rollback.get("rollback_triggered") and rollback.get("next_parent_reward_path"):
            return {"candidate_id": rollback.get("next_parent_reward_id"), "reward_path": rollback.get("next_parent_reward_path"), "selection_score": rollback.get("hard_conditions", {}).get("best_elite_score", 0.0)}
        selected_id = selection.get("selected_candidate")
        return next((r for r in records if r.get("candidate_id") == selected_id), None)

    def _train_selected_and_update_memory(self, *, round_idx: int, round_dir: Path, selected_record: dict[str, Any] | None, artifacts: list[str]) -> dict[str, Any]:
        if selected_record is None:
            save_json(round_dir / "full_training_skipped.json", {"skipped": True, "reason": "No selected reward candidate."})
            artifacts.append("full_training_skipped.json")
            return {"executed": False, "success": False, "num_cards": 0}
        if self.dry_run:
            save_json(round_dir / "full_training_skipped.json", {"skipped": True, "reason": "dry_run=True"})
            artifacts.append("full_training_skipped.json")
            return {"executed": False, "success": True, "num_cards": 0}
        full_training_dir = ensure_dir(round_dir / "full_training")
        result = LongTrainer(env_id=self.config.training.env_id, reward_path=Path(selected_record["reward_path"]), output_dir=full_training_dir, selected_candidate_id=str(selected_record["candidate_id"]), seed=self.config.training.seed + round_idx * 100, full_timesteps=self.config.training.full_timesteps, final_eval_episodes=self.config.training.final_eval_episodes).run()
        artifacts.extend(["full_training/training_config.json", "full_training/long_training_result.json"])
        if result.model_path:
            artifacts.append(str(result.model_path.relative_to(round_dir)))
        if result.eval_summary_path:
            artifacts.extend([str(result.eval_summary_path.relative_to(round_dir)), "full_training/trajectory_logs/", "full_training/component_logs/"])
        num_cards = 0
        if result.success:
            card_result = EvidenceCardBuilder(env_id=self.config.training.env_id, confidence_threshold=self.config.trajectory_memory.label_confidence_threshold).build_from_training_dir(full_training_dir=full_training_dir, round_dir=round_dir, memory_dir=self.layout.memory_dir, source_round=round_idx, source_reward_id=str(selected_record["candidate_id"]))
            num_cards = int(card_result.get("num_cards", 0))
            artifacts.extend(["trajectory_cards.jsonl", "trajectory_judgment.jsonl", "trajectory_judgment_summary.json", "memory/trajectory_cards.jsonl"])
            score = float(selected_record.get("selection_score", selected_record.get("tage_score", selected_record.get("selection_static_score", 0.0))) or 0.0)
            EliteArchive(self.layout.memory_dir / "elite_archive.json", self.layout.elite_rewards_dir).add_or_update(reward_id=str(selected_record["candidate_id"]), reward_path=selected_record["reward_path"], score=score, round_idx=round_idx, metadata={"num_trajectory_cards": num_cards})
            artifacts.append("memory/elite_archive.json")
        return {"executed": True, "success": result.success, "num_cards": num_cards}

    def _write_reflection(self, *, round_idx: int, round_dir: Path, analyzer_report: dict[str, Any], selection_report: dict[str, Any], coverage_report: dict[str, Any], rollback_report: dict[str, Any], artifacts: list[str]) -> None:
        ReflectionAgent(output_dir=ensure_dir(round_dir / "reflection"), failure_memory_path=self.layout.memory_dir / "failure_repair_memory.jsonl", archival_lessons_path=self.layout.memory_dir / "archival_lessons.jsonl").run(round_idx=round_idx, analyzer_report=analyzer_report, selection_report=selection_report, coverage_report=coverage_report, rollback_report=rollback_report)
        artifacts.extend(["reflection/reflection.json", "memory/failure_repair_memory.jsonl", "memory/archival_lessons.jsonl"])

    def _round0_static_score(self, mutation_family: str, valid: bool) -> float:
        if not valid:
            return -1.0
        return {"progress_conditioned": 0.90, "component_recomposition": 0.80, "local_repair": 0.70}.get(mutation_family, 0.50)

    def _write_round_readme(self, round_dir: Path, round_idx: int, message: str) -> None:
        save_text(round_dir / "README.md", f"# ASE-MTAGE Round {round_idx}\n\n{message}\n")

    def _mark_round_completed(self, round_idx: int) -> None:
        self.state.current_round = round_idx
        self.state.completed_rounds.append(round_idx)
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
    raw_config = load_config(config_path)
    if experiment_name:
        raw_config["experiment_name"] = experiment_name
    pipeline = ASEMTAGEPipeline(raw_config, config_path=config_path, output_root=output_root, dry_run=False)
    return pipeline.run(n_rounds=n_rounds)
