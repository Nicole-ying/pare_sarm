"""ASE-MTAGE closed-loop pipeline."""

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
from ase_mtage.tools.env_sanitizer import EnvSanitizer
from ase_mtage.tools.evidence_card_builder import EvidenceCardBuilder
from ase_mtage.tools.memory_coverage import MemoryCoverageAnalyzer
from ase_mtage.tools.mtage_evaluator import MemoryTAGEEvaluator
from ase_mtage.tools.reward_validator import RewardValidator
from ase_mtage.tools.rollback import RollbackManager
from ase_mtage.tools.selector import CandidateSelector
from ase_mtage.training.long_trainer import LongTrainer
from ase_mtage.utils.io import ensure_dir, load_config, load_json, load_jsonl, now_timestamp, save_json, save_text


class ASEMTAGEPipeline:
    """Closed-loop ASE-MTAGE pipeline with strict LLM fallback control."""

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
            self.config = ASEMTAGEConfig.from_dict(config if config is not None else load_config(config_path))
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
                "Closed-loop ASE-MTAGE with long training and trajectory-memory accumulation.",
                "If llm.enabled=true and llm.fallback_on_error=false, LLM failures stop the run.",
                "Analyzer receives full round evidence: coverage, trajectory labels, components, selection, TAGE, and memory.",
                "No historical-best-health gate is used.",
            ],
        )

    @property
    def fallback_on_error(self) -> bool:
        return bool(getattr(self.config.llm, "fallback_on_error", True))

    def _resolve_exp_dir(self) -> Path:
        if self.resume_from is not None:
            return self.resume_from
        name = self.config.experiment_name
        if not name:
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
        sanitized = EnvSanitizer().sanitize_env(env_id=self.config.training.env_id, output_dir=self.layout.core_memory_dir)
        EnvPerceptionAgent(
            self.layout.core_memory_dir,
            llm_client=self.llm_client,
            temperature=float(self.config.llm.temperature.get("env_perception", 0.2)),
            fallback_on_error=self.fallback_on_error,
        ).run(env_id=self.config.training.env_id, sanitized_env_summary=sanitized)
        save_text(self.layout.memory_dir / "trajectory_cards.jsonl", "")
        save_text(self.layout.memory_dir / "failure_repair_memory.jsonl", "")
        save_text(self.layout.memory_dir / "archival_lessons.jsonl", "")
        EliteArchive(self.layout.memory_dir / "elite_archive.json", self.layout.elite_rewards_dir)
        save_json(self.layout.memory_dir / "coverage_report.json", {"num_trajectories": 0, "coverage_type": "empty_or_too_small"})
        self._init_logger()
        self._save_state("INIT")

    def _init_logger(self) -> None:
        from ase_mtage.utils.logger import setup_logger

        log_path = self.layout.exp_dir / "run.log"
        log = setup_logger(log_path=log_path, enabled=True)
        log.section(f"ASE-MTAGE | {self.config.method.name} | {self.config.training.env_id}")
        log.info(f"max_rounds={self.config.method.max_rounds} | k_candidates={self.config.method.k_candidates} | full_timesteps={self.config.training.full_timesteps}")
        log.info(f"LLM: enabled={self.config.llm.enabled} | provider={self.config.llm.provider} | model={self.config.llm.model} | timeout={self.config.llm.timeout_seconds}s | fallback_on_error={self.fallback_on_error}")
        log.info(f"exp_dir={self.layout.exp_dir}")

    def run(self, n_rounds: int | None = None) -> dict[str, Any]:
        from ase_mtage.utils.logger import get_logger, setup_logger

        if self.resume_from is not None and (self.layout.exp_dir / "experiment_state.json").exists():
            log_path = self.layout.exp_dir / "run.log"
            setup_logger(log_path=log_path, enabled=True)
            log = get_logger()
            log.info("Resuming experiment...")
            existing_state = load_json(self.layout.exp_dir / "experiment_state.json", default={})
            completed = existing_state.get("completed_rounds", [])
            log.info(f"Previously completed rounds: {completed}")
            self.state = ExperimentState(
                method=self.config.method.name,
                env_id=self.config.training.env_id,
                exp_dir=str(self.exp_dir),
                max_rounds=self.config.method.max_rounds,
                use_short_training=self.config.method.use_short_training,
                selected_long_train_per_round=self.config.method.selected_long_train_per_round,
                current_round=existing_state.get("current_round", -1),
                completed_rounds=completed,
                last_completed_node=existing_state.get("last_completed_node", "INIT"),
            )
        else:
            self.setup_experiment()
            log = get_logger()
        total_rounds = self.config.method.max_rounds if n_rounds is None else n_rounds
        summaries: list[dict[str, Any]] = []
        previous_selection_report: dict[str, Any] | None = None
        for round_idx in range(total_rounds):
            if round_idx == 0:
                log.round_start(round_idx, total_rounds, "bootstrap")
                summary = self.run_round0(round_idx)
            else:
                log.round_start(round_idx, total_rounds, "self_evolution")
                summary = self.run_self_evolution_round(round_idx, previous_selection_report=previous_selection_report)
            log.round_done(round_idx, summary.message)
            summaries.append(summary.to_dict())
            sel_path = self.layout.exp_dir / f"round{round_idx}" / "selection_report.json"
            previous_selection_report = load_json(sel_path, default={}) if sel_path.exists() else None
        log.section("Experiment completed")
        self._save_state("EXPERIMENT_COMPLETED")
        result = {
            "success": True,
            "phase": "closed_loop_ase_mtage",
            "method": self.config.method.name,
            "env_id": self.config.training.env_id,
            "llm_enabled": bool(self.llm_client),
            "llm_fallback_on_error": self.fallback_on_error,
            "exp_dir": str(self.layout.exp_dir),
            "rounds": summaries,
        }
        save_json(self.layout.exp_dir / "summary.json", result)
        return result

    def _load_core_context(self) -> tuple[dict[str, Any], str]:
        env_manifest = load_json(self.layout.core_memory_dir / "env_manifest.json")
        task_manifest_path = self.layout.core_memory_dir / "task_manifest.md"
        task_manifest = task_manifest_path.read_text(encoding="utf-8") if task_manifest_path.exists() else ""
        return env_manifest, task_manifest

    def _generate_and_validate_candidates(
        self,
        *,
        round_idx: int,
        round_dir: Path,
        analyzer_report: dict[str, Any] | None = None,
        parent_reward_code: str | None = None,
        coverage_report: dict[str, Any] | None = None,
        reflection_guidance: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        candidates_root = ensure_dir(round_dir / "candidates")
        artifacts: list[str] = []
        env_manifest, task_manifest = self._load_core_context()
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
            reflection_guidance=reflection_guidance,
        )
        records: list[dict[str, Any]] = []
        validation_results: list[dict[str, Any]] = []
        validator = RewardValidator()
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

    def run_round0(self, round_idx: int) -> RoundSummary:
        from ase_mtage.utils.logger import get_logger as _get_logger

        log = _get_logger()
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []
        selection_report_path = round_dir / "selection_report.json"
        if selection_report_path.exists() and (round_dir / "full_training" / "long_training_result.json").exists():
            log.info(f"Round {round_idx}: selection and training already exist, loading from disk")
            selection = load_json(selection_report_path, default={})
            records = selection.get("candidate_scores", [])
            selected_id = selection.get("selected_candidate")
            selected = next((r for r in records if r.get("candidate_id") == selected_id), None)
            artifacts.extend(["selection_report.json", "candidate_generation_report.json"])
        else:
            records, validation_results, created = self._generate_and_validate_candidates(round_idx=round_idx, round_dir=round_dir)
            artifacts.extend(created)
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
            save_json(selection_report_path, selection)
            save_json(round_dir / "candidate_generation_report.json", {"round": round_idx, "candidates": validation_results, "selected_candidate": selection["selected_candidate"], "llm_called": bool(self.llm_client), "llm_fallback_on_error": self.fallback_on_error})
            artifacts.extend(["selection_report.json", "candidate_generation_report.json"])
        training_result = self._train_selected_and_update_memory(round_idx=round_idx, round_dir=round_dir, selected_record=selected, artifacts=artifacts)
        real_coverage = self._coverage_for_reflection(round_dir)
        self._write_reflection(
            round_idx=round_idx,
            round_dir=round_dir,
            analyzer_report={
                "parent_reward_id": None,
                "self_evaluation_lesson": "Round 0 bootstrapped initial trajectory memory.",
                "mutation_intent": {"primary_family": "bootstrap", "secondary_family": "bootstrap", "required_changes": [], "preserve_components": [], "remove_or_gate_components": []},
            },
            selection_report=selection,
            coverage_report=real_coverage,
            rollback_report={"rollback_triggered": False, "next_parent_reward_id": selection["selected_candidate"], "reason": "bootstrap"},
            artifacts=artifacts,
        )
        self._write_round_readme(round_dir, round_idx, "Round 0 generated, selected, long-trained one reward, and built initial trajectory memory.")
        artifacts.append("README.md")
        save_json(round_dir / "round_state.json", {"round": round_idx, "phase": "round0_bootstrap", "selected_candidate": selection["selected_candidate"], "long_training_executed": training_result["executed"], "long_training_success": training_result["success"], "num_trajectory_cards": training_result["num_cards"], "llm_called": bool(self.llm_client), "llm_fallback_on_error": self.fallback_on_error})
        artifacts.append("round_state.json")
        summary = RoundSummary(round=round_idx, status="completed" if training_result["success"] else "completed_with_training_failure", phase="round0_bootstrap", message=f"Round 0 selected {selection['selected_candidate']}; trajectory_cards={training_result['num_cards']}.", round_dir=str(round_dir), artifacts_created=artifacts, long_training_executed=training_result["executed"], short_training_executed=False)
        save_json(round_dir / "round_summary.json", summary.to_dict())
        self._mark_round_completed(round_idx)
        return summary

    def run_self_evolution_round(self, round_idx: int, previous_selection_report: dict[str, Any] | None = None) -> RoundSummary:
        from ase_mtage.utils.logger import get_logger

        log = get_logger()
        round_dir = ensure_dir(self.layout.exp_dir / f"round{round_idx}")
        artifacts: list[str] = []
        log.info(f"Round {round_idx}: analyzing memory coverage...")
        coverage = self._analyze_coverage(round_dir, artifacts)
        log.info(f"Round {round_idx}: coverage={coverage.get('coverage_type')} | decision={coverage.get('decision_level')} | labeled={coverage.get('num_labeled_non_ambiguous', 0)}")
        archive = EliteArchive(self.layout.memory_dir / "elite_archive.json", self.layout.elite_rewards_dir)
        best = archive.best()  # success_like_count → partial_progress → tage_score
        env_manifest, task_manifest = self._load_core_context()
        parent_code = ""
        if best and best.get("reward_path") and Path(best["reward_path"]).exists():
            parent_code = Path(best["reward_path"]).read_text(encoding="utf-8")
        # Read cross-round memory: previous training results, archival lessons, reflection guidance
        previous_training_results = self._read_previous_training_results(round_idx)
        archival_lessons = load_jsonl(self.layout.memory_dir / "archival_lessons.jsonl")
        recent_archival_lessons = archival_lessons[-5:] if len(archival_lessons) > 5 else archival_lessons
        reflection_guidance = self._read_previous_reflection_guidance(round_idx)
        analyzer_dir = ensure_dir(round_dir / "analyzer")
        latest_summary = self._latest_trajectory_judgment_summary(round_idx)
        latest_component_summary = self._component_summary_from_memory()
        log.info(f"Round {round_idx}: running Analyzer (LLM)...")
        analyzer_report = AnalyzerAgent(
            analyzer_dir,
            llm_client=self.llm_client,
            temperature=float(self.config.llm.temperature.get("analyzer", 0.4)),
            fallback_on_error=self.fallback_on_error,
        ).run(
            round_idx=round_idx,
            parent_reward_id=(best or {}).get("reward_id"),
            coverage_report=coverage,
            previous_selection_report=previous_selection_report,
            trajectory_judgment_summary=latest_summary,
            component_summary=latest_component_summary,
            failure_memory_records=FailureRepairMemory(self.layout.memory_dir / "failure_repair_memory.jsonl").read_recent(limit=5),
            elite_archive=archive.read(),
            task_manifest=task_manifest,
            env_manifest=env_manifest,
            parent_reward_code=parent_code,
            previous_training_results=previous_training_results,
            archival_lessons=recent_archival_lessons,
        )
        artifacts.extend(["analyzer/prompt.txt", "analyzer/response.txt", "analyzer/self_evaluation.json"])
        log.info(f"Round {round_idx}: analyzer done | judgment={analyzer_report.get('overall_judgment', '?')}")
        records, validation_results, created = self._generate_and_validate_candidates(round_idx=round_idx, round_dir=round_dir, analyzer_report=analyzer_report, parent_reward_code=parent_code, coverage_report=coverage, reflection_guidance=reflection_guidance)
        artifacts.extend(created)
        log.info(f"Round {round_idx}: generated {len(records)} candidates | valid={sum(1 for r in records if r['valid'])}")
        save_json(round_dir / "candidate_generation_report.json", {"round": round_idx, "analyzer_self_evaluation_path": str(analyzer_dir / "self_evaluation.json"), "candidates": validation_results, "llm_called": bool(self.llm_client), "llm_fallback_on_error": self.fallback_on_error})
        artifacts.append("candidate_generation_report.json")
        tage_reports = self._run_tage(round_idx=round_idx, round_dir=round_dir, records=records, coverage=coverage, artifacts=artifacts)
        selection = CandidateSelector().select(round_idx=round_idx, candidate_records=records, coverage_report=coverage, output_path=round_dir / "selection_report.json", selection_mode="memory_tage_offline_selection")
        # Backfill selection_score into records so elite archive gets the right score
        sel_scores = {s["candidate_id"]: s.get("selection_score") for s in selection.get("candidate_scores", [])}
        for rec in records:
            if rec.get("candidate_id") in sel_scores:
                rec["selection_score"] = sel_scores[rec["candidate_id"]]
        log.info(f"Round {round_idx}: selected={selection.get('selected_candidate')} | score={selection.get('candidate_scores', [{}])[0].get('selection_score', '?')}")
        artifacts.append("selection_report.json")
        rollback = RollbackManager().check(round_idx=round_idx, current_selection_report=selection, coverage_report=coverage, elite_archive=archive, output_path=round_dir / "rollback_report.json")
        artifacts.append("rollback_report.json")
        selected = self._resolve_selected_record(selection, records, rollback)
        training_result = self._train_selected_and_update_memory(round_idx=round_idx, round_dir=round_dir, selected_record=selected, artifacts=artifacts)
        self._write_reflection(round_idx=round_idx, round_dir=round_dir, analyzer_report=analyzer_report, selection_report=selection, coverage_report=coverage, rollback_report=rollback, artifacts=artifacts, tage_summary={"candidate_tage_reports": tage_reports}, elite_archive=archive.read())
        self._write_round_readme(round_dir, round_idx, "Self-evolution round: Analyzer -> Mutator -> Memory-TAGE -> Rollback -> Long training -> New trajectory memory -> Reflection.")
        artifacts.append("README.md")
        save_json(round_dir / "round_state.json", {"round": round_idx, "phase": "closed_loop_self_evolution", "llm_called": bool(self.llm_client), "llm_fallback_on_error": self.fallback_on_error, "memory_coverage_type": coverage.get("coverage_type"), "decision_level": coverage.get("decision_level"), "selected_candidate": selection.get("selected_candidate"), "rollback_triggered": rollback.get("rollback_triggered"), "next_parent_reward_id": rollback.get("next_parent_reward_id"), "long_training_executed": training_result["executed"], "long_training_success": training_result["success"], "num_trajectory_cards": training_result["num_cards"], "reflection_written": True})
        artifacts.append("round_state.json")
        msg = f"Round {round_idx} selected {selection.get('selected_candidate')}; decision_level={coverage.get('decision_level')}; new_cards={training_result['num_cards']}."
        summary = RoundSummary(round=round_idx, status="completed" if training_result["success"] else "completed_with_training_failure", phase="closed_loop_self_evolution", message=msg, round_dir=str(round_dir), artifacts_created=artifacts, long_training_executed=training_result["executed"], short_training_executed=False)
        save_json(round_dir / "round_summary.json", summary.to_dict())
        self._mark_round_completed(round_idx)
        return summary

    def _latest_trajectory_judgment_summary(self, round_idx: int) -> dict[str, Any]:
        for r in range(round_idx - 1, -1, -1):
            path = self.layout.exp_dir / f"round{r}" / "trajectory_judgment_summary.json"
            if path.exists():
                return load_json(path, default={})
        return {}

    def _read_previous_training_results(self, round_idx: int) -> dict[str, Any] | None:
        """Read training results from all previous rounds to show trend."""
        results: list[dict[str, Any]] = []
        for r in range(round_idx - 1, -1, -1):
            path = self.layout.exp_dir / f"round{r}" / "full_training" / "training_result.json"
            if not path.exists():
                path = self.layout.exp_dir / f"round{r}" / "full_training" / "long_training_result.json"
            if path.exists():
                data = load_json(path, default={})
                data["round"] = r
                results.append(data)
        if not results:
            return None
        results.reverse()  # chronological order
        returns = [r.get("mean_candidate_return") for r in results if r.get("mean_candidate_return") is not None]
        if len(returns) >= 4:
            mid = len(returns) // 2
            first_half_avg = sum(returns[:mid]) / mid
            second_half_avg = sum(returns[mid:]) / (len(returns) - mid)
            if second_half_avg > first_half_avg * 1.05:
                trend = "improving"
            elif second_half_avg < first_half_avg * 0.95:
                trend = "declining"
            else:
                trend = "flat_or_oscillating"
        elif len(returns) >= 2:
            trend = "improving" if returns[-1] > returns[0] else ("declining" if returns[-1] < returns[0] else "flat_or_insufficient_data")
        else:
            trend = "flat_or_insufficient_data"
        return {"previous_rounds": results, "num_rounds_with_training": len(results), "mean_returns_chronological": returns, "trend": trend}

    def _read_previous_reflection_guidance(self, round_idx: int) -> list[str]:
        """Read future_guidance from the most recent previous round's reflection."""
        for r in range(round_idx - 1, -1, -1):
            path = self.layout.exp_dir / f"round{r}" / "reflection" / "reflection.json"
            if path.exists():
                reflection = load_json(path, default={})
                guidance = reflection.get("future_guidance", [])
                if guidance:
                    return guidance
        return []

    def _component_summary_from_memory(self) -> dict[str, Any]:
        cards = load_jsonl(self.layout.memory_dir / "trajectory_cards.jsonl")
        stats: dict[str, dict[str, list[float]]] = {}
        for card in cards:
            label = str(card.get("coarse_label") or (card.get("final_label") or {}).get("coarse_label", "ambiguous"))
            is_failure = label in {"early_failure", "low_progress_survival"}
            is_positive = label in {"partial_progress", "success_like"}
            for name, value in dict(card.get("reward_component_totals") or {}).items():
                item = stats.setdefault(str(name), {"failure_values": [], "positive_values": []})
                try:
                    v = float(value)
                except Exception:
                    continue
                if is_failure:
                    item["failure_values"].append(v)
                if is_positive:
                    item["positive_values"].append(v)
        compact: dict[str, Any] = {}
        for name, item in stats.items():
            fvals = item["failure_values"]
            pvals = item["positive_values"]
            compact[name] = {"failure_mean": sum(fvals) / len(fvals) if fvals else 0.0, "positive_mean": sum(pvals) / len(pvals) if pvals else 0.0, "num_failure": len(fvals), "num_positive": len(pvals)}
        return {"component_stats": compact}

    def _coverage_for_reflection(self, round_dir: Path) -> dict[str, Any]:
        summary_path = round_dir / "trajectory_judgment_summary.json"
        if summary_path.exists():
            summary = load_json(summary_path, default={})
            return {"coverage_type": "bootstrap_after_training", "label_counts": summary.get("label_counts", {}), "num_trajectories": summary.get("num_trajectories", 0), "num_use_for_tage_pair": summary.get("num_use_for_tage_pair", 0)}
        return {"coverage_type": "bootstrap", "label_counts": {}}

    def _analyze_coverage(self, round_dir: Path, artifacts: list[str]) -> dict[str, Any]:
        coverage = MemoryCoverageAnalyzer(
            min_trajectories=self.config.trajectory_memory.min_trajectories,
            min_labeled_trajectories=self.config.trajectory_memory.min_labeled_trajectories,
        ).analyze_file(memory_cards_path=self.layout.memory_dir / "trajectory_cards.jsonl", output_path=round_dir / "coverage_report.json")
        save_json(self.layout.memory_dir / "coverage_report.json", coverage)
        artifacts.extend(["coverage_report.json", "memory/coverage_report.json"])
        return coverage

    def _run_tage(self, *, round_idx: int, round_dir: Path, records: list[dict[str, Any]], coverage: dict[str, Any], artifacts: list[str]) -> list[dict[str, Any]]:
        evaluator = MemoryTAGEEvaluator()
        memory_cards_path = self.layout.memory_dir / "trajectory_cards.jsonl"

        # First pass: collect reward vectors from all valid candidates for novelty comparison
        all_vectors: dict[str, list[float]] = {}
        for record in records:
            if record["valid"]:
                try:
                    all_vectors[record["candidate_id"]] = evaluator.collect_reward_vector(
                        record["reward_path"], memory_cards_path
                    )
                except Exception:
                    all_vectors[record["candidate_id"]] = []

        # Second pass: evaluate each candidate with other vectors for real novelty
        reports: list[dict[str, Any]] = []
        for record in records:
            report_path = Path(record["candidate_dir"]) / "tage_report.json"
            if record["valid"]:
                other_vectors = {k: v for k, v in all_vectors.items() if k != record["candidate_id"]}
                report = evaluator.evaluate_candidate(candidate_id=record["candidate_id"], reward_path=record["reward_path"], memory_cards_path=memory_cards_path, coverage_report=coverage, output_path=report_path, other_reward_vectors=other_vectors)
                record["tage_report_path"] = str(report_path)
                record["tage_score"] = report.get("tage_score")
                reports.append(report)
            else:
                save_json(report_path, {"candidate_id": record["candidate_id"], "valid": False, "tage_score": -1.0, "decision_level": coverage.get("decision_level")})
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
        from ase_mtage.utils.logger import get_logger

        log = get_logger()
        if selected_record is None:
            save_json(round_dir / "full_training_skipped.json", {"skipped": True, "reason": "No selected reward candidate."})
            artifacts.append("full_training_skipped.json")
            return {"executed": False, "success": False, "num_cards": 0}
        if self.dry_run:
            save_json(round_dir / "full_training_skipped.json", {"skipped": True, "reason": "dry_run=True"})
            artifacts.append("full_training_skipped.json")
            return {"executed": False, "success": True, "num_cards": 0}
        full_training_dir = ensure_dir(round_dir / "full_training")
        training_result_path = full_training_dir / "long_training_result.json"
        resume_training = training_result_path.exists()
        if resume_training:
            training_result = load_json(training_result_path, default={})
            if training_result.get("success") and training_result.get("model_path"):
                log.info(f"Round {round_idx}: training already completed, skipping to evidence cards")
                model_path = Path(training_result["model_path"])
                eval_summary_path = Path(training_result["eval_summary_path"]) if training_result.get("eval_summary_path") else None
            else:
                resume_training = False
        if not resume_training:
            result = LongTrainer(env_id=self.config.training.env_id, reward_path=Path(selected_record["reward_path"]), output_dir=full_training_dir, selected_candidate_id=str(selected_record["candidate_id"]), seed=self.config.training.seed + round_idx * 100, full_timesteps=self.config.training.full_timesteps, final_eval_episodes=self.config.training.final_eval_episodes, round_idx=round_idx, device=getattr(self.config.training, "device", "cpu")).run()
            artifacts.extend(["full_training/training_config.json", "full_training/long_training_result.json"])
            if result.model_path:
                artifacts.append(str(result.model_path.relative_to(round_dir)))
            if result.eval_summary_path:
                artifacts.extend([str(result.eval_summary_path.relative_to(round_dir)), "full_training/trajectory_logs/", "full_training/component_logs/"])
            success = result.success
            model_path = result.model_path
            eval_summary_path = result.eval_summary_path
        else:
            log.info(f"Round {round_idx}: resuming from trained model={model_path}")
            artifacts.extend(["full_training/training_config.json", "full_training/long_training_result.json"])
            if model_path and model_path.exists():
                artifacts.append(str(model_path.relative_to(round_dir)))
            if eval_summary_path and eval_summary_path.exists():
                artifacts.extend([str(eval_summary_path.relative_to(round_dir)), "full_training/trajectory_logs/", "full_training/component_logs/"])
            success = True
        num_cards = 0
        if success:
            log.info(f"Round {round_idx}: building evidence cards & trajectory judgments...")
            env_manifest, task_manifest = self._load_core_context()
            card_result = EvidenceCardBuilder(env_id=self.config.training.env_id, llm_client=self.llm_client, judge_temperature=float(self.config.llm.temperature.get("trajectory_judge", 0.2)), task_manifest=task_manifest, env_manifest=env_manifest, output_dir=round_dir / "trajectory_judge", fallback_on_error=self.fallback_on_error).build_from_training_dir(full_training_dir=full_training_dir, round_dir=round_dir, memory_dir=self.layout.memory_dir, source_round=round_idx, source_reward_id=str(selected_record["candidate_id"]))
            num_cards = int(card_result.get("num_cards", 0))
            label_counts = card_result.get("summary", {}).get("label_counts", {})
            num_success_like = int(label_counts.get("success_like", 0))
            num_partial_progress = int(label_counts.get("partial_progress", 0))
            log.info(f"Round {round_idx}: {num_cards} trajectory cards built | labels={label_counts}")
            artifacts.extend(["trajectory_cards.jsonl", "trajectory_judgment.jsonl", "trajectory_judgment_summary.json", "trajectory_judge/", "memory/trajectory_cards.jsonl"])
            score = float(selected_record.get("selection_score", selected_record.get("tage_score", selected_record.get("selection_static_score", 0.0))) or 0.0)
            if not resume_training:
                mean_return = result.mean_candidate_return
                mean_ep_len = result.mean_episode_length
                EliteArchive(self.layout.memory_dir / "elite_archive.json", self.layout.elite_rewards_dir).add_or_update(reward_id=str(selected_record["candidate_id"]), reward_path=selected_record["reward_path"], score=score, round_idx=round_idx, metadata={"num_trajectory_cards": num_cards, "label_counts": label_counts}, training_return=mean_return, num_success_like=num_success_like, num_partial_progress=num_partial_progress)
                save_json(full_training_dir / "training_result.json", {"mean_candidate_return": mean_return, "mean_episode_length": mean_ep_len, "candidate_id": selected_record["candidate_id"], "round": round_idx, "num_success_like": num_success_like, "num_partial_progress": num_partial_progress, "label_counts": label_counts})
                artifacts.append("full_training/training_result.json")
            artifacts.append("memory/elite_archive.json")
        return {"executed": True, "success": success, "num_cards": num_cards}

    def _write_reflection(self, *, round_idx: int, round_dir: Path, analyzer_report: dict[str, Any], selection_report: dict[str, Any], coverage_report: dict[str, Any], rollback_report: dict[str, Any], artifacts: list[str], trajectory_judgment_summary: dict[str, Any] | None = None, tage_summary: dict[str, Any] | None = None, elite_archive: dict[str, Any] | None = None) -> None:
        summary_path = round_dir / "trajectory_judgment_summary.json"
        if trajectory_judgment_summary is None and summary_path.exists():
            trajectory_judgment_summary = load_json(summary_path, default={})
        ReflectionAgent(output_dir=ensure_dir(round_dir / "reflection"), failure_memory_path=self.layout.memory_dir / "failure_repair_memory.jsonl", archival_lessons_path=self.layout.memory_dir / "archival_lessons.jsonl", llm_client=self.llm_client, temperature=float(self.config.llm.temperature.get("reflector", 0.3)), fallback_on_error=self.fallback_on_error).run(round_idx=round_idx, analyzer_report=analyzer_report, selection_report=selection_report, coverage_report=coverage_report, rollback_report=rollback_report, trajectory_judgment_summary=trajectory_judgment_summary, tage_summary=tage_summary, elite_archive=elite_archive)
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


def run_phase1(*, config_path: str | Path | None = None, output_root: str | Path | None = None, n_rounds: int | None = None, experiment_name: str | None = None) -> dict[str, Any]:
    raw_config = load_config(config_path)
    if experiment_name:
        raw_config["experiment_name"] = experiment_name
    pipeline = ASEMTAGEPipeline(raw_config, config_path=config_path, output_root=output_root, dry_run=False)
    return pipeline.run(n_rounds=n_rounds)
