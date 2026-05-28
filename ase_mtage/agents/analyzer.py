"""Analyzer Agent for ASE-MTAGE.

Analyzer is the reward-diagnosis agent. It must not only read a coverage report;
it also receives previous selection, trajectory judgment summaries, component
summaries, TAGE reports, failure memory, and elite archive evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase_mtage.llm_client import LLMClient, extract_json_object, load_prompt
from ase_mtage.utils.io import ensure_dir, save_json, save_text


class AnalyzerAgent:
    """Produce structured reward self-evaluation and mutation intent."""

    def __init__(self, output_dir: str | Path, *, llm_client: LLMClient | None = None, temperature: float = 0.4, fallback_on_error: bool = True) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.llm_client = llm_client
        self.temperature = temperature
        self.fallback_on_error = fallback_on_error

    def run(
        self,
        *,
        round_idx: int,
        parent_reward_id: str | None,
        coverage_report: dict[str, Any],
        previous_selection_report: dict[str, Any] | None = None,
        trajectory_judgment_summary: dict[str, Any] | None = None,
        component_summary: dict[str, Any] | None = None,
        failure_memory_records: list[dict[str, Any]] | None = None,
        elite_archive: dict[str, Any] | None = None,
        task_manifest: str | None = None,
        env_manifest: dict[str, Any] | None = None,
        parent_reward_code: str | None = None,
        tage_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.llm_client is not None:
            try:
                return self._run_llm(
                    round_idx=round_idx,
                    parent_reward_id=parent_reward_id,
                    coverage_report=coverage_report,
                    previous_selection_report=previous_selection_report,
                    trajectory_judgment_summary=trajectory_judgment_summary,
                    component_summary=component_summary,
                    failure_memory_records=failure_memory_records,
                    elite_archive=elite_archive,
                    task_manifest=task_manifest,
                    env_manifest=env_manifest,
                    parent_reward_code=parent_reward_code,
                    tage_summary=tage_summary,
                )
            except Exception as exc:
                save_text(self.output_dir / "llm_error.txt", str(exc) + "\n")
                if not self.fallback_on_error:
                    raise RuntimeError("AnalyzerAgent LLM failed and fallback_on_error=false") from exc
        return self._run_deterministic(
            round_idx=round_idx,
            parent_reward_id=parent_reward_id,
            coverage_report=coverage_report,
            previous_selection_report=previous_selection_report,
            trajectory_judgment_summary=trajectory_judgment_summary,
            component_summary=component_summary,
            failure_memory_records=failure_memory_records,
            elite_archive=elite_archive,
            env_manifest=env_manifest,
            tage_summary=tage_summary,
        )

    def _run_llm(self, **kwargs: Any) -> dict[str, Any]:
        template = load_prompt("analyzer.md")
        input_artifacts = {
            "round": kwargs["round_idx"],
            "parent_reward_id": kwargs.get("parent_reward_id"),
            "task_manifest": kwargs.get("task_manifest") or "",
            "env_manifest": kwargs.get("env_manifest") or {},
            "parent_reward_code": kwargs.get("parent_reward_code") or "",
            "coverage_report": kwargs.get("coverage_report") or {},
            "trajectory_judgment_summary": kwargs.get("trajectory_judgment_summary") or {},
            "component_summary": kwargs.get("component_summary") or {},
            "tage_summary": kwargs.get("tage_summary") or {},
            "previous_selection_report": kwargs.get("previous_selection_report") or {},
            "failure_repair_memory_recent": kwargs.get("failure_memory_records") or [],
            "elite_archive": kwargs.get("elite_archive") or {},
        }
        user_prompt = template.replace("{input_artifacts}", json.dumps(input_artifacts, ensure_ascii=False, indent=2))
        save_text(self.output_dir / "prompt.txt", user_prompt)
        resp = self.llm_client.chat(system_prompt="You are the ASE-MTAGE Analyzer Agent. Output only valid JSON.", user_prompt=user_prompt, temperature=self.temperature)
        save_text(self.output_dir / "response.txt", resp.content)
        save_json(self.output_dir / "llm_raw_response.json", resp.raw)
        evaluation = extract_json_object(resp.content)
        evaluation.setdefault("agent_mode", "llm_analyzer")
        save_json(self.output_dir / "self_evaluation.json", evaluation)
        return evaluation

    def _run_deterministic(
        self,
        *,
        round_idx: int,
        parent_reward_id: str | None,
        coverage_report: dict[str, Any],
        previous_selection_report: dict[str, Any] | None = None,
        trajectory_judgment_summary: dict[str, Any] | None = None,
        component_summary: dict[str, Any] | None = None,
        failure_memory_records: list[dict[str, Any]] | None = None,
        elite_archive: dict[str, Any] | None = None,
        env_manifest: dict[str, Any] | None = None,
        tage_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        coverage_type = str(coverage_report.get("coverage_type", "ambiguous"))
        label_counts = dict(coverage_report.get("label_counts") or (trajectory_judgment_summary or {}).get("label_counts") or {})
        known_failures = [label for label in ["early_failure", "low_progress_survival"] if label_counts.get(label, 0) > 0]
        useful_patterns: list[str] = []
        if label_counts.get("partial_progress", 0) > 0:
            useful_patterns.append("partial_progress trajectories exist and should be preserved or improved")
        if label_counts.get("success_like", 0) > 0:
            useful_patterns.append("success_like trajectories exist and should be strongly preferred")
        component_diagnosis = self._component_diagnosis(previous_selection_report, tage_summary, coverage_report, component_summary)
        preserve_components, remove_or_gate_components = self._component_actions(component_diagnosis, known_failures, coverage_type, env_manifest)
        mutation_intent = self._mutation_intent(coverage_type, known_failures, useful_patterns, preserve_components, remove_or_gate_components)
        overall = "memory_insufficient_for_strong_diagnosis" if coverage_type in {"empty_or_too_small", "ambiguous"} else "parent_reward_provides_usable_memory_but_needs_evolution"
        uncertainties = list(coverage_report.get("forbidden_assumptions") or [])
        if not useful_patterns:
            uncertainties.append("Memory may still be too weak for positive reward-design conclusions.")
        if (trajectory_judgment_summary or {}).get("num_use_for_tage_pair", 0) == 0:
            uncertainties.append("The latest round contributed no usable TAGE preference trajectories.")
        evaluation = {
            "round": round_idx,
            "parent_reward_id": parent_reward_id,
            "overall_judgment": overall,
            "failure_summary": f"coverage_type={coverage_type}; label_counts={label_counts}; decision_level={coverage_report.get('decision_level', 'unknown')}.",
            "memory_interpretation": {
                "coverage_type": coverage_type,
                "decision_level": coverage_report.get("decision_level", "unknown"),
                "usable_preference_level": self._usable_preference_level(coverage_report.get("decision_level")),
                "main_known_failures": known_failures,
                "main_useful_patterns": useful_patterns,
                "uncertainties": sorted(set(uncertainties)),
                "label_counts": label_counts,
                "trajectory_judgment_summary": trajectory_judgment_summary or {},
            },
            "component_diagnosis": component_diagnosis,
            "mutation_intent": mutation_intent,
            "rollback_decision": {"recommend_rollback": False, "rollback_target": None, "reason": "Hard rollback is handled by RollbackManager, not Analyzer."},
            "self_evaluation_lesson": self._lesson(coverage_type, known_failures, useful_patterns),
            "previous_selection_used": previous_selection_report or {},
            "component_summary_used": component_summary or {},
            "recent_failure_memory_used": failure_memory_records or [],
            "elite_archive_summary": elite_archive or {},
            "agent_mode": "deterministic_full_evidence_analyzer",
        }
        save_text(self.output_dir / "prompt.txt", "Deterministic full-evidence AnalyzerAgent; no LLM prompt was sent.\n")
        save_text(self.output_dir / "response.txt", "Deterministic self_evaluation.json generated from coverage, trajectories, components, selection, and memory.\n")
        save_json(self.output_dir / "self_evaluation.json", evaluation)
        return evaluation

    def _component_diagnosis(self, selection_report: dict[str, Any] | None, tage_summary: dict[str, Any] | None, coverage_report: dict[str, Any], component_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
        diagnosis: list[dict[str, Any]] = []
        selected_report = self._selected_tage_report(selection_report, tage_summary)
        if selected_report:
            pref = (selected_report.get("preference_consistency") or {}).get("score")
            avoid_score = (selected_report.get("failure_avoidance") or {}).get("normalized_score")
            comp = selected_report.get("component_alignment") or {}
            if pref is not None:
                diagnosis.append({"component": "global_reward_ranking", "verdict": "keep" if float(pref or 0) >= 0.6 else "restructure", "evidence": f"preference_consistency={pref}"})
            if avoid_score is not None:
                diagnosis.append({"component": "failure_avoidance", "verdict": "keep" if float(avoid_score or 0) >= 0.55 else "strengthen", "evidence": f"avoidance_score={avoid_score}"})
            for name, report in (comp.get("components") or {}).items():
                consistency = float(report.get("pair_consistency", 0.0) or 0.0)
                verdict = "keep" if consistency >= 0.65 else ("remove_or_gate" if consistency <= 0.35 else "unknown")
                diagnosis.append({"component": name, "verdict": verdict, "evidence": f"component_pair_consistency={consistency:.3f}; {report.get('diagnosis', '')}"})
        if component_summary:
            for name, item in dict(component_summary.get("component_stats", {}) or {}).items():
                failure_mean = float(item.get("failure_mean", 0.0) or 0.0)
                positive_mean = float(item.get("positive_mean", 0.0) or 0.0)
                if failure_mean > positive_mean and item.get("num_failure", 0) > 0:
                    diagnosis.append({"component": name, "verdict": "remove_or_gate", "evidence": f"component_summary: failure_mean={failure_mean:.3f} > positive_mean={positive_mean:.3f}"})
                elif positive_mean > failure_mean and item.get("num_positive", 0) > 0:
                    diagnosis.append({"component": name, "verdict": "keep", "evidence": f"component_summary: positive_mean={positive_mean:.3f} > failure_mean={failure_mean:.3f}"})
        if not diagnosis:
            coverage_type = str(coverage_report.get("coverage_type", "ambiguous"))
            label_counts = dict(coverage_report.get("label_counts") or {})
            if label_counts.get("low_progress_survival", 0) > 0:
                diagnosis.append({"component": "ungated_survival_or_contact_bonus", "verdict": "remove_or_gate", "evidence": "low_progress_survival trajectories exist in memory; avoid rewarding time/contact without progress"})
            if label_counts.get("early_failure", 0) > 0:
                diagnosis.append({"component": "terminal_failure_penalty", "verdict": "strengthen", "evidence": "early_failure trajectories exist in memory"})
            if label_counts.get("partial_progress", 0) > 0:
                diagnosis.append({"component": "progress_delta", "verdict": "keep", "evidence": "partial_progress trajectories exist in memory"})
            if coverage_type in {"empty_or_too_small", "ambiguous"}:
                diagnosis.append({"component": "all_components", "verdict": "unknown", "evidence": f"coverage_type={coverage_type}; component-specific evidence is weak"})
        return diagnosis

    def _selected_tage_report(self, selection_report: dict[str, Any] | None, tage_summary: dict[str, Any] | None) -> dict[str, Any] | None:
        if not tage_summary:
            return None
        selected_id = (selection_report or {}).get("selected_candidate")
        reports = tage_summary.get("candidate_tage_reports") or []
        if selected_id:
            for report in reports:
                if report.get("candidate_id") == selected_id:
                    return report
        return reports[0] if reports else None

    def _component_actions(self, component_diagnosis: list[dict[str, Any]], known_failures: list[str], coverage_type: str, env_manifest: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        preserve, remove = [], []
        for item in component_diagnosis:
            name = str(item.get("component", ""))
            verdict = str(item.get("verdict", "unknown"))
            if verdict in {"keep", "keep_or_improve", "strengthen"} and name not in {"all_components", "global_reward_ranking", "failure_avoidance"}:
                preserve.append(name)
            if verdict in {"remove_or_gate", "restructure"} and name not in {"all_components", "global_reward_ranking"}:
                remove.append(name)
        if "early_failure" in known_failures:
            preserve.append("terminal_failure_penalty")
        if "low_progress_survival" in known_failures:
            remove.extend(["ungated_survival_bonus", "ungated_contact_bonus", "low_progress_survival_gate"])
        if coverage_type in {"failure_plus_partial_progress", "balanced"}:
            preserve.append("progress_delta")
        if "bipedalwalker" in str((env_manifest or {}).get("env_name", "")).lower():
            preserve.append("forward_velocity")
        return sorted(set(preserve)), sorted(set(remove))

    def _mutation_intent(self, coverage_type: str, known_failures: list[str], useful_patterns: list[str], preserve: list[str], remove: list[str]) -> dict[str, Any]:
        decision_family = {
            "balanced": "progress_conditioned",
            "failure_plus_partial_progress": "progress_conditioned",
            "failure_plus_weak_or_noisy_partial": "component_recomposition",
            "single_failure_mode": "component_recomposition",
            "multiple_failure_modes": "component_recomposition",
        }.get(coverage_type, "local_repair")
        required = []
        if known_failures:
            required.append(f"explicitly avoid known failures: {', '.join(known_failures)}")
        if useful_patterns:
            required.append("preserve useful patterns: " + "; ".join(useful_patterns))
        if remove:
            required.append("gate or remove components: " + ", ".join(remove))
        if preserve:
            required.append("preserve or strengthen components: " + ", ".join(preserve))
        if not required:
            required.append("keep mutation conservative and gather more reliable trajectory evidence")
        return {"primary_family": decision_family, "secondary_family": "component_recomposition" if decision_family != "component_recomposition" else "progress_conditioned", "forbidden_changes": ["do not use official reward", "do not only scale all coefficients", "do not add global survival bonus without progress gating", "do not treat noisy partial_progress as success_like"], "required_changes": required, "preserve_components": preserve, "remove_or_gate_components": remove}

    def _usable_preference_level(self, decision_level: Any) -> str:
        return {"no_decision": "none", "failure_filter_only": "failure_only", "weak_pairwise_selection": "weak_pairwise", "strong_pairwise_selection": "strong_pairwise"}.get(str(decision_level), "none")

    def _lesson(self, coverage_type: str, known_failures: list[str], useful_patterns: list[str]) -> str:
        if coverage_type in {"failure_plus_partial_progress", "balanced"}:
            return "Trajectory memory supports preference-guided mutation, but partial progress must remain distinct from success."
        if known_failures:
            return "Current memory mainly supports avoiding known failures, not claiming success."
        return "Memory is weak or noisy; mutate conservatively and collect more long-training trajectories."
