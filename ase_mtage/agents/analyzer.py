"""Analyzer Agent for ASE-MTAGE.

When an LLM client is provided, this agent uses `prompts/analyzer.md` and writes
LLM prompt/response artifacts. Otherwise it falls back to deterministic schema-
compatible analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase_mtage.llm_client import LLMClient, extract_json_object, load_prompt
from ase_mtage.utils.io import ensure_dir, save_json, save_text


class AnalyzerAgent:
    """Produce structured reward self-evaluation and mutation intent."""

    def __init__(self, output_dir: str | Path, *, llm_client: LLMClient | None = None, temperature: float = 0.4) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.llm_client = llm_client
        self.temperature = temperature

    def run(
        self,
        *,
        round_idx: int,
        parent_reward_id: str | None,
        coverage_report: dict[str, Any],
        previous_selection_report: dict[str, Any] | None = None,
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
                    failure_memory_records=failure_memory_records,
                    elite_archive=elite_archive,
                    task_manifest=task_manifest,
                    env_manifest=env_manifest,
                    parent_reward_code=parent_reward_code,
                    tage_summary=tage_summary,
                )
            except Exception as exc:
                save_text(self.output_dir / "llm_error.txt", str(exc) + "\n")
        return self._run_deterministic(
            round_idx=round_idx,
            parent_reward_id=parent_reward_id,
            coverage_report=coverage_report,
            previous_selection_report=previous_selection_report,
            failure_memory_records=failure_memory_records,
            elite_archive=elite_archive,
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
            "tage_summary": kwargs.get("tage_summary") or {},
            "selection_report": kwargs.get("previous_selection_report") or {},
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

    def _run_deterministic(self, *, round_idx: int, parent_reward_id: str | None, coverage_report: dict[str, Any], previous_selection_report: dict[str, Any] | None = None, failure_memory_records: list[dict[str, Any]] | None = None, elite_archive: dict[str, Any] | None = None) -> dict[str, Any]:
        coverage_type = str(coverage_report.get("coverage_type", "ambiguous"))
        label_counts = dict(coverage_report.get("label_counts") or {})
        known_failures = [label for label in ["early_failure", "low_progress_survival"] if label_counts.get(label, 0) > 0]
        useful_patterns = []
        if label_counts.get("partial_progress", 0) > 0:
            useful_patterns.append("partial_progress trajectories exist and should be preserved or improved")
        if label_counts.get("success_like", 0) > 0:
            useful_patterns.append("success_like trajectories exist and should be strongly preferred")
        component_diagnosis = self._component_diagnosis(previous_selection_report)
        mutation_intent = self._mutation_intent(coverage_type, known_failures, useful_patterns, component_diagnosis)
        overall = "memory_insufficient_for_strong_diagnosis" if coverage_type in {"empty_or_too_small", "ambiguous"} else "parent_reward_provides_usable_memory_but_needs_evolution"
        evaluation = {
            "round": round_idx,
            "parent_reward_id": parent_reward_id,
            "overall_judgment": overall,
            "failure_summary": f"Memory coverage is {coverage_type}; label_counts={label_counts}.",
            "memory_interpretation": {
                "coverage_type": coverage_type,
                "usable_preference_level": "strong_pairwise" if coverage_type == "balanced" else ("weak_pairwise" if coverage_type == "failure_plus_partial_progress" else "weak_or_none"),
                "main_known_failures": known_failures,
                "main_useful_patterns": useful_patterns,
                "uncertainties": [] if useful_patterns else ["Memory may still be too weak for strong conclusions."],
                "label_counts": label_counts,
            },
            "component_diagnosis": component_diagnosis,
            "mutation_intent": mutation_intent,
            "rollback_decision": {"recommend_rollback": False, "rollback_target": None, "reason": "Hard rollback is handled by RollbackManager, not Analyzer."},
            "self_evaluation_lesson": self._lesson(coverage_type, known_failures, useful_patterns),
            "recent_failure_memory_used": failure_memory_records or [],
            "elite_archive_summary": elite_archive or {},
            "agent_mode": "deterministic_analyzer",
        }
        save_text(self.output_dir / "prompt.txt", "Deterministic AnalyzerAgent; no LLM prompt was sent.\n")
        save_text(self.output_dir / "response.txt", "Deterministic self_evaluation.json generated.\n")
        save_json(self.output_dir / "self_evaluation.json", evaluation)
        return evaluation

    def _component_diagnosis(self, selection_report: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not selection_report:
            return []
        selected_id = selection_report.get("selected_candidate")
        selected = next((item for item in selection_report.get("candidate_scores", []) or [] if item.get("candidate_id") == selected_id), None)
        if not selected:
            return []
        diagnosis = []
        pref = selected.get("preference_consistency")
        fail = selected.get("failure_avoidance")
        if pref is not None:
            diagnosis.append({"component": "global_reward_ranking", "verdict": "keep_or_improve" if float(pref or 0) >= 0.5 else "restructure", "evidence": f"selected preference_consistency={pref}"})
        if fail is not None:
            diagnosis.append({"component": "failure_avoidance", "verdict": "keep" if float(fail or 0) >= 0.5 else "strengthen", "evidence": f"selected failure_avoidance={fail}"})
        return diagnosis

    def _mutation_intent(self, coverage_type: str, known_failures: list[str], useful_patterns: list[str], component_diagnosis: list[dict[str, Any]]) -> dict[str, Any]:
        if coverage_type in {"failure_plus_partial_progress", "balanced"}:
            primary = "progress_conditioned"
            required = ["preserve components that favor partial_progress over known failures", "reduce reward assigned to known failure labels"]
        elif coverage_type in {"single_failure_mode", "multiple_failure_modes"}:
            primary = "component_recomposition"
            required = ["escape known failure modes", "increase structural diversity"]
        else:
            primary = "local_repair"
            required = ["use conservative interpretable components", "avoid relying on unreliable memory coverage"]
        if known_failures:
            required.append(f"explicitly avoid known failures: {', '.join(known_failures)}")
        if useful_patterns:
            required.append("preserve useful patterns: " + "; ".join(useful_patterns))
        return {"primary_family": primary, "secondary_family": "component_recomposition" if primary != "component_recomposition" else "progress_conditioned", "forbidden_changes": ["do not use official reward", "do not only scale all coefficients", "do not add global survival bonus without progress gating"], "required_changes": required, "preserve_components": [], "remove_or_gate_components": []}

    def _lesson(self, coverage_type: str, known_failures: list[str], useful_patterns: list[str]) -> str:
        if coverage_type in {"failure_plus_partial_progress", "balanced"}:
            return "Trajectory memory is informative enough to guide reward evolution using preference consistency."
        if known_failures:
            return "Current memory mainly supports avoiding known failures, not claiming success."
        return "Memory is still weak; keep mutation conservative and collect more long-training trajectories."
