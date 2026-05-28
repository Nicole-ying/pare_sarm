"""Trajectory Judge Agent for ASE-MTAGE.

The judge is rule-first and LLM-assisted. High-confidence rule labels are accepted
after hard consistency checks. Low-confidence cards are sent to the LLM when a
client is available. In strict paper runs, LLM failure raises instead of silently
falling back to ambiguous labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase_mtage.llm_client import LLMClient, extract_json_object, load_prompt
from ase_mtage.tools.label_consistency import LabelConsistencyChecker
from ase_mtage.utils.io import ensure_dir, save_json, save_text


@dataclass(slots=True)
class TrajectoryJudgment:
    trajectory_id: str
    final_label: dict[str, Any]
    agree_with_rule: bool
    evidence_used: list[str]
    conflict_warnings: list[str]
    use_for_memory: bool
    use_for_tage_pair: bool
    allowed_preference_role: str
    do_not_use_reason: str
    judge_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "final_label": self.final_label,
            "agree_with_rule": self.agree_with_rule,
            "evidence_used": self.evidence_used,
            "conflict_warnings": self.conflict_warnings,
            "use_for_memory": self.use_for_memory,
            "use_for_tage_pair": self.use_for_tage_pair,
            "allowed_preference_role": self.allowed_preference_role,
            "do_not_use_reason": self.do_not_use_reason,
            "judge_mode": self.judge_mode,
        }


class TrajectoryJudgeAgent:
    """Guarded trajectory label finalizer with optional LLM judgment."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.70,
        llm_client: LLMClient | None = None,
        temperature: float = 0.2,
        output_dir: str | Path | None = None,
        task_manifest: str | None = None,
        env_manifest: dict[str, Any] | None = None,
        fallback_on_error: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.llm_client = llm_client
        self.temperature = temperature
        self.output_dir = ensure_dir(output_dir) if output_dir else None
        self.task_manifest = task_manifest or ""
        self.env_manifest = env_manifest or {}
        self.fallback_on_error = fallback_on_error
        self.consistency_checker = LabelConsistencyChecker(confidence_threshold=confidence_threshold)

    def judge(self, card: dict[str, Any]) -> TrajectoryJudgment:
        trajectory_id = str(card.get("trajectory_id", "unknown"))
        rule_label = dict(card.get("rule_label") or {})
        episode = dict(card.get("episode") or {})
        features = dict(card.get("features") or {})
        rule_conf = float(rule_label.get("confidence", 0.0) or 0.0)

        if rule_conf >= self.confidence_threshold:
            label = {
                "coarse_label": rule_label.get("coarse_label", "ambiguous"),
                "detail_label": rule_label.get("detail_label", "rule_high_confidence"),
                "confidence": rule_conf,
                "use_for_tage_pair": True,
            }
            judge_mode = "rule_high_confidence"
            agree_with_rule = True
            do_not_use_reason = ""
            evidence_used = list(rule_label.get("evidence") or [])
        elif self.llm_client is not None:
            try:
                llm = self._judge_with_llm(card)
                label = dict(llm.get("final_label") or {})
                label["use_for_tage_pair"] = bool(llm.get("use_for_tage_pair", False))
                judge_mode = "llm_low_confidence_judge"
                agree_with_rule = bool(llm.get("agree_with_rule", False))
                do_not_use_reason = str(llm.get("do_not_use_reason", ""))
                evidence_used = list(llm.get("evidence_used") or rule_label.get("evidence") or [])
            except Exception as exc:
                self._save_llm_error(trajectory_id, exc)
                if not self.fallback_on_error:
                    raise RuntimeError("TrajectoryJudgeAgent LLM failed and fallback_on_error=false") from exc
                label, judge_mode, agree_with_rule, do_not_use_reason, evidence_used = self._ambiguous_fallback(rule_conf, rule_label)
        else:
            label, judge_mode, agree_with_rule, do_not_use_reason, evidence_used = self._ambiguous_fallback(rule_conf, rule_label)

        checked = self.consistency_checker.check(label=label, episode=episode, features=features)
        warnings = list(checked.get("consistency_warnings") or [])
        use_for_tage_pair = bool(checked.get("use_for_tage_pair", False))
        preference_role = self._preference_role(str(checked.get("coarse_label", "ambiguous"))) if use_for_tage_pair else "none"
        if not use_for_tage_pair and not do_not_use_reason:
            do_not_use_reason = "Label is ambiguous, low confidence, or conflicts with hard evidence."

        return TrajectoryJudgment(
            trajectory_id=trajectory_id,
            final_label={
                "coarse_label": checked.get("coarse_label", "ambiguous"),
                "detail_label": checked.get("detail_label", "unknown"),
                "confidence": float(checked.get("confidence", 0.0) or 0.0),
            },
            agree_with_rule=agree_with_rule and not warnings,
            evidence_used=evidence_used,
            conflict_warnings=warnings,
            use_for_memory=True,
            use_for_tage_pair=use_for_tage_pair,
            allowed_preference_role=preference_role,
            do_not_use_reason=do_not_use_reason,
            judge_mode=judge_mode,
        )

    def _judge_with_llm(self, card: dict[str, Any]) -> dict[str, Any]:
        template = load_prompt("trajectory_judge.md")
        input_artifacts = {
            "task_manifest": self.task_manifest,
            "env_manifest": self.env_manifest,
            "trajectory_evidence_card": card,
            "rule_label": card.get("rule_label"),
            "allowed_labels": ["early_failure", "low_progress_survival", "partial_progress", "success_like", "ambiguous"],
        }
        user_prompt = template.replace("{input_artifacts}", json.dumps(input_artifacts, ensure_ascii=False, indent=2))
        safe_id = str(card.get("trajectory_id", "unknown")).replace("/", "_")
        if self.output_dir:
            save_text(self.output_dir / f"{safe_id}_judge_prompt.txt", user_prompt)
        resp = self.llm_client.chat(
            system_prompt="You are the ASE-MTAGE Trajectory Judge Agent. Output only valid JSON.",
            user_prompt=user_prompt,
            temperature=self.temperature,
        )
        if self.output_dir:
            save_text(self.output_dir / f"{safe_id}_judge_response.txt", resp.content)
            save_json(self.output_dir / f"{safe_id}_judge_raw_response.json", resp.raw)
        return extract_json_object(resp.content)

    def _ambiguous_fallback(self, rule_conf: float, rule_label: dict[str, Any]) -> tuple[dict[str, Any], str, bool, str, list[str]]:
        return (
            {"coarse_label": "ambiguous", "detail_label": "low_confidence_rule_label", "confidence": min(rule_conf, 0.49), "use_for_tage_pair": False},
            "guarded_no_llm_or_llm_failed",
            False,
            "Rule confidence is low and LLM judge is unavailable or failed.",
            list(rule_label.get("evidence") or []),
        )

    def _save_llm_error(self, trajectory_id: str, exc: Exception) -> None:
        if self.output_dir:
            safe_id = trajectory_id.replace("/", "_")
            save_text(self.output_dir / f"{safe_id}_judge_error.txt", str(exc) + "\n")

    def _preference_role(self, coarse_label: str) -> str:
        if coarse_label in {"early_failure", "low_progress_survival"}:
            return "negative_reference"
        if coarse_label == "partial_progress":
            return "mid_reference"
        if coarse_label == "success_like":
            return "positive_reference"
        return "none"
