"""Trajectory Judge Agent for ASE-MTAGE.

Judges trajectories in mini-batches via LLM. No rule labels, no confidence scores.
Outputs only: coarse_label, use_for_tage_pair, allowed_preference_role.
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
    coarse_label: str
    use_for_tage_pair: bool
    allowed_preference_role: str


class TrajectoryJudgeAgent:
    """Batch trajectory judge via LLM."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        temperature: float = 0.2,
        output_dir: str | Path | None = None,
        task_manifest: str | None = None,
        env_manifest: dict[str, Any] | None = None,
        fallback_on_error: bool = True,
        batch_size: int = 10,
    ) -> None:
        self.llm_client = llm_client
        self.temperature = temperature
        self.output_dir = ensure_dir(output_dir) if output_dir else None
        self.task_manifest = task_manifest or ""
        self.env_manifest = env_manifest or {}
        self.fallback_on_error = fallback_on_error
        self.batch_size = int(batch_size)
        self.consistency_checker = LabelConsistencyChecker()

    def judge_batch(self, cards: list[dict[str, Any]]) -> list[TrajectoryJudgment]:
        """Judge a batch of evidence cards in a single LLM call.

        Cards are grouped into sub-batches of ``batch_size``.
        """
        results: list[TrajectoryJudgment] = []
        for i in range(0, len(cards), self.batch_size):
            batch = cards[i : i + self.batch_size]
            results.extend(self._judge_one_batch(batch))
        return results

    def _judge_one_batch(self, cards: list[dict[str, Any]]) -> list[TrajectoryJudgment]:
        if self.llm_client is None or not cards:
            return [self._make_ambiguous(c) for c in cards]

        template = load_prompt("trajectory_judge.md")
        input_artifacts = {
            "task_manifest": self.task_manifest,
            "env_manifest": self.env_manifest,
            "cards": cards,
        }
        user_prompt = template.replace("{input_artifacts}", json.dumps(input_artifacts, ensure_ascii=False, indent=2))

        try:
            resp = self.llm_client.chat(
                system_prompt="You are the ASE-MTAGE Trajectory Judge Agent. Output only a valid JSON array.",
                user_prompt=user_prompt,
                temperature=self.temperature,
                agent_name="trajectory_judge",
            )
            raw = extract_json_object(resp.content)
            if isinstance(raw, list):
                llm_results = raw
            elif isinstance(raw, dict) and "judgments" in raw:
                llm_results = raw["judgments"]
            else:
                raise RuntimeError(f"Expected JSON array, got: {type(raw).__name__}")

            if self.output_dir:
                batch_id = f"batch_{hash(user_prompt) & 0xFFFFFF:06x}"
                save_text(self.output_dir / f"{batch_id}_prompt.txt", user_prompt)
                save_text(self.output_dir / f"{batch_id}_response.txt", resp.content)
                save_json(self.output_dir / f"{batch_id}_raw.json", resp.raw)
        except Exception as exc:
            if not self.fallback_on_error:
                raise RuntimeError("TrajectoryJudgeAgent LLM failed and fallback_on_error=false") from exc
            return [self._make_ambiguous(c) for c in cards]

        return [self._build_judgment(card, llm_results) for card in cards]

    def _build_judgment(self, card: dict[str, Any], llm_results: list[dict[str, Any]]) -> TrajectoryJudgment:
        tid = str(card.get("trajectory_id", "unknown"))
        match = next((r for r in llm_results if r.get("trajectory_id") == tid), None)

        if match is None:
            return self._make_ambiguous(card)

        coarse_label = str(match.get("coarse_label", "ambiguous"))
        if coarse_label not in {"early_failure", "low_progress_survival", "partial_progress", "success_like", "ambiguous"}:
            coarse_label = "ambiguous"

        use_for_tage_pair = bool(match.get("use_for_tage_pair", False)) and coarse_label != "ambiguous"

        episode = dict(card.get("episode") or {})
        features = dict(card.get("features") or {})
        checked = self.consistency_checker.check(
            coarse_label=coarse_label,
            episode=episode,
            features=features,
        )
        final_label = str(checked.get("coarse_label", coarse_label))
        final_use = use_for_tage_pair and final_label != "ambiguous" and bool(checked.get("use_for_tage_pair", True))

        return TrajectoryJudgment(
            trajectory_id=tid,
            coarse_label=final_label,
            use_for_tage_pair=final_use,
            allowed_preference_role=self._preference_role(final_label) if final_use else "none",
        )

    def _make_ambiguous(self, card: dict[str, Any]) -> TrajectoryJudgment:
        return TrajectoryJudgment(
            trajectory_id=str(card.get("trajectory_id", "unknown")),
            coarse_label="ambiguous",
            use_for_tage_pair=False,
            allowed_preference_role="none",
        )

    def _preference_role(self, coarse_label: str) -> str:
        if coarse_label in {"early_failure", "low_progress_survival"}:
            return "negative_reference"
        if coarse_label == "partial_progress":
            return "mid_reference"
        if coarse_label == "success_like":
            return "positive_reference"
        return "none"

    def judgment_to_dict(self, j: TrajectoryJudgment, *, include_metadata: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trajectory_id": j.trajectory_id,
            "coarse_label": j.coarse_label,
            "use_for_tage_pair": j.use_for_tage_pair,
            "allowed_preference_role": j.allowed_preference_role,
        }
        if include_metadata:
            d["judge_mode"] = "llm_batch"
        return d
