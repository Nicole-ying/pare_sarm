"""
Moderator Agent — debate and convergence with task context + memory.
"""
import json, sys
from pathlib import Path

_mr = Path(__file__).resolve().parent.parent
if str(_mr) not in sys.path: sys.path.insert(0, str(_mr))

from infra.llm_client import call_llm, parse_json_response
from infra.file_utils import save_json, load_text, load_json


class Moderator:
    def __init__(self, api_key=None, model="deepseek-reasoner", temperature=0.3):
        self.api_key = api_key; self.model = model; self.temperature = temperature

    def generate_agenda(self, diagnosis_a, diagnosis_b, evidence_board):
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        system = load_text(prompts_dir / "moderator_phase1_system.txt")
        prompt = (
            f"{system}\n\n---\n\n"
            f"## Evidence\nRound: {evidence_board['meta']['round']}\n"
            f"## Diagnostician A\n```json\n{json.dumps(diagnosis_a, indent=2, ensure_ascii=False)}\n```\n\n"
            f"## Diagnostician B\n```json\n{json.dumps(diagnosis_b, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Output debate agenda as JSON."
        )
        print(f"  [Moderator] Phase 1 ({len(prompt)} chars)")
        resp = call_llm(prompt, self.api_key, self.model, self.temperature)
        agenda = parse_json_response(resp)
        if "_parse_error" in agenda:
            resp = call_llm(prompt + "\nOutput ONLY valid JSON.", self.api_key, self.model, self.temperature - 0.1)
            agenda = parse_json_response(resp)
        return agenda

    def decide_convergence(self, agenda, revised_a, revised_b, evidence_board,
                            current_reward_code="", task_understanding=None,
                            memory_context="", max_revisions_reached=False):
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        system = load_text(prompts_dir / "moderator_phase2_system.txt")

        code_block = ""
        if current_reward_code:
            body = current_reward_code.split("def compute_reward", 1)
            code_block = f"def compute_reward{body[1]}" if len(body) > 1 else current_reward_code

        tu_block = ""
        if task_understanding:
            ti = task_understanding.get("task_identity", {})
            traps = task_understanding.get("reward_trap_warnings", [])
            tu_block = (
                f"Task: {ti.get('primary_objective','')}\n"
                f"Success: {ti.get('success_condition','')}\n"
                f"Traps: {'; '.join(traps[:5])}"
            )

        prompt = (
            f"{system}\n\n---\n\n"
            f"## Task Context\n{tu_block}\n\n"
            f"## Current Reward Code\n```python\n{code_block}\n```\n\n"
            f"## Debate Agenda\n```json\n{json.dumps(agenda, indent=2, ensure_ascii=False)}\n```\n\n"
            f"## Revised A\n```json\n{json.dumps(revised_a, indent=2, ensure_ascii=False)}\n```\n\n"
            f"## Revised B\n```json\n{json.dumps(revised_b, indent=2, ensure_ascii=False)}\n```\n\n"
            f"{f'## Cross-Round Memory{chr(10)}{memory_context}{chr(10)}{chr(10)}' if memory_context else ''}"
            f"Max revisions: {max_revisions_reached}\n\n"
            f"---\n\nOutput convergence + full_code JSON."
        )
        print(f"  [Moderator] Phase 2 ({len(prompt)} chars)")
        resp = call_llm(prompt, self.api_key, self.model, self.temperature)
        decision = parse_json_response(resp)
        if "_parse_error" in decision:
            resp = call_llm(prompt + "\nOutput ONLY valid JSON.", self.api_key, self.model, self.temperature - 0.1)
            decision = parse_json_response(resp)
        if "decision" not in decision:
            decision["decision"] = "converge"
            decision["final_diagnosis"] = revised_a.get("diagnosis", revised_a)
        return decision


def run_moderator_phase1(dia, dib, board, api_key=None, model="deepseek-reasoner"):
    return Moderator(api_key, model).generate_agenda(dia, dib, board)

def run_moderator_phase2(agenda, ra, rb, board, current_code="", task_understanding=None,
                          memory_context="", max_revisions=False, api_key=None,
                          model="deepseek-reasoner"):
    return Moderator(api_key, model).decide_convergence(
        agenda, ra, rb, board, current_code, task_understanding,
        memory_context, max_revisions)
