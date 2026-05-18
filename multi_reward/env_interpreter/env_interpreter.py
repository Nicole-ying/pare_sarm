"""
EnvInterpreter — LLM agent that understands environment constraints.

Runs ONCE at Round 0. Reads step.py + env.py + exploration.json.
Outputs task_understanding.json — the permanent task description used
by all subsequent agents.

This agent never sees training data or reward functions.
"""

import json
import sys
from pathlib import Path

# Ensure multi_reward is on path
_mr = Path(__file__).resolve().parent.parent
if str(_mr) not in sys.path:
    sys.path.insert(0, str(_mr))

from infra.llm_client import call_llm, parse_json_response
from infra.file_utils import save_json, load_text


class EnvInterpreter:
    """LLM agent that produces task_understanding.json from environment sources."""

    def __init__(self, env_dir: Path, exploration_path: Path,
                 api_key: str = None, model: str = "deepseek-reasoner",
                 temperature: float = 0.3):
        self.env_dir = Path(env_dir)
        self.exploration_path = Path(exploration_path)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def interpret(self) -> dict:
        """Run the EnvInterpreter and return task_understanding.json as a dict.

        Also saves the result to the output directory.
        """
        # Load inputs
        env_source = load_text(self.env_dir / "env.py")
        step_source = load_text(self.env_dir / "step.py")
        exploration_data = load_text(self.exploration_path)

        if not step_source:
            raise FileNotFoundError(f"step.py not found in {self.env_dir}")

        # Load prompts
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        system_prompt = load_text(prompts_dir / "env_interpreter_system.txt")

        # Build user prompt
        user_prompt = (
            f"{system_prompt}\n\n"
            f"## Environment Source Code\n\n"
            f"### env.py\n```python\n{env_source[:8000]}\n```\n\n"
            f"### step.py\n```python\n{step_source[:8000]}\n```\n\n"
            f"## Random Exploration Data\n\n"
            f"```json\n{exploration_data[:6000]}\n```\n\n"
            f"## Task\n\n"
            f"Analyze this environment and produce a structured "
            f"task_understanding.json. Follow your system prompt "
            f"instructions exactly. Output ONLY the JSON object."
        )

        print(f"  [EnvInterpreter] Prompt: {len(user_prompt)} chars")

        # Call LLM
        response = call_llm(
            user_prompt, self.api_key, self.model, self.temperature
        )

        # Parse JSON
        result = parse_json_response(response)

        if "_parse_error" in result:
            print(f"  [EnvInterpreter] Parse error, retrying ...")
            retry_prompt = (
                user_prompt
                + "\n\nYour previous response could not be parsed as JSON. "
                + "Output ONLY a valid JSON object inside ```json ... ``` blocks."
            )
            response = call_llm(
                retry_prompt, self.api_key, self.model, self.temperature - 0.1
            )
            result = parse_json_response(response)

        # Validate required keys
        required_keys = [
            "task_identity", "physical_constraints",
            "design_implications", "critical_variables"
        ]
        missing = [k for k in required_keys if k not in result]
        if missing:
            print(f"  [EnvInterpreter] Warning: missing keys: {missing}")

        return result
