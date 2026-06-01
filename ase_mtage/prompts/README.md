# ASE-MTAGE Prompt Templates

These prompts define the LLM-facing protocol for ASE-MTAGE agents.

Current code can run without LLM calls, but these templates are the intended
prompt contracts for the future DeepSeek/GPT-backed implementation.

Core principles:

1. Do not use or infer the official environment reward.
2. Ground every decision in provided artifacts.
3. Output strict JSON or Python code according to the required schema.
4. State uncertainty explicitly instead of inventing evidence.
5. Preserve reproducibility: every prompt and response should be saved under `round_k/<agent>/`.

Prompt files:

- `env_perception.md`: build sanitized task/env manifests.
- `trajectory_judge.md`: classify ambiguous trajectory evidence cards.
- `analyzer.md`: diagnose reward failures and produce mutation intent.
- `mutator.md`: generate reward function candidates from analyzer intent.
- `reflector.md`: write failure-repair memory and archival lessons.
