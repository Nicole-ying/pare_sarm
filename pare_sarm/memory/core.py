"""Core Memory: permanent facts about the environment and task."""

import json
from pathlib import Path


class CoreMemory:
    """Stores permanent facts: TaskManifest, reward signature, env metadata.

    These do NOT change across rounds — they're set once during EnvPerception.
    """

    def __init__(self, exp_dir: Path):
        self._path = Path(exp_dir) / "memory" / "core.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._facts: dict[str, str] = {}
        self._load()

    def add_fact(self, key: str, value: str):
        """Store a key fact (e.g., 'task_manifest', 'reward_signature')."""
        self._facts[key] = value

    def get_fact(self, key: str) -> str:
        """Retrieve a stored fact."""
        return self._facts.get(key, "")

    def get_all_facts(self) -> dict[str, str]:
        """Return all stored facts."""
        return dict(self._facts)

    def save(self):
        """Persist to disk."""
        self._path.write_text(
            json.dumps(self._facts, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self):
        """Load from disk if exists."""
        if self._path.exists():
            try:
                self._facts = json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._facts = {}
