"""Memory system: core, episodic, behavioral, and archival stores.

- Core Memory: Permanent TaskManifest + key facts (one per experiment)
- Episodic Memory: Per-round records with simple keyword search
- Behavior Memory: Cross-round agent behavior tracking (hovering? crashing? improving?)
- Archival Memory: Cross-experiment patterns and design principles
"""

from .core import CoreMemory
from .episodic import EpisodicMemory
from .behavior import BehaviorMemory
from .archival import ArchivalMemory


class MemorySystem:
    """Integrated memory system for the PARE-SARM pipeline."""

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.core = CoreMemory(exp_dir)
        self.episodic = EpisodicMemory(exp_dir)
        self.behavior = BehaviorMemory(exp_dir)
        self.archival = ArchivalMemory(exp_dir)

    def save(self):
        """Persist all memory stores to disk."""
        self.core.save()
        self.episodic.save()
        self.behavior.save()

    def save_task_manifest(self, manifest: str):
        """Store the TaskManifest in core memory."""
        self.core.add_fact("task_manifest", manifest)

    def store_episode(self, round_num: int, data: dict):
        """Store one round's results in episodic memory."""
        self.episodic.store(round_num, data)

    def query(self, query: str, max_results: int = 5) -> list[dict]:
        """Search episodic memory by keyword overlap. Returns relevant entries."""
        return self.episodic.search(query, max_results)

    def add_archival_pattern(self, pattern: str, source_round: int):
        """Store a learned design principle across experiments."""
        self.archival.add(pattern, source_round)

    def get_archival_patterns(self, query: str = "", max_results: int = 5) -> list[str]:
        """Retrieve relevant archival patterns."""
        return self.archival.search(query, max_results)

    def get_round_summary(self, round_num: int) -> dict:
        """Get stored data for a specific round."""
        return self.episodic.get_round(round_num)
