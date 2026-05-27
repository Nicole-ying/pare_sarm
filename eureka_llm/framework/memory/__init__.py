"""Memory system — three-layer design (Core, Episodic, Archival)."""
import sys
from pathlib import Path

_framework_dir = Path(__file__).resolve().parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from memory.memory_system import MemorySystem, RoundMemory
from memory.core_memory import CoreMemory
from memory.episodic_memory import EpisodicMemory
from memory.archival_memory import ArchivalMemory
from memory.context import build_memory_context, inject_memory_into_prompt

__all__ = [
    "MemorySystem", "RoundMemory",
    "CoreMemory", "EpisodicMemory", "ArchivalMemory",
    "build_memory_context", "inject_memory_into_prompt",
]
