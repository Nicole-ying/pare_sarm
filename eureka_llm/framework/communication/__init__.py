"""Communication layer: typed message schemas and shared message pool."""

from .schemas import (
    AgentMessage,
    TaskManifest,
    RewardCode,
    EvaluationReport,
    ReflectionReport,
    TrainingResult,
    MemoryQuery,
    MemoryResult,
    GeneratorProposal,
)
from .message_pool import MessagePool
