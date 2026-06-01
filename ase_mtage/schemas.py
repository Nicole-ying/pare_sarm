"""Core data schemas for ASE-MTAGE.

These dataclasses define the stable artifact protocol used by the pipeline.
They intentionally keep LLM credentials in env vars by default, so config files
can be committed without leaking API keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


PhaseName = Literal[
    "INIT",
    "ROUND_STARTED",
    "ROUND_COMPLETED",
    "EXPERIMENT_COMPLETED",
    "FAILED",
]


@dataclass(slots=True)
class MethodConfig:
    """High-level method configuration."""

    name: str = "ASE-MTAGE"
    k_candidates: int = 3
    max_rounds: int = 3
    selected_long_train_per_round: int = 1
    use_short_training: bool = False


@dataclass(slots=True)
class TrainingConfig:
    """Training-related configuration."""

    env_id: str = "CartPole-v1"
    full_timesteps: int = 100_000
    eval_interval: int = 20_000
    eval_episodes_per_interval: int = 5
    final_eval_episodes: int = 20
    seed: int = 42
    device: str = "cpu"


@dataclass(slots=True)
class LLMConfig:
    """LLM configuration.

    Prefer setting api_key_env="DEEPSEEK_API_KEY" and exporting that environment
    variable locally. Do not commit literal API keys. If api_key is provided, the
    saved normalized config masks it.

    fallback_on_error controls whether an LLM failure may fall back to deterministic
    logic. For paper/main experiments set fallback_on_error=false so any LLM-path
    issue fails fast instead of silently becoming a deterministic run.
    """

    enabled: bool = False
    provider: str = "none"
    model: str = "dry-run"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: int = 120
    max_tokens: int = 4096
    fallback_on_error: bool = True
    temperature: dict[str, float] = field(
        default_factory=lambda: {
            "env_perception": 0.2,
            "trajectory_judge": 0.2,
            "analyzer": 0.4,
            "mutator": 0.6,
            "reflector": 0.3,
        }
    )


@dataclass(slots=True)
class TrajectoryMemoryConfig:
    """Trajectory-memory thresholds."""

    min_trajectories: int = 10
    min_labeled_trajectories: int = 8
    use_ambiguous_for_tage: bool = False


@dataclass(slots=True)
class ASEMTAGEConfig:
    """Top-level config object."""

    method: MethodConfig = field(default_factory=MethodConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    trajectory_memory: TrajectoryMemoryConfig = field(default_factory=TrajectoryMemoryConfig)
    output_root: str = "outputs"
    experiment_name: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ASEMTAGEConfig":
        """Build config from a nested dict while tolerating unknown future keys."""
        raw = raw or {}
        method = MethodConfig(**{k: v for k, v in raw.get("method", {}).items() if k in MethodConfig.__dataclass_fields__})
        training = TrainingConfig(**{k: v for k, v in raw.get("training", {}).items() if k in TrainingConfig.__dataclass_fields__})
        llm = LLMConfig(**{k: v for k, v in raw.get("llm", {}).items() if k in LLMConfig.__dataclass_fields__})
        trajectory_memory = TrajectoryMemoryConfig(
            **{k: v for k, v in raw.get("trajectory_memory", {}).items() if k in TrajectoryMemoryConfig.__dataclass_fields__}
        )
        return cls(
            method=method,
            training=training,
            llm=llm,
            trajectory_memory=trajectory_memory,
            output_root=raw.get("output_root", "outputs"),
            experiment_name=raw.get("experiment_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("llm", {}).get("api_key"):
            data["llm"]["api_key"] = "***MASKED***"
        return data


@dataclass(slots=True)
class ExperimentState:
    """Serializable experiment resume state."""

    method: str
    env_id: str
    exp_dir: str
    current_round: int = -1
    last_completed_node: PhaseName | str = "INIT"
    completed_rounds: list[int] = field(default_factory=list)
    max_rounds: int = 0
    use_short_training: bool = False
    selected_long_train_per_round: int = 1
    historical_best_health_used_as_gate: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RoundSummary:
    """Minimal round summary produced by each pipeline round."""

    round: int
    status: str
    phase: str
    message: str
    round_dir: str
    artifacts_created: list[str] = field(default_factory=list)
    long_training_executed: bool = False
    short_training_executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentLayout:
    """Important paths inside an ASE-MTAGE experiment directory."""

    exp_dir: Path
    memory_dir: Path
    core_memory_dir: Path
    raw_trajectories_dir: Path
    elite_rewards_dir: Path

    @classmethod
    def from_exp_dir(cls, exp_dir: str | Path) -> "ExperimentLayout":
        root = Path(exp_dir)
        memory = root / "memory"
        return cls(
            exp_dir=root,
            memory_dir=memory,
            core_memory_dir=memory / "core",
            raw_trajectories_dir=memory / "raw_trajectories",
            elite_rewards_dir=memory / "elite_rewards",
        )
