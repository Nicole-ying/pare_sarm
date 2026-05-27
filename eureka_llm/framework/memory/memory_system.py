"""
memory_system.py — Unified three-layer memory system.

Layer 1 — CoreMemory: Token-limited context always in agent prompts.
         Task Manifest + key facts + agent scratchpad.
Layer 2 — EpisodicMemory: Per-experiment round storage with TF-IDF search.
         All round artifacts indexed and searchable.
Layer 3 — ArchivalMemory: Cross-experiment pattern library.
         Abstract lessons shared across all experiments.

Usage:
    mem = MemorySystem(run_dir)
    mem.initialize_task_manifest(step_source, task_description)  # Layer 1
    mem.episodic.store_round(round_num, artifacts)                # Layer 2
    mem.add_lesson("overconstrained reward → agent stopped")      # Layer 2 legacy
    patterns = mem.archival.search("reward hacking", k=3)         # Layer 3
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .core_memory import CoreMemory
from .episodic_memory import EpisodicMemory
from .archival_memory import ArchivalMemory

MEMORY_HEADER = """# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.

"""

# ── Phase-2 v2 Belief Schemas ─────────────────────────────────────────────

PERCEPTION_BELIEF_V2 = {
    "agent": "perception", "version": 2,
    "history": [],
    "trends": {},
    "dynamics_profile": "",
    "cross_round_insights": "",
}

ANALYST_BELIEF_V2 = {
    "agent": "analyst", "version": 2,
    "history": [],
    "validated_hypotheses": [],
    "causal_model": "",
}

GENERATOR_BELIEF_V2 = {
    "agent": "generator", "version": 2,
    "history": [],
    "code_patterns_worked": [],
    "code_patterns_failed": [],
}

BELIEF_V2_SCHEMAS = {
    "perception": PERCEPTION_BELIEF_V2,
    "analyst": ANALYST_BELIEF_V2,
    "generator": GENERATOR_BELIEF_V2,
}


class RoundMemory:
    """Per-round artifact storage and retrieval (backward compat wrapper)."""

    def __init__(self, round_dir: Path):
        self.dir = round_dir

    @property
    def reward_fn_source(self) -> Optional[str]:
        p = self.dir / "reward_fn_source.py"
        return p.read_text("utf-8") if p.exists() else None

    @property
    def training_summary(self) -> Optional[dict]:
        import csv
        evals = []
        csv_path = self.dir / "evaluations" / "history.csv"
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["timesteps"] = int(row["timesteps"]) if row.get("timesteps") else 0
                    raw = row.get("env_metrics", "{}")
                    try:
                        row["env_metrics"] = json.loads(raw)
                    except json.JSONDecodeError:
                        row["env_metrics"] = {}
                    evals.append(row)
        return {"eval_history": evals} if evals else None

    @property
    def perception_report(self) -> Optional[str]:
        p = self.dir / "perception_report.md"
        return p.read_text("utf-8") if p.exists() else None

    @property
    def analyst_proposal(self) -> Optional[dict]:
        p = self.dir / "analyzer_proposal.json"
        return json.loads(p.read_text("utf-8")) if p.exists() else None

    @property
    def reflection(self) -> Optional[str]:
        p = self.dir / "reflection.md"
        return p.read_text("utf-8") if p.exists() else None

    @property
    def gif_path(self) -> Optional[Path]:
        gif_dir = self.dir / "gifs"
        if gif_dir.exists():
            gifs = sorted(gif_dir.glob("*.gif"))
            if gifs:
                return gifs[-1]
        return None


class MemorySystem:
    """Unified three-layer memory for the multi-agent reward framework.

    Provides:
    - core (Layer 1): Token-limited context for agent prompts
    - episodic (Layer 2): Per-experiment searchable round storage
    - archival (Layer 3): Cross-experiment pattern library
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.memory_dir = self.run_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.belief_dir = self.memory_dir / "beliefs"
        self.belief_dir.mkdir(parents=True, exist_ok=True)

        # Three layers
        self.core = CoreMemory(max_tokens=2000)
        self.episodic = EpisodicMemory(self.memory_dir)
        self.archival = ArchivalMemory()

        # Load persisted core memory if it exists
        core_path = self.memory_dir / "core.json"
        if core_path.exists():
            self.core.load(core_path)
        self.episodic.load_index()

    # ── Layer 1: Core Memory (delegated) ───────────────────────────────────

    @property
    def task_manifest_path(self) -> Path:
        return self.memory_dir / "TASK_MANIFEST.md"

    def get_task_manifest(self) -> str:
        """Get Task Manifest from core memory (primary) or file fallback."""
        if self.core.task_manifest:
            return self.core.task_manifest
        p = self.task_manifest_path
        return p.read_text("utf-8") if p.exists() else ""

    def save_task_manifest(self, manifest_markdown: str) -> str:
        """Save Task Manifest to both core memory and disk."""
        content = manifest_markdown.strip() + "\n"
        self.core.set_task_manifest(content)
        self.task_manifest_path.write_text(content, encoding="utf-8")
        self.core.save(self.memory_dir / "core.json")
        return content

    def initialize_task_manifest(self, step_source: str,
                                  env_description: str = "",
                                  termination_analysis: str = "",
                                  obs_description: str = "",
                                  action_description: str = "") -> str:
        """Legacy manifest initialization — routes to core memory."""
        content = f"""# Task Manifest

## Environment Description
{env_description or "Inferred from step() source."}

## Termination Analysis
{termination_analysis or "See step() source for termination conditions."}

## Observation Space
{obs_description or "See step() source for observation structure."}

## Action Space
{action_description or "See step() source for action structure."}

## Step Source Code
```python
{step_source}
```
"""
        return self.save_task_manifest(content)

    # ── Layer 2: Episodic Memory (delegated + backward compat) ─────────────

    @property
    def memory_md_path(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    def get_lessons(self, max_lines: int = 200) -> str:
        p = self.memory_md_path
        if not p.exists():
            return ""
        return p.read_text("utf-8")

    def add_lesson(self, lesson_text: str) -> None:
        """Append a lesson to MEMORY.md (Layer 2 legacy path)."""
        p = self.memory_md_path
        header = MEMORY_HEADER if not p.exists() else ""
        with p.open("a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(lesson_text.rstrip() + "\n\n")

        # Also add to archival memory (Layer 3) for cross-experiment learning
        self.archival.add_lesson(
            lesson_text,
            round_num=0,
            experiment_id=self.run_dir.name,
            env_type=self._infer_env_type(),
        )

    def query_lessons(self, keyword: str, max_results: int = 5) -> list[str]:
        """Search Layer 2 MEMORY.md by keyword (backward compat).

        Also searches Layer 3 archival memory and merges results.
        """
        # Search MEMORY.md (Layer 2)
        p = self.memory_md_path
        local_matches = []
        if p.exists():
            text = p.read_text("utf-8")
            lessons = re.split(r"\n\n+", text)
            for lesson in lessons:
                lesson = lesson.strip()
                if not lesson or lesson.startswith("#"):
                    continue
                if keyword.lower() in lesson.lower():
                    local_matches.append(lesson)

        # Search EpisodicMemory (Layer 2 new)
        episodic_results = self.episodic.search(keyword, k=max_results)

        # Search ArchivalMemory (Layer 3)
        archival_results = self.archival.search(keyword, k=max_results)

        # Merge: local matches first, then episodic, then archival
        results = []
        seen = set()
        for m in local_matches[:max_results]:
            fp = m[:100].strip().lower()
            if fp not in seen:
                seen.add(fp)
                results.append(m)

        for r in episodic_results:
            fp = r.get("snippet", "")[:100].strip().lower()
            if fp not in seen:
                seen.add(fp)
                results.append(f"[Round {r['round_num']}] {r['snippet']}")

        for r in archival_results:
            fp = r.get("content", "")[:100].strip().lower()
            if fp not in seen:
                seen.add(fp)
                results.append(f"[Archival/{r.get('env_type', 'general')}] {r['content']}")

        return results[:max_results]

    # ── Layer 2: Storage & Round Access ────────────────────────────────────

    def store_round(self, round_num: int, artifacts: dict) -> None:
        """Store round artifacts in episodic memory."""
        self.episodic.store_round(round_num, artifacts)

    def round_path(self, round_num: int) -> Path:
        return self.run_dir / f"round{round_num}"

    def get_round(self, round_num: int) -> RoundMemory:
        return RoundMemory(self.round_path(round_num))

    def get_available_rounds(self) -> list[int]:
        rounds = []
        for d in self.run_dir.iterdir():
            if d.name.startswith("round") and d.name[5:].isdigit():
                rounds.append(int(d.name[5:]))
        return sorted(rounds)

    def get_recent_lessons(self, n: int = 3) -> str:
        """Get the n most recent complete rounds as a summarized history string."""
        rounds = self.get_available_rounds()
        recent = rounds[-n:] if len(rounds) > n else rounds
        parts = []
        for r in recent:
            rm = self.get_round(r)
            summary_parts = [f"### Round {r}"]
            rsrc = rm.reward_fn_source
            if rsrc:
                lines = rsrc.splitlines()
                doc_lines = [l for l in lines if l.strip().startswith(("#", '"""'))][:5]
                summary_parts.append("Reward: " + " ".join(doc_lines)[:200])

            ts = rm.training_summary
            if ts and ts.get("eval_history"):
                last = ts["eval_history"][-1]
                mean_len = last.get('mean_length', '?')
                env_m = last.get("env_metrics", {}) or {}
                if isinstance(env_m, str):
                    try:
                        env_m = json.loads(env_m)
                    except (json.JSONDecodeError, TypeError):
                        env_m = {}
                metric_strs = []
                for k, v in list(env_m.items())[:3]:
                    if isinstance(v, dict):
                        metric_strs.append(f"{k}={v.get('mean', '?')}")
                    elif isinstance(v, (int, float)):
                        metric_strs.append(f"{k}={v}")
                extra = ", ".join(metric_strs)
                if extra:
                    summary_parts.append(f"Metrics: mean_len={mean_len}, {extra}")
                else:
                    summary_parts.append(f"Metrics: mean_len={mean_len}")

            pr = rm.perception_report
            if pr:
                lines = pr.splitlines()
                behavior_lines = [l for l in lines if "behavior" in l.lower()
                                  or "trend" in l.lower() or "summary" in l.lower()][:3]
                if behavior_lines:
                    summary_parts.append("Perception: " + " ".join(behavior_lines)[:200])

            ref = rm.reflection
            if ref:
                ref_lines = ref.splitlines()[:3]
                summary_parts.append("Lesson: " + " ".join(ref_lines)[:200])

            parts.append("\n".join(summary_parts))
        return "\n\n".join(parts)

    # ── Layer 3: Cross-experiment ──────────────────────────────────────────

    def consolidate_to_archival(self, reflection_text: str, round_num: int,
                                 env_description: str = "") -> int:
        """Extract patterns from a reflection and add to archival memory."""
        return self.archival.consolidate_from_reflection(
            reflection_text, round_num,
            experiment_id=self.run_dir.name,
            env_description=env_description,
        )

    def get_archival_patterns(self, query: str = "", k: int = 5,
                               env_type: str = "") -> list[dict]:
        """Search archival memory for relevant patterns."""
        return self.archival.search(query, k=k, env_type=env_type)

    # ── Belief States (backward compat) ────────────────────────────────────

    def belief_path(self, agent_name: str) -> Path:
        return self.belief_dir / f"{agent_name}.json"

    def get_belief(self, agent_name: str) -> dict:
        p = self.belief_path(agent_name)
        if not p.exists():
            return {"agent": agent_name, "version": 1, "history": []}
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return {"agent": agent_name, "version": 1, "history": []}

    def update_belief(self, agent_name: str, entry: dict, max_entries: int = 50) -> dict:
        belief = self.get_belief(agent_name)
        belief.setdefault("agent", agent_name)
        belief.setdefault("version", 1)
        hist = belief.setdefault("history", [])
        hist.append(entry)
        if len(hist) > max_entries:
            belief["history"] = hist[-max_entries:]
        self.belief_path(agent_name).write_text(
            json.dumps(belief, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return belief

    def _migrate_v1_to_v2(self, belief: dict, expected_agent: str) -> dict:
        if belief.get("version", 1) >= 2:
            return belief
        base = dict(BELIEF_V2_SCHEMAS.get(expected_agent, {
            "agent": expected_agent, "version": 2, "history": [],
        }))
        base["history"] = belief.get("history", [])
        return base

    def get_agent_beliefs(self, agent_names: list[str]) -> dict[str, dict]:
        result = {}
        for name in agent_names:
            raw = self.get_belief(name)
            result[name] = self._migrate_v1_to_v2(raw, name)
        return result

    def format_beliefs_for_prompt(self, agent_names: list[str],
                                   max_history: int = 3) -> str:
        beliefs = self.get_agent_beliefs(agent_names)
        parts = []
        for name, b in beliefs.items():
            history = b.get("history", [])[-max_history:]
            if not history:
                continue
            lines = [f"### {name.title()} Agent Belief State"]
            for h in history:
                lines.append(
                    f"- Round {h.get('round', h.get('round_num', '?'))}: "
                    f"{str(h.get('diagnosis', h.get('summary', h.get('status', ''))))[:120]}"
                )
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # ── Internal ────────────────────────────────────────────────────────────

    def _infer_env_type(self) -> str:
        """Infer environment archetype from experiment context."""
        # Check task manifest first
        manifest = self.get_task_manifest()
        text = manifest.lower() if manifest else self.run_dir.name.lower()
        from .archival_memory import _classify_env_type
        return _classify_env_type(text)

    def save(self) -> None:
        """Persist all memory layers."""
        self.core.save(self.memory_dir / "core.json")
        # Episodic is saved on each store_round call
        # Archival is auto-persisted
