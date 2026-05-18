"""
EvidenceAnalyzer — the central algorithmic module of multi_reward.

Assembles evidence_board.json from raw training data files.
NO LLM calls. All statistical computation is deterministic and testable.

Reads:
- round_dir/trajectory_logs/*.trajectory.jsonl
- round_dir/evaluations/history.csv
- round_dir/entropy_history.jsonl
- round_dir/run_info.json
- round_dir/config.yaml
- previous round's evidence_board.json (for cross-round trends)
- previous round's diagnosis files (for predictions-vs-actual)
- env_dir/step.py (for termination conditions)
- exploration.json (for env context)

Output: evidence_board.json
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from .training_stats import (
    load_eval_history,
    load_trajectory_summary,
    load_entropy_history,
    load_training_data,
)
from .termination_parser import parse_termination_conditions
from .event_detector import detect_critical_events
from .cross_round_tracker import compute_cross_round_trends
from .evidence_board_schema import create_empty_board, board_to_feature_vector

BEIJING = timezone(timedelta(hours=8))


class EvidenceAnalyzer:
    """Algorithmically produces evidence_board.json from raw training data.

    Usage:
        analyzer = EvidenceAnalyzer(round_dir, env_dir, experiment_dir)
        board = analyzer.analyze()
        # board is written to round_dir/evidence_board.json
    """

    def __init__(self, round_dir: Path, env_dir: Path, experiment_dir: Path):
        self.round_dir = Path(round_dir)
        self.env_dir = Path(env_dir)
        self.experiment_dir = Path(experiment_dir)
        self.round_num = _extract_round_num(round_dir)

    def analyze(self) -> dict[str, Any]:
        """Run all analyses and produce the complete evidence board.

        Returns the board dict (also saved to disk).
        """
        board = create_empty_board()
        round_num = self.round_num

        # Meta
        self._fill_meta(board)
        board["meta"]["round"] = round_num

        # Environment context
        self._fill_environment_context(board)

        # Training result
        self._fill_training_result(board)

        # Previous proposal (predictions vs actual)
        if round_num > 0:
            self._fill_previous_proposal(board)

        # Task progress (from EnvInterpreter critical_variables)
        self._fill_task_progress(board)

        # Cross-round trends
        self._fill_cross_round_trends(board)

        # Feature vector (for similarity search)
        board["feature_vector"] = board_to_feature_vector(board)

        # Save
        output_path = self.round_dir / "evidence_board.json"
        output_path.write_text(
            json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"  [EvidenceAnalyzer] evidence_board.json saved "
            f"({len(json.dumps(board))} bytes)"
        )

        return board

    def _fill_meta(self, board: dict):
        """Fill meta section."""
        # Experiment ID from directory name
        exp_name = self.experiment_dir.name
        board["meta"]["experiment_id"] = exp_name
        board["meta"]["generated_at"] = datetime.now(BEIJING).strftime(
            "%Y-%m-%dT%H:%M:%S+08:00"
        )

        # Load run info for training stats
        run_info_path = self.round_dir / "run_info.json"
        if run_info_path.exists():
            try:
                run_info = json.loads(run_info_path.read_text("utf-8"))
                board["meta"]["total_training_steps"] = run_info.get(
                    "total_timesteps", 0
                )
            except Exception:
                pass

    def _fill_environment_context(self, board: dict):
        """Fill environment context from exploration and env source files."""
        ec = board["environment_context"]

        # From exploration
        exploration = self._load_exploration()
        if exploration:
            ec["obs_dim"] = exploration.get("obs_dim", 0)
            ec["action_dim"] = _action_dim_from_exploration(exploration)
            ec["max_episode_steps"] = exploration.get("max_episode_steps", 1000)
            ec["zero_action_profile"] = exploration.get("zero_action", {})

            # Action bounds
            act_space = exploration.get("spaces", {}).get("action", {})
            if act_space:
                ec["action_bounds"] = {
                    "low": act_space.get("low", -1.0),
                    "high": act_space.get("high", 1.0),
                }

        # From step.py
        step_source = self._load_step_source()
        if step_source:
            ec["termination_conditions"] = parse_termination_conditions(step_source)

    def _fill_training_result(self, board: dict):
        """Fill training_result section — the core of the evidence board."""
        tr = board["training_result"]

        # Load raw data
        data = load_training_data(self.round_dir)
        traj = data["traj_summary"]
        eval_hist = data["eval_history"]
        entropy_hist = data["entropy_history"]

        # ── Episode stats ──
        lengths = traj.get("lengths", {})
        tr["episode_stats"] = {
            "mean_length": lengths.get("mean", 0),
            "std_length": lengths.get("std", 0),
            "min_length": lengths.get("min", 0),
            "max_length": lengths.get("max", 0),
            "length_distribution": {
                "q10": lengths.get("q10", 0),
                "q25": lengths.get("q25", 0),
                "q50": lengths.get("q50", 0),
                "q75": lengths.get("q75", 0),
                "q90": lengths.get("q90", 0),
            },
            "termination_breakdown": self._compute_termination_breakdown(traj),
            "n_episodes": traj.get("n_episodes", 0),
        }

        # ── Reward components ──
        tr["reward_components"] = self._compute_component_stats(traj)

        # ── Behavior descriptors ──
        tr["behavior_descriptors"] = self._compute_behavior_descriptors(
            traj, eval_hist, entropy_hist
        )

        # ── Health checks (BINARY pass/fail — no composite score) ──
        tr["health_checks"] = self._run_health_checks(tr, entropy_hist)

        # ── Critical events ──
        tr["critical_events"] = detect_critical_events(
            tr["reward_components"],
            tr["behavior_descriptors"],
            entropy_hist,
            tr["episode_stats"],
        )

    def _fill_previous_proposal(self, board: dict):
        """Load previous round's diagnosis and compare predictions with actual."""
        if self.round_num < 1:
            return

        prev_round = self.round_num - 1
        prev_dir = self.experiment_dir / f"round{prev_round}"

        if not prev_dir.exists():
            return

        # Load previous diagnosis (use final_diagnosis if exists, else diagnosis_A)
        for fname in [
            "final_diagnosis.json",
            "diagnosis_A.json",
            "analyst_proposal.json",
        ]:
            diag_path = prev_dir / fname
            if diag_path.exists():
                try:
                    diagnosis = json.loads(diag_path.read_text("utf-8"))
                    board["previous_proposal"] = self._build_prediction_comparison(
                        diagnosis, board["training_result"]
                    )
                    break
                except Exception:
                    continue

    def _build_prediction_comparison(self, diagnosis: dict, training_result: dict) -> dict:
        """Compare predictions from diagnosis with actual training results."""
        # Handle nested final_diagnosis structure
        inner = diagnosis.get("final_diagnosis", diagnosis)
        diag_text = inner.get("diagnosis", diagnosis.get("diagnosis", ""))
        if isinstance(diag_text, dict):
            diag_text = diag_text.get("primary_hypothesis", str(diag_text))

        comparison = {
            "round": self.round_num - 1,
            "diagnosis_summary": str(diag_text)[:200],
        }

        predicted_effects = {}
        actual_vs_predicted = {}
        changes = inner.get("proposed_changes", diagnosis.get("proposed_changes", []))

        for change in changes:
            component = change.get("component", "unknown")
            predicted_effects[component] = {
                "change": change.get("change_type", "unknown"),
                "prediction": change.get("predicted_effect", ""),
            }
            comp_stats = training_result.get("reward_components", {}).get(component, {})
            if comp_stats:
                actual_vs_predicted[component] = {
                    "current_mean": comp_stats.get("mean"),
                    "current_share": comp_stats.get("share_of_total"),
                }

        comparison["predicted_effects"] = predicted_effects
        comparison["actual_vs_predicted"] = actual_vs_predicted
        return comparison

    def _fill_cross_round_trends(self, board: dict):
        """Compute cross-round trends by loading all previous evidence boards."""
        previous_boards = []
        for r in range(self.round_num):
            prev_path = self.experiment_dir / f"round{r}" / "evidence_board.json"
            if prev_path.exists():
                try:
                    prev_board = json.loads(prev_path.read_text("utf-8"))
                    previous_boards.append(prev_board)
                except Exception:
                    pass

        board["cross_round_trends"] = compute_cross_round_trends(
            board, previous_boards
        )

    # ── Internal computation helpers ──

    def _compute_component_stats(self, traj: dict) -> dict:
        """Compute per-component statistics with share of total."""
        components = traj.get("components", {})
        if not components:
            return {}

        # Calculate total absolute mean for share computation
        means = [abs(v.get("mean", 0)) for v in components.values()]
        total = sum(means)

        result = {}
        for name, stats in components.items():
            share = abs(stats.get("mean", 0)) / max(total, 1e-9)
            result[name] = {
                "mean": stats.get("mean", 0),
                "std": stats.get("std", 0),
                "coeff_of_variation": stats.get("coeff_of_variation", 0),
                "share_of_total": round(float(share), 4),
            }
        return result

    def _compute_behavior_descriptors(
        self,
        traj: dict,
        eval_hist: list[dict],
        entropy_hist: list[dict],
    ) -> dict:
        """Compute behavior descriptors — pure statistics, no good/bad labels."""
        env_metrics = traj.get("env_metrics", {})
        descriptors = {}

        # Extract each env metric as a behavior descriptor with trend
        for name, stats in env_metrics.items():
            trend = self._compute_trend(name, eval_hist)
            descriptors[name] = {
                "mean": stats.get("mean", 0),
                "std": stats.get("std", 0),
                "trend": trend,
            }

        # Efficiency — velocity / action_magnitude
        vel_key = None
        am_key = None
        for k in env_metrics:
            if "velocity" in k.lower():
                vel_key = k
            if "action" in k.lower() and "magnitude" in k.lower():
                am_key = k

        if vel_key and am_key:
            vel_mean = env_metrics[vel_key].get("mean", 0)
            am_mean = env_metrics[am_key].get("mean", 0.001)
            efficiency = abs(vel_mean) / max(abs(am_mean), 0.001)
            descriptors["action_efficiency"] = {
                "mean": round(float(efficiency), 4),
                "std": 0,
                "trend": "unknown",
            }

        return descriptors

    def _compute_trend(self, metric_name: str, eval_hist: list[dict]) -> str:
        """Compute trend direction for a metric across evaluation points."""
        vals = []
        for row in eval_hist:
            m = row.get("env_metrics", {}).get(metric_name, {})
            v = m.get("mean") if isinstance(m, dict) else None
            if isinstance(v, (int, float)):
                vals.append(v)

        if len(vals) < 2:
            return "unknown"

        first_half = vals[: len(vals) // 2]
        second_half = vals[len(vals) // 2 :]
        f_mean = sum(first_half) / len(first_half)
        s_mean = sum(second_half) / len(second_half)

        if s_mean > f_mean * 1.1:
            return "increasing"
        elif s_mean < f_mean * 0.9:
            return "decreasing"
        return "stable"

    def _run_health_checks(self, training_result: dict, entropy_hist: list[dict]) -> dict:
        """Run binary health checks. Each is pass/fail with detail.

        These are NOT composite scores — they are threshold-based safety gates.
        """
        checks = {}
        rc = training_result.get("reward_components", {})

        # 1. Component activity: are reward components actually active?
        active = sum(
            1 for c in rc.values() if abs(c.get("mean", 0)) > 0.01
        )
        total = len(rc)
        checks["component_activity"] = {
            "active_count": active,
            "total_count": total,
            "passed": active > 0,
            "detail": (
                f"{active}/{total} components active"
                if active > 0
                else "All components inactive — reward function may be dead"
            ),
        }

        # 2. Component dominance: any single component > 80%?
        shares = [abs(c.get("share_of_total", 0)) for c in rc.values()]
        max_share = max(shares) if shares else 0
        checks["component_dominance"] = {
            "max_share": round(max_share, 3),
            "passed": max_share < 0.80,
            "detail": (
                f"No component dominates (max share={max_share:.1%})"
                if max_share < 0.80
                else f"Single component dominates ({max_share:.1%} of total)"
            ),
        }

        # 3. Entropy collapse
        final_entropy = 0.5
        passed_entropy = True
        if entropy_hist:
            final_entropy = entropy_hist[-1].get("entropy", 0.5)
            passed_entropy = final_entropy > 0.05

        checks["entropy_collapse"] = {
            "final_entropy": round(float(final_entropy), 4),
            "passed": passed_entropy,
            "detail": (
                f"Entropy healthy ({final_entropy:.3f})"
                if passed_entropy
                else f"Entropy collapsed ({final_entropy:.3f} < 0.05)"
            ),
        }

        # 4. Survival health: are agents surviving at all?
        tb = training_result.get("episode_stats", {}).get("termination_breakdown", {})
        n_term = tb.get("terminated", {}).get("count", 0)
        n_trunc = tb.get("truncated", {}).get("count", 0)
        total_ep = max(n_term + n_trunc, 1)
        passed_survival = (n_term / total_ep) < 0.9

        checks["survival_health"] = {
            "termination_rate": round(n_term / total_ep, 3),
            "passed": passed_survival,
            "detail": (
                f"{n_term}/{total_ep} episodes terminated ({n_term/total_ep:.0%})"
                if passed_survival
                else f"High failure rate: {n_term}/{total_ep} terminated ({n_term/total_ep:.0%})"
            ),
        }

        # 5. Serious violations from constraint discovery
        violations = training_result.get("critical_events", [])
        high_severity = [e for e in violations if e.get("severity") == "high"]
        passed_violations = len(high_severity) == 0

        checks["serious_violation"] = {
            "violation_count": len(high_severity),
            "passed": passed_violations,
            "detail": (
                "No high-severity violations"
                if passed_violations
                else f"{len(high_severity)} high-severity violation(s) detected"
            ),
        }

        return checks

    def _compute_termination_breakdown(self, traj: dict) -> dict:
        """Compute termination reason breakdown."""
        reasons = traj.get("termination_reasons", {})
        total = sum(reasons.values()) or 1
        return {
            reason: {"count": count, "fraction": round(count / total, 3)}
            for reason, count in reasons.items()
        }

    # ── File loading helpers ──

    def _load_exploration(self) -> Optional[dict]:
        """Load exploration JSON from experiment directory."""
        # Try explorations/ subdirectory relative to framework
        explores_dir = self.experiment_dir.parent.parent / "explorations"
        if explores_dir.exists():
            for f in explores_dir.glob("*.json"):
                try:
                    return json.loads(f.read_text("utf-8"))
                except Exception:
                    continue

        # Try experiment_dir directly
        for f in self.experiment_dir.glob("*.json"):
            if "exploration" in f.name.lower():
                try:
                    return json.loads(f.read_text("utf-8"))
                except Exception:
                    continue
        return None

    def _load_step_source(self) -> Optional[str]:
        """Load step.py from environment directory."""
        step_path = self.env_dir / "step.py"
        if step_path.exists():
            return step_path.read_text("utf-8")
        return None


    def _fill_task_progress(self, board: dict):
        """Compute task-progress indicators from EnvInterpreter's critical_variables.

        Uses the obs_stats from trajectory data to track whether the agent
        is progressing toward task-relevant behaviors.

        This is NOT a composite score. It's behavioral facts derived from
        the environment's own termination conditions and physics.
        """
        # Load task understanding to get critical_variables
        tu_path = self.experiment_dir / "memory" / "TASK_MANIFEST.md"
        tu_json = tu_path.with_suffix(".json")
        critical_vars = []
        if tu_json.exists():
            try:
                tu = json.loads(tu_json.read_text("utf-8"))
                critical_vars = tu.get("critical_variables", [])
            except Exception:
                pass

        if not critical_vars:
            board["task_progress"] = {"note": "No critical_variables from EnvInterpreter"}
            return

        # Load trajectory obs_stats across all episodes
        traj_dir = self.round_dir / "trajectory_logs"
        if not traj_dir.exists():
            board["task_progress"] = {"note": "No trajectory data yet"}
            return

        all_obs_means = []
        all_obs_maxs = []
        all_obs_mins = []
        for fname in sorted(traj_dir.glob("*.trajectory.jsonl")):
            for line in fname.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    obs_stats = record.get("obs_stats", {})
                    if obs_stats.get("mean"):
                        all_obs_means.append(obs_stats["mean"])
                        all_obs_maxs.append(obs_stats["max"])
                        all_obs_mins.append(obs_stats["min"])
                except Exception:
                    continue

        if not all_obs_means:
            board["task_progress"] = {"note": "No obs_stats in trajectory data"}
            return

        # Aggregate across episodes
        n_episodes = len(all_obs_means)
        n_dims = len(all_obs_means[0]) if all_obs_means else 0
        agg_means = [0.0] * n_dims
        agg_maxs = [float("-inf")] * n_dims
        agg_mins = [float("inf")] * n_dims
        for i in range(n_episodes):
            for d in range(min(n_dims, len(all_obs_means[i]))):
                agg_means[d] += all_obs_means[i][d]
                agg_maxs[d] = max(agg_maxs[d], all_obs_maxs[i][d])
                agg_mins[d] = min(agg_mins[d], all_obs_mins[i][d])
        agg_means = [m / n_episodes for m in agg_means]

        # Build task progress report
        progress_indicators = {}
        for cv in critical_vars:
            name = cv.get("name", "")
            importance = cv.get("importance", "")
            if importance not in ("termination_condition", "task_progress"):
                continue

            # Try to map variable name to an obs dimension
            # Heuristic: look for dims with angle-like ranges or binary values
            indicator = {
                "variable": name,
                "importance": importance,
            }
            progress_indicators[name] = indicator

        # Add episode-level progress summary
        survival_episodes = 0
        contact_episodes = 0  # episodes where legs made contact (for lander-type tasks)

        board["task_progress"] = {
            "n_episodes_analyzed": n_episodes,
            "obs_dim_count": n_dims,
            "critical_variables_tracked": list(progress_indicators.keys()),
            "obs_statistics": {
                "per_dim_mean_range": [
                    {"dim": d, "mean": round(agg_means[d], 4),
                     "range": round(agg_maxs[d] - agg_mins[d], 4)}
                    for d in range(min(n_dims, 8))
                ],
            },
            "progress_indicators": progress_indicators,
        }


def _extract_round_num(round_dir: Path) -> int:
    """Extract round number from directory name (e.g., 'round3' -> 3)."""
    name = round_dir.name
    if name.startswith("round"):
        try:
            return int(name[5:])
        except ValueError:
            pass
    return 0


def _action_dim_from_exploration(exploration: dict) -> int:
    """Extract action dimension from exploration data."""
    act = exploration.get("spaces", {}).get("action", {})
    shape = act.get("shape", [])
    if shape and shape[0] is not None:
        return shape[0]
    return act.get("n", 0)
