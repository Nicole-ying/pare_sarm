import math


def _safe_float(x, default=0.0):
    try:
        value = float(x)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value


def compute_reward(obs, action, next_obs, terminated, truncated, info):
    # Extract state from current and next observation.
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    angle_vel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distance to the landing pad (origin).
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    near_threshold = 0.35
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Progress component (only for far stage) ----
    approach_progress = far_stage * progress_delta

    # ---- Proximity stability reward (only for near stage) ----
    # Reward being close with low speeds and small tilt.
    # Stability score in [0, 1]; higher is better.
    stability_score = max(0.0, 1.0 - abs(angle1) - abs(vy1) - abs(vx1))
    proximity_reward = near_stage * 2.0 * stability_score

    # ---- Leg contact bonus (only for near stage) ----
    leg_contact = 1.0 if (left_leg1 > 0.5 and right_leg1 > 0.5) else 0.0
    leg_bonus = near_stage * leg_contact

    # ---- Terminal event handling ----
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        # Crash heuristic: high vertical speed, large tilt, or not both legs.
        crash_cond = (
            abs(vy1) > 0.8 or
            abs(angle1) > 0.5 or
            not (left_leg1 > 0.5 and right_leg1 > 0.5)
        )
        if crash_cond:
            terminal_penalty = -10.0
        else:
            # Safe landing: both legs contact, moderate speeds, small angle.
            safe_cond = (
                left_leg1 > 0.5 and right_leg1 > 0.5 and
                abs(vy1) < 0.3 and abs(vx1) < 0.3 and
                abs(angle1) < 0.3
            )
            if safe_cond:
                terminal_bonus = 15.0
            # ambiguous termination -> no bonus/penalty

    # ---- Low‑progress timeout penalty ----
    low_progress_timeout = -0.5 if truncated and next_dist >= near_threshold else 0.0

    components = {
        "approach_progress": approach_progress,
        "proximity_reward": proximity_reward,
        "leg_bonus": leg_bonus,
        "terminal_penalty": terminal_penalty,
        "terminal_bonus": terminal_bonus,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        4.0 * approach_progress
        + 1.0 * proximity_reward
        + 1.0 * leg_bonus
        + terminal_penalty
        + terminal_bonus
        + low_progress_timeout
    )
    return float(total_reward), components