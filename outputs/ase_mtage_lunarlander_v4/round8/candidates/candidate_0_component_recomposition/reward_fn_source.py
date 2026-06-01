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
    # Extract features for current and next state
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    angvel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Prior and current distance to origin (landing pad)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive when moving closer

    # Stage threshold
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Restructured approach_progress ----
    # Penalize high speed/angle during approach to reduce early_failure misreward.
    instability_penalty = 0.1 * (abs(vx1) + abs(vy1) + abs(angle1))
    # Quality-adjusted approach: progress minus instability, only in far stage.
    approach_progress = 3.0 * far_stage * max(0.0, progress_delta - instability_penalty)

    # ---- Proximity bonus (kept) ----
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0

    # ---- Near_stability (kept) ----
    speed_mag = abs(vx1) + abs(vy1)
    angle_mag = abs(angle1)
    stability_score = max(0.0, 1.0 - speed_mag - angle_mag)
    near_stability = near_stage * stability_score * 1.5

    # ---- Leg contact bonus (kept) ----
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 0.0
    if near_stage > 0.5 and both_legs:
        leg_bonus = 3.0

    # ---- Terminal handling: separate crash_penalty and safe_landing_bonus ----
    crash_penalty = 0.0
    safe_landing_bonus = 0.0
    if terminated:
        # Stringent safe landing conditions: both legs, low speeds, small angle.
        safe = (
            both_legs and
            abs(vy1) < 0.3 and
            abs(vx1) < 0.3 and
            abs(angle1) < 0.2
        )
        if safe:
            safe_landing_bonus = 15.0
        else:
            # Any other termination is considered a crash.
            crash_penalty = -10.0

    # ---- Low-progress timeout penalty (kept) ----
    low_progress_timeout = -0.5 if truncated and next_dist > 0.5 else 0.0

    components = {
        "approach_progress": approach_progress,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "crash_penalty": crash_penalty,
        "safe_landing_bonus": safe_landing_bonus,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        approach_progress
        + proximity_bonus
        + near_stability
        + leg_bonus
        + crash_penalty
        + safe_landing_bonus
        + low_progress_timeout
    )
    return float(total_reward), components