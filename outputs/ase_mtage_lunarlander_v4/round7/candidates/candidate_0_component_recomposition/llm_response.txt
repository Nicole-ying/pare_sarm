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
    # Extract observations
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

    # Distances
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage thresholds
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---------- Progress terms ----------
    # Restructured approach_progress: penalize early failure by subtracting a
    # term that grows with speed/angle when progress is small.
    # Base progress reward, scaled.
    raw_progress = far_stage * progress_delta
    # Penalty if progress is low AND craft is unstable: high speed or tilt.
    instability_penalty = 0.0
    if far_stage > 0.5:
        # Instability measure: sum of absolute velocities and angle.
        instability = abs(vx1) + abs(vy1) + abs(angle1)
        # If progress is below threshold, apply a negative penalty proportional to instability.
        if progress_delta < 0.0:
            # Going away – penalize harder.
            instability_penalty = -0.5 * instability
        elif progress_delta < 0.01:
            # Very little progress – still penalize instability.
            instability_penalty = -0.3 * instability
    # Combine: progress bonus minus instability penalty.
    approach_progress = 3.0 * raw_progress + instability_penalty

    # ---------- Proximity / safety terms (near stage) ----------
    # Proximity bonus: fraction of how close to pad (max 1).
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0

    # Stability: low speed and small angle.
    speed_mag = abs(vx1) + abs(vy1)
    angle_mag = abs(angle1)
    stability_score = max(0.0, 1.0 - speed_mag - angle_mag)
    near_stability = near_stage * stability_score * 1.5

    # Leg contact bonus only when very close and stable.
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 0.0
    if near_stage > 0.5 and both_legs:
        leg_bonus = 3.0

    # ---------- Terminal bonuses / penalties ----------
    crash_penalty = 0.0
    safe_landing_bonus = 0.0
    if terminated:
        # Determine if it's a safe landing.
        # Stringent conditions: both legs, very low speeds, small angle.
        safe = (
            both_legs and
            abs(vy1) < 0.4 and
            abs(vx1) < 0.4 and
            abs(angle1) < 0.2
        )
        if safe:
            safe_landing_bonus = 20.0
        else:
            # Anything else is a crash or dangerous landing.
            crash_penalty = -15.0

    # ---------- Penalty for low-progress timeout ----------
    low_progress_timeout = -0.5 if truncated and next_dist > 0.5 else 0.0

    # ---------- Assemble components and total reward ----------
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