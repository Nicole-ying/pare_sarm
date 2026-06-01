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
    # Current state features
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    # Next state features
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    angvel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distances to landing pad (origin)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage definition: near when distance < 0.4
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Restructured approach progress with quality scaling ----
    # Quality factor discourages high speed and large tilt during approach.
    speed_mag = abs(vx1) + abs(vy1)
    angle_mag = abs(angle1)
    quality = max(0.0, 1.0 - (speed_mag + angle_mag) / 2.0)  # in [0,1]
    approach_progress = 3.0 * far_stage * progress_delta * quality

    # ---- Proximity bonus (preserved) ----
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0

    # ---- Near stability (preserved) ----
    stability_score = max(0.0, 1.0 - speed_mag - angle_mag)
    near_stability = near_stage * stability_score * 1.5

    # ---- Leg contact bonus (preserved) ----
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 3.0 if (near_stage > 0.5 and both_legs) else 0.0

    # ---- Safe landing bonus (replaces old terminal_reward) ----
    # Only awarded on true safe landing: both legs, low speeds, upright.
    safe_landing_bonus = 0.0
    if terminated:
        if (both_legs and
            abs(vy1) < 0.3 and
            abs(vx1) < 0.3 and
            abs(angle1) < 0.2):
            safe_landing_bonus = 15.0
        # No penalty for crashes to avoid rewarding non-landing terminations.

    # ---- Low-progress timeout penalty (preserved) ----
    low_progress_timeout = -0.5 if (truncated and next_dist > 0.5) else 0.0

    components = {
        "approach_progress": approach_progress,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "safe_landing_bonus": safe_landing_bonus,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        approach_progress
        + proximity_bonus
        + near_stability
        + leg_bonus
        + safe_landing_bonus
        + low_progress_timeout
    )
    return float(total_reward), components