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
    # Extract current state
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    # Extract next state
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
    progress_delta = prev_dist - next_dist  # positive if moving closer

    # Stage definition (progress-conditioned gating)
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ----- Far stage: approach progress with quality weighting -----
    # Quality factor penalizes high speed/angle in the far stage
    speed_mag = abs(vx1) + abs(vy1)
    angle_mag = abs(angle1) + abs(angvel1)
    quality = max(0.0, 1.0 - (speed_mag * 0.5 + angle_mag * 0.3))
    approach_progress = far_stage * progress_delta * quality * 1.5

    # Clear penalty for instability in the far stage to discourage early_failure
    far_instability_penalty = far_stage * (-0.5 * (speed_mag + angle_mag))

    # ----- Near stage: proximity and stability rewards -----
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0
    near_stability = near_stage * max(0.0, 1.0 - (speed_mag + angle_mag)) * 1.5

    # Leg contact bonus (only near and both legs in contact)
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 3.0 if (near_stage > 0.5 and both_legs) else 0.0

    # ----- Terminal events -----
    safe_landing_bonus = 0.0
    crash_penalty = 0.0
    if terminated:
        safe = (
            both_legs and
            abs(vy1) < 0.5 and
            abs(vx1) < 0.5 and
            abs(angle1) < 0.3
        )
        if safe:
            safe_landing_bonus = 15.0
        else:
            crash = (
                abs(vy1) > 1.0 or
                abs(angle1) > 0.8 or
                abs(vx1) > 1.0 or
                not both_legs
            )
            if crash:
                crash_penalty = -5.0

    # Timeout penalty for low progress
    low_progress_timeout = -0.5 if (truncated and next_dist > 0.5) else 0.0

    components = {
        "approach_progress": approach_progress,
        "far_instability_penalty": far_instability_penalty,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "safe_landing_bonus": safe_landing_bonus,
        "crash_penalty": crash_penalty,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        approach_progress
        + far_instability_penalty
        + proximity_bonus
        + near_stability
        + leg_bonus
        + safe_landing_bonus
        + crash_penalty
        + low_progress_timeout
    )
    return float(total_reward), components