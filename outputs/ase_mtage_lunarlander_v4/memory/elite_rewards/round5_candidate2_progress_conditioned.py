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
    # Extract features from current and next observation.
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

    # Distance to landing pad (origin).
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist

    # Stage thresholds: far vs near based on distance.
    near_threshold = 0.4
    near_stage = 1.0 if next_dist < near_threshold else 0.0
    far_stage = 1.0 - near_stage

    # ---- Early stage (far): reward approach progress, penalize retreat ----
    # Squash negative progress to penalize moving away; amplify positive progress for discrimination.
    if progress_delta > 0.01:
        approach_progress = far_stage * 4.0 * progress_delta
    else:
        # Penalize lack of progress or retreat lightly.
        approach_progress = far_stage * -0.5 * max(0.0, -progress_delta)

    # ---- Late stage (near): reward proximity and gentle stability ----
    # Proximity bonus: closer is better.
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0

    # Stability proxy: low speeds, small angle, small angular velocity.
    stability_score = max(0.0, 1.0 - (abs(vx1) + abs(vy1) + abs(angle1) + abs(angvel1)))
    near_stability = near_stage * stability_score * 1.5

    # Leg contact bonus only when very close and stable.
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = near_stage * (3.0 if both_legs else 0.0)

    # ---- Terminal handling: strict safe landing bonus only ----
    safe_landing_bonus = 0.0
    if terminated:
        # Stringent conditions: both legs, low speeds, small angle, minimal angular velocity.
        safe = (
            both_legs
            and abs(vy1) < 0.3
            and abs(vx1) < 0.3
            and abs(angle1) < 0.2
            and abs(angvel1) < 0.2
        )
        if safe:
            safe_landing_bonus = 15.0
        # No penalty for non-safe terminations; low_progress_timeout covers timeouts.

    # ---- Low-progress timeout penalty ----
    low_progress_timeout = -0.5 if truncated and next_dist > 0.5 else 0.0

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