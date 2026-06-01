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
    # Extract current and next state features from the observation array.
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

    # Distance to the landing pad (origin).
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage definition: near when distance < 0.4.
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Early stage (far): reward approach/progress but penalize instability ----
    # Basic progress term.
    early_progress = far_stage * progress_delta
    # Instability penalty: penalize high speed or large angle during far stage.
    speed_mag = abs(vx1) + abs(vy1)
    angle_mag = abs(angle1)
    # Penalty for exceeding moderate thresholds (speed > 0.5, angle > 0.2 rad).
    instability_penalty = far_stage * (
        max(0.0, speed_mag - 0.5) + max(0.0, angle_mag - 0.2)
    ) * 1.5
    # Combined approach reward with negative bias for instability.
    approach_progress = 3.0 * early_progress - instability_penalty

    # ---- Late stage (near): reward proximity, stability, and leg contact ----
    # Proximity bonus: closer is better.
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0
    # Stability reward: low speed and small angle.
    stability_score = max(0.0, 1.0 - speed_mag - angle_mag)
    near_stability = near_stage * stability_score * 1.5
    # Leg contact bonus: both legs in contact.
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 0.0
    if near_stage > 0.5 and both_legs:
        leg_bonus = 3.0

    # ---- Terminal event handling: crash penalty and safe-landing bonus ----
    crash_penalty = 0.0
    safe_landing_bonus = 0.0
    if terminated:
        # Stringent safe landing conditions:
        #   - both legs contacting
        #   - low vertical speed (<0.3)
        #   - low horizontal speed (<0.3)
        #   - small angle (<0.2 rad ≈ 11 deg)
        safe = (
            both_legs
            and abs(vy1) < 0.3
            and abs(vx1) < 0.3
            and abs(angle1) < 0.2
        )
        if safe:
            safe_landing_bonus = 15.0
        else:
            # Any termination that is not a safe landing is a crash.
            crash_penalty = -10.0

    # ---- Low-progress timeout penalty ----
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