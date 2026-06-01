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
    # extract features from current (obs) and next (next_obs) states
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

    # distances to landing pad (origin)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # stage gating (far vs near)
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- approach progress (far stage) with quality gating to penalize instability ----
    speed_mag = abs(vx1) + abs(vy1)
    angle_mag = abs(angle1)
    quality = max(0.1, 1.0 - (speed_mag + angle_mag) / 2.5)  # soft penalty for high speed/tilt
    approach_progress = 3.0 * far_stage * progress_delta * quality

    # ---- near stage components ----
    # proximity bonus: reward for being close
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0

    # stability score for near stage (low speed, small angle)
    stability_score = max(0.0, 1.0 - speed_mag - angle_mag)
    near_stability = near_stage * stability_score * 1.5

    # leg contact bonus when near and both legs touch
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 0.0
    if near_stage > 0.5 and both_legs:
        leg_bonus = 3.0

    # ---- terminal event handling (replaces composite terminal_reward) ----
    crash_penalty = 0.0
    safe_landing_bonus = 0.0
    if terminated:
        # strict safe landing conditions
        safe = (
            both_legs
            and abs(vy1) < 0.5
            and abs(vx1) < 0.5
            and abs(angle1) < 0.3
            and abs(angvel1) < 0.5
        )
        if safe:
            safe_landing_bonus = 15.0
        else:
            # crash detection: high speed, large tilt, or no leg contact
            crash = (
                abs(vy1) > 1.0
                or abs(angle1) > 0.8
                or abs(vx1) > 1.0
                or not both_legs
            )
            if crash:
                crash_penalty = -10.0

    # ---- low-progress timeout penalty ----
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