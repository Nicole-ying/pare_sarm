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

    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist

    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    speed_mag = abs(vx1) + abs(vy1)
    angle_mag = abs(angle1)
    instability = speed_mag + angle_mag

    # Restructured approach_progress: penalize early_failure (low progress, high speed/angle)
    # Use a baseline that is negative when progress_delta is small or instability high.
    # Quality-aware scaling: if progress_delta < 0.02 and instability > 0.5 => negative.
    progress_quality = max(0.0, 1.0 - 2.0 * instability)
    approach_progress = far_stage * (
        2.0 * progress_delta * progress_quality - 0.05 * (1.0 - progress_quality)
    )

    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 3.0

    stability_score = max(0.0, 1.0 - instability)
    near_stability = near_stage * stability_score * 2.0

    both_legs = (left_leg1 > 0.5) and (right_leg1 > 0.5)
    leg_bonus = (3.0 if near_stage > 0.5 and both_legs else 0.0)

    # Replace terminal_reward with crash_penalty and safe_landing_bonus
    crash_penalty = 0.0
    safe_landing_bonus = 0.0
    if terminated:
        # Safe landing conditions (stringent)
        safe = (
            both_legs
            and abs(vy1) < 0.3
            and abs(vx1) < 0.3
            and abs(angle1) < 0.2
            and abs(angvel1) < 0.1
        )
        if safe:
            safe_landing_bonus = 15.0
        else:
            # Crash: high vertical speed, large tilt, high horizontal speed, or no leg contact
            crash = (
                abs(vy1) > 1.0
                or abs(angle1) > 0.8
                or abs(vx1) > 1.0
                or not both_legs
            )
            if crash:
                crash_penalty = -10.0

    # Gate crash_penalty and safe_landing_bonus by near_stage (they only apply near pad)
    crash_penalty = crash_penalty if near_stage > 0.5 else 0.0
    safe_landing_bonus = safe_landing_bonus if near_stage > 0.5 else 0.0

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