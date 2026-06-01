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
    # current and next state features
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

    # distances
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # stage thresholds
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Restructured early-stage reward (far) ----
    # Quality factor: penalizes high speed and large tilt during descent.
    # Normalize speeds and angle to [0,1] range.
    speed_norm = abs(vx1) + abs(vy1)  # typical max ~2, so cap at 3
    angle_norm = abs(angle1)          # max ~1.57, so cap at 2
    quality = max(0.0, 1.0 - (speed_norm / 3.0 + angle_norm / 2.0))
    # Approach progress scaled by quality – reduces reward for fast/tilted progress.
    approach_progress = far_stage * progress_delta * quality * 3.0

    # Additional early-stage penalty for high vertical speed (dangerous)
    high_speed_penalty = -far_stage * (abs(vy1) * 0.5 + abs(vx1) * 0.2 + abs(angle1) * 0.2)

    # ---- Late-stage components (near) ----
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0
    stability_score = max(0.0, 1.0 - (abs(vx1) + abs(vy1) + abs(angle1)))
    near_stability = near_stage * stability_score * 1.5

    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 3.0 if (near_stage > 0.5 and both_legs) else 0.0

    # ---- Terminal handling (replace old terminal_reward) ----
    crash_penalty = 0.0
    safe_landing_bonus = 0.0
    if terminated:
        # Stringent safe landing conditions
        safe = (
            both_legs and
            abs(vy1) < 0.3 and
            abs(vx1) < 0.3 and
            abs(angle1) < 0.2
        )
        if safe:
            safe_landing_bonus = 15.0
        else:
            # Clear crash: high speed, large tilt, or no leg contact
            crash = (
                abs(vy1) > 0.8 or
                abs(angle1) > 0.6 or
                abs(vx1) > 1.0 or
                not both_legs
            )
            if crash:
                crash_penalty = -5.0
            # else ambiguous termination – no reward/penalty

    # ---- Low-progress timeout penalty ----
    low_progress_timeout = -0.5 if (truncated and next_dist > 0.5) else 0.0

    components = {
        "approach_progress": approach_progress,
        "high_speed_penalty": high_speed_penalty,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "safe_landing_bonus": safe_landing_bonus,
        "crash_penalty": crash_penalty,
        "low_progress_timeout": low_progress_timeout,
    }

    total_reward = (
        approach_progress
        + high_speed_penalty
        + proximity_bonus
        + near_stability
        + leg_bonus
        + safe_landing_bonus
        + crash_penalty
        + low_progress_timeout
    )
    return float(total_reward), components