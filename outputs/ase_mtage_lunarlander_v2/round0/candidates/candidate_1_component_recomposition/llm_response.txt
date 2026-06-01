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
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    leg2 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Previous distance (from obs)
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)

    # Progress: negative delta means moving closer
    progress_delta = prev_dist - curr_dist

    # Speed and angle magnitude
    speed = math.sqrt(vx * vx + vy * vy)
    angle_abs = abs(angle)
    ang_vel_abs = abs(ang_vel)

    # Landing conditions
    both_legs = 1.0 if (leg1 > 0.5 and leg2 > 0.5) else 0.0
    near_pad = 1.0 if curr_dist < 0.35 else 0.0

    # Approach progress (only meaningful when not near pad)
    approach_progress = (1.0 - near_pad) * progress_delta

    # Stability penalty (discourage high speed and angular velocity)
    # Scale: penalize speed > 1, angle > 0.5 rad, angular velocity > 1 (raw)
    stability_penalty = -0.1 * (speed + angle_abs * 2.0 + ang_vel_abs * 1.0)

    # Landing reward: if near pad, both legs contact, low speed and angle
    safe_landing = near_pad * both_legs * (
        1.0 if (speed < 0.5 and angle_abs < 0.3) else 0.0
    ) * 2.0

    # Terminal penalty for crash (terminated but not safe landing)
    crash_penalty = 0.0
    if terminated:
        # If not a safe landing, it's a crash
        if not (near_pad and both_legs and speed < 0.5 and angle_abs < 0.3):
            crash_penalty = -2.0

    # Timeout penalty (truncated without progress)
    timeout_penalty = 0.0
    if truncated and curr_dist > 0.5:
        timeout_penalty = -1.0

    components = {
        "approach_progress": approach_progress,
        "stability_penalty": stability_penalty,
        "safe_landing": safe_landing,
        "crash_penalty": crash_penalty,
        "timeout_penalty": timeout_penalty,
    }

    total_reward = (
        4.0 * approach_progress
        + stability_penalty
        + safe_landing
        + crash_penalty
        + timeout_penalty
    )

    return float(total_reward), components