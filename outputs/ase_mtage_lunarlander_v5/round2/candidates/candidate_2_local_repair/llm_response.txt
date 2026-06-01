import math

def _safe_float(x, default=0.0):
    try:
        val = float(x)
    except Exception:
        return default
    if not math.isfinite(val):
        return default
    return val

def compute_reward(obs, action, next_obs, terminated, truncated, info):
    # Unpack observations (index 0-7)
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    ang = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    leg_left = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    leg_right = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distance from pad (0,0)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist   # positive if closer

    # Stage definitions
    FAR_THRESHOLD = 0.35
    VERY_NEAR_THRESHOLD = 0.15
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    very_near_stage = 1.0 if curr_dist < VERY_NEAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage

    # ---- Early stage: approach progress (preserved) ----
    progress = max(delta_dist, 0.0) * far_stage * 5.0

    # ---- Late stage: stability & precision (gated – now only applies when very near) ----
    stability_penalty = very_near_stage * (
        -0.3 * (abs(vx) + abs(vy))
        -0.3 * abs(ang)
        -0.2 * abs(ang_vel)
    )

    # Leg contact bonus (kept, but only when near threshold)
    leg_bonus = near_stage * (leg_left + leg_right) * 0.5

    # ---- Terminal events ----
    safe_landing = (
        leg_left > 0.5 and leg_right > 0.5
        and abs(vy) < 0.1
        and abs(vx) < 0.1
        and abs(ang) < 0.1
        and abs(ang_vel) < 0.1
        and curr_dist < VERY_NEAR_THRESHOLD
    )
    safe_landing_bonus = 10.0 if terminated and safe_landing else 0.0

    # Crash penalty (gated: reduced magnitude, only penalize hard crashes)
    crash_penalty = -2.0 if terminated and not safe_landing else 0.0

    # Timeout penalty (kept)
    timeout_penalty = -5.0 if truncated and curr_dist > FAR_THRESHOLD else 0.0

    # ---- No survival reward ----
    time_alive_bonus = 0.0

    components = {
        "approach_progress": progress,
        "stability_penalty": stability_penalty,
        "leg_bonus": leg_bonus,
        "safe_landing_bonus": safe_landing_bonus,
        "crash_penalty": crash_penalty,
        "timeout_penalty": timeout_penalty,
    }

    total_reward = progress + stability_penalty + leg_bonus + safe_landing_bonus + crash_penalty + timeout_penalty
    return float(total_reward), components