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
    # Unpack observations (indices 0-7)
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
    delta_dist = prev_dist - curr_dist   # positive = moving closer

    # Stage thresholds
    FAR_THRESHOLD = 0.35
    NEAR_THRESHOLD = 0.2   # for very near / safe landing zone
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage

    # ---- Stage 1: Far stage : approach progress ----
    progress = max(delta_dist, 0.0) * far_stage * 5.0

    # ---- Stage 2: Near stage: stability and leg contact ----
    # Penalize instability (high velocities, angle, angular velocity)
    near_instability = near_stage * (abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)) * (-0.5)

    # Reward leg contact when near the pad
    leg_bonus = near_stage * (leg_left + leg_right) * 1.0

    # Partial safe-landing progress: count met conditions (continuous shape)
    conditions_met = 0.0
    if leg_left > 0.5 and leg_right > 0.5:
        conditions_met += 1.0
    if abs(vy) < 0.15:
        conditions_met += 1.0
    if abs(vx) < 0.1:
        conditions_met += 1.0
    if abs(ang) < 0.05:
        conditions_met += 1.0
    if abs(ang_vel) < 0.05:
        conditions_met += 1.0
    if curr_dist < NEAR_THRESHOLD:
        conditions_met += 1.0

    safe_landing_progress = near_stage * (conditions_met / 6.0) * 2.0  # max 2.0 when all met

    # ---- Terminal events ----
    # Safe landing detection (all 6 conditions)
    safe_landing = (
        leg_left > 0.5 and leg_right > 0.5
        and abs(vy) < 0.15
        and abs(vx) < 0.1
        and abs(ang) < 0.05
        and abs(ang_vel) < 0.05
        and curr_dist < NEAR_THRESHOLD
    )
    safe_landing_bonus = 10.0 if terminated and safe_landing else 0.0

    # Terminal failure penalty (mild)
    failure_penalty = -1.0 if terminated and not safe_landing else 0.0

    # Timeout penalty (no progress far from pad)
    timeout_penalty = -5.0 if truncated and curr_dist > FAR_THRESHOLD else 0.0

    # ---- Total reward (no survival bonus) ----
    total_reward = (
        progress
        + near_instability
        + leg_bonus
        + safe_landing_progress
        + safe_landing_bonus
        + failure_penalty
        + timeout_penalty
    )

    components = {
        "approach_progress": progress,
        "near_instability_penalty": near_instability,
        "leg_contact_bonus": leg_bonus,
        "safe_landing_progress": safe_landing_progress,
        "safe_landing_bonus": safe_landing_bonus,
        "terminal_failure_penalty": failure_penalty,
        "timeout_penalty": timeout_penalty,
    }

    return float(total_reward), components