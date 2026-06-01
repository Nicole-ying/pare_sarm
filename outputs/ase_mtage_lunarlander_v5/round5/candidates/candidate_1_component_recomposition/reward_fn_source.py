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

    # Distances from landing pad (0,0)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist   # positive = moving closer

    # Stage thresholds (unchanged from parent)
    FAR_THRESHOLD = 0.35
    NEAR_THRESHOLD = 0.2
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage

    # ---- Progress term (preserved and strengthened) ----
    # Positive delta only, scaled for far stage
    approach_progress = max(delta_dist, 0.0) * far_stage * 5.0

    # ---- Stability term (preserved) ----
    # Exponential reward for low velocities and small angle when near pad
    stability_score = math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)))
    near_stage_stability = near_stage * stability_score * 1.5   # increased weight slightly

    # ---- Leg contact bonus (preserved, gated by near stage) ----
    leg_contact_bonus = near_stage * (leg_left + leg_right) * 0.5

    # ---- Landing bonuses (restructured) ----
    # Partial landing: both legs in contact, moderate speeds, small angle
    partial_landing = (
        leg_left > 0.5 and leg_right > 0.5
        and abs(vy) < 0.5
        and abs(vx) < 0.5
        and abs(ang) < 0.3
        and abs(ang_vel) < 0.5
        and curr_dist < NEAR_THRESHOLD
    )
    partial_landing_bonus = 2.0 if partial_landing else 0.0

    # Safe landing: strict conditions (relaxed slightly from parent)
    safe_landing = (
        leg_left > 0.5 and leg_right > 0.5
        and abs(vy) < 0.15
        and abs(vx) < 0.15
        and abs(ang) < 0.15
        and abs(ang_vel) < 0.15
        and curr_dist < NEAR_THRESHOLD
    )
    safe_landing_bonus = 10.0 if terminated and safe_landing else 0.0

    # ---- Penalties (gated to avoid penalizing partial progress) ----
    # Crash penalty: only for violent crashes (high vertical speed or extreme angle)
    crash_penalty = 0.0
    if terminated and not safe_landing:
        if abs(vy) > 1.0 or abs(ang) > 0.5:
            crash_penalty = -2.0

    # Timeout penalty: mild penalty for being far when truncated
    timeout_penalty = 0.0
    if truncated:
        if curr_dist > 0.5:
            timeout_penalty = -3.0
        elif curr_dist > 0.2:
            timeout_penalty = -1.0

    # No survival bonus
    total_reward = (
        approach_progress
        + near_stage_stability
        + leg_contact_bonus
        + partial_landing_bonus
        + safe_landing_bonus
        + crash_penalty
        + timeout_penalty
    )

    components = {
        "approach_progress": approach_progress,
        "near_stage_stability": near_stage_stability,
        "leg_contact_bonus": leg_contact_bonus,
        "partial_landing_bonus": partial_landing_bonus,
        "safe_landing_bonus": safe_landing_bonus,
        "crash_penalty": crash_penalty,
        "timeout_penalty": timeout_penalty,
    }

    return float(total_reward), components