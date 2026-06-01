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

    # Distances from pad (0,0)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist   # positive = moving closer

    # Stage thresholds: far (>0.35), mid (0.1 to 0.35), near (<0.1)
    FAR_THRESHOLD = 0.35
    NEAR_THRESHOLD = 0.1
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    mid_stage = 1.0 if (curr_dist <= FAR_THRESHOLD and curr_dist > NEAR_THRESHOLD) else 0.0
    near_stage = 1.0 if curr_dist <= NEAR_THRESHOLD else 0.0

    # ---- Stage 1: Far - reward only approach progress ----
    # Positive delta only (no negative), scaled by 6 to encourage progress
    approach_progress = max(delta_dist, 0.0) * far_stage * 6.0

    # ---- Stage 2: Mid - reward progress + mild stability ----
    # Still reward approach, but also encourage leg readiness and small angle
    mid_progress = max(delta_dist, 0.0) * mid_stage * 4.0
    mid_stability = mid_stage * (
        0.5 * (leg_left + leg_right)  # leg contact bonus
        + math.exp(-(abs(ang) + abs(ang_vel) + abs(vx)))  # orientation & lateral calmness
    ) * 1.0

    # ---- Stage 3: Near - reward precision and landing readiness ----
    # No approach reward; only stability and leg contact matter
    near_stability = near_stage * math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel))) * 2.0
    near_leg_bonus = near_stage * (leg_left + leg_right) * 1.0  # encourage both legs down

    # ---- Terminal events ----
    # True safe landing: both legs, near-zero speeds, small angle, on pad
    safe_landing = (
        leg_left > 0.5 and leg_right > 0.5
        and abs(vy) < 0.08
        and abs(vx) < 0.08
        and abs(ang) < 0.08
        and abs(ang_vel) < 0.08
        and curr_dist < 0.15  # slightly relaxed from NEAR_THRESHOLD to allow extra final step
    )
    safe_landing_bonus = 25.0 if terminated and safe_landing else 0.0

    # Terminal failure penalty (only when not a safe landing)
    failure_penalty = -3.0 if terminated and not safe_landing else 0.0

    # Timeout penalty (avoid surviving by just staying alive)
    timeout_penalty = -5.0 if truncated and curr_dist > 0.5 else 0.0

    # No survival bonus – strictly progress/stability gated

    total_reward = (
        approach_progress
        + mid_progress
        + mid_stability
        + near_stability
        + near_leg_bonus
        + safe_landing_bonus
        + failure_penalty
        + timeout_penalty
    )

    components = {
        "far_approach_progress": approach_progress,
        "mid_approach_progress": mid_progress,
        "mid_stability": mid_stability,
        "near_stability": near_stability,
        "near_leg_bonus": near_leg_bonus,
        "safe_landing_bonus": safe_landing_bonus,
        "terminal_failure_penalty": failure_penalty,
        "timeout_penalty": timeout_penalty,
    }

    return float(total_reward), components