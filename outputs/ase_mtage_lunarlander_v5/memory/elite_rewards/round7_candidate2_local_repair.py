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
    # Unpack current and next observations
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

    # Distances to pad (assumed at (0,0))
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist  # positive = moving closer

    # ----- Stage gates -----
    # Far stage: above 0.35
    FAR_THRESHOLD = 0.35
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage  # within 0.35

    # Sub-stages within near stage for progressive rewards
    MID_THRESHOLD = 0.2
    CLOSE_THRESHOLD = 0.1
    mid_stage = 1.0 if near_stage and curr_dist > MID_THRESHOLD else 0.0
    close_stage = 1.0 if near_stage and curr_dist <= MID_THRESHOLD else 0.0

    # ----- Approach progress (far stage) -----
    # Positive delta only, scaled
    approach_progress = max(delta_dist, 0.0) * far_stage * 5.0

    # ----- Near-stage stability (close stage only) -----
    # Exponential decay on sum of absolute velocities and angular terms
    stability_score = math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)))
    near_stage_stability = close_stage * stability_score * 2.0

    # ----- Leg contact bonus (mid and close stages) -----
    # Mild incentive for leg contact when near pad
    leg_bonus = near_stage * (leg_left + leg_right) * 0.5

    # ----- Intermediate safe bonus (truncated, close, low vertical speed) -----
    intermediate_safe = 0.0
    if truncated and curr_dist < FAR_THRESHOLD and abs(vy) < 0.5:
        # Removed leg contact requirement to fire more often
        intermediate_safe = 2.0

    # ----- Safe landing bonus (terminated successfully) -----
    safe_landing_bonus = 0.0
    if terminated:
        # Relaxed safe landing condition: both legs, low speeds, small angle
        relaxed_safe = (leg_left > 0.5 and leg_right > 0.5 and
                        abs(vy) < 0.5 and abs(vx) < 0.2 and
                        abs(ang) < 0.2 and abs(ang_vel) < 0.2 and
                        curr_dist < 0.3)
        if relaxed_safe:
            safe_landing_bonus = 5.0
            # Stricter condition for full bonus (thresholds slightly relaxed)
            strict_safe = (abs(vy) < 0.15 and abs(vx) < 0.15 and
                           abs(ang) < 0.15 and abs(ang_vel) < 0.15 and
                           curr_dist < 0.15)
            if strict_safe:
                safe_landing_bonus = 10.0

    # ----- Gated failure penalty (removed, never effective) -----
    gated_failure_penalty = 0.0  # removed completely

    # ----- Gated timeout penalty (only for negative progress) -----
    gated_timeout_penalty = 0.0
    if truncated and curr_dist > FAR_THRESHOLD:
        # Only penalize if actively moving away (delta_dist < 0)
        if delta_dist < 0:
            # Scale penalty by magnitude of negative progress
            gated_timeout_penalty = -max(0.1, min(1.0, -delta_dist * 2.0))
        elif delta_dist < 0.01 and curr_dist > 0.8:
            # Very small progress and far away: minor penalty
            gated_timeout_penalty = -0.25

    # ----- Total reward -----
    total_reward = (
        approach_progress
        + near_stage_stability
        + leg_bonus
        + intermediate_safe
        + safe_landing_bonus
        + gated_failure_penalty
        + gated_timeout_penalty
    )

    components = {
        "approach_progress": approach_progress,
        "near_stage_stability": near_stage_stability,
        "leg_contact_bonus": leg_bonus,
        "intermediate_safe_bonus": intermediate_safe,
        "safe_landing_bonus": safe_landing_bonus,
        "gated_failure_penalty": gated_failure_penalty,
        "gated_timeout_penalty": gated_timeout_penalty,
    }

    return float(total_reward), components