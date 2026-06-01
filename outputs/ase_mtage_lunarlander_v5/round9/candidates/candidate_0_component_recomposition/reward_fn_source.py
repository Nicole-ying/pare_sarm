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
    FAR_THRESHOLD = 0.35
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage  # within 0.35

    CLOSE_THRESHOLD = 0.2
    close_stage = 1.0 if curr_dist <= CLOSE_THRESHOLD else 0.0

    # ----- Approach progress (far stage, positive delta only) -----
    approach_progress = max(delta_dist, 0.0) * far_stage * 5.0

    # ----- Near-stage stability (close stage only) -----
    stability_score = math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)))
    near_stage_stability = close_stage * stability_score * 2.0

    # ----- Leg contact bonus (near stage only, mild) -----
    leg_contact_bonus = near_stage * (leg_left + leg_right) * 0.5

    # ----- Continuous landing progress (near stage, stability + leg contact) -----
    # Stability factor using exponentiated negatives
    stability_factor = math.exp(-(abs(vx) * 3.0 + abs(vy) * 5.0 + abs(ang) * 5.0 + abs(ang_vel) * 5.0))
    # Leg factor: 0 if no contact, 0.5 if one leg, 1 if both
    leg_factor = (leg_left + leg_right) * 0.5
    landing_progress = near_stage * stability_factor * leg_factor * 2.0

    # ----- Safe-landing bonus (only on termination) -----
    # Continuous reward based on final state quality
    safe_landing_bonus = 0.0
    if terminated:
        final_quality = math.exp(-(abs(vx) * 5.0 + abs(vy) * 5.0 + abs(ang) * 10.0 + abs(ang_vel) * 10.0))
        # Both legs must be in contact to trigger bonus
        both_legs = leg_left * leg_right
        safe_landing_bonus = both_legs * final_quality * 10.0

    # ----- Gated timeout penalty (only for negligible progress) -----
    gated_timeout_penalty = 0.0
    if truncated and curr_dist > 0.5 and delta_dist <= 0.0:
        gated_timeout_penalty = -0.5

    # ----- Total reward -----
    total_reward = (
        approach_progress
        + near_stage_stability
        + leg_contact_bonus
        + landing_progress
        + safe_landing_bonus
        + gated_timeout_penalty
    )

    components = {
        "approach_progress": approach_progress,
        "near_stage_stability": near_stage_stability,
        "leg_contact_bonus": leg_contact_bonus,
        "landing_progress": landing_progress,
        "safe_landing_bonus": safe_landing_bonus,
        "gated_timeout_penalty": gated_timeout_penalty,
    }

    return float(total_reward), components