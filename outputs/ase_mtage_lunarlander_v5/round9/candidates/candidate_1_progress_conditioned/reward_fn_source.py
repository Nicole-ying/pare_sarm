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

    # ----- Stage gates based on distance -----
    FAR_THRESHOLD = 0.35
    MID_THRESHOLD = 0.2
    CLOSE_THRESHOLD = 0.1

    # Stage flags (exclusive categories)
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 if curr_dist <= FAR_THRESHOLD else 0.0
    mid_stage = 1.0 if (curr_dist > MID_THRESHOLD and curr_dist <= FAR_THRESHOLD) else 0.0
    close_stage = 1.0 if curr_dist <= MID_THRESHOLD else 0.0

    # ----- Far stage: approach progress (only reward positive progress) -----
    approach_progress = max(delta_dist, 0.0) * far_stage * 5.0

    # ----- Near stage: stability and leg contact bonuses -----
    # Continuous stability score based on velocities and angular terms
    stability_score = math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)))
    near_stage_stability = near_stage * stability_score * 2.0

    # Leg contact bonus: small positive incentive when near pad and legs are in contact
    leg_bonus = near_stage * (leg_left + leg_right) * 0.5

    # ----- Continuous landing quality bonus for close stage (progressive terminal shaping) -----
    landing_quality = 0.0
    if close_stage > 0.5:
        # Increasingly generous bonus as conditions improve toward safe landing
        # Base bonus for being close
        close_bonus = 1.0
        # Speed bonus: reward low horizontal and vertical speeds
        speed_bonus = math.exp(-(abs(vx) * 3.0 + abs(vy) * 2.0))
        # Angular bonus: reward small angle and angular velocity
        angle_bonus = math.exp(-(abs(ang) * 5.0 + abs(ang_vel) * 3.0))
        # Leg contact bonus: reward both legs in contact
        leg_contact_bonus_component = 0.0
        if leg_left > 0.5 and leg_right > 0.5:
            leg_contact_bonus_component = 2.0
        total_bonus = (close_bonus + speed_bonus * 2.0 + angle_bonus * 2.0 + leg_contact_bonus_component)
        landing_quality = total_bonus * 0.5  # Scale to keep in reasonable range

    # ----- Gated terminal rewards (instead of discrete bonuses) -----
    # Safe landing bonus: progressive when terminated with good state
    safe_landing_bonus = 0.0
    if terminated:
        # Only give bonus if close and stable
        if curr_dist < 0.3:
            # Continuous measure of landing success
            landing_perfect = (
                (leg_left > 0.5 and leg_right > 0.5) and
                abs(vy) < 0.5 and abs(vx) < 0.2 and
                abs(ang) < 0.2 and abs(ang_vel) < 0.2
            )
            if landing_perfect:
                # Strong bonus for safe landing
                safe_landing_bonus = 10.0
            else:
                # Partial bonus for being close but not perfect
                safe_landing_bonus = 2.0 * math.exp(-(abs(vy) + abs(vx) + abs(ang) + abs(ang_vel)))
        else:
            # Catastrophic crash: small penalty
            safe_landing_bonus = -1.0

    # ----- Gated timeout penalty (only for negligible progress) -----
    gated_timeout_penalty = 0.0
    if truncated:
        # Only penalize if moving away or no progress and still far
        if curr_dist > 0.5 and delta_dist <= 0.0:
            gated_timeout_penalty = -0.5
        # If close but not safe, no penalty (partial progress allowed)
        # else no penalty

    # ----- Total reward -----
    total_reward = (
        approach_progress
        + near_stage_stability
        + leg_bonus
        + landing_quality
        + safe_landing_bonus
        + gated_timeout_penalty
    )

    components = {
        "approach_progress": approach_progress,
        "near_stage_stability": near_stage_stability,
        "leg_contact_bonus": leg_bonus,
        "landing_quality": landing_quality,
        "safe_landing_bonus": safe_landing_bonus,
        "gated_timeout_penalty": gated_timeout_penalty,
    }

    return float(total_reward), components