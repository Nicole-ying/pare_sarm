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

    # ----- Stage gates (unchanged from parent) -----
    FAR_THRESHOLD = 0.35
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage

    MID_THRESHOLD = 0.2
    CLOSE_THRESHOLD = 0.1
    mid_stage = 1.0 if near_stage and curr_dist > MID_THRESHOLD else 0.0
    close_stage = 1.0 if near_stage and curr_dist <= MID_THRESHOLD else 0.0

    # ----- Component 1: approach_progress (preserved as-is) -----
    approach_progress = max(delta_dist, 0.0) * far_stage * 5.0

    # ----- Component 2: near_stage_stability (preserved as-is) -----
    stability_score = math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)))
    near_stage_stability = close_stage * stability_score * 2.0

    # ----- Component 3: leg_contact_bonus (preserved as-is) -----
    leg_bonus = near_stage * (leg_left + leg_right) * 0.5

    # ----- New Component 4: progressive_safe_bonus (continuous when close and stable) -----
    # Rewards being in a safe, near-pad state each step, regardless of termination.
    # This replaces intermediate_safe_bonus and provides shaping toward success.
    safe_state = 0.0
    if curr_dist < FAR_THRESHOLD and abs(vy) < 0.6 and abs(vx) < 0.3 and abs(ang) < 0.3 and abs(ang_vel) < 0.3:
        # Base reward for being in safe zone
        safe_state = 0.5
        # Extra for both legs contacting the ground (indicates stable landing attitude)
        if leg_left > 0.5 and leg_right > 0.5:
            safe_state += 0.5
    progressive_safe_bonus = safe_state * 1.0  # scale factor

    # ----- New Component 5: landing_bonus (progressive terminal bonus) -----
    # At termination, scale reward based on quality of landing, replacing rigid thresholds.
    landing_bonus = 0.0
    if terminated:
        # Determine if lander is within landing pad area and stable
        on_pad = curr_dist < 0.25
        both_legs = (leg_left > 0.5 and leg_right > 0.5)
        stability_factor = math.exp(-(abs(vy) * 0.5 + abs(vx) * 0.3 + abs(ang) * 0.4 + abs(ang_vel) * 0.2))
        if on_pad and both_legs and abs(vy) < 0.8 and abs(vx) < 0.5:
            # Base bonus for plausible landing
            base_bonus = 3.0
            # Additional bonus for very stable landing (low velocities, small angle)
            extra_bonus = 5.0 * stability_factor* (1.0 if abs(vy) < 0.15 and abs(vx) < 0.1 and abs(ang) < 0.1 else 0.5)
            landing_bonus = base_bonus + extra_bonus
        elif on_pad or both_legs:
            # Partial landing: reward proximity or leg contact moderately
            landing_bonus = 1.0 * stability_factor

    # ----- Restructured Component 6: gated_timeout_penalty (only for truly stagnant trajectories) -----
    gated_timeout_penalty = 0.0
    if truncated:
        # Only penalize if the lander is far from pad AND moving away (delta_dist < 0)
        # This ensures only low-progress survival (e.g., hovering far away) gets penalized.
        if curr_dist > 0.5 and delta_dist < 0.0:
            # Scale penalty by distance and amount of regression
            gated_timeout_penalty = -min(1.0, abs(delta_dist) * 2.0)

    # ----- Total reward = sum of components -----
    total_reward = (
        approach_progress
        + near_stage_stability
        + leg_bonus
        + progressive_safe_bonus
        + landing_bonus
        + gated_timeout_penalty
    )

    components = {
        "approach_progress": approach_progress,
        "near_stage_stability": near_stage_stability,
        "leg_contact_bonus": leg_bonus,
        "progressive_safe_bonus": progressive_safe_bonus,
        "landing_bonus": landing_bonus,
        "gated_timeout_penalty": gated_timeout_penalty,
    }

    return float(total_reward), components