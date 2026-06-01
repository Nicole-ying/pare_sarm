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

    # Compute distances to landing pad (centered at (0,0))
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist  # positive = moving closer

    # ----- Stage determination -----
    # Far stage: distance > FAR_THRESHOLD
    # Near stage: distance <= FAR_THRESHOLD
    # Close stage: distance <= CLOSE_THRESHOLD (sub-stage of near)
    FAR_THRESHOLD = 0.35
    CLOSE_THRESHOLD = 0.15
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage
    close_stage = 1.0 if curr_dist <= CLOSE_THRESHOLD else 0.0
    intermediate_stage = near_stage - close_stage  # near but not close

    # ----- Component: approach_progress (far stage only) -----
    # Only positive progress matters; scaled to encourage movement toward pad
    approach_progress = max(delta_dist, 0.0) * far_stage * 6.0

    # ----- Component: near_stage_stability (intermediate sub-stage) -----
    # Penalize high velocities, large angles and angular velocity
    intermediate_stability = math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)))
    near_stage_stability = intermediate_stage * intermediate_stability * 5.0

    # ----- Component: leg_contact_bonus (intermediate sub-stage) -----
    # Small bonus for leg contact when near pad
    leg_contact = leg_left + leg_right
    leg_bonus = intermediate_stage * leg_contact * 0.5

    # ----- Component: close_stage_stability (close sub-stage) -----
    # Continuous terminal-like reward that rewards smooth, stable landing
    # Quality score based on multiple factors
    # Parameters tuned to give high value only when all conditions are met
    landing_quality = math.exp(
        - (abs(vy) * 8.0
           + abs(vx) * 6.0
           + abs(ang) * 10.0
           + abs(ang_vel) * 8.0
           + (2.0 - leg_contact) * 2.0
           + curr_dist * 3.0)
    )
    # The quality is high (near 1) only when very stable and close
    close_stability_bonus = close_stage * landing_quality * 8.0

    # ----- Component: progressive_landing_bonus (close sub-stage) -----
    # Additional bonus for being truly landed (both legs, near-zero velocities)
    # This acts as a progressive terminal success reward that fires earlier
    # Progressive thresholds:
    #  - Minimal condition: both legs in contact, low vertical speed, small angle
    both_legs = 1.0 if leg_left > 0.5 and leg_right > 0.5 else 0.0
    low_vertical = 1.0 if abs(vy) < 0.4 else 0.0
    small_angle = 1.0 if abs(ang) < 0.3 else 0.0
    low_horizontal = 1.0 if abs(vx) < 0.3 else 0.0
    low_ang_vel = 1.0 if abs(ang_vel) < 0.3 else 0.0
    # Count how many conditions are satisfied
    cond_count = both_legs + low_vertical + small_angle + low_horizontal + low_ang_vel
    # Progressive bonus scales linearly with number of satisfied conditions
    # Only activate when at least 3 conditions are met and close
    progressive_bonus = close_stage * max(0.0, (cond_count - 2.0) * 1.5)

    # ----- Terminal failures (only for clear bad outcomes) -----
    # Gated failure penalty: only if terminated and conditions are clearly catastrophic
    gated_failure_penalty = 0.0
    if terminated:
        # Only penalize hard crash or out-of-bounds
        if curr_dist > 1.0 or abs(vy) > 2.0 or abs(vx) > 2.0:
            gated_failure_penalty = -1.0
        elif curr_dist > 0.5 and (abs(vy) > 1.5 or abs(ang) > 0.8):
            gated_failure_penalty = -0.5

    # Gated timeout penalty: only if truncated with negligible progress
    gated_timeout_penalty = 0.0
    if truncated and delta_dist < 0.0 and curr_dist > 0.5:
        # No progress and still far away -> penalize
        gated_timeout_penalty = -0.5

    # ----- Total reward -----
    total_reward = (
        approach_progress
        + near_stage_stability
        + leg_bonus
        + close_stability_bonus
        + progressive_bonus
        + gated_failure_penalty
        + gated_timeout_penalty
    )

    components = {
        "approach_progress": approach_progress,
        "near_stage_stability": near_stage_stability,
        "leg_contact_bonus": leg_bonus,
        "close_stability_bonus": close_stability_bonus,
        "progressive_landing_bonus": progressive_bonus,
        "gated_failure_penalty": gated_failure_penalty,
        "gated_timeout_penalty": gated_timeout_penalty,
    }

    return float(total_reward), components