import math

def _safe_float(x, default=0.0):
    try:
        v = float(x)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(v):
        return default
    return v

def compute_reward(obs, action, next_obs, terminated, truncated, info):
    # Extract observations
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    ang1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    angv1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distances to origin (landing pad)
    prev_dist = math.sqrt(x0*x0 + y0*y0)
    next_dist = math.sqrt(x1*x1 + y1*y1)

    # Stage gate: near if distance < 0.35
    near_thresh = 0.35
    far_stage = 1.0 if next_dist >= near_thresh else 0.0
    near_stage = 1.0 - far_stage

    # ---- Early stage (far from pad) ----
    # Approach progress: positive when moving closer
    progress_delta = prev_dist - next_dist
    # Penalise high speeds to discourage early_failure
    speed_penalty_early = - (abs(vx1) + abs(vy1)) * 0.5
    approach_progress = far_stage * (progress_delta + speed_penalty_early)

    # ---- Late stage (near pad) ----
    # Reward both legs contact and moderate stability
    legs_both = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    soft_angle_penalty = -abs(ang1) * 2.0          # mild penalty for tilt
    soft_speed_penalty = - (abs(vx1) + abs(vy1)) * 0.5
    leg_bonus = 5.0 * legs_both
    stability_near = near_stage * (leg_bonus + soft_angle_penalty + soft_speed_penalty)

    # ---- Terminal handling ----
    term_bonus = 0.0
    term_penalty = 0.0
    if terminated:
        # Crash indicators: high vertical speed, large tilt, or missing both legs
        crash = (
            abs(vy1) > 0.8 or
            abs(ang1) > 0.5 or
            abs(vx1) > 0.5 or
            not (left_leg > 0.5 and right_leg > 0.5)
        )
        if crash:
            term_penalty = -15.0
        else:
            # Safe landing: both legs contact, moderate speeds, small angle
            safe = (
                left_leg > 0.5 and right_leg > 0.5 and
                abs(vy1) < 0.2 and
                abs(vx1) < 0.3 and
                abs(ang1) < 0.3
            )
            if safe:
                term_bonus = 20.0
            # else: ambiguous termination (no penalty/bonus)

    # ---- Low progress timeout ----
    low_progress_penalty = -0.5 if truncated and next_dist >= near_thresh else 0.0

    components = {
        "approach_progress": approach_progress,
        "stability_near": stability_near,
        "terminal_penalty": term_penalty,
        "terminal_bonus": term_bonus,
        "low_progress_timeout": low_progress_penalty,
    }

    total = (
        4.0 * approach_progress
        + 1.0 * stability_near
        + term_penalty
        + term_bonus
        + low_progress_penalty
    )
    return float(total), components