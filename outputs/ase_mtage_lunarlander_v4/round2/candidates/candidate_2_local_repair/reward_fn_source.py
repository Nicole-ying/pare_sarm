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
    # Extract current and next state features from the observation array.
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    angvel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distance to the landing pad (origin).
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage definition: far vs near threshold
    near_threshold = 0.35
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Early stage: reward approach/progress, penalize instability ----
    # Small penalty for high speed/tilt to discourage early_failure trajectories
    approach_instability = 0.02 * (abs(vx1) + abs(vy1) + abs(angle1))
    approach_progress = far_stage * (progress_delta - approach_instability)

    # ---- Late stage: soft stability shaping near pad (remove punitive penalties) ----
    legs_both = (left_leg1 > 0.5 and right_leg1 > 0.5)
    stability_near = near_stage * (-0.2 * (abs(vx1) + abs(vy1) + abs(angle1)) + 1.0 * legs_both)

    # ---- Terminal event handling ----
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        legs_both_contact = (left_leg1 > 0.5 and right_leg1 > 0.5)
        # Safe landing: both legs contact, moderate speeds, small tilt
        if legs_both_contact and abs(vy1) < 0.3 and abs(vx1) < 0.3 and abs(angle1) < 0.2:
            terminal_bonus = 10.0
        # Clear crash: extreme speed or angle
        elif abs(vy1) > 1.0 or abs(vx1) > 1.0 or abs(angle1) > 1.0:
            terminal_penalty = -10.0
        # else ambiguous termination – no reward/penalty

    # ---- Low-progress timeout ----
    low_progress_timeout = -0.5 if truncated and next_dist >= near_threshold else 0.0

    components = {
        "approach_progress": approach_progress,
        "stability_near": stability_near,
        "terminal_penalty": terminal_penalty,
        "terminal_bonus": terminal_bonus,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        4.0 * approach_progress
        + 1.0 * stability_near
        + terminal_penalty
        + terminal_bonus
        + low_progress_timeout
    )
    return float(total_reward), components