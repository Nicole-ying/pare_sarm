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
    # Extract features
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

    # Distance to landing pad (origin)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    near_threshold = 0.35
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ----- Progress terms (dense approach) -----
    approach_progress = far_stage * progress_delta

    # ----- Stability/safety terms (only near pad) -----
    # Weakened penalties compared to parent: reduce coefficients
    speed_penalty = - (abs(vx1) * 0.5 + abs(vy1) * 1.0)
    angle_penalty = -abs(angle1) * 2.0
    angvel_penalty = -abs(angvel1) * 0.5
    leg_bonus = 5.0 if (left_leg1 > 0.5 and right_leg1 > 0.5) else 0.0
    near_stability = near_stage * (speed_penalty + angle_penalty + angvel_penalty + leg_bonus)

    # ----- Terminal penalties (clear crashes only) -----
    crash_penalty = 0.0
    if terminated:
        # Clear crash: high vertical speed > 1.0, large tilt > 0.5, or no leg contact
        crash_condition = (
            abs(vy1) > 1.0 or
            abs(angle1) > 0.5 or
            abs(vx1) > 0.5 or
            not (left_leg1 > 0.5 and right_leg1 > 0.5)
        )
        if crash_condition:
            crash_penalty = -10.0

    # ----- Terminal bonuses (safe landing with moderate conditions) -----
    terminal_bonus = 0.0
    if terminated and not crash_penalty:
        # Check safe landing: both legs contact, moderate speeds, small angle
        safe_condition = (
            left_leg1 > 0.5 and right_leg1 > 0.5 and
            abs(vy1) < 1.0 and abs(vx1) < 0.5 and
            abs(angle1) < 0.4
        )
        if safe_condition:
            terminal_bonus = 20.0
        # otherwise no bonus (ambiguous termination)

    # ----- Low-progress timeout penalty -----
    low_progress_timeout = -0.5 if truncated and next_dist >= near_threshold else 0.0

    # ----- Assemble components and total reward -----
    components = {
        "approach_progress": approach_progress,
        "near_stability": near_stability,
        "crash_penalty": crash_penalty,
        "terminal_bonus": terminal_bonus,
        "low_progress_timeout": low_progress_timeout,
    }

    total_reward = (
        4.0 * approach_progress
        + 1.0 * near_stability
        + crash_penalty
        + terminal_bonus
        + low_progress_timeout
    )

    return float(total_reward), components