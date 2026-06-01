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
    # Current and next state features
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
    progress_delta = prev_dist - next_dist  # positive = closer

    # Stage threshold
    near_threshold = 0.35
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Early (far) stage: controlled approach ----
    # Reward progress but penalize excessive speed to avoid early crashes
    approach_progress = far_stage * max(progress_delta, 0.0)  # only positive progress
    speed_penalty_far = far_stage * (abs(vx1) + abs(vy1)) * 0.3
    far_quality = approach_progress - speed_penalty_far

    # ---- Late (near) stage: precision and stability ----
    # Soft penalties for instability, bonus for good leg contact
    speed_penalty_near = (abs(vx1) + abs(vy1)) * 1.0
    angle_penalty_near = abs(angle1) * 4.0
    angvel_penalty_near = abs(angvel1) * 0.5
    leg_bonus_near = 2.0 if (left_leg1 > 0.5 and right_leg1 > 0.5) else 0.0
    near_stability = -speed_penalty_near - angle_penalty_near - angvel_penalty_near + leg_bonus_near
    near_reward = near_stage * near_stability

    # ---- Terminal events: clear separation ----
    terminal_bonus = 0.0
    terminal_penalty = 0.0
    if terminated:
        safe_landing = (
            left_leg1 > 0.5 and right_leg1 > 0.5 and
            abs(vy1) < 0.5 and abs(vx1) < 0.5 and
            abs(angle1) < 0.2
        )
        if safe_landing:
            terminal_bonus = 20.0
        # unmistakable crash: high vertical speed or large tilt or high horizontal speed
        elif (
            abs(vy1) > 1.0 or abs(vx1) > 1.0 or abs(angle1) > 0.5
        ):
            terminal_penalty = -15.0
        # else ambiguous termination (e.g., both legs but unstable) – neutral
    # Timeout penalty only if still far from pad
    low_progress_timeout = -1.0 if (truncated and next_dist > 0.5) else 0.0

    components = {
        "approach_progress": approach_progress,
        "far_quality": far_quality,
        "near_stability": near_reward,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        5.0 * far_quality
        + 1.0 * near_reward
        + terminal_bonus
        + terminal_penalty
        + low_progress_timeout
    )
    return float(total_reward), components