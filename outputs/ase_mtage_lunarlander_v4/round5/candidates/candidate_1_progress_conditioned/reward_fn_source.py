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
    # Extract current and next state features
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

    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage definitions
    near_threshold = 0.35
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Early stage (far): reward approach, penalize instability ----
    # Strongly reward positive progress; penalize moving away
    approach_progress = far_stage * max(0.0, progress_delta) * 5.0 - far_stage * max(0.0, -progress_delta) * 1.0

    # Quality penalty: high speed or angle when far
    far_instability = (
        abs(vx1) + abs(vy1) + abs(angle1)
    )
    far_quality_penalty = far_stage * far_instability * -0.5

    # ---- Late stage (near): reward proximity, stability, leg contact ----
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 3.0

    # Stability: low speed, small angle
    stability_score = max(0.0, 1.0 - (abs(vx1) + abs(vy1) + abs(angle1)))
    near_stability = near_stage * stability_score * 2.0

    # Leg contact bonus
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = near_stage * (1.0 if both_legs else 0.0) * 1.0

    # ---- Terminal handling ----
    terminal_bonus = 0.0
    crash_penalty = 0.0
    if terminated:
        # Safe landing conditions: both legs, moderate vertical speed, small horizontal speed, small angle
        safe_landing = (
            both_legs
            and abs(vy1) < 0.5
            and abs(vx1) < 0.5
            and abs(angle1) < 0.3
        )
        if safe_landing:
            terminal_bonus = 10.0
        else:
            # Clear crash: high vertical speed, large tilt, high horizontal speed, or no leg contact
            crash = (
                abs(vy1) > 1.0
                or abs(angle1) > 0.8
                or abs(vx1) > 1.0
                or not both_legs
            )
            if crash:
                crash_penalty = -5.0
            # else ambiguous termination – no bonus/penalty

    # ---- Low-progress timeout penalty ----
    low_progress_timeout = -1.0 if truncated and next_dist > 0.5 else 0.0

    components = {
        "approach_progress": approach_progress,
        "far_quality_penalty": far_quality_penalty,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "crash_penalty": crash_penalty,
        "low_progress_timeout": low_progress_timeout,
    }

    total_reward = (
        approach_progress
        + far_quality_penalty
        + proximity_bonus
        + near_stability
        + leg_bonus
        + terminal_bonus
        + crash_penalty
        + low_progress_timeout
    )

    return float(total_reward), components