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
    # current observation
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    # next observation
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    angvel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # distances to origin (landing pad)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # stage thresholds
    far_threshold = 0.5
    far_stage = 1.0 if next_dist >= far_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Far-stage components ----
    # reward for moving closer (applied only if making progress)
    far_approach = far_stage * max(0.0, progress_delta) * 3.0
    # penalty for regressing (moving away)
    far_regress_penalty = far_stage * max(0.0, -progress_delta) * 5.0
    # quality penalty: high speed or large tilt are bad
    far_quality_penalty = far_stage * (
        -abs(vy1) - abs(vx1) - abs(angle1)
    ) * 1.0

    # ---- Near-stage components ----
    # stability reward: low speed, small angle
    near_stability = near_stage * max(
        0.0, 1.0 - abs(vy1) - abs(vx1) - abs(angle1)
    ) * 2.0
    # proximity bonus: closer to pad is better
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / far_threshold) * 2.0
    # leg contact bonus
    both_legs = (left_leg1 > 0.5) and (right_leg1 > 0.5)
    leg_bonus = near_stage * (3.0 if both_legs else 0.0)

    # ---- Terminal handling ----
    safe_landing_bonus = 0.0
    crash_penalty = 0.0
    if terminated:
        safe = (
            both_legs
            and abs(vy1) < 0.5
            and abs(vx1) < 0.5
            and abs(angle1) < 0.3
        )
        if safe:
            safe_landing_bonus = 15.0
        else:
            # crash if high speed, large tilt, or no legs contact
            crash = (
                abs(vy1) > 1.0
                or abs(vx1) > 1.0
                or abs(angle1) > 0.8
                or not both_legs
            )
            if crash:
                crash_penalty = -5.0
            # else ambiguous termination: no reward/penalty

    # ---- Low-progress penalty (far stage, no significant progress) ----
    low_progress_penalty = 0.0
    if far_stage > 0.5 and progress_delta < 0.01 and next_dist > 0.5:
        low_progress_penalty = -0.5

    components = {
        "far_approach": far_approach,
        "far_regress_penalty": far_regress_penalty,
        "far_quality_penalty": far_quality_penalty,
        "near_stability": near_stability,
        "proximity_bonus": proximity_bonus,
        "leg_bonus": leg_bonus,
        "safe_landing_bonus": safe_landing_bonus,
        "crash_penalty": crash_penalty,
        "low_progress_penalty": low_progress_penalty,
    }

    total_reward = (
        far_approach
        + far_regress_penalty
        + far_quality_penalty
        + near_stability
        + proximity_bonus
        + leg_bonus
        + safe_landing_bonus
        + crash_penalty
        + low_progress_penalty
    )
    return float(total_reward), components