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

    # Distances to landing pad (origin)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage threshold
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # --- Far stage components ---
    # 1. far_approach: positive progress reward when far
    far_approach = 3.0 * far_stage * max(0.0, progress_delta)  # only reward improvement
    # 2. far_quality_penalty: penalize high speed and large tilt when far (discourages wild behavior)
    speed_mag = abs(vx1) + abs(vy1)
    angle_abs = abs(angle1)
    far_quality_penalty = -2.0 * far_stage * (speed_mag + 2.0 * angle_abs)

    # --- Near stage components ---
    # 3. near_proximity: bonus for being close to pad
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0
    # 4. near_stability: reward for low speed and small tilt when near
    stability_score = max(0.0, 1.0 - (speed_mag + angle_abs))
    near_stability = near_stage * stability_score * 1.5
    # 5. leg_bonus: reward when both legs touch (only in near stage)
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 3.0 if near_stage > 0.5 and both_legs else 0.0

    # --- Terminal components ---
    terminal_success_bonus = 0.0
    terminal_crash_penalty = 0.0
    if terminated:
        safe_landing = (
            both_legs and
            abs(vy1) < 0.5 and
            abs(vx1) < 0.5 and
            abs(angle1) < 0.3 and
            abs(angvel1) < 0.5
        )
        if safe_landing:
            terminal_success_bonus = 15.0
        else:
            # Crash: high vertical speed, large tilt, or no leg contact
            crash = (
                abs(vy1) > 1.0 or
                abs(angle1) > 0.8 or
                abs(vx1) > 1.0 or
                not both_legs
            )
            if crash:
                terminal_crash_penalty = -5.0
            # else: ambiguous termination – no bonus or penalty

    # --- Timeout penalty ---
    low_progress_timeout_penalty = -0.5 if truncated and next_dist > 0.5 else 0.0

    components = {
        "far_approach": far_approach,
        "far_quality_penalty": far_quality_penalty,
        "near_proximity": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "terminal_success_bonus": terminal_success_bonus,
        "terminal_crash_penalty": terminal_crash_penalty,
        "low_progress_timeout_penalty": low_progress_timeout_penalty,
    }
    total_reward = (
        far_approach
        + far_quality_penalty
        + proximity_bonus
        + near_stability
        + leg_bonus
        + terminal_success_bonus
        + terminal_crash_penalty
        + low_progress_timeout_penalty
    )
    return float(total_reward), components