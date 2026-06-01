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
    # Unpack observations
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

    # ---- Stage gates ----
    FAR_THRESHOLD = 0.35
    CLOSE_THRESHOLD = 0.2

    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage  # curr_dist <= FAR_THRESHOLD

    # Proximity factor (continuous from 0 to 1 as lander approaches pad)
    proximity = 0.0
    if near_stage:
        proximity = max(0.0, 1.0 - curr_dist / FAR_THRESHOLD)

    # ---- Approach progress (far stage only) ----
    approach_progress = max(delta_dist, 0.0) * far_stage * 5.0

    # ---- Near-stage stability (only when close to pad) ----
    stability_score = math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)))
    near_stage_stability = near_stage * stability_score * 2.0

    # ---- Leg contact bonus (gated by near stage and moderate stability) ----
    leg_bonus = 0.0
    if near_stage and stability_score > 0.3:
        leg_bonus = (leg_left + leg_right) * 0.5

    # ---- Continuous safe bonus (per-step reward for good landing condition) ----
    # Combines proximity, stability, and leg contact
    continuous_bonus = proximity * stability_score * (leg_left + leg_right) * 0.25

    # ---- Terminal safe landing bonus (progressive) ----
    # Conditions for relaxed and strict safe landing
    both_legs = (leg_left > 0.5 and leg_right > 0.5)
    low_speed = (abs(vy) < 0.5 and abs(vx) < 0.2)
    low_angle = (abs(ang) < 0.2 and abs(ang_vel) < 0.2)
    on_pad = curr_dist < 0.3

    strict_vel = (abs(vy) < 0.1 and abs(vx) < 0.1)
    strict_ang = (abs(ang) < 0.1 and abs(ang_vel) < 0.1)
    strict_pad = curr_dist < 0.15

    terminal_safe_bonus = 0.0
    if terminated:
        if both_legs and strict_vel and strict_ang and strict_pad:
            terminal_safe_bonus = 10.0
        elif both_legs and low_speed and low_angle and on_pad:
            terminal_safe_bonus = 5.0
        elif (leg_left > 0.5 or leg_right > 0.5) and low_speed and on_pad:
            terminal_safe_bonus = 2.0

    # ---- Progressive intermediate bonus for truncated episodes (close to pad) ----
    intermediate_bonus = 0.0
    if truncated and curr_dist < FAR_THRESHOLD:
        # Reward proportional to closeness and stability
        intermediate_bonus = proximity * stability_score * 1.0
        # Additional if legs make contact
        if leg_left > 0.5 or leg_right > 0.5:
            intermediate_bonus += 0.5

    # ---- Mild crash penalty (only for catastrophic crashes) ----
    crash_penalty = 0.0
    if terminated:
        # Catastrophic: high vertical speed or horizontal speed, far from pad
        if abs(vy) > 1.5 or abs(vx) > 1.5:
            crash_penalty = -0.5
        elif curr_dist > 1.0 and (abs(vy) > 1.0 or abs(vx) > 1.0):
            crash_penalty = -0.2

    # ---- Gated timeout penalty (only when no progress) ----
    timeout_penalty = 0.0
    if truncated and delta_dist < -0.01 and curr_dist > 0.5:
        timeout_penalty = -0.5

    # ---- Total reward ----
    total_reward = (
        approach_progress
        + near_stage_stability
        + leg_bonus
        + continuous_bonus
        + terminal_safe_bonus
        + intermediate_bonus
        + crash_penalty
        + timeout_penalty
    )

    components = {
        "approach_progress": approach_progress,
        "near_stage_stability": near_stage_stability,
        "leg_contact_bonus": leg_bonus,
        "continuous_safe_bonus": continuous_bonus,
        "terminal_safe_bonus": terminal_safe_bonus,
        "intermediate_bonus": intermediate_bonus,
        "crash_penalty": crash_penalty,
        "timeout_penalty": timeout_penalty
    }

    return float(total_reward), components