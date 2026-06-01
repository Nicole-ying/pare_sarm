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
    # current state
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    # next state
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
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- far stage: reward approach/progress ----
    approach_progress = far_stage * max(0.0, progress_delta) * 5.0
    # small negative for moving away
    if far_stage > 0.5 and progress_delta < 0.0:
        approach_progress += progress_delta * 2.0  # negative penalty

    # ---- near stage: proximity + stability + leg contact ----
    # proximity: closer is better up to 2x when at pad
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0

    # stability: low speed and small angle
    speed_mag = abs(vx1) + abs(vy1)
    angle_mag = abs(angle1)
    stability_score = max(0.0, 1.0 - speed_mag - angle_mag)
    near_stability = near_stage * stability_score * 1.5

    # leg contact bonus only when both legs touch and very close
    both_legs = left_leg1 > 0.5 and right_leg1 > 0.5
    leg_bonus = 0.0
    if near_stage > 0.5 and both_legs:
        leg_bonus = 3.0

    # ---- terminal handling ----
    terminal_bonus = 0.0
    terminal_penalty = 0.0
    if terminated:
        safe_landing = (
            both_legs
            and abs(vy1) < 0.5
            and abs(vx1) < 0.5
            and abs(angle1) < 0.3
        )
        if safe_landing:
            terminal_bonus = 20.0
        else:
            # clear crash: high downward speed, large tilt, or high lateral speed
            crash = (
                abs(vy1) > 1.0
                or abs(angle1) > 0.8
                or abs(vx1) > 1.0
                or (abs(vy1) > 0.8 and not both_legs)
            )
            if crash:
                terminal_penalty = -10.0
            # else ambiguous termination, no bonus or penalty

    # ---- low-progress timeout ----
    low_progress_timeout = -0.5 if truncated and next_dist > 0.5 else 0.0

    components = {
        "approach_progress": approach_progress,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        approach_progress
        + proximity_bonus
        + near_stability
        + leg_bonus
        + terminal_bonus
        + terminal_penalty
        + low_progress_timeout
    )
    return float(total_reward), components