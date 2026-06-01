import math

def _safe_float(x, default=0.0):
    try:
        value = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value

def compute_reward(obs, action, next_obs, terminated, truncated, info):
    # current state from obs
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    # next state from next_obs
    x_pos = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_pos = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # distances
    curr_dist = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_dist = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # progress delta (improvement)
    progress_delta = curr_dist - next_dist  # positive if moving closer

    # stage indicators
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    near_pad = 1.0 if (y_pos < 0.2 and abs(x_pos) < 0.2) else 0.0
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    low_vel = 1.0 if (abs(y_vel) < 0.3 and abs(x_vel) < 0.3) else 0.0
    upright = 1.0 if (abs(angle) < 0.15 and abs(ang_vel) < 0.3) else 0.0

    # 1) approach_progress: positive delta when high altitude
    approach_progress = high_alt * max(progress_delta, 0.0) * 5.0

    # 2) progress_delta: raw improvement (always, but clipped to avoid negative)
    raw_progress = max(progress_delta, 0.0) * 2.0

    # 3) pad_approach_stability: near pad reward for stable state
    pad_stability = near_pad * (1.0 - min(abs(angle) * 4.0 + abs(ang_vel) * 3.0, 1.0)) * 8.0

    # 4) leg_contact_bonus: only when near pad and both legs contact
    leg_bonus = near_pad * both_legs * 12.0

    # 5) terminal assessment (combined bonus/penalty)
    terminal_bonus = 0.0
    failure_severity = 0.0
    if terminated:
        success = (near_pad > 0.5 and both_legs > 0.5 and low_vel > 0.5 and upright > 0.5)
        if success:
            terminal_bonus = 20.0
        else:
            # partial or crash: penalty scaled by velocity and angle, but capped
            failure_severity = min(abs(y_vel) * 4.0 + abs(angle) * 4.0 + abs(x_vel) * 2.0, 15.0)
    elif truncated:
        # timeout: small penalty if far from pad
        if next_dist > 0.5:
            failure_severity = 2.0

    failure_penalty = -failure_severity

    components = {
        "approach_progress": approach_progress,
        "raw_progress": raw_progress,
        "pad_approach_stability": pad_stability,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "failure_penalty": failure_penalty,
    }

    total_reward = (2.0 * approach_progress +
                    1.5 * raw_progress +
                    2.0 * pad_stability +
                    leg_bonus +
                    terminal_bonus +
                    failure_penalty)

    return float(total_reward), components