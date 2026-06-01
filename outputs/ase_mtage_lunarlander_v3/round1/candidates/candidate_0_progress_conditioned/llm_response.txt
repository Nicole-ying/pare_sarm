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
    # Extract next_obs components
    x_pos = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_pos = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Current obs for progress calculation
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    # Distances from landing pad (pad at x=0, y=0)
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)
    progress = curr_distance - next_distance  # positive when moving toward pad

    # Stage determination
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    low_alt = 1.0 - high_alt
    near_pad = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2) else 0.0
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0

    # Early (high altitude) components
    progress_reward_high = high_alt * max(progress, 0.0) * 3.0
    speed_penalty_high = high_alt * (-abs(y_vel) * 0.2)

    # Late (low altitude) components: stability and centering
    stability_low = low_alt * (
        -abs(x_vel) * 0.5
        - abs(angle) * 2.0
        - abs(ang_vel) * 0.3
    )
    centering_low = low_alt * (-abs(x_pos) * 1.0)
    progress_reward_low = low_alt * max(progress, 0.0) * 1.0

    # Leg contact reward near pad (both legs)
    leg_bonus = near_pad * both_legs * 5.0

    # Terminal handling (only on termination, not truncation)
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        safe_landing = (
            both_legs > 0.5
            and abs(angle) < 0.3
            and abs(y_vel) < 0.5
            and abs(x_vel) < 0.3
            and abs(x_pos) < 0.2
        )
        if safe_landing:
            terminal_bonus = 10.0
        else:
            terminal_penalty = -5.0  # crash or poor landing
    elif truncated:
        # Penalize timeout when far from pad
        if next_distance > 0.4:
            terminal_penalty = -2.0

    # Assemble components
    components = {
        "progress_reward_high": progress_reward_high,
        "speed_penalty_high": speed_penalty_high,
        "stability_low": stability_low,
        "centering_low": centering_low,
        "progress_reward_low": progress_reward_low,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
    }

    total_reward = (
        progress_reward_high
        + speed_penalty_high
        + stability_low
        + centering_low
        + progress_reward_low
        + leg_bonus
        + terminal_bonus
        + terminal_penalty
    )

    return float(total_reward), components