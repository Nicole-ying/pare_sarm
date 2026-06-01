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
    # Extract current and next observations
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x_pos = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_pos = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distances to landing pad
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # Stage indicators
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    low_alt = 1.0 if y_pos <= 0.3 else 0.0
    near_pad = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2) else 0.0
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0

    # Approach progress (high altitude only)
    progress = curr_distance - next_distance
    early_approach = high_alt * max(progress, 0.0) * 5.0

    # Low-altitude stability reward (preserved)
    low_alt_stability = low_alt * max(0.0, 1.0 - (abs(angle) * 3.0 + abs(ang_vel) * 2.0)) * 2.0

    # Leg contact bonus (only near pad)
    leg_bonus = near_pad * both_legs * 10.0

    # Terminal evaluation (adjusted)
    terminal_bonus = 0.0
    terminal_penalty = 0.0