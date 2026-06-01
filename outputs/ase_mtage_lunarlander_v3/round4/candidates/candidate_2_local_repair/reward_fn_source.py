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
    x_pos = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_pos = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    near_pad = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2) else 0.0
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    low_alt = 1.0 - high_alt

    progress = curr_distance - next_distance

    early_approach = high_alt * max(progress, 0.0) * 5.0

    # Softened early speed penalty: only penalize excessive vertical speed
    early_speed_penalty = high_alt * (-max(y_vel - 0.5, 0.0) * 0.2)

    # Reduced late stability coefficients
    late_stability = low_alt * (
        -abs(x_vel) * 1.0
        -abs(angle) * 2.0
        -abs(ang_vel) * 1.0
    )

    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    leg_bonus = near_pad * both_legs * 10.0

    # Adjusted terminal penalty: less severe for partial progress
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        crashed = 1.0 if (left_leg < 0.5 or right_leg < 0.5) else 0.0
        if crashed or abs(y_vel) > 1.0 or abs(angle) > 0.6:
            terminal_penalty = -10.0
        else:
            # More attainable terminal bonus: reward both legs contact near pad with moderate velocity
            velocity_ok = 1.0 - min(abs(y_vel) * 0.5, 1.0)
            angle_ok = 1.0 - min(abs(angle) * 1.0, 1.0)
            terminal_bonus = 15.0 * near_pad * both_legs * velocity_ok * angle_ok
    elif truncated:
        if next_distance > 0.3:
            terminal_penalty = -3.0

    # Reduced flight stability penalty
    flight_stability = -abs(angle) * 0.5 - abs(ang_vel) * 0.5

    components = {
        "early_approach": early_approach,
        "early_speed_penalty": early_speed_penalty,
        "late_stability": late_stability,
        "leg_bonus": leg_bonus,
        "terminal_penalty": terminal_penalty,
        "terminal_bonus": terminal_bonus,
        "flight_stability": flight_stability,
    }

    total_reward = early_approach + early_speed_penalty + late_stability + leg_bonus + terminal_penalty + terminal_bonus + flight_stability
    return float(total_reward), components