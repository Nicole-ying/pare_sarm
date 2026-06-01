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

    # Also extract current obs for progress if needed
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    # Distance from landing pad (pad is at x=0, y=0)
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # Stage determination: near the pad (within 0.15 of ground, small horizontal offset)
    near_pad = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2) else 0.0
    # Also consider altitude high vs low
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    low_alt = 1.0 - high_alt

    # Progress components
    progress = curr_distance - next_distance  # positive for moving toward pad

    # Early stage (high altitude): reward approach and gentle descent
    early_approach = high_alt * max(progress, 0.0) * 5.0
    # Penalize high vertical speed when high (reduced weight)
    early_speed_penalty = high_alt * (-abs(y_vel) * 0.3)

    # Late stage (low altitude): reward stability and leg contact (reduced weights)
    late_stability = low_alt * (
        -abs(x_vel) * 1.0  # horizontal velocity penalty reduced
        -abs(angle) * 2.0  # angle penalty reduced
        -abs(ang_vel) * 1.0  # angular velocity penalty reduced
    )
    # Leg contact bonus: only if both legs contact near pad
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    leg_bonus = near_pad * both_legs * 10.0

    # Terminal evaluation based on termination/truncation
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        # Check for crash: high velocity, large angle, or both legs not contacting
        # Use more conservative thresholds
        clear_crash = (left_leg < 0.5 and right_leg < 0.5) or abs(y_vel) > 1.0 or abs(angle) > 0.8
        if clear_crash:
            terminal_penalty = -5.0
        else:
            # Soft landing bonus: only if low velocity, upright, near pad, any leg contact
            soft_landing = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2 and abs(y_vel) < 0.5 and abs(angle) < 0.3) else 0.0
            terminal_bonus = 10.0 * soft_landing
    elif truncated:
        # Timeout: penalize if not near pad
        if next_distance > 0.3:
            terminal_penalty = -5.0

    # Stability during flight (reduced weights)
    flight_stability = -abs(angle) * 1.0 - abs(ang_vel) * 0.5

    # Component dictionary
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