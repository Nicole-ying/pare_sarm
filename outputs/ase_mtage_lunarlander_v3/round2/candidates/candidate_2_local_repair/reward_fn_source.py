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
    early_approach = high_alt * max(progress, 0.0) * 6.0  # slightly increased coefficient
    # Early speed penalty – gated by excessive vertical speed (threshold 0.8) to avoid penalising normal descent
    early_speed_penalty = high_alt * (-abs(y_vel) * 0.3 if abs(y_vel) > 0.8 else 0.0)

    # Late stage (low altitude): redesigned stability component with positive reward for good behaviour
    # Reward small horizontal/vertical speed and small angle; mild penalty for large deviations
    good_stability_low = 1.0 if (abs(x_vel) < 0.3 and abs(y_vel) < 0.5 and abs(angle) < 0.2 and abs(ang_vel) < 0.5) else 0.0
    late_stability = low_alt * (
        good_stability_low * 4.0  # positive reward for being stable
        - abs(x_vel) * 1.5        # mild penalty for horizontal velocity
        - abs(angle) * 2.0        # mild angle penalty
        - abs(ang_vel) * 1.0      # mild angular velocity penalty
    )

    # Leg contact bonus – strengthened with velocity/angle condition and increased reward
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    stable_landing_condition = (abs(y_vel) < 0.5 and abs(x_vel) < 0.2 and abs(angle) < 0.2)
    leg_bonus = near_pad * both_legs * (15.0 if stable_landing_condition else 2.0)

    # Terminal evaluation based on termination/truncation
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        # Check for crash: high velocity or large angle at ground contact
        crashed = 1.0 if (left_leg < 0.5 or right_leg < 0.5) else 0.0
        # Severe crash only if really bad
        if crashed and (abs(y_vel) > 1.0 or abs(angle) > 0.6):
            terminal_penalty = -10.0  # reduced magnitude
        elif not crashed:
            # Successful landing (both legs, near upright, low velocity)
            terminal_bonus = 20.0 * near_pad * both_legs * (1.0 - abs(angle) * 2.0) * (1.0 - abs(y_vel) * 2.0)
    elif truncated:
        # Timeout: penalize if not near pad
        if next_distance > 0.3:
            terminal_penalty = -3.0  # reduced from -5.0

    # Remove flight_stability to avoid double penalty (angle/angular velocity already penalized in late_stability)
    flight_stability = 0.0

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