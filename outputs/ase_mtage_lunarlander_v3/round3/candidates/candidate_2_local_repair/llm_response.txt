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
    # Use max(0, progress) to avoid penalizing overshoot slightly
    early_approach = high_alt * max(progress, 0.0) * 5.0
    # Penalize high vertical speed when high (to encourage controlled descent) - reduced coefficient
    early_speed_penalty = high_alt * (-abs(y_vel) * 0.2)

    # Late stage (low altitude): reward stability and leg contact - reduced penalties
    late_stability = low_alt * (
        -abs(x_vel) * 1.0  # horizontal velocity penalty (was 2.0)
        -abs(angle) * 2.0  # angle penalty (was 5.0)
        -abs(ang_vel) * 1.0  # angular velocity penalty (was 2.0)
    )
    # Leg contact bonus: only if both legs contact near pad
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    leg_bonus = near_pad * both_legs * 10.0

    # Terminal evaluation based on termination/truncation
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        # Check for crash: high velocity or large angle at ground contact
        # If legs are not both contacting, likely a crash
        crashed = 1.0 if (left_leg < 0.5 or right_leg < 0.5) else 0.0
        # Also possible side crash or high impact
        # Gate: only penalize if not in near-pad condition with low velocity
        if crashed or abs(y_vel) > 0.8 or abs(angle) > 0.4:
            # Only apply penalty if not near pad with low speed (mild penalty)
            if not (near_pad and abs(y_vel) < 0.5 and abs(angle) < 0.3):
                terminal_penalty = -5.0  # was -15.0
        else:
            # Successful landing (both legs, near upright, low velocity)
            terminal_bonus = 20.0 * near_pad * both_legs * (1.0 - abs(angle) * 2.0) * (1.0 - abs(y_vel) * 2.0)
    elif truncated:
        # Timeout: penalize if not near pad
        if next_distance > 0.3:
            terminal_penalty = -5.0

    # Stability during flight: penalize excessive angular velocity and angle - reduced coefficients
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