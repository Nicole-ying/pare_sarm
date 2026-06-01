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

    # Also extract current obs to measure progress
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    # Distances to pad (0,0)
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)
    progress = curr_distance - next_distance  # positive = moving toward pad

    # Stage detection: far (high altitude or far horizontally) vs near (low altitude and close horizontally)
    # Use altitude as primary stage gate; also consider horizontal distance for near condition
    far_stage = 1.0 if y_pos > 0.35 or abs(x_pos) > 0.4 else 0.0
    near_stage = 1.0 - far_stage

    # Components
    # 1. Approach progress (only in far stage)
    approach_progress = far_stage * max(progress, 0.0) * 5.0

    # 2. Near-pad stability (only in near stage) – softer than parent
    near_stability = near_stage * (
        -abs(x_vel) * 1.0
        -abs(angle) * 2.0
        -abs(ang_vel) * 1.0
    )

    # 3. Leg contact bonus (both legs, near pad)
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    on_pad = 1.0 if (abs(x_pos) < 0.2 and y_pos < 0.15) else 0.0
    leg_bonus = both_legs * on_pad * 10.0

    # 4. Terminal bonus (successful landing)
    terminal_bonus = 0.0
    if terminated and both_legs and abs(angle) < 0.3 and y_vel > -0.5 and abs(x_vel) < 0.3 and abs(x_pos) < 0.3 and y_pos < 0.2:
        terminal_bonus = 15.0

    # 5. Terminal penalty (crash or timeout)
    terminal_penalty = 0.0
    if terminated:
        # Crash if not both legs, or high impact/angle
        if not both_legs or abs(y_vel) > 1.0 or abs(angle) > 0.5:
            terminal_penalty = -10.0
    elif truncated:
        # Timeout penalty proportional to distance from pad
        terminal_penalty = -2.0 * min(1.0, next_distance / 0.5)

    # 6. Mild speed penalty in far stage (to guide controlled descent without over-penalizing)
    far_speed_penalty = far_stage * (-abs(y_vel) * 0.2 - abs(x_vel) * 0.1)

    components = {
        "approach_progress": approach_progress,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
        "far_speed_penalty": far_speed_penalty,
    }

    total_reward = (
        approach_progress
        + near_stability
        + leg_bonus
        + terminal_bonus
        + terminal_penalty
        + far_speed_penalty
    )
    return float(total_reward), components