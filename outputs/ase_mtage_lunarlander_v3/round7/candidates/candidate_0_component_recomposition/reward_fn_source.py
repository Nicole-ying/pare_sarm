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
    # Current state for delta progress
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    # Next state
    x_pos = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_pos = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distances to pad
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # Stage indicators
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    near_pad_region = 1.0 if (abs(x_pos) < 0.3 and y_pos < 0.5) else 0.0
    low_alt = 1.0 if y_pos <= 0.3 else 0.0
    near_pad = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2) else 0.0
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0

    # 1. early_approach (preserved high-alt progress)
    progress_high = max(curr_distance - next_distance, 0.0)
    early_approach = high_alt * progress_high * 5.0

    # 2. pad_approach (progress near the pad)
    pad_approach = 0.0
    if near_pad_region > 0.5 and high_alt < 0.5:
        delta = max(curr_distance - next_distance, 0.0)
        pad_approach = delta * 3.0

    # 3. stable_descent (replaces low_alt_stability, gated by pad proximity and orientation)
    stable_descent = 0.0
    if low_alt > 0.5 and abs(x_pos) < 0.4:
        angle_penalty = min(abs(angle) / 0.3, 1.0)
        ang_vel_penalty = min(abs(ang_vel) / 0.5, 1.0)
        horiz_penalty = min(abs(x_vel) / 0.5, 1.0)
        stability = 1.0 - (angle_penalty + ang_vel_penalty + horiz_penalty) / 3.0
        if stability > 0.0:
            stable_descent = low_alt * stability * 2.0

    # 4. leg_bonus (preserved)
    leg_bonus = near_pad * both_legs * 10.0

    # 5. terminal_bonus (preserved)
    success_condition = (near_pad > 0.5 and both_legs > 0.5 and
                         abs(y_vel) < 0.5 and abs(angle) < 0.2 and abs(x_vel) < 0.3)
    terminal_bonus = 0.0
    if terminated and success_condition:
        terminal_bonus = 15.0

    # 6. landing_progress (reward near-landing terminations that are not perfect)
    landing_progress = 0.0
    if terminated and not success_condition:
        if abs(x_pos) < 0.5 and y_pos < 0.5:
            dist_pen = abs(x_pos) * 10.