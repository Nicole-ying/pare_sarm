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

    # Compute distances to landing pad (pad at (0,0))
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # Stage indicators
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    low_alt = 1.0 if y_pos <= 0.3 else 0.0  # near ground
    near_pad = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2) else 0.0
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0

    # Progress term: reward for moving toward pad (delta improvement)
    progress = curr_distance - next_distance  # positive = moving closer
    early_approach = high_alt * max(progress, 0.0) * 5.0

    # Stability/safety term: positive reward for stable near-landing state
    # Gated by proximity to pad to avoid rewarding low-altitude survival away from pad
    near_pad_for_stability = 1.0 if abs(x_pos) < 0.3 else 0.0
    low_alt_stability = low_alt * near_pad_for_stability * max(0.0, 1.0 - (abs(angle) * 3.0 + abs(ang_vel) * 2.0)) * 2.0

    # Leg contact bonus: only when near pad and both legs contact
    leg_bonus = near_pad * both_legs * 10.0

    # Terminal evaluation
    terminal_bonus = 0.0
    terminal_penalty = 0.0
    if terminated:
        # Determine if successful soft landing
        success = (near_pad > 0.5 and both_legs > 0.5 and 
                   abs(y_vel) < 0.5 and abs(angle) < 0.2 and abs(x_vel) < 0.3)
        if success:
            terminal_bonus = 15.0
        else:
            # Softened crash penalty: only penalize severe crashes (high velocity or high angle)
            severe_crash = (abs(y_vel) > 0.5 or abs(angle) > 0.4)
            if severe_crash:
                crash_severity = min(abs(y_vel) * 2.0 + abs(angle) * 2.0, 6.0)
                terminal_penalty = -crash_severity
            else:
                terminal_penalty = -0.5  # small penalty for moderate end
    elif truncated:
        # Timeout penalty if far from pad
        if next_distance > 0.5:
            terminal_penalty = -1.0

    # Assemble components dictionary
    components = {
        "early_approach": early_approach,
        "low_alt_stability": low_alt_stability,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
    }

    total_reward = (2.0 * early_approach + 
                    1.5 * low_alt_stability + 
                    leg_bonus + 
                    terminal_bonus + 
                    terminal_penalty)
    return float(total_reward), components