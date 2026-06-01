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
    # Current and next observations
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

    # Distances to landing pad (pad at (0,0))
    curr_dist = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_dist = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # Stage indicators
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    low_alt = 1.0 if y_pos <= 0.3 else 0.0  # near ground
    near_pad = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2) else 0.0
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    far_from_pad = 1.0 if abs(x_pos) > 0.2 else 0.0

    # 1. Approach progress: delta improvement (positive when moving closer)
    progress_delta = curr_dist - next_dist
    # Only reward when above a certain altitude to avoid spurious signals near ground
    early_approach = high_alt * max(progress_delta, 0.0) * 5.0

    # 2. Progress per step: always reward moving closer, scaled moderately
    step_progress = max(progress_delta, 0.0) * 1.0

    # 3. Low‑alt stability: gated by being near the pad to prevent hovering far from pad
    low_alt_near_pad = low_alt * (1.0 - far_from_pad)
    low_alt_stability = low_alt_near_pad * max(0.0, 1.0 - (abs(angle) * 3.0 + abs(ang_vel) * 2.0)) * 2.0

    # 4. Leg contact bonus: only when near pad and both legs contact
    leg_bonus = near_pad * both_legs * 10.0

    # 5. Terminal evaluation
    terminal_bonus = 0.0
    failure_penalty = 0.0
    if terminated:
        success = (near_pad > 0.5 and both_legs > 0.5 and
                   abs(y_vel) < 0.5 and abs(angle) < 0.2 and abs(x_vel) < 0.3)
        if success:
            terminal_bonus = 20.0
        else:
            # Crash penalty, less severe to avoid over‑penalizing partial progress
            crash_severity = min(abs(y_vel) * 2.0 + abs(angle) * 2.0, 8.0)
            failure_penalty = -crash_severity
    elif truncated:
        # Timeout penalty only if still far from pad
        if next_dist > 0.5:
            failure_penalty = -2.0

    # 6. Hovering penalty: discourage staying at low altitude but far from pad
    hovering_penalty = 0.0
    if low_alt > 0.5 and far_from_pad > 0.5:
        hovering_penalty = -0.5

    # Components dictionary
    components = {
        "early_approach": early_approach,
        "step_progress": step_progress,
        "low_alt_stability": low_alt_stability,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "failure_penalty": failure_penalty,
        "hovering_penalty": hovering_penalty,
    }

    total_reward = (2.0 * early_approach +
                    1.0 * step_progress +
                    1.5 * low_alt_stability +
                    leg_bonus +
                    terminal_bonus +
                    failure_penalty +
                    hovering_penalty)
    return float(total_reward), components