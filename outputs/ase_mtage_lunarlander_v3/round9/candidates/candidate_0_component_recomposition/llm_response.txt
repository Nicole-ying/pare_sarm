import math


def _safe_float(x, default=0.0):
    try:
        value = float(x)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value


def compute_reward(obs, action, next_obs, terminated, truncated, info):
    # Extract next observation components
    x_pos = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_pos = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Current position from previous observation
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    # Distances to landing pad (pad at (0,0))
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # --- Progress term (preserve from parent) ---
    progress_delta = curr_distance - next_distance  # positive = moving closer
    approach_progress = max(progress_delta, 0.0) * 5.0

    # --- Proximity-gated stability ---
    near_pad = 1.0 if (abs(x_pos) < 0.3 and y_pos < 0.5) else 0.0
    # Stability score when near pad: encourage low angle, low velocity, low angular velocity
    stability_score = 0.0
    if near_pad > 0.5:
        stability_score = max(0.0, 1.0 - (abs(angle) * 3.0 + abs(ang_vel) * 2.0 + abs(x_vel) * 1.5 + abs(y_vel) * 0.8))
    stability_near_pad = near_pad * stability_score * 3.0

    # --- Leg contact bonus (soft, attainable for partial progress) ---
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    any_leg = 1.0 if (left_leg > 0.5 or right_leg > 0.5) else 0.0
    leg_contact_bonus = near_pad * (any_leg * 0.5 + both_legs * 2.0)

    # --- Terminal evaluation ---
    terminal_approach_penalty = 0.0
    terminal_success_bonus = 0.0
    terminal_crash_penalty = 0.0
    timeout_penalty = 0.0

    if terminated:
        # Determine landing zone quality
        on_pad = abs(x_pos) < 0.25 and y_pos < 0.2
        soft_landing = on_pad and both_legs > 0.5 and abs(y_vel) < 0.5 and abs(angle) < 0.2 and abs(x_vel) < 0.3
        crash_severity = abs(y_vel) * 3.0 + abs(angle) * 3.0 + next_distance * 2.0
        if soft_landing:
            terminal_success_bonus = 12.0
        elif on_pad:
            # Partial landing with some velocity/angle - reward progress but penalize severity
            terminal_approach_penalty = -min(crash_severity, 6.0)
        else:
            # Crashed far from pad - heavy penalty
            terminal_crash_penalty = -min(crash_severity + 5.0, 12.0)
    elif truncated:
        # Timeout penalty: larger if far from pad
        if next_distance > 0.8:
            timeout_penalty = -4.0
        elif next_distance > 0.4:
            timeout_penalty = -2.0
        else:
            timeout_penalty = -0.5

    # Assemble components
    components = {
        "approach_progress": approach_progress,
        "stability_near_pad": stability_near_pad,
        "leg_contact_bonus": leg_contact_bonus,
        "terminal_success_bonus": terminal_success_bonus,
        "terminal_approach_penalty": terminal_approach_penalty,
        "terminal_crash_penalty": terminal_crash_penalty,
        "timeout_penalty": timeout_penalty,
    }

    total_reward = (approach_progress +
                    stability_near_pad +
                    leg_contact_bonus +
                    terminal_success_bonus +
                    terminal_approach_penalty +
                    terminal_crash_penalty +
                    timeout_penalty)

    return float(total_reward), components