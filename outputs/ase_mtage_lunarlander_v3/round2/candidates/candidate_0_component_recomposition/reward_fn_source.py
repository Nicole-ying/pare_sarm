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
    # Current and next state
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x_vel_next = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel_next = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle_next = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel_next = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg_next = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg_next = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    x_next = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_next = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0

    # Distances to pad (pad at x=0, y=0)
    curr_dist = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_dist = math.sqrt(x_next * x_next + y_next * y_next)

    # Progress toward pad (positive is good)
    progress = curr_dist - next_dist

    # ----- Component definitions -----
    # 1. Approach progress (always positive if moving closer)
    approach_progress = max(progress, 0.0) * 3.0  # scale to be meaningful

    # 2. Crash prevention: penalize only excessive vertical speed
    vertical_speed_excess = max(0.0, abs(y_vel_next) - 0.8)  # threshold 0.8
    crash_prevention_penalty = -vertical_speed_excess * 5.0

    # 3. Low-altitude stability: positive reward when near pad and stable
    low_alt = 1.0 if (y_next < 0.2 and abs(x_next) < 0.3) else 0.0
    stability_bonus = low_alt * (
        -0.5 * abs(x_vel_next)  # small penalty for horizontal drift
        - 2.0 * abs(angle_next) + 1.0  # reward for upright (+1 max)
        - 1.0 * abs(ang_vel_next) + 0.2  # small bonus for low angular vel
    )
    stability_bonus = max(0.0, stability_bonus)  # only nonnegative

    # 4. Leg contact bonus: both legs on ground near pad and low vertical speed
    both_legs = 1.0 if (left_leg_next > 0.5 and right_leg_next > 0.5) else 0.0
    safe_landing = 1.0 if (both_legs and low_alt > 0.5 and abs(y_vel_next) < 0.5 and abs(angle_next) < 0.2) else 0.0
    leg_landing_bonus = safe_landing * 8.0

    # 5. Terminal penalties/bonuses
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        # Check for crash: high velocity, large angle, or no leg contact on ground
        crashed = 1.0 if (abs(y_vel_next) > 0.8 or abs(angle_next) > 0.4 or both_legs < 0.5) else 0.0
        # Weigh severity: if both legs contact after crash? improbable, so simple
        if crashed:
            terminal_penalty = -8.0  # moderate penalty
        else:
            # Successful landing (both legs, near pad, low velocity, upright)
            if safe_landing:
                terminal_bonus = 15.0
    elif truncated:
        # Timeout: penalize only if far from pad
        if next_dist > 0.4:
            terminal_penalty = -3.0

    # Combine components
    components = {
        "approach_progress": approach_progress,
        "crash_prevention_penalty": crash_prevention_penalty,
        "stability_bonus": stability_bonus,
        "leg_landing_bonus": leg_landing_bonus,
        "terminal_penalty": terminal_penalty,
        "terminal_bonus": terminal_bonus,
    }

    total_reward = (approach_progress +
                    crash_prevention_penalty +
                    stability_bonus +
                    leg_landing_bonus +
                    terminal_penalty +
                    terminal_bonus)

    return float(total_reward), components