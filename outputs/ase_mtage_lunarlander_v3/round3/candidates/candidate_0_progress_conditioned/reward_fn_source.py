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
    # Extract current and next observation features
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distances to origin (landing pad at origin)
    curr_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)

    # Stage determination: high altitude (y > 0.3) vs low altitude (y <= 0.3)
    high_alt = 1.0 if y1 > 0.3 else 0.0
    low_alt = 1.0 - high_alt

    # ---- Early stage (high altitude) ----
    # Approach progress: positive improvement in distance
    progress = curr_dist - next_dist
    approach_bonus = high_alt * max(progress, 0.0) * 4.0

    # Speed penalty for high altitude: discourage excessive vertical speed (to reduce crashes)
    speed_penalty_high = high_alt * (-abs(vy) * 0.8 - abs(vx) * 0.4)

    # ---- Late stage (low altitude) ----
    # Stability: reward upright, low angular velocity, low horizontal velocity
    low_alt_stability = low_alt * (
        -abs(angle) * 3.0
        - abs(ang_vel) * 1.5
        - abs(vx) * 2.0
    )
    # Vertical speed penalty only when high (to discourage hard landing)
    vert_penalty_low = low_alt * (-max(0.0, -vy - 0.5) * 1.0)  # only penalize downward speed > 0.5

    # Leg contact bonus (only when both legs contact and very near pad)
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    near_pad = 1.0 if (abs(x1) < 0.2 and y1 < 0.15) else 0.0
    leg_bonus = near_pad * both_legs * 8.0

    # ---- Terminal evaluation ----
    terminal_bonus = 0.0
    terminal_crash = 0.0
    if terminated:
        # Check for crash: either no leg contact or excessive velocity/angle
        crashed = 1.0 if (left_leg < 0.5 or right_leg < 0.5) else 0.0
        if crashed or abs(vy) > 1.0 or abs(angle) > 0.5:
            terminal_crash = -10.0
        else:
            # Successful landing
            terminal_bonus = 15.0 * near_pad * both_legs * (1.0 - abs(angle) * 2.0) * (1.0 - abs(vy) * 2.0)
            terminal_bonus = max(terminal_bonus, 0.0)  # clamp negative to zero
    elif truncated:
        # Timeout penalty only if far from pad
        if next_dist > 0.4:
            terminal_crash = -3.0

    # Component dictionary
    components = {
        "approach_bonus": approach_bonus,
        "speed_penalty_high": speed_penalty_high,
        "low_alt_stability": low_alt_stability,
        "vert_penalty_low": vert_penalty_low,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "terminal_crash": terminal_crash,
    }

    total_reward = approach_bonus + speed_penalty_high + low_alt_stability + vert_penalty_low + leg_bonus + terminal_bonus + terminal_crash
    return float(total_reward), components