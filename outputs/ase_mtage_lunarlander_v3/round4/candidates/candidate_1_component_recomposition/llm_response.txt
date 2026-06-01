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
    # Extract state from next observation
    x_pos = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_pos = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Current observation for progress
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    # Distances to pad (pad at (0,0) in viewport coordinates, ground at y=0)
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # Stage indicators
    high_alt = 1.0 if y_pos > 0.5 else 0.0
    low_alt = 1.0 - high_alt
    near_pad = 1.0 if (y_pos < 0.15 and abs(x_pos) < 0.2) else 0.0
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0

    # 1. Approach progress (positive for moving toward pad at high altitude)
    progress = curr_distance - next_distance  # positive means moving closer
    approach_progress = high_alt * max(progress, 0.0) * 4.0

    # 2. Descent control (softer penalty on vertical speed at high altitude)
    descent_control = high_alt * (-abs(y_vel) * 0.3)

    # 3. Stability during flight (mild penalty for excessive angle/angular velocity)
    flight_stability = -abs(angle) * 1.0 - abs(ang_vel) * 0.5

    # 4. Low-altitude stability (soft penalty on horizontal speed, angle, angular velocity)
    low_stability = low_alt * (-abs(x_vel) * 1.0 - abs(angle) * 2.0 - abs(ang_vel) * 1.0)

    # 5. Leg bonus (unchanged, positive when both legs on pad)
    leg_bonus = near_pad * both_legs * 10.0

    # 6. Near-landing proximity reward (continuous, replaces hard terminal bonus)
    # Score scales smoothly with proximity to ideal landing conditions
    if both_legs > 0.5 and y_pos < 0.2 and abs(x_pos) < 0.3:
        u = 1.0 - abs(x_pos) / 0.2
        v = 1.0 - y_pos / 0.15  # want y small and negative? Actually y near 0 is ground; treat y=0 as ideal, y small positive as good
        w = 1.0 - abs(angle) / 0.3
        s = 1.0 - abs(y_vel) / 0.6
        t = 1.0 - abs(x_vel) / 0.6
        landing_proximity = max(0.0, u) * max(0.0, v) * max(0.0, w) * max(0.0, s) * max(0.0, t)
        landing_proximity = min(landing_proximity, 1.0) * 5.0
    else:
        landing_proximity = 0.0

    # 7. Terminal handling
    terminal_penalty = 0.0
    if terminated:
        # Crash: no or single leg contact, or high speed/angle at ground
        if not (left_leg > 0.5 and right_leg > 0.5) or abs(y_vel) > 1.5 or abs(angle) > 0.5:
            terminal_penalty = -12.0
        else:
            # Landed with both legs but perhaps not perfect
            terminal_penalty = -abs(x_vel) * 2.0 - abs(angle) * 5.0  # small penalty for imperfections
    elif truncated:
        # Timeout: penalty proportional to distance from pad
        terminal_penalty = -min(max(next_distance - 0.3, 0.0) * 3.0, 6.0)

    components = {
        "approach_progress": approach_progress,
        "descent_control": descent_control,
        "flight_stability": flight_stability,
        "low_stability": low_stability,
        "leg_bonus": leg_bonus,
        "landing_proximity": landing_proximity,
        "terminal_penalty": terminal_penalty,
    }

    total_reward = (approach_progress + descent_control + flight_stability +
                    low_stability + leg_bonus + landing_proximity + terminal_penalty)
    return float(total_reward), components