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
    # Unpack observations
    x = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Previous state for delta
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    # Distances to landing pad (0,0)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x * x + y * y)
    progress = prev_dist - curr_dist  # positive → moving toward pad

    # Stage flags
    low_alt = 1.0 if y < 0.2 else 0.0
    high_alt = 1.0 - low_alt

    # --- Progress terms ---
    approach_progress = progress * 5.0  # scaled

    # --- Stability / Safety (throughout) ---
    # Light flight penalty for high angle or spin
    flight_stability = -(abs(angle) * 1.5 + abs(ang_vel) * 0.8)

    # Extra ground stability: discourage horizontal drift and large angle near pad
    ground_stability = low_alt * (-abs(vx) * 3.0 - abs(angle) * 4.0 - abs(ang_vel) * 1.5)

    # --- Leg contact bonus ---
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    near_pad = 1.0 if (y < 0.15 and abs(x) < 0.2) else 0.0
    leg_bonus = near_pad * both_legs * 12.0

    # --- Terminal evaluation ---
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        # Safe landing criteria (relaxed from parent – does not require both legs)
        safe_landing = (
            y < 0.15
            and abs(x) < 0.2
            and abs(vy) < 0.5
            and abs(vx) < 0.5
            and abs(angle) < 0.4
        )
        if safe_landing:
            terminal_bonus = 15.0
            # Additional bonus for both legs touching
            if both_legs:
                terminal_bonus += 5.0
        else:
            # Clear crash: high impact velocity, large angle, or off‑pad with no legs
            hard_crash = (abs(vy) > 1.0 or abs(angle) > 0.6 or (y < 0.05 and not both_legs))
            if hard_crash:
                terminal_penalty = -12.0
            else:
                terminal_penalty = -4.0
    elif truncated:
        # Timeout: penalize if far from pad
        if curr_dist > 0.5:
            terminal_penalty = -5.0

    components = {
        "approach_progress": approach_progress,
        "flight_stability": flight_stability,
        "ground_stability": ground_stability,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
    }

    total_reward = (
        approach_progress
        + flight_stability
        + ground_stability
        + leg_bonus
        + terminal_bonus
        + terminal_penalty
    )
    return float(total_reward), components