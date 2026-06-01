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

    # Extract current obs for progress
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    # Distance from landing pad (pad at (0,0))
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # Progress toward pad (positive when moving closer)
    progress = curr_distance - next_distance

    # Stage determination: far vs near
    # Use both distance and altitude to define stages
    near_pad = 1.0 if (next_distance < 0.4 and y_pos < 0.3) else 0.0
    far_stage = 1.0 - near_pad

    # ---------- Stage 1: Far from pad ----------
    # Reward progress (approach) – positive delta, gated by far stage
    early_approach = far_stage * max(progress, 0.0) * 5.0

    # Mild angular stability penalty to discourage spinning at high altitude
    early_stability = far_stage * (-abs(ang_vel) * 1.0)

    # ---------- Stage 2: Near the pad ----------
    # Reward low horizontal velocity and upright orientation (less penalty)
    late_stability = near_pad * (
        -0.5 * abs(x_vel)      # low coefficient for horizontal speed
        - 2.0 * abs(angle)     # angle penalty (radians)
    )

    # Leg contact bonus: only if both legs contact near the pad
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    leg_bonus = near_pad * both_legs * 10.0

    # ---------- Terminal evaluation ----------
    terminal_reward = 0.0
    terminal_penalty = 0.0
    terminal_bonus = 0.0

    if terminated:
        # Check for successful landing: both legs, low velocity, upright
        if (both_legs > 0.5 and abs(angle) < 0.2 and
            abs(y_vel) < 0.5 and abs(x_vel) < 0.5):
            terminal_bonus = 20.0
        else:
            # Clear crash or bad landing
            terminal_penalty = -5.0
    elif truncated:
        # Timeout: penalize if still far from pad, small bonus if near
        if next_distance > 0.3:
            terminal_penalty = -2.0
        else:
            terminal_bonus = 1.0

    # ---------- Assemble components ----------
    components = {
        "early_approach": early_approach,
        "early_stability": early_stability,
        "late_stability": late_stability,
        "leg_bonus": leg_bonus,
        "terminal_penalty": terminal_penalty,
        "terminal_bonus": terminal_bonus,
    }

    total_reward = (early_approach + early_stability + late_stability +
                    leg_bonus + terminal_penalty + terminal_bonus)

    return float(total_reward), components