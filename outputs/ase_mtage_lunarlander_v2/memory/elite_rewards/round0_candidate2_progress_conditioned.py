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
    # Extract current and next observations
    x_cur = _safe_float(obs[0])
    y_cur = _safe_float(obs[1])
    vx_cur = _safe_float(obs[2])
    vy_cur = _safe_float(obs[3])
    angle_cur = _safe_float(obs[4])
    # angular_velocity_cur = _safe_float(obs[5])  # not used directly
    leg1_cur = _safe_float(obs[6])
    leg2_cur = _safe_float(obs[7])

    x_next = _safe_float(next_obs[0])
    y_next = _safe_float(next_obs[1])
    vx_next = _safe_float(next_obs[2])
    vy_next = _safe_float(next_obs[3])
    angle_next = _safe_float(next_obs[4])
    # angular_velocity_next = _safe_float(next_obs[5])  # not used directly
    leg1_next = _safe_float(next_obs[6])
    leg2_next = _safe_float(next_obs[7])

    # Distances to landing pad (0,0)
    dist_cur = math.sqrt(x_cur * x_cur + y_cur * y_cur)
    dist_next = math.sqrt(x_next * x_next + y_next * y_next)

    # Progress: negative delta means moving closer to pad
    distance_delta = dist_cur - dist_next  # positive if moving closer

    # Speed magnitude
    speed_next = math.sqrt(vx_next * vx_next + vy_next * vy_next)

    # Angle absolute (corrected to be within [0, pi] but we just take abs)
    angle_abs = abs(angle_next)

    # Leg contact indicator (both legs grounded)
    both_legs = 1.0 if (leg1_next > 0.5 and leg2_next > 0.5) else 0.0

    # ---- Stage definitions based on height (y) ----
    # High altitude: y > 0.5 (still far from ground)
    # Medium altitude: 0.1 < y <= 0.5
    # Low altitude / near ground: y <= 0.1 and x near 0
    near_ground = 1.0 if y_next <= 0.1 else 0.0
    near_pad_x = 1.0 if abs(x_next) <= 0.15 else 0.0
    landing_zone = near_ground * near_pad_x  # both conditions

    high_alt = 1.0 if y_next > 0.5 else 0.0
    medium_alt = 1.0 if (y_next > 0.1 and y_next <= 0.5) else 0.0
    low_alt = 1.0 if y_next <= 0.1 else 0.0

    # ---- Reward components ----

    # 1. Approach progress: only when high or medium altitude (not low)
    approach_progress = (high_alt + 0.5 * medium_alt) * max(distance_delta, 0.0) * 5.0

    # 2. Velocity control reward: encourage reducing speed when near ground (medium and low)
    # For medium altitude, reward reducing vertical speed (positive vy means moving up, we want negative small magnitude)
    # For low altitude, reward small speed magnitude overall
    vel_ctrl_med = medium_alt * max(0.0, 1.0 - abs(vy_cur) ) * 0.5  # reward staying with small vertical speed
    vel_ctrl_low = low_alt * max(0.0, 1.0 - speed_next * 2.0) * 1.0  # soft landing: speed < 0.5 gives reward
    velocity_control = vel_ctrl_med + vel_ctrl_low

    # 3. Angle stability: penalize large angle when near ground
    angle_penalty = -abs(angle_next) * (medium_alt + low_alt) * 2.0

    # 4. Landing bonus: both legs on ground, near pad, low speed
    landing_bonus = 0.0
    if terminated and landing_zone > 0.5 and both_legs > 0.5 and speed_next < 0.5 and angle_abs < 0.1:
        landing_bonus = 10.0

    # 5. Crash penalty: terminated with high speed or large angle or out of bounds
    crash_penalty = 0.0
    if terminated:
        # Check for crash: high speed (>1.5) or large angle (>0.8) or out of pad (|x|>1)
        if speed_next > 1.5 or angle_abs > 0.8 or abs(x_next) > 1.0:
            crash_penalty = -10.0
        # Also penalize if terminated without both legs (crash into ground)
        if both_legs < 0.5:
            crash_penalty = -8.0

    # 6. Timeout penalty: truncated with poor progress (low distance reduction)
    timeout_penalty = 0.0
    if truncated:
        # If after many steps we are still far from pad or have high speed
        if dist_next > 0.8:
            timeout_penalty = -2.0
        elif speed_next > 1.0:
            timeout_penalty = -1.0

    # 7. Small survival bonus for staying in the game (but only if making progress)
    survival_bonus = 0.0
    if not terminated and not truncated:
        # Only if we are not stuck near starting position for too long
        # Here we don't have history, so we give a small constant to encourage exploration
        survival_bonus = 0.01  # very small to avoid over-rewarding survival

    total_reward = (approach_progress + velocity_control + angle_penalty +
                    landing_bonus + crash_penalty + timeout_penalty + survival_bonus)

    components = {
        "approach_progress": approach_progress,
        "velocity_control": velocity_control,
        "angle_stability": angle_penalty,
        "landing_bonus": landing_bonus,
        "crash_penalty": crash_penalty,
        "timeout_penalty": timeout_penalty,
        "survival_bonus": survival_bonus
    }

    return float(total_reward), components