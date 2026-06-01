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
    leg1_cur = _safe_float(obs[6])
    leg2_cur = _safe_float(obs[7])

    x_next = _safe_float(next_obs[0])
    y_next = _safe_float(next_obs[1])
    vx_next = _safe_float(next_obs[2])
    vy_next = _safe_float(next_obs[3])
    angle_next = _safe_float(next_obs[4])
    leg1_next = _safe_float(next_obs[6])
    leg2_next = _safe_float(next_obs[7])

    # Distances to landing pad (0,0)
    dist_cur = math.sqrt(x_cur * x_cur + y_cur * y_cur)
    dist_next = math.sqrt(x_next * x_next + y_next * y_next)

    # Progress: positive if moving closer
    distance_delta = dist_cur - dist_next

    # Speed magnitude
    speed_next = math.sqrt(vx_next * vx_next + vy_next * vy_next)

    # Angle absolute
    angle_abs = abs(angle_next)

    # Leg contact indicator (both legs grounded)
    both_legs = 1.0 if (leg1_next > 0.5 and leg2_next > 0.5) else 0.0

    # ---- Stage definitions based on height (y) ----
    high_alt = 1.0 if y_next > 0.5 else 0.0
    medium_alt = 1.0 if (y_next > 0.1 and y_next <= 0.5) else 0.0
    low_alt = 1.0 if y_next <= 0.1 else 0.0

    near_ground = 1.0 if y_next <= 0.1 else 0.0
    near_pad_x = 1.0 if abs(x_next) <= 0.15 else 0.0
    landing_zone = near_ground * near_pad_x

    # ---- Reward components (local_repair adjustments) ----

    # 1. Approach progress (slightly increased medium coefficient)
    approach_progress = (high_alt * 5.0 + medium_alt * 3.0) * max(distance_delta, 0.0)

    # 2. Velocity control (unchanged)
    vel_ctrl_med = medium_alt * max(0.0, 1.0 - abs(vy_cur)) * 0.5
    vel_ctrl_low = low_alt * max(0.0, 1.0 - speed_next * 2.0) * 1.0
    velocity_control = vel_ctrl_med + vel_ctrl_low

    # 3. Angle stability (increased weight for low altitude)
    angle_penalty = -angle_abs * (medium_alt + low_alt) * 2.5

    # 4. Landing bonus (slightly relaxed thresholds, increased reward)
    landing_bonus = 0.0
    if terminated and landing_zone > 0.5 and both_legs > 0.5 and speed_next < 0.6 and angle_abs < 0.15:
        landing_bonus = 12.0

    # 5. Crash penalty (unchanged)
    crash_penalty = 0.0
    if terminated:
        if speed_next > 1.5 or angle_abs > 0.8 or abs(x_next) > 1.0:
            crash_penalty = -10.0
        if both_legs < 0.5:
            crash_penalty = -8.0

    # 6. Timeout penalty (increased penalty for far distance)
    timeout_penalty = 0.0
    if truncated:
        if dist_next > 0.8:
            timeout_penalty = -3.0
        elif speed_next > 1.0:
            timeout_penalty = -1.0

    # 7. Survival bonus (very small, unchanged)
    survival_bonus = 0.01 if (not terminated and not truncated) else 0.0

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