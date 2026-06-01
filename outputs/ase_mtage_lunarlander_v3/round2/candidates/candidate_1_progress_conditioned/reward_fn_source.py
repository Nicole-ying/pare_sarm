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
    # extract next_obs components
    x_pos = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y_pos = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # extract current obs for progress
    x_curr = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y_curr = _safe_float(obs[1]) if len(obs) > 1 else 0.0

    # distance to landing pad (0,0)
    curr_distance = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_distance = math.sqrt(x_pos * x_pos + y_pos * y_pos)

    # stage gating: high altitude if y_pos > 0.3 (above ground contact zone)
    high_alt = 1.0 if y_pos > 0.3 else 0.0
    low_alt = 1.0 - high_alt

    # progress toward pad (positive = moving closer)
    progress = curr_distance - next_distance

    # ---- early stage (high altitude) ----
    # reward approach (same as parent, positive only)
    early_approach = high_alt * max(progress, 0.0) * 5.0

    # penalize only excessive vertical speed (>0.8) to deter crashes while allowing normal descent
    high_speed_penalty = high_alt * (0.0 if abs(y_vel) <= 0.8 else -0.5 * (abs(y_vel) - 0.8))

    # ---- late stage (low altitude) ----
    # stability bonus: positive when near upright, low horizontal speed, low angular velocity
    # define a "stable" condition
    stable_angle = abs(angle) < 0.15
    stable_xvel = abs(x_vel) < 0.15
    stable_angvel = abs(ang_vel) < 0.3
    good_stability = 1.0 if (stable_angle and stable_xvel and stable_angvel) else 0.0

    # positive reward for being stable at low altitude (encourages gentle landing)
    low_alt_stability_bonus = low_alt * good_stability * 3.0

    # penalty for poor stability at low altitude: scaled down compared to parent
    low_alt_stability_penalty = low_alt * ( -abs(angle) * 1.0 - abs(x_vel) * 1.0 - abs(ang_vel) * 0.5 )
    # note: we keep both bonus and penalty to give clear signal

    # leg contact bonus: reward contacting with both legs near pad and low velocity
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    near_pad = 1.0 if (abs(x_pos) < 0.15 and y_pos < 0.15) else 0.0
    safe_contact = (both_legs and near_pad and abs(y_vel) < 0.5 and abs(angle) < 0.2)
    leg_bonus = 1.0 * safe_contact * 15.0

    # ---- terminal evaluation ----
    terminal_penalty = 0.0
    terminal_bonus = 0.0
    if terminated:
        # crash condition: not both legs in contact or high impact
        crash = (left_leg < 0.5 or right_leg < 0.5) or (abs(y_vel) > 0.8 or abs(angle) > 0.4)
        if crash:
            terminal_penalty = -15.0
        else:
            # successful landing
            if near_pad and both_legs and abs(y_vel) < 0.5 and abs(angle) < 0.2:
                terminal_bonus = 20.0
    elif truncated:
        # timeout: penalize only if far from pad
        if next_distance > 0.5:
            terminal_penalty = -3.0

    # combine components
    components = {
        "early_approach": early_approach,
        "high_speed_penalty": high_speed_penalty,
        "low_alt_stability_bonus": low_alt_stability_bonus,
        "low_alt_stability_penalty": low_alt_stability_penalty,
        "leg_bonus": leg_bonus,
        "terminal_penalty": terminal_penalty,
        "terminal_bonus": terminal_bonus,
    }

    total_reward = (early_approach + high_speed_penalty + low_alt_stability_bonus
                    + low_alt_stability_penalty + leg_bonus
                    + terminal_penalty + terminal_bonus)
    return float(total_reward), components