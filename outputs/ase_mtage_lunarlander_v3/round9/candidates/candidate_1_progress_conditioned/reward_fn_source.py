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
    # Observations: x, y, x_vel, y_vel, angle, angular_vel, left_leg, right_leg
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

    # Distances to landing pad (pad at origin)
    prev_dist = math.sqrt(x_curr * x_curr + y_curr * y_curr)
    next_dist = math.sqrt(x_pos * x_pos + y_pos * y_pos)
    progress = prev_dist - next_dist  # positive = moving closer

    # Stage gate: far = distance > threshold, near = otherwise
    far_threshold = 0.5
    is_far = 1.0 if next_dist > far_threshold else 0.0
    is_near = 1.0 - is_far

    # ---- Stage 1: Far from pad ----
    # Reward for making progress toward pad; no reward for staying still or moving away
    approach_progress = is_far * max(progress, 0.0) * 6.0
    # Penalize moving away (optional but encourages direction)
    away_penalty = is_far * min(progress, 0.0) * 2.0  # negative

    # ---- Stage 2: Near pad (distance <= threshold) ----
    # Stability: reward low velocity, upright angle, low angular velocity
    stability = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel) + abs(angle) * 2.0 + abs(ang_vel) * 1.5))
    near_stability = is_near * stability * 4.0

    # Leg contact bonus: only when both legs touch and near pad
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    leg_bonus = is_near * both_legs * 8.0

    # ---- Terminal evaluation (use softer criteria for success) ----
    terminal_bonus = 0.0
    terminal_penalty = 0.0
    if terminated:
        # Success: both legs contact, near pad, moderate velocities and angle
        success = (both_legs > 0.5 and
                   abs(x_pos) < 0.25 and y_pos < 0.15 and
                   abs(y_vel) < 0.8 and abs(x_vel) < 0.5 and abs(angle) < 0.3)
        if success:
            terminal_bonus = 12.0
        else:
            # Crash penalty: severity based on impact velocity and angle
            severity = min(abs(y_vel) * 5.0 + abs(angle) * 5.0, 15.0)
            terminal_penalty = -severity
    elif truncated:
        # Timeout penalty if far from pad with no progress
        if next_dist > 0.5:
            terminal_penalty = -2.0
        elif next_dist > 0.3:
            terminal_penalty = -1.0

    # ---- Assemble components ----
    components = {
        "approach_progress": approach_progress,
        "away_penalty": away_penalty,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty
    }

    total_reward = (approach_progress + away_penalty +
                    near_stability + leg_bonus +
                    terminal_bonus + terminal_penalty)
    return float(total_reward), components