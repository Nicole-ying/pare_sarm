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
    # Extract observations
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distances to landing pad (center at (0,0))
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)

    # --- Progress component (delta approach) ---
    approach_delta = prev_dist - next_dist
    # Scale to be roughly in range [-1,1] per step (max possible delta ~0.1 per step? Use 10x)
    approach_progress = 10.0 * approach_delta

    # --- Stability/safety component ---
    speed = math.sqrt(vx1 * vx1 + vy1 * vy1)
    angle_penalty = abs(angle1) * 5.0  # radian
    ang_vel_penalty = abs(ang_vel1) * 2.0
    raw_stability = speed + angle_penalty + ang_vel_penalty
    stability_penalty = -0.5 * raw_stability  # negative penalty

    # --- Terminal component ---
    terminal_value = 0.0
    if terminated:
        # Check for successful landing: both legs contact, close to pad, low speed, low tilt
        both_legs = (left_leg > 0.5 and right_leg > 0.5)
        near_pad = next_dist < 0.35
        low_speed = speed < 0.5
        upright = abs(angle1) < 0.15
        if both_legs and near_pad and low_speed and upright:
            terminal_value = 10.0
        else:
            terminal_value = -10.0  # crash or out-of-bounds
    # Truncated penalty for not making enough progress
    if truncated and next_dist > 0.5:
        terminal_value = -2.0

    # --- Components dictionary ---
    components = {
        "approach_progress": approach_progress,
        "stability_penalty": stability_penalty,
        "terminal_value": terminal_value,
    }

    total_reward = 1.0 * approach_progress + 1.0 * stability_penalty + terminal_value
    return float(total_reward), components