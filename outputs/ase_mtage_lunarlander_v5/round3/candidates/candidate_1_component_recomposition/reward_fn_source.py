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
    # Unpack previous observations (before action)
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    vx0 = _safe_float(obs[2]) if len(obs) > 2 else 0.0
    vy0 = _safe_float(obs[3]) if len(obs) > 3 else 0.0
    ang0 = _safe_float(obs[4]) if len(obs) > 4 else 0.0
    ang_vel0 = _safe_float(obs[5]) if len(obs) > 5 else 0.0
    leg_left0 = _safe_float(obs[6]) if len(obs) > 6 else 0.0
    leg_right0 = _safe_float(obs[7]) if len(obs) > 7 else 0.0

    # Unpack current observations (after action)
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    ang1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    leg_left1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    leg_right1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0