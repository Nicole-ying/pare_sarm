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
    # Extract relevant observations
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    vx0 = _safe_float(obs[2]) if len(obs) > 2 else 0.0
    vy0 = _safe_float(obs[3]) if len(obs) > 3 else 0.0
    angle0 = _safe_float(obs[4]) if len(obs) > 4 else 0.0
    angvel0 = _safe_float(obs[5]) if len(obs) > 5 else 0.0
    left_leg0 = _safe_float(obs[6]) if len(obs) > 6 else 0.0
    right_leg0 = _safe_float(obs[7]) if len(obs) > 7 else 0.0

    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    angvel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distance to landing pad (assumed at (0,0))
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)

    # Progress: reduction in distance from previous step
    progress_delta = prev_dist - next_dist

    # Stability: penalize large angle, angular velocity, and side velocity
    stability_penalty = -(abs(angle1) + 0.1 * abs(angvel1) + 0.5 * abs(vx1))

    # High impact penalty: if terminated and harsh landing
    crash_penalty = 0.0
    if terminated:
        # Detect crash: either out-of-bounds or too fast/angled at ground
        out_of_bounds = abs(x1) > 1.0 or y1 < -0.5
        harsh_landing = (abs(vy1) > 1.0 or abs(angle1) > 0.8)
        if out_of_bounds or harsh_landing:
            crash_penalty = -1.0

    # Success bonus: both legs contact, near upright, near zero velocity, near pad
    success_bonus = 0.0
    if terminated:
        both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
        near_upright = abs(angle1) < 0.1
        low_velocity = abs(vy1) < 0.1 and abs(vx1) < 0.1
        near_pad = next_dist < 0.15
        if both_legs and near_upright and low_velocity and near_pad:
            success_bonus = 2.0

    # Truncation penalty for failing to approach pad
    low_progress_timeout = 0.0
    if truncated and next_dist > 0.5:
        low_progress_timeout = -0.5

    # Progress component (gated: only positive when moving towards pad, else negative)
    progress_reward = progress_delta  # positive if closing in

    # Compose components dictionary
    components = {
        "progress": progress_reward,
        "stability": stability_penalty,
        "crash_penalty": crash_penalty,
        "success_bonus": success_bonus,
        "low_progress_timeout": low_progress_timeout
    }

    # Weights: emphasize progress, stability, and terminal events
    total_reward = (
        3.0 * progress_reward +
        1.5 * stability_penalty +
        crash_penalty +
        success_bonus +
        low_progress_timeout
    )

    return float(total_reward), components