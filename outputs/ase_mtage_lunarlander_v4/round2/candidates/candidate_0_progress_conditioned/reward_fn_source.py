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
    # Extract current and next state features from the observation array.
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx1 = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy1 = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle1 = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    angvel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distance to the landing pad (origin).
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage definition: near when distance < 0.4.
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Early stage (far): reward approach/progress ----
    approach_progress = far_stage * progress_delta

    # ---- Late stage (near): reward stability and precision ----
    # Use moderate penalties to avoid over-punishing partial-progress trajectories.
    speed_penalty = - (abs(vx1) + abs(vy1)) * 1.0
    angle_penalty = - abs(angle1) * 2.0
    angvel_penalty = - abs(angvel1) * 0.5
    leg_bonus = 10.0 if (left_leg1 > 0.5 and right_leg1 > 0.5) else 0.0
    stability_near = near_stage * (speed_penalty + angle_penalty + angvel_penalty + leg_bonus)

    # ---- Terminal event handling ----
    terminal_reward = 0.0
    if terminated:
        # Safe landing: both legs contact, moderate speeds, small angle.
        safe = (
            left_leg1 > 0.5 and
            right_leg1 > 0.5 and
            abs(vy1) < 0.3 and
            abs(vx1) < 0.3 and
            abs(angle1) < 0.2
        )
        if safe:
            terminal_reward = 20.0
        else:
            # Crash heuristic: high speed, large tilt, or no leg contact.
            crash = (
                abs(vy1) > 0.8 or
                abs(angle1) > 0.5 or
                abs(vx1) > 0.5 or
                not (left_leg1 > 0.5 and right_leg1 > 0.5)
            )
            if crash:
                terminal_reward = -5.0
            # else ambiguous termination – no reward or penalty.

    # ---- Low-progress timeout penalty ----
    low_progress_timeout = -1.0 if truncated and next_dist > 0.5 else 0.0

    components = {
        "approach_progress": approach_progress,
        "stability_near": stability_near,
        "terminal_reward": terminal_reward,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        6.0 * approach_progress
        + 1.0 * stability_near
        + terminal_reward
        + low_progress_timeout
    )
    return float(total_reward), components