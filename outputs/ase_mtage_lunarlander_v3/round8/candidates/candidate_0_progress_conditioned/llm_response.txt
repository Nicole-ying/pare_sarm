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
    # observations: [x, y, x_vel, y_vel, angle, ang_vel, left_leg, right_leg]
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    x_vel = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    y_vel = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    angle = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # distances to landing pad (center at 0,0)
    curr_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)

    # stage detection: far (high altitude or far horizontally) vs near
    far_stage = 1.0 if (y1 > 0.3 or abs(x1) > 0.4) else 0.0
    near_stage = 1.0 - far_stage

    # far stage: reward progress toward pad
    progress_delta = curr_dist - next_dist  # positive is moving closer
    far_progress = far_stage * max(progress_delta, 0.0) * 5.0

    # near stage: reward stability (low speeds, upright)
    stability_score = max(0.0, 1.0 - (abs(x_vel) * 0.5 + abs(y_vel) * 0.5 + abs(angle) * 2.0 + abs(ang_vel) * 1.0))
    near_stability = near_stage * stability_score * 2.0

    # leg contact bonus (only in near stage)
    both_legs = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
    leg_contact_bonus = near_stage * both_legs * 5.0

    # step progress (preserve component)
    step_progress = max(progress_delta * 2.0, 0.0)

    # terminal success: specific criteria
    success = 0.0
    if terminated and both_legs > 0.5 and abs(y_vel) < 0.5 and abs(x_vel) < 0.3 and abs(angle) < 0.2:
        success = 15.0
    terminal_success_bonus = success

    # crash penalty (if terminated but not successful)
    crash_penalty = 0.0
    if terminated and success == 0.0:
        crash_severity = min(abs(y_vel) * 3.0 + abs(angle) * 4.0, 10.0)
        crash_penalty = -crash_severity

    # timeout penalty (truncated and still far from pad)
    timeout_penalty = 0.0
    if truncated and next_dist > 0.5:
        timeout_penalty = -2.0

    components = {
        "far_progress": far_progress,
        "near_stability": near_stability,
        "leg_contact_bonus": leg_contact_bonus,
        "step_progress": step_progress,
        "terminal_success_bonus": terminal_success_bonus,
        "crash_penalty": crash_penalty,
        "timeout_penalty": timeout_penalty,
    }

    total_reward = (far_progress + near_stability + leg_contact_bonus +
                    step_progress + terminal_success_bonus +
                    crash_penalty + timeout_penalty)

    return float(total_reward), components