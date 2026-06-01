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
    angvel1 = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    left_leg1 = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    right_leg1 = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0

    # Distance to landing pad (origin)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage definitions: far (beyond 0.4) / near (within 0.4)
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ----- Early stage (far): reward approach, penalize sideways/tilt -----
    # Reward for distance improvement (capped at 0 to avoid negative progress)
    far_approach = far_stage * max(0.0, progress_delta) * 2.0
    # Penalty for high horizontal velocity or tilt (discourages early_failure)
    far_quality_penalty = far_stage * (0.2 * abs(vx1) + 0.2 * abs(angle1))
    # Low‑progress gate: small negative if no improvement at all
    low_progress_penalty = -0.1 * far_stage if progress_delta <= 0.0 else 0.0
    far_stage_reward = far_approach - far_quality_penalty + low_progress_penalty

    # ----- Late stage (near): proximity, stability, leg contact -----
    # Proximity bonus: closer to pad is better
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0
    # Stability: low speed and small angle
    speed_mag = abs(vx1) + abs(vy1)
    stability_score = max(0.0, 1.0 - speed_mag - abs(angle1))
    near_stability = near_stage * stability_score * 1.5
    # Leg contact bonus (only when both legs in contact)
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 3.0 * near_stage if both_legs else 0.0

    # ----- Terminal event handling -----
    terminal_reward = 0.0
    if terminated:
        # Safe landing conditions (relaxed: moderate speeds, small tilt, both legs)
        safe = (
            both_legs
            and abs(vy1) < 0.7
            and abs(vx1) < 0.7
            and abs(angle1) < 0.4
        )
        if safe:
            terminal_reward = 15.0
        else:
            # Crash: high velocity or extreme tilt
            crash = (
                abs(vy1) > 1.0
                or abs(vx1) > 1.0
                or abs(angle1) > 0.9
            )
            if crash:
                terminal_reward = -5.0
            # else ambiguous – no reward or penalty

    # Low‑progress timeout penalty (only when truncated and still far)
    low_progress_timeout = -0.5 if truncated and next_dist > 0.5 else 0.0

    components = {
        "far_approach": far_approach,
        "far_quality_penalty": -far_quality_penalty,
        "low_progress_penalty": low_progress_penalty,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "terminal_reward": terminal_reward,
        "low_progress_timeout": low_progress_timeout,
    }

    total_reward = (
        far_stage_reward
        + proximity_bonus
        + near_stability
        + leg_bonus
        + terminal_reward
        + low_progress_timeout
    )

    return float(total_reward), components