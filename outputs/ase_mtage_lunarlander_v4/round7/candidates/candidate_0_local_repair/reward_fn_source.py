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
    # Extract state features from the observation array.
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

    # Distance to landing pad (origin).
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive means moving closer

    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # ---- Approach progress (only for far stage) ----
    # We gate this component: if the agent is unstable (high speed or large tilt),
    # we penalise instead of rewarding progress.
    speed = abs(vx1) + abs(vy1)
    angle_abs = abs(angle1)
    # Penalty factor: 1.0 for stable, decreasing if unstable.
    stability_factor = max(0.0, 1.0 - speed - angle_abs * 1.5)
    # Only reward progress when stable; otherwise maybe penalise.
    if far_stage > 0.5:
        if stability_factor > 0.3:
            approach_progress = 3.0 * far_stage * progress_delta
        else:
            # Unstable: penalise any positive progress or reward less.
            approach_progress = -0.5 * far_stage * (abs(progress_delta) if progress_delta < 0 else 0.0)
    else:
        approach_progress = 0.0

    # ---- Near stage components (preserved from parent) ----
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0

    speed_mag = abs(vx1) + abs(vy1)
    stability_score = max(0.0, 1.0 - speed_mag - abs(angle1))
    near_stability = near_stage * stability_score * 1.5

    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 0.0
    if near_stage > 0.5 and both_legs:
        leg_bonus = 3.0

    # ---- Terminal event (restructured to be more stringent) ----
    terminal_reward = 0.0
    if terminated:
        # Much stricter safe landing conditions: both legs contact, very low speed, small angle.
        safe = (
            both_legs and
            abs(vy1) < 0.3 and
            abs(vx1) < 0.3 and
            abs(angle1) < 0.2
        )
        if safe:
            terminal_reward = 15.0
        else:
            # Clear crash: high vertical speed or large tilt or no legs.
            crash = (
                abs(vy1) > 0.8 or
                abs(angle1) > 0.5 or
                abs(vx1) > 0.8 or
                not both_legs
            )
            if crash:
                terminal_reward = -10.0  # stronger penalty
            # else ambiguous: no reward/penalty.

    # ---- Low-progress timeout penalty ----
    # Increase penalty if far from pad at truncation.
    if truncated:
        if next_dist > 0.5:
            # Heavier penalty for being far when truncated.
            low_progress_timeout = -1.0 * (1.0 + (next_dist - 0.5) * 2.0)
        else:
            low_progress_timeout = 0.0
    else:
        low_progress_timeout = 0.0

    components = {
        "approach_progress": approach_progress,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "terminal_reward": terminal_reward,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        approach_progress
        + proximity_bonus
        + near_stability
        + leg_bonus
        + terminal_reward
        + low_progress_timeout
    )
    return float(total_reward), components