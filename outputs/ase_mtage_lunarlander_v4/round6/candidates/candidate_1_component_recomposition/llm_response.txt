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
    # Extract state variables from the observation array.
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

    # Distances to the landing pad (origin).
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    next_dist = math.sqrt(x1 * x1 + y1 * y1)
    progress_delta = prev_dist - next_dist  # positive = moving closer

    # Stage definitions.
    near_threshold = 0.4
    far_stage = 1.0 if next_dist >= near_threshold else 0.0
    near_stage = 1.0 - far_stage

    # Stability proxy: high speed or large angle is undesirable.
    speed = abs(vx1) + abs(vy1)
    angle_abs = abs(angle1)
    speed_angle_penalty = min(1.0, speed + angle_abs)  # 0 = ideal, higher = worse

    # ---- 1. Approach progress (far stage) - gated by stability ----
    # Reward progress only if the lander is not too unstable.
    # If unstable, the progress reward is suppressed sharply.
    stability_gate = max(0.0, 1.0 - 2.0 * speed_angle_penalty)  # 1 for ideal, 0 for very bad
    approach_progress = 3.0 * far_stage * progress_delta * stability_gate

    # ---- 2. Far-stage instability penalty ----
    # Directly penalize high speed or large angle in the far stage.
    far_instability = -1.0 * far_stage * speed_angle_penalty

    # ---- 3. Proximity bonus (near stage) ----
    # The closer to the pad, the better.
    proximity_bonus = near_stage * max(0.0, 1.0 - next_dist / near_threshold) * 2.0

    # ---- 4. Near-stage stability reward ----
    # Encourage calm movement when close to the pad.
    near_stability = near_stage * max(0.0, 1.0 - speed_angle_penalty) * 1.5

    # ---- 5. Leg-contact bonus (only when stable and near) ----
    both_legs = (left_leg1 > 0.5 and right_leg1 > 0.5)
    leg_bonus = 0.0
    if near_stage > 0.5 and both_legs and speed < 1.0 and angle_abs < 0.5:
        leg_bonus = 3.0

    # ---- 6. Terminal handling (replace terminal_reward) ----
    crash_penalty = 0.0
    safe_landing_bonus = 0.0
    if terminated:
        safe = (
            both_legs
            and abs(vy1) < 0.5
            and abs(vx1) < 0.5
            and abs(angle1) < 0.3
        )
        if safe:
            safe_landing_bonus = 15.0
        else:
            # Penalize clear crashes: high vertical speed, large tilt, no leg contact.
            crash = (
                abs(vy1) > 1.0
                or abs(angle1) > 0.8
                or abs(vx1) > 1.0
                or not both_legs
            )
            if crash:
                crash_penalty = -5.0
    # else: no terminal reward/penalty for timeouts (handled below)

    # ---- 7. Low-progress timeout penalty ----
    low_progress_timeout = -0.5 if truncated and next_dist > 0.5 else 0.0

    components = {
        "approach_progress": approach_progress,
        "far_instability": far_instability,
        "proximity_bonus": proximity_bonus,
        "near_stability": near_stability,
        "leg_bonus": leg_bonus,
        "crash_penalty": crash_penalty,
        "safe_landing_bonus": safe_landing_bonus,
        "low_progress_timeout": low_progress_timeout,
    }
    total_reward = (
        approach_progress
        + far_instability
        + proximity_bonus
        + near_stability
        + leg_bonus
        + crash_penalty
        + safe_landing_bonus
        + low_progress_timeout
    )
    return float(total_reward), components