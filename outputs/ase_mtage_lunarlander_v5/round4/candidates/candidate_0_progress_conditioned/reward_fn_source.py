import math

# Thresholds for stage gating
FAR_THRESHOLD = 0.35
NEAR_THRESHOLD = 0.2
STABLE_CONTACT_THRESHOLD = 0.5  # leg contact > 0.5 means true

def _safe_float(x, default=0.0):
    try:
        value = float(x)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value

def compute_reward(obs, action, next_obs, terminated, truncated, info):
    # Unpack observations (indices 0-7)
    x0 = _safe_float(obs[0]) if len(obs) > 0 else 0.0
    y0 = _safe_float(obs[1]) if len(obs) > 1 else 0.0
    x1 = _safe_float(next_obs[0]) if len(next_obs) > 0 else 0.0
    y1 = _safe_float(next_obs[1]) if len(next_obs) > 1 else 0.0
    vx = _safe_float(next_obs[2]) if len(next_obs) > 2 else 0.0
    vy = _safe_float(next_obs[3]) if len(next_obs) > 3 else 0.0
    ang = _safe_float(next_obs[4]) if len(next_obs) > 4 else 0.0
    ang_vel = _safe_float(next_obs[5]) if len(next_obs) > 5 else 0.0
    leg_left = _safe_float(next_obs[6]) if len(next_obs) > 6 else 0.0
    leg_right = _safe_float(next_obs[7]) if len(next_obs) > 7 else 0.0
    # Previous leg contact for delta
    prev_leg_left = _safe_float(obs[6]) if len(obs) > 6 else 0.0
    prev_leg_right = _safe_float(obs[7]) if len(obs) > 7 else 0.0

    # Distance from pad (0,0)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist   # positive = moving closer

    # Stage detection
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    both_legs = 1.0 if (leg_left > STABLE_CONTACT_THRESHOLD and leg_right > STABLE_CONTACT_THRESHOLD) else 0.0
    approach_stage = 1.0 if (curr_dist <= FAR_THRESHOLD and both_legs < 0.5) else 0.0
    landing_stage = 1.0 if (curr_dist <= FAR_THRESHOLD and both_legs > 0.5) else 0.0

    # 1) Approach progress (far stage only)
    progress = max(delta_dist, 0.0) * far_stage * 5.0

    # 2) Orientation penalty (far stage) – discourage spinning
    orientation_penalty = far_stage * (-abs(ang_vel) * 0.5)

    # 3) Leg gain bonus (approach stage) – reward gaining new leg contact
    leg_gain = max(0.0, (leg_left + leg_right) - (prev_leg_left + prev_leg_right))
    leg_gain_bonus = leg_gain * approach_stage * 1.0

    # 4) Leg contact bonus (approach stage) – sustain contact
    leg_bonus = (leg_left + leg_right) * approach_stage * 0.5

    # 5) Exponential stability reward (approach + landing stages)
    stability_score = math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel)))
    near_stability = (approach_stage + landing_stage) * stability_score * 1.0

    # 6) Dense landing reward (landing stage only)
    landing_dense = landing_stage * 1.0

    # 7) Safe landing condition for terminal events
    safe_landing_cond = (
        both_legs > 0.5
        and abs(vy) < 0.1
        and abs(vx) < 0.1
        and abs(ang) < 0.1
        and abs(ang_vel) < 0.1
        and curr_dist < NEAR_THRESHOLD
    )
    safe_landing_bonus = 10.0 if terminated and safe_landing_cond else 0.0
    failure_penalty = -1.0 if terminated and not safe_landing_cond else 0.0
    timeout_penalty = -2.0 if truncated and curr_dist > FAR_THRESHOLD else 0.0

    total_reward = (
        progress
        + orientation_penalty
        + leg_gain_bonus
        + leg_bonus
        + near_stability
        + landing_dense
        + safe_landing_bonus
        + failure_penalty
        + timeout_penalty
    )

    components = {
        "approach_progress": progress,
        "orientation_penalty": orientation_penalty,
        "leg_gain_bonus": leg_gain_bonus,
        "leg_contact_bonus": leg_bonus,
        "near_stage_stability": near_stability,
        "landing_dense_bonus": landing_dense,
        "safe_landing_bonus": safe_landing_bonus,
        "terminal_failure_penalty": failure_penalty,
        "timeout_penalty": timeout_penalty,
    }

    return float(total_reward), components