import math


def _safe_float(x, default=0.0):
    try:
        val = float(x)
    except Exception:
        return default
    if not math.isfinite(val):
        return default
    return val


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

    # Distance from pad (0,0)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist   # positive if closer

    # Smooth near-staging factor: 1 when at pad, 0 when far, linear decrease
    FAR_THRESHOLD = 0.5
    near_factor = max(0.0, 1.0 - curr_dist / FAR_THRESHOLD) if FAR_THRESHOLD > 0 else 0.0
    far_stage = 1.0 - near_factor  # 1 when far, 0 when very close

    # ---- Category 1: Progress (only when far from pad) ----
    approach_progress = far_stage * max(delta_dist, 0.0) * 5.0

    # ---- Category 2: Stability (positive reward for being calm near pad) ----
    # Reward increases as the lander becomes more stable: low velocities, small angle
    calm_score = max(0.0, 1.0 - abs(vy) - 0.5 * abs(vx) - 0.5 * abs(ang) - 0.3 * abs(ang_vel))
    near_stage_stability = near_factor * calm_score * 2.0   # max ~2 when perfect

    # ---- Category 2: Leg contact bonus (additional signal when near) ----
    leg_contact_bonus = near_factor * (leg_left + leg_right) * 0.5

    # ---- Category 3: Terminal bonuses / penalties ----
    # Soft landing: moderate requirements, gives bonus for partial success
    soft_landing = (
        leg_left > 0.5 and leg_right > 0.5
        and abs(vy) < 0.5
        and abs(vx) < 0.5
        and abs(ang) < 0.3
        and curr_dist < 0.2
    )
    soft_landing_bonus = 5.0 if terminated and soft_landing else 0.0

    # Strict landing: high precision – harder to achieve
    strict_landing = (
        leg_left > 0.5 and leg_right > 0.5
        and abs(vy) < 0.1
        and abs(vx) < 0.1
        and abs(ang) < 0.1
        and abs(ang_vel) < 0.1
        and curr_dist < 0.15
    )
    strict_landing_bonus = 10.0 if terminated and strict_landing else 0.0

    # Crash penalty: differentiated to avoid heavy exploration suppression
    # Early failures (still far from pad) get a smaller penalty
    far_crash = terminated and not soft_landing and not strict_landing and curr_dist > 1.0
    near_crash = terminated and not soft_landing and not strict_landing and curr_dist <= 1.0
    crash_penalty = -5.0 if far_crash else (-8.0 if near_crash else 0.0)

    # Timeout penalty: truncated without being close to pad
    timeout_penalty = -3.0 if truncated and curr_dist > FAR_THRESHOLD else 0.0

    # ---- No unconditional survival reward ----
    # ---- No official reward leak ----

    components = {
        "approach_progress": approach_progress,
        "near_stage_stability": near_stage_stability,
        "leg_contact_bonus": leg_contact_bonus,
        "soft_landing_bonus": soft_landing_bonus,
        "strict_landing_bonus": strict_landing_bonus,
        "crash_penalty": crash_penalty,
        "timeout_penalty": timeout_penalty,
    }

    total_reward = (
        approach_progress
        + near_stage_stability
        + leg_contact_bonus
        + soft_landing_bonus
        + strict_landing_bonus
        + crash_penalty
        + timeout_penalty
    )
    return float(total_reward), components