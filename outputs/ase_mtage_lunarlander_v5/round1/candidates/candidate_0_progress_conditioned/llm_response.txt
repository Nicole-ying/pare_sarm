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

    # Distance to pad (0,0)
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist  # positive if closer

    # Stage definitions
    FAR_THRESHOLD = 0.35
    NEAR_THRESHOLD = 0.25
    # far_stage is 1 if far from pad, 0 otherwise
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    # near_stage is 1 if close to pad, 0 otherwise
    near_stage = 1.0 if curr_dist <= FAR_THRESHOLD else 0.0

    # ---- Early stage: approach progress ----
    # Only reward positive progress when far from pad; cap progress to avoid excessive reward
    approach_progress = far_stage * max(delta_dist, 0.0) * 5.0

    # ---- Late stage: positive stability reward ----
    # Normalize instability: sum of abs values, bounded 0-1 per component, weighted average
    # Max plausible values: vx 2.0, vy 2.0, ang 0.8, ang_vel 5.0
    # We scale to get a smooth reward between 0 (unstable) and 2 (perfectly stable)
    norm_v = (abs(vx) + abs(vy)) / 4.0  # 0..1
    norm_ang = abs(ang) / 0.8
    norm_ang_vel = abs(ang_vel) / 5.0
    instability = 0.4 * norm_v + 0.4 * norm_ang + 0.2 * norm_ang_vel
    stability_score = max(0.0, 1.0 - instability)  # 0..1

    # Stability reward is active only in near stage, with a small bonus for leg contact
    stability_reward = near_stage * (2.0 * stability_score)
    leg_bonus = near_stage * (leg_left + leg_right) * 0.3

    # ---- Gentle terminal stability bonus ----
    # More lenient than strict safe_landing: require low vertical speed (<0.3), low horizontal speed (<0.3),
    # small angle (<0.2), small angular velocity (<0.5), at least one leg contact, near pad
    gentle_landing = (
        abs(vy) < 0.3
        and abs(vx) < 0.3
        and abs(ang) < 0.2
        and abs(ang_vel) < 0.5
        and (leg_left > 0.5 or leg_right > 0.5)
        and curr_dist < NEAR_THRESHOLD
    )
    gentle_landing_bonus = 8.0 if terminated and gentle_landing else 0.0

    # Strict safe landing (full success) bonus remains but with slightly relaxed angle/velocity thresholds
    strict_safe = (
        leg_left > 0.5 and leg_right > 0.5
        and abs(vy) < 0.1
        and abs(vx) < 0.1
        and abs(ang) < 0.1
        and abs(ang_vel) < 0.1
        and curr_dist < NEAR_THRESHOLD
    )
    strict_safe_bonus = 12.0 if terminated and strict_safe else 0.0

    # Crash penalty: terminated but not gentle landing (so we encourage at least gentle)
    crash_penalty = -8.0 if terminated and not gentle_landing else 0.0

    # Timeout penalty: truncated without being near pad
    timeout_penalty = -5.0 if truncated and curr_dist > FAR_THRESHOLD else 0.0

    # No pure survival reward
    time_alive = 0.0

    components = {
        "approach_progress": approach_progress,
        "stability_reward": stability_reward,
        "leg_bonus": leg_bonus,
        "gentle_landing_bonus": gentle_landing_bonus,
        "strict_safe_bonus": strict_safe_bonus,
        "crash_penalty": crash_penalty,
        "timeout_penalty": timeout_penalty,
    }

    total_reward = (
        approach_progress
        + stability_reward
        + leg_bonus
        + gentle_landing_bonus
        + strict_safe_bonus
        + crash_penalty
        + timeout_penalty
    )
    return float(total_reward), components