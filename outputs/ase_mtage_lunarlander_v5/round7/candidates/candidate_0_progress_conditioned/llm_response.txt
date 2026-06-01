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
    # Unpack observations
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

    # Distance to pad (assumed at (0,0))
    prev_dist = math.sqrt(x0 * x0 + y0 * y0)
    curr_dist = math.sqrt(x1 * x1 + y1 * y1)
    delta_dist = prev_dist - curr_dist  # positive = moving closer

    # Stage gates
    FAR_THRESHOLD = 0.35
    far_stage = 1.0 if curr_dist > FAR_THRESHOLD else 0.0
    near_stage = 1.0 - far_stage

    # --- Far-stage approach progress ---
    approach_progress = far_stage * max(delta_dist, 0.0) * 5.0

    # --- Near-stage stability (continuous) ---
    stability_bonus = near_stage * math.exp(-(abs(vx) + abs(vy) + abs(ang) + abs(ang_vel))) * 2.0

    # --- Per-step landing bonus (progress-gated: only when near, both legs, low speeds) ---
    landing_bonus = 0.0
    if near_stage > 0.5 and leg_left > 0.5 and leg_right > 0.5 and abs(vy) < 0.5 and abs(vx) < 0.2:
        landing_bonus = 0.5

    # --- Truncated but close & stable (progressive intermediate bonus) ---
    truncated_progress_bonus = 0.0
    if truncated and near_stage > 0.5 and (leg_left > 0.5 or leg_right > 0.5) and abs(vy) < 0.5 and abs(vx) < 0.2:
        truncated_progress_bonus = 2.0

    # --- Terminal safe landing bonus (progressive thresholds) ---
    terminal_landing_bonus = 0.0
    if terminated:
        # Relaxed safe landing condition
        if (leg_left > 0.5 and leg_right > 0.5 and
            abs(vy) < 0.5 and abs(vx) < 0.2 and
            abs(ang) < 0.2 and abs(ang_vel) < 0.2 and
            curr_dist < 0.3):
            terminal_landing_bonus = 5.0
            # Stricter condition for full bonus
            if (abs(vy) < 0.1 and abs(vx) < 0.1 and
                abs(ang) < 0.1 and abs(ang_vel) < 0.1 and
                curr_dist < 0.15):
                terminal_landing_bonus = 10.0

    # --- Gated timeout penalty (only for negative progress while far) ---
    timeout_penalty = 0.0
    if truncated and delta_dist < 0.0 and curr_dist > 0.5:
        timeout_penalty = -1.0

    # --- Total reward ---
    total_reward = (
        approach_progress +
        stability_bonus +
        landing_bonus +
        truncated_progress_bonus +
        terminal_landing_bonus +
        timeout_penalty
    )

    components = {
        "approach_progress": approach_progress,
        "stability_bonus": stability_bonus,
        "landing_bonus": landing_bonus,
        "truncated_progress_bonus": truncated_progress_bonus,
        "terminal_landing_bonus": terminal_landing_bonus,
        "timeout_penalty": timeout_penalty,
    }

    return float(total_reward), components