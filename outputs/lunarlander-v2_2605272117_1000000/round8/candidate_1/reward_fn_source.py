"""Proxy reward candidate."""

import math
import numpy as np

def compute_reward(state, m_power, s_power, terminated):
    # Unpack observation
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = abs(state[4])
    left_leg = state[6]
    right_leg = state[7]

    # Compute task progress
    progress = progress_fn(state)  # in [0, 1]

    # ---- Per-step components ----
    # 1. Alive bonus: decreases as progress increases to encourage initial survival
    alive_bonus = 0.5 * (1.0 - progress)

    # 2. Progress reward: linear in progress, relatively small to avoid hovering
    progress_reward = 0.2 * progress

    # 3. Speed penalty: scaled by progress, so mild early, stronger near landing
    combined_speed = abs(vx) + abs(vy)
    speed_penalty = -0.1 * progress * min(combined_speed, 5.0)

    # 4. Angle penalty: scaled by progress, mild early, stronger near landing
    angle_penalty = -0.05 * progress * angle

    # 5. Fuel penalty: small, to discourage excessive thrust
    fuel_penalty = -0.01 * (abs(m_power) + abs(s_power))

    # 6. Time penalty: small constant to encourage finishing the episode
    time_penalty = -0.05

    per_step = alive_bonus + progress_reward + speed_penalty + angle_penalty + fuel_penalty + time_penalty

    # ---- Terminal bonus ----
    if terminated:
        # Success: both legs on ground, near pad, low speed, upright
        success = (
            left_leg == 1.0 and right_leg == 1.0 and
            abs(x) < 0.2 and
            abs(vx) < 0.3 and abs(vy) < 0.3 and
            angle < 0.1
        )
        terminal_bonus = 200.0 if success else 0.0
        outcome_val = 1.0 if success else -1.0
    else:
        terminal_bonus = 0.0
        outcome_val = 0.0

    total = per_step + terminal_bonus

    # ---- Component dictionary ----
    components = {
        "alive_bonus": alive_bonus,
        "progress_reward": progress_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "fuel_penalty": fuel_penalty,
        "time_penalty": time_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome_val,
    }

    return float(total), components
