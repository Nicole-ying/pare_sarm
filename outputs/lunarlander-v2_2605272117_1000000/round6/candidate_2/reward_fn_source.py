"""Proxy reward candidate."""

import math
import numpy as np


def compute_reward(state, m_power, s_power, terminated):
    # Unpack observation
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    left_leg = state[6]
    right_leg = state[7]

    # Get task progress from available helper
    progress = progress_fn(state)  # in [0, 1]

    # ----- Per-step components (all positive or balanced) -----
    # 1. Base survival bonus (small positive to keep per-step positive early)
    alive_bonus = 0.3

    # 2. Progress reward: increases as lander approaches pad
    progress_reward = 0.5 * progress

    # 3. Leg contact bonus (small, prevents exploitation)
    leg_bonus = 0.1 * (left_leg + right_leg)

    # 4. Speed penalty: only significant when near pad (progress > 0.7)
    total_speed = abs(vx) + abs(vy)
    speed_penalty = -0.2 * total_speed * max(0.0, progress - 0.7) / 0.3  # scale to max -0.2

    # 5. Angle penalty: encourage upright posture near pad
    angle_penalty = -0.1 * abs(angle) * max(0.0, progress - 0.7) / 0.3

    # 6. Hovering penalty: discourage staying near pad without landing
    #   Activates when progress > 0.9 and at least one leg not touching
    hover_penalty = 0.0
    if progress > 0.9 and (left_leg < 0.5 or right_leg < 0.5):
        # Strong negative to make hovering unprofitable
        hover_penalty = -1.5 * (progress - 0.9)

    per_step = alive_bonus + progress_reward + leg_bonus + speed_penalty + angle_penalty + hover_penalty

    # ----- Terminal bonus (moderate, to reduce dominance) -----
    if terminated:
        # Success: both legs on ground, near pad, upright, nearly zero velocity
        success = (
            left_leg == 1.0 and right_leg == 1.0 and
            abs(x) < 0.3 and abs(angle) < 0.2 and
            abs(vx) < 0.5 and abs(vy) < 0.5
        )
        outcome_val = 1.0 if success else -1.0
        terminal_bonus = 50.0 if success else -50.0
    else:
        outcome_val = 0.0
        terminal_bonus = 0.0

    total = per_step + terminal_bonus

    components = {
        "alive_bonus": alive_bonus,
        "progress_reward": progress_reward,
        "leg_bonus": leg_bonus,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "hover_penalty": hover_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome_val
    }

    return float(total), components
