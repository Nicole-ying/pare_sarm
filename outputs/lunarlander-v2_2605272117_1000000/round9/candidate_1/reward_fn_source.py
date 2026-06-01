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

    # Task progress (0..1), includes position, angle, leg contact
    progress = progress_fn(state)

    # --- Per-step components ---
    # Progress reward – strong differential signal for moving towards pad
    progress_reward = 3.0 * progress  # [0, 3.0]

    # Leg contact bonus – encourages stable contact with both legs
    leg_bonus = 0.5 if (left_leg == 1.0 and right_leg == 1.0) else 0.0

    # Speed penalty – only applied when vertical speed is dangerously high
    # or horizontal speed is extreme. Thresholds chosen to avoid normal operation.
    speed_penalty = 0.0
    if vy < -2.0:  # fast downward descent
        speed_penalty -= 0.5 * (vy + 2.0) ** 2
    if abs(vx) > 2.0:  # extreme horizontal velocity
        speed_penalty -= 0.5 * (abs(vx) - 2.0) ** 2

    per_step = progress_reward + leg_bonus + speed_penalty

    # --- Terminal bonus ---
    if terminated:
        # Success: both legs on pad, upright, low speed, near origin
        success = (
            left_leg == 1.0 and right_leg == 1.0 and
            abs(x) < 0.3 and angle < 0.2 and
            abs(vx) < 0.5 and abs(vy) < 0.5
        )
        terminal_bonus = 200.0 if success else -100.0
        outcome_val = 1.0 if success else 0.0
    else:
        terminal_bonus = 0.0
        outcome_val = 0.0

    total = per_step + terminal_bonus

    components = {
        "progress_reward": progress_reward,
        "leg_bonus": leg_bonus,
        "speed_penalty": speed_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome_val,
    }

    return float(total), components
