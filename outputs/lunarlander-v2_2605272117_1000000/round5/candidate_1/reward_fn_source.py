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

    # Compute task progress (defined externally, returns 0..1)
    progress = progress_fn(state)

    # ----- Per-step components -----
    # Small constant penalty to encourage fast completion
    time_penalty = -0.2

    # Small positive reward for making progress (never enough to outweigh time_penalty)
    progress_bonus = 0.1 * progress

    # Light fuel penalty to discourage wasteful engine use
    fuel_penalty = -0.02 * (m_power + abs(s_power))

    # ----- Terminal bonus (only applied when episode ends) -----
    if terminated:
        # Success: both legs on ground, low velocities, near pad, upright
        success = (left_leg == 1.0 and right_leg == 1.0 and
                   abs(vx) < 0.5 and abs(vy) < 0.5 and
                   abs(x) < 0.3 and abs(angle) < 0.2)
        outcome_val = 1.0 if success else -1.0
        terminal_bonus = 250.0 if success else -250.0
    else:
        outcome_val = 0.0
        terminal_bonus = 0.0

    # Total per-step reward (including terminal bonus if applicable)
    total = time_penalty + progress_bonus + fuel_penalty + terminal_bonus

    # Component dictionary
    components = {
        "time_penalty": time_penalty,
        "progress_bonus": progress_bonus,
        "fuel_penalty": fuel_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome_val
    }

    return float(total), components
