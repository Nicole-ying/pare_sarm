"""LLM-generated reward function.
"""

import math
import numpy as np


def compute_reward(state, m_power, s_power, terminated):
    # Unpack observation
    x = state[0]
    angle = state[4]
    vx = state[2]
    vy = state[3]
    left_leg = state[6]
    right_leg = state[7]

    # ----- Per-step shaping components -----
    # 1. Proximity: how close to the pad in x and angle
    x_progress = 1.0 - min(1.0, abs(x) / 1.5)
    angle_progress = 1.0 - min(1.0, abs(angle) / math.pi)
    proximity = 0.5 * x_progress + 0.5 * angle_progress

    # 2. Speed penalty: discourage high horizontal/vertical velocity
    total_speed = abs(vx) + abs(vy)
    speed_penalty = -0.5 * min(5.0, total_speed)

    # 3. Leg contact bonus: reward stable ground contact
    leg_bonus = 2.0 * (left_leg + right_leg)

    # 4. Fuel penalty: small cost for firing engines
    fuel_penalty = -0.2 * (m_power + abs(s_power))

    # 5. Terminal bonus: large reward/penalty at episode end
    # Determine success or failure based on termination condition
    if terminated:
        success = (
            left_leg == 1.0 and right_leg == 1.0 and
            abs(x) < 0.3 and abs(angle) < 0.2 and
            abs(vx) < 0.5 and abs(vy) < 0.5
        )
        outcome_val = 1.0 if success else -1.0
        terminal_bonus = 10.0 if success else -10.0
    else:
        outcome_val = 0.0
        terminal_bonus = 0.0

    # Total per-step reward
    total = proximity + speed_penalty + leg_bonus + fuel_penalty + terminal_bonus

    # Component dictionary
    components = {
        "proximity": proximity,
        "speed_penalty": speed_penalty,
        "leg_bonus": leg_bonus,
        "fuel_penalty": fuel_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome_val
    }

    return float(total), components
