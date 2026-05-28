"""LLM-generated reward function.
"""

import math
import numpy as np


def compute_reward(state, m_power, s_power, terminated):
    # Extract observation components
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    angvel = state[5]
    left_leg = state[6]
    right_leg = state[7]

    # 1. Proximity to landing pad (Gaussian peak at (0,0))
    dist_sq = x*x + y*y
    proximity = 10.0 * math.exp(-5.0 * dist_sq)

    # 2. Combined penalty for velocity and orientation errors
    speed_err = 0.5 * (vx*vx + vy*vy)
    angle_err = 1.0 * (angle*angle + angvel*angvel)
    error_penalty = -(speed_err + angle_err)

    # 3. Reward for leg contact
    leg_contact = 2.0 * (left_leg + right_leg)

    # 4. Penalty for firing engines (fuel cost)
    fuel_penalty = -2.0 * m_power - 1.0 * abs(s_power)

    # 5. Terminal bonus (large positive for safe landing, large negative for failure)
    # Define safe landing: both legs contact, near pad, upright
    safe = (terminated and
            left_leg == 1.0 and right_leg == 1.0 and
            abs(x) < 0.1 and abs(y) < 0.1 and
            abs(angle) < 0.1)
    if terminated:
        landing_bonus = 100.0 if safe else -100.0
    else:
        landing_bonus = 0.0

    # Sum all components
    total = proximity + error_penalty + leg_contact + fuel_penalty + landing_bonus

    # Build component dictionary with outcome for diagnosis
    components = {
        "proximity": proximity,
        "error_penalty": error_penalty,
        "leg_contact": leg_contact,
        "fuel_penalty": fuel_penalty,
        "landing_bonus": landing_bonus,
        "_outcome": 1.0 if safe else (-1.0 if terminated else 0.0)
    }

    return float(total), components
