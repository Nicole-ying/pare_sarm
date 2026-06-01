"""Proxy reward candidate."""

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

    # 1. Proximity to landing pad - reduced from 10 to 3
    dist_sq = x * x + y * y
    proximity = 3.0 * math.exp(-5.0 * dist_sq)

    # 2. Velocity and orientation penalty - scaled down significantly
    speed_err = 0.5 * (vx * vx + vy * vy)
    angle_err = 1.0 * (angle * angle + angvel * angvel)
    error_penalty = -0.02 * (speed_err + angle_err)

    # 3. Reward for leg contact (kept)
    leg_contact = 2.0 * (left_leg + right_leg)

    # 4. Fuel penalty - reduced from -2*m_power -1*abs(s_power)
    fuel_penalty = -0.2 * m_power - 0.1 * abs(s_power)

    # 5. Terminal bonus - large positive for success, small negative for failure
    safe = (terminated and
            left_leg == 1.0 and right_leg == 1.0 and
            abs(x) < 0.1 and abs(y) < 0.1 and
            abs(angle) < 0.1)
    if terminated:
        landing_bonus = 100.0 if safe else -10.0
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
