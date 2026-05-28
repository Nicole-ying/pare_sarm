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

    # Progress measure [0,1] – combines position, angle, leg contact
    progress = progress_fn(state)

    # --- Per-step components ---
    # 1. Survival bonus proportional to progress
    progress_reward = 1.0 * progress

    # 2. Leg contact bonus – only meaningful when near ground (y < 0.25)
    leg_near_ground = (left_leg + right_leg) * (1.0 if y < 0.25 else 0.0)
    leg_bonus = 0.5 * leg_near_ground

    # 3. Controlled descent: reward low vertical speed when close to ground
    #    and low angular velocity at all times.
    descent_quality = 0.0
    # Vertical speed shaping: penalize high |vy| when near ground
    if y < 0.5:
        descent_quality -= 0.5 * min(abs(vy), 2.0)  # cap at 2 for scale
    # Angular velocity penalty (always)
    descent_quality -= 0.3 * abs(angvel)
    # Angle penalty (always)
    descent_quality -= 0.3 * abs(angle)
    # Clip to avoid negative domination
    descent_quality = max(descent_quality, -2.0)

    # 4. Fuel penalty – scaled by (1 - progress) so early usage is cheaper
    fuel_usage = m_power + abs(s_power)
    fuel_penalty = -0.2 * fuel_usage * (1.0 - progress)

    # --- Terminal bonus ---
    if terminated:
        safe = (left_leg == 1.0 and right_leg == 1.0 and
                abs(x) < 0.1 and abs(y) < 0.1 and
                abs(angle) < 0.1)
        if safe:
            landing_bonus = 200.0
        else:
            landing_bonus = -50.0
    else:
        landing_bonus = 0.0

    # Sum all components (terminal bonus added after per-step)
    total = (progress_reward + leg_bonus + descent_quality + fuel_penalty
             + landing_bonus)

    # Build component dictionary
    components = {
        "progress_reward": progress_reward,
        "leg_bonus": leg_bonus,
        "descent_quality": descent_quality,
        "fuel_penalty": fuel_penalty,
        "landing_bonus": landing_bonus,
        "_outcome": 1.0 if (terminated and safe) else (
            -1.0 if terminated else 0.0
        )
    }

    return float(total), components
