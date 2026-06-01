"""LLM-generated reward function.
"""

import math
import numpy as np


def compute_reward(state, m_power, s_power, terminated):
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    angvel = state[5]
    left_leg = state[6]
    right_leg = state[7]

    # Composite state cost: distance to origin, velocity, and orientation
    dist_sq = x * x + y * y
    v_sq = vx * vx + vy * vy
    angle_sq = angle * angle + angvel * angvel
    # Exponentiated reward: high when all are small
    descent_reward = math.exp(-(dist_sq + v_sq + angle_sq))

    # Leg contact bonus (encourages landing on legs)
    leg_bonus = 0.5 * (left_leg + right_leg)

    # Small fuel penalty (to discourage waste but not prevent necessary use)
    fuel_penalty = -0.01 * (m_power + abs(s_power))

    # Small survival penalty to avoid indefinite hovering
    survival_penalty = -0.05

    # Terminal bonus: large positive for a safe landing
    safe_landing = (
        terminated and
        left_leg == 1.0 and right_leg == 1.0 and
        abs(x) < 0.1 and abs(y) < 0.1 and
        abs(angle) < 0.1 and
        v_sq < 0.1
    )
    terminal_bonus = 500.0 if safe_landing else 0.0

    # Sum all components (terminal added after per-step)
    total = descent_reward + leg_bonus + fuel_penalty + survival_penalty + terminal_bonus

    # Outcome for diagnosis: 1.0 = safe landing, -1.0 = termination without landing, 0.0 = ongoing
    if safe_landing:
        outcome = 1.0
    elif terminated:
        outcome = -1.0
    else:
        outcome = 0.0

    components = {
        "descent_reward": descent_reward,
        "leg_bonus": leg_bonus,
        "fuel_penalty": fuel_penalty,
        "survival_penalty": survival_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome
    }

    return float(total), components
