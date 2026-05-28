"""Proxy reward candidate."""

import math
import numpy as np

def compute_reward(state, m_power, s_power, terminated):
    # Extract state components
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    angvel = state[5]
    left_leg = state[6]
    right_leg = state[7]

    # --- Per-step penalties (negative) and bonuses ---

    # 1. Distance penalty: linear in distance from pad (normalized coordinates)
    dist = math.sqrt(x*x + y*y)
    distance_penalty = -1.0 * dist  # range [-2.12, 0]

    # 2. Vertical speed penalty: penalize high downward speed (vy negative is down)
    # Normalized vy range ~[-5,5], penalty proportional to absolute value
    vertical_speed_penalty = -0.1 * abs(vy)  # range [-0.5, 0]

    # 3. Angle penalty: deviation from upright (radians, normalized)
    angle_penalty = -0.1 * abs(angle)  # range [-0.314, 0] approx

    # 4. Angular velocity penalty: discourage spinning
    angular_velocity_penalty = -0.01 * abs(angvel)  # range [-0.05, 0]

    # 5. Leg contact bonus: small reward when both legs are on ground (near landing)
    leg_bonus = 0.1 if (left_leg == 1.0 and right_leg == 1.0) else 0.0

    # 6. Survival penalty: negative constant to encourage finishing quickly
    survival_penalty = -0.1

    # 7. Fuel penalty: discourage wasteful engine use
    fuel_penalty = -0.005 * (m_power + abs(s_power))  # typical range [-0.01, 0]

    # --- Terminal bonus ---
    safe_landing = (
        terminated
        and left_leg == 1.0
        and right_leg == 1.0
        and abs(x) < 0.1
        and abs(y) < 0.1
        and abs(angle) < 0.1
        and (vx*vx + vy*vy) < 0.1
    )
    if safe_landing:
        terminal_bonus = 500.0
        outcome = 1.0
    elif terminated:
        terminal_bonus = -100.0   # penalty for crash/out-of-bounds/sleep
        outcome = -1.0
    else:
        terminal_bonus = 0.0
        outcome = 0.0

    # --- Sum per-step components ---
    per_step = (
        distance_penalty
        + vertical_speed_penalty
        + angle_penalty
        + angular_velocity_penalty
        + leg_bonus
        + survival_penalty
        + fuel_penalty
    )

    total = per_step + terminal_bonus

    components = {
        "distance_penalty": distance_penalty,
        "vertical_speed_penalty": vertical_speed_penalty,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angular_velocity_penalty,
        "leg_bonus": leg_bonus,
        "survival_penalty": survival_penalty,
        "fuel_penalty": fuel_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome
    }

    return float(total), components
