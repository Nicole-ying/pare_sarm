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

    # --- Per-step components ---

    # Survival penalty: encourages the agent to finish quickly
    # (negative constant, typical range [-0.5])
    survival_penalty = -0.5

    # Position shaping: linear distance to the pad (normalized coordinates)
    # Provides a constant gradient toward (0,0)
    # Typical range: [-3.0, 0] for each axis, combined up to -6.0
    distance_penalty = -2.0 * abs(x) - 2.0 * abs(y)

    # Speed penalty: quadratic to discourage high velocity
    # speed_sq up to ~25, penalty up to -5.0
    speed_sq = vx * vx + vy * vy
    speed_penalty = -0.2 * speed_sq

    # Angle penalty: absolute values to discourage tilt and spin
    # angle up to π, angvel up to 5, penalty up to ~-4.0
    angle_penalty = -0.5 * abs(angle) - 0.5 * abs(angvel)

    # Fuel penalty: small negative to discourage waste, negligible magnitude
    fuel_penalty = -0.001 * (m_power + abs(s_power))

    # --- Terminal bonus ---
    safe_landing = (
        terminated
        and left_leg >= 0.5
        and right_leg >= 0.5
        and abs(x) < 0.1
        and abs(y) < 0.1
        and abs(angle) < 0.1
        and speed_sq < 0.1
    )
    terminal_bonus = 500.0 if safe_landing else 0.0

    # --- Outcome for diagnosis ---
    if safe_landing:
        outcome = 1.0
    elif terminated:
        outcome = -1.0
    else:
        outcome = 0.0

    # Sum all per-step components (terminal bonus is added separately)
    total = survival_penalty + distance_penalty + speed_penalty + angle_penalty + fuel_penalty + terminal_bonus

    components = {
        "survival_penalty": survival_penalty,
        "distance_penalty": distance_penalty,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "fuel_penalty": fuel_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome
    }

    return float(total), components
