"""Proxy reward candidate."""

import math
import numpy as np


def compute_reward(state, m_power, s_power, terminated):
    # Unpack state
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    angvel = state[5]
    left_leg = state[6]
    right_leg = state[7]

    # --- Per-step components ---

    # 1. Alive bonus: encourages survival (counteracts the previous survival penalty)
    alive_bonus = 0.05

    # 2. Quadratic state cost: penalizes deviation from ideal landing state
    #    (position, velocity, orientation). This is like an LQR-style shaping.
    #    The agent learns to minimize this negative reward, guiding it toward the pad.
    state_cost = (
        x * x +                # horizontal position
        y * y +                # vertical position
        0.5 * (vx * vx + vy * vy) +  # velocity (scaled slightly less to avoid over-penalizing needed speed)
        angle * angle +        # angle
        0.2 * angvel * angvel  # angular velocity (less weight)
    )
    state_penalty = -0.2 * state_cost  # magnitude ~ -1 to -10 in normal operation

    # 3. Minimal fuel usage cost (almost negligible – only to discourage extreme waste)
    fuel_cost = -0.001 * (m_power + abs(s_power))

    # Terminal conditions
    safe_landing = (
        terminated and
        left_leg == 1.0 and right_leg == 1.0 and
        abs(x) < 0.1 and abs(y) < 0.1 and
        abs(angle) < 0.1 and
        (vx * vx + vy * vy) < 0.1
    )
    terminal_bonus = 500.0 if safe_landing else 0.0

    # Total per-step (terminal bonus added after, as required)
    total = alive_bonus + state_penalty + fuel_cost + terminal_bonus

    # Outcome
    if safe_landing:
        outcome = 1.0
    elif terminated:
        outcome = -1.0
    else:
        outcome = 0.0

    components = {
        "alive_bonus": alive_bonus,
        "state_penalty": state_penalty,
        "fuel_cost": fuel_cost,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome
    }

    return float(total), components
