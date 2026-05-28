"""Proxy reward candidate."""

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

    # Dense reward for being in a desirable state (position, speed, orientation)
    dist_sq = x * x + y * y
    speed_sq = vx * vx + vy * vy
    angle_sq = angle * angle + angvel * angvel
    desired_state_reward = 5.0 * math.exp(-(dist_sq + speed_sq + angle_sq))

    # Small positive alive bonus to encourage survival
    alive_bonus = 0.1

    # Conditional leg bonus: only when both legs contact and the lander is near the pad
    leg_bonus = 0.0
    if left_leg == 1.0 and right_leg == 1.0:
        if abs(y) < 0.2 and speed_sq < 0.5:
            leg_bonus = 0.5

    # Minimal fuel penalty to avoid waste but not discourage necessary thrust
    fuel_penalty = -0.001 * (m_power + abs(s_power))

    # Terminal bonus for a safe landing (large sparse reward)
    safe_landing = (
        terminated
        and left_leg == 1.0
        and right_leg == 1.0
        and abs(x) < 0.1
        and abs(y) < 0.1
        and abs(angle) < 0.1
        and speed_sq < 0.1
    )
    terminal_bonus = 500.0 if safe_landing else 0.0

    # Sum all per-step components (terminal bonus added after per-step)
    total = desired_state_reward + alive_bonus + leg_bonus + fuel_penalty + terminal_bonus

    # Outcome for diagnosis: 1.0 = safe landing, -1.0 = termination without landing, 0.0 = ongoing
    if safe_landing:
        outcome = 1.0
    elif terminated:
        outcome = -1.0
    else:
        outcome = 0.0

    components = {
        "desired_state_reward": desired_state_reward,
        "alive_bonus": alive_bonus,
        "leg_bonus": leg_bonus,
        "fuel_penalty": fuel_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome
    }

    return float(total), components
