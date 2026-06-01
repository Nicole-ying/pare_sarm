"""Proxy reward candidate."""

import math
import numpy as np


def compute_reward(state, m_power, s_power, terminated):
    # Unpack observation (all normalized)
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = abs(state[4])          # radians
    left_leg = state[6]
    right_leg = state[7]

    # Task progress [0,1]
    progress = progress_fn(state)

    # ----- Per-step components -----
    # Small survival bonus to keep episodes alive but weak
    alive_bonus = 0.2

    # Progress reward: high when far from pad, drops to near zero when near
    # This encourages the agent to move toward the pad quickly
    progress_reward = 2.0 * (1.0 - progress)

    # Leg contact bonus (only when both legs touch the ground)
    leg_bonus = 0.5 if (left_leg == 1.0 and right_leg == 1.0) else 0.0

    # Speed penalty: discourage fast horizontal and vertical velocities
    # vx, vy are normalized; typical max magnitude is ~5.0
    speed_penalty = -0.02 * (vx * vx + vy * vy)

    # Angle penalty: discourage tilting (radians, max ~3.14)
    angle_penalty = -0.1 * (angle * angle)

    # Constant per-step time penalty to gently push for faster completions
    time_penalty = -0.05

    # Hovering penalty: applied when the agent is near the pad (progress > 0.8)
    # but not yet in contact with both legs. This discourages staying aloft near success.
    hovering_penalty = 0.0
    if progress > 0.8 and (left_leg == 0.0 or right_leg == 0.0):
        hovering_penalty = -0.5

    # Sum per-step components
    per_step = (alive_bonus + progress_reward + leg_bonus +
                speed_penalty + angle_penalty + time_penalty +
                hovering_penalty)

    # ----- Terminal bonus -----
    if terminated:
        # Success: both legs on pad, close to (0,0), upright, low speed
        success = (
            left_leg == 1.0 and right_leg == 1.0 and
            abs(x) < 0.3 and angle < 0.2 and
            abs(vx) < 0.5 and abs(vy) < 0.5
        )
        terminal_bonus = 200.0 if success else -100.0
        outcome_val = 1.0 if success else -1.0
    else:
        terminal_bonus = 0.0
        outcome_val = 0.0

    total = per_step + terminal_bonus

    components = {
        "alive_bonus": alive_bonus,
        "progress_reward": progress_reward,
        "leg_bonus": leg_bonus,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "hovering_penalty": hovering_penalty,
        "terminal_bonus": terminal_bonus,
        "_outcome": outcome_val,
    }

    return float(total), components
