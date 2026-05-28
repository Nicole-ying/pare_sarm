"""Proxy reward candidate."""

import math
import numpy as np

def compute_reward(state, m_power, s_power, terminated):
    # Extract normalized observation components
    x, y, vx, vy, angle, angvel, leg_left, leg_right = state

    # 1. Distance reward: encourage being near the pad (0,0)
    dist = math.sqrt(x**2 + y**2)
    dist_reward = -0.5 * dist**2   # max penalty ~ -2.25

    # 2. Velocity penalty: discourage high speed (especially downward)
    vel = math.sqrt(vx**2 + vy**2)
    vel_penalty = -0.2 * vel**2    # max penalty ~ -5.0 for max speed 5

    # 3. Angle penalty: encourage upright orientation
    angle_penalty = -0.5 * angle**2   # max ~ -4.93 for π rad

    # 4. Leg contact reward: strong signal for both legs on ground
    leg_contact_reward = 0.5 * (leg_left + leg_right)   # max 1.0

    # 5. Engine usage penalty: small cost to discourage wasteful firing
    # m_power ∈ [0,1] for main engine, s_power ∈ [-1,1] for side
    eng_penalty = -0.02 * m_power - 0.01 * abs(s_power)

    # Total per-step reward (before terminal outcome)
    total = dist_reward + vel_penalty + angle_penalty + leg_contact_reward + eng_penalty

    # Construct component dict
    components = {
        "distance_reward": dist_reward,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "leg_contact_reward": leg_contact_reward,
        "engine_penalty": eng_penalty
    }

    # Terminal outcome (not added to total)
    if terminated:
        # Heuristic for safe landing: both legs contact, near vertical, near pad, low vertical speed
        safe_landing = (leg_left == 1.0 and leg_right == 1.0 and
                        abs(angle) < 0.1 and
                        abs(vy) < 0.1 and
                        abs(x) < 0.1)
        if safe_landing:
            components["_outcome"] = 1.0
        else:
            components["_outcome"] = -1.0
    else:
        components["_outcome"] = 0.0

    return float(total), components
