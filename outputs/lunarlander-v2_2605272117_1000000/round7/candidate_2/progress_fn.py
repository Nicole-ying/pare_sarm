def progress_fn(obs):
    """
    Measures task progress from observation alone.
    Returns a float in [0, 1] that increases monotonically as the lander
    approaches a safe landing on the pad.
    """
    # Normalized position (ideal is (0,0) when on pad)
    x = obs[0]
    y = obs[1]
    # Angle (ideal is 0 radians, upright)
    angle = abs(obs[4])
    # Leg contacts (both should be 1 when landed)
    leg_contact = (obs[6] + obs[7]) / 2.0

    # Distance in the normalized coordinate system (max possible ~2.12)
    dist = (x**2 + y**2) ** 0.5
    max_dist = 2.12  # sqrt(2 * 1.5^2)

    # Progress based on position: 1 when at origin, 0 at max distance
    pos_progress = 1.0 - dist / max_dist

    # Penalty for large angle (scaled so 180° gives ~0.3 penalty)
    angle_penalty = angle / (2 * 3.14159)  # max penalty ~0.5
    angle_progress = 1.0 - angle_penalty

    # Combine: position (60%), angle (20%), leg contact (20%)
    progress = 0.6 * pos_progress + 0.2 * angle_progress + 0.2 * leg_contact

    # Clip to [0, 1]
    return max(0.0, min(1.0, progress))
