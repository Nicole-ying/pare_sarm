# Task Manifest

## Environment

`LunarLander-v2`

## Task Goal

land safely on the landing pad using observable state evidence, without using the official reward

## Reward Leakage Policy

Official environment reward is not visible and must not be used.

## Coarse Outcome Labels

- `early_failure`
- `low_progress_survival`
- `partial_progress`
- `success_like`
- `ambiguous`

## Trajectory Features To Extract

- `episode_length`
- `terminated`
- `truncated`
- `initial_distance_to_landing_pad`
- `final_distance_to_landing_pad`
- `distance_improvement`
- `final_speed_magnitude`
- `final_angle_abs`
- `final_legs_contact_both`
- `final_x_position`
- `final_y_position`
- `max_x_position`
- `min_x_position`
- `max_y_position`
- `min_y_position`
- `landing_pad_proximity_flag`

## Labeling Cautions

- Do not use or infer the official environment reward.
- Do not treat long episode length alone as success_like.
- Do not treat high approach progress with unstable final speed or large angle as success_like.
- Do not treat a crash outside the pad as success_like even if legs touch ground.
- Both legs must be in contact with the ground and the lander must be near the pad (x near 0) for a safe landing; unstable final state (high speed, large angle) should not be classified as success_like.
- The angular velocity observation is scaled by 0.4 internally; raw values should be interpreted with caution.
