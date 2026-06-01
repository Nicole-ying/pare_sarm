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
- `initial_distance_to_pad`
- `final_distance_to_pad`
- `distance_improvement`
- `final_vertical_velocity`
- `final_horizontal_velocity`
- `final_angle_abs`
- `final_angular_velocity_abs`
- `final_left_leg_contact`
- `final_right_leg_contact`
- `both_legs_contact_final`
- `final_y_position`

## Labeling Cautions

- Do not use or infer the official environment reward.
- Do not treat long episode length alone as success_like.
- Do not treat stable hover at low altitude without both legs contacting ground as success_like.
- Do not treat low velocity with high angle or off-pad location as success_like.
- Success_like requires strong evidence: both legs in contact, lander near upright (angle close to 0), near zero velocity, and position near landing pad (x near 0, y near ground within viewport).
- If termination occurred by landing on pad but with significant velocity or angle, classify as partial_progress or ambiguous.
