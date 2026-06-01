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
- `terminated_flag`
- `truncated_flag`
- `initial_x_position`
- `initial_y_position`
- `final_x_position`
- `final_y_position`
- `final_x_velocity`
- `final_y_velocity`
- `final_angle_abs`
- `final_angular_velocity_abs`
- `final_left_leg_contact`
- `final_right_leg_contact`
- `distance_to_landing_pad_initial`
- `distance_to_landing_pad_final`
- `distance_improvement`
- `final_speed`
- `minimum_distance_to_landing_pad`
- `proportion_of_steps_with_both_legs_contact`
- `smoothness_of_approach`

## Labeling Cautions

- Do not use or infer the official environment reward.
- Do not treat episode length alone as success_like.
- Do not treat high approach progress with unstable final speed, large tilt, or lack of leg contact as success_like.
- Verify that termination is due to explicit task completion (landing safely) rather than crash or out-of-bounds.
- Use the two leg contact booleans as primary indicators of safe landing (both legs in contact).
- Consider final angle and angular velocity to ensure stability at landing.
- Do not treat a landing with high horizontal velocity or significant tilt as success.
- Label as ambiguous if termination reason is unclear (e.g., time limit reached with partial progress).
