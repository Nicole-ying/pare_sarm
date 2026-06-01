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
- `initial_x_position`
- `initial_y_position`
- `final_x_position`
- `final_y_position`
- `final_x_velocity`
- `final_y_velocity`
- `final_angle_abs`
- `final_angular_velocity_abs`
- `final_leg_contacts_sum (0,1,2)`
- `distance_to_pad_initial`
- `distance_to_pad_final`
- `distance_improvement (initial - final)`
- `proportion_of_steps_with_leg_contact_in_last_10%`
- `proportion_of_steps_with_low_vertical_speed_in_last_20%`
- `energy_used_approximation (steps with engine fire weighted)`

## Labeling Cautions

- Do not use or infer the official environment reward.
- Do not treat long episode length alone as success_like.
- Do not treat high approach progress with unstable final speed as success_like.
- Do not treat final leg contact as sufficient for safe landing; the lander must have low speed and be stable (near zero velocity and angular velocity).
- Safe landing requires both legs in contact, low vertical speed (<0.5?), low horizontal speed, small angle and angular velocity, and lander not going out of viewport.
- A success_like outcome should be based on observable final state that matches a safe landing: both legs contact, near-zero velocities, small angle, and within the landing pad area.
