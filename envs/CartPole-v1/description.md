# CartPole-v1

## Task Objective
Balance a pole on a cart by moving the cart left or right. The pole starts upright and the agent must prevent it from falling.

## Success Criteria
Keep the pole angle within ±12 degrees and cart position within ±2.4 for as long as possible, up to 500 steps.

## Failure Conditions
- Pole angle exceeds ±12 degrees (terminated)
- Cart position exceeds ±2.4 (terminated)

## Observation Space (4 dims)
| Dim | Name | Description |
|-----|------|-------------|
| 0 | cart_pos | Cart position (-2.4 to 2.4) |
| 1 | cart_vel | Cart velocity (unbounded) |
| 2 | pole_angle | Pole angle in radians (-0.209 to 0.209) |
| 3 | pole_ang_vel | Pole angular velocity (unbounded) |

## Action Space (Discrete 2)
- 0: Push cart left
- 1: Push cart right

## Key Dynamics
- Pole falls due to gravity — the agent must counteract
- Cart has momentum — can't stop instantly
- Under random policy: mean episode length ~27 steps
- Zero-action baseline: pole falls almost immediately
