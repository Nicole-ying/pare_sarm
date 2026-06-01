### Task Objective
The agent must control a lunar lander to safely land on a designated landing pad at coordinates (0,0) in the environment’s internal coordinate system. The lander starts at the top center of the viewport with a random initial impulse.

### Success Criteria
- The episode is considered a solution if the cumulative reward is **at least 200 points**.
- A safe landing occurs when the lander touches down on the pad with both legs in contact with the ground and the lander comes to rest (awake flag becomes False).

### Failure Conditions
The episode terminates immediately if any of the following occur:
1. **Crash**: Any part of the lander body touches the ground (detected via contact listener).
2. **Out of bounds**: The lander’s normalized x-coordinate (`state[0]`) exceeds ±1.0 (i.e., leaves the viewport horizontally).
3. **Not awake**: The lander body enters a sleep state (Box2D awake flag becomes False). This happens when the lander comes to rest, either on the pad or off it.

### Observation Space
The observation is an 8‑dimensional vector. Dimensions and their raw meanings:

| Index | Raw Quantity | Normalization | Range (low..high) | Unit |
|-------|--------------|---------------|-------------------|------|
| 0 | Horizontal position (x) | `(pos.x - center) / (half_viewport_width)` | -1.5 … 1.5 | – |
| 1 | Vertical position (y) | `(pos.y - (helipad_y+LEG_DOWN/SCALE)) / (half_viewport_height)` | -1.5 … 1.5 | – |
| 2 | Horizontal velocity (vx) | `vel.x * (half_viewport_width/SCALE) / FPS` | -5.0 … 5.0 | – |
| 3 | Vertical velocity (vy) | `vel.y * (half_viewport_height/SCALE) / FPS` | -5.0 … 5.0 | – |
| 4 | Angle | `self.lander.angle` (radians) | -π … π | radians |
| 5 | Angular velocity | `20.0 * self.lander.angularVelocity / FPS` | -5.0 … 5.0 | – |
| 6 | Left leg ground contact | 1.0 if contact, else 0.0 | 0 … 1 | bool |
| 7 | Right leg ground contact | 1.0 if contact, else 0.0 | 0 … 1 | bool |

*Note:* The raw physical values can be recovered by applying the inverse scale factors given in the environment notes (x:10, y:6.666, vx:5, vy:7.5, angular velocity:2.5).

### Action Space
Two variants are available:
- **Discrete** (`Discrete(4)`):
  - 0: Do nothing
  - 1: Fire left orientation engine
  - 2: Fire main engine
  - 3: Fire right orientation engine
- **Continuous** (`Box(-1, +1, (2,), dtype=np.float32)`):
  - `action[0]` = main engine throttle: `<0` off, `0…1` maps to 50%–100% power.
  - `action[1]` = lateral boosters: `<-0.5` fires left, `>0.5` fires right, throttle scales 50%–100%; values in `[-0.5, 0.5]` are off.

### Key Dynamics
- **Gravity**: Default -10.0, can be set between -12 and 0.
- **Wind & turbulence** (optional): Simulated via `tanh(sin(…)+sin(…))`. Wind applies a horizontal force; turbulence applies torque.
- **Engine impulses**: Applied at an offset from the center of mass, creating torque. The side engine offset changes with lander orientation (a known “bug”).
- **Legs**: Two spring‑loaded legs. Ground contact gives +10 reward per step; losing contact gives -10.
- **Termination on sleep**: The lander must come to rest (awake=False) to end the episode without crashing. This means the lander must settle, not just touch the ground briefly.

### Reward Design Hints
The environment’s built‑in reward (which is overridden by `compute_reward`) suggests:
- Reward should increase as the lander gets closer to the pad (distance penalty).
- Reward should increase as the lander slows down (velocity penalty).
- Reward should increase when the lander is upright (angle penalty).
- Strong positive signal for each leg making ground contact (+10 per leg).
- Small negative penalty for firing side engines (-0.03 per frame) and main engine (-0.3 per frame).
- Large terminal reward: +100 for safe landing, -100 for crash.
A good reward function combines these components, scaled appropriately. The episode is considered solved if the cumulative reward reaches 200.

---