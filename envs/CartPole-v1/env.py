"""CartPole-v1 environment with LLM-replaceable compute_reward.

Wraps the standard gymnasium CartPole-v1 environment.
The compute_reward method is a stub — it gets replaced at runtime
by the LLM-generated reward function via inject_and_register().

Observation space (4 dims):
    0: Cart position (-4.8 to 4.8)
    1: Cart velocity (-inf to inf)
    2: Pole angle (-24 deg to 24 deg, in radians)
    3: Pole angular velocity (-inf to inf)

Action space (2 actions):
    0: Push cart left
    1: Push cart right

Termination:
    - Pole angle exceeds ±12 degrees
    - Cart position exceeds ±2.4
    - Episode length > 500 (truncation, not termination)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CartPoleEnv(gym.Env):
    """CartPole with replaceable compute_reward for LLM reward design."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # These constants are exposed to the reward function scope
    MAX_POS = 2.4
    MAX_ANGLE = 12 * math.pi / 180  # radians
    MAX_STEPS = 500

    def __init__(self, render_mode=None):
        super().__init__()
        self._env = gym.make("CartPole-v1", render_mode=render_mode)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._pre_step_state = {}
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self._env.step(action)

        # Capture pre-step state for reward function
        self._pre_step_state = {
            "_cart_pos": float(obs[0]),
            "_pole_angle": float(obs[2]),
        }

        # Build state vector for compute_reward
        state = np.array(obs, dtype=np.float32)

        # Call compute_reward (will be replaced by LLM-generated code)
        x, x_dot, theta, theta_dot = state
        reward, components = self.compute_reward(state, action, terminated)

        return state, reward, terminated, truncated, {
            "reward_components": components,
            "env_reward": float(env_reward),
        }

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @staticmethod
    def compute_reward(state, action, terminated):
        """Stub — replaced at runtime by LLM-generated reward function.

        Args:
            state: np.array of [cart_pos, cart_vel, pole_angle, pole_ang_vel]
            action: int (0=left, 1=right)
            terminated: bool (True if pole fell or cart out of bounds)

        Returns:
            total_reward: float
            components: dict[str, float]
        """
        return 0.0, {"_outcome": 0.0}
