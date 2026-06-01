def step(self, action):
    obs, env_reward, terminated, truncated, info = self._env.step(action)

    # Capture pre-step state variables for cross-step metrics
    self._pre_step_state = {
        "_cart_pos": float(obs[0]),
        "_pole_angle": float(obs[2]),
    }

    state = np.array(obs, dtype=np.float32)

    # ============================================================
    # LLM generates this function
    x, x_dot, theta, theta_dot = state
    reward, components = self.compute_reward(state, action, terminated)
    # ============================================================

    return state, reward, terminated, truncated, {
        "reward_components": components,
        "env_reward": float(env_reward),
    }
