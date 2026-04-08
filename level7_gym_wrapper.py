"""
Level 7: Gym Wrapper
Wraps SimpleDeltaHeatEnv into a Gymnasium-compatible environment.
Observation dimension is auto-detected from core env.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GymWrapper(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, core_env):
        super().__init__()
        self.core = core_env

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        obs, _ = self.core.reset()
        obs_dim = len(obs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = self.core.reset(seed=seed)
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.core.step(action)
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass