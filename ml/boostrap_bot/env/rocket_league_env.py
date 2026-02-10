"""
gym environment for Rocket League.

obervation and action spaces are aligned with RLBot and with the Go inference
layer. This module provides:

- RocketLeagueEnv: base env with correct spaces. default implementation is a
  dummy (synthetic) env for pipeline testing and debugging.

  TODO: use a real env that connects to RLBot when training against the game
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from boostrap_bot.env.spaces import (
    OBS_SIZE,
    ACTION_SIZE,
    obs_space_bounds,
    action_space_bounds,
)


class RocketLeagueEnv(gym.Env):
    """
    rocket League environment with RLBot-compatible observation and action spaces.

    observation: vector of shape (OBS_SIZE,) — ball state, car state, padding.
    action: vector of shape (ACTION_SIZE,) — throttle, steer, pitch, yaw, roll, jump, boost.

    this base implementation is a dummy env (random transitions) so that the
    training and ONNX export pipeline can run without a live game
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 5000, seed: int | None = None):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        low_obs, high_obs = obs_space_bounds()
        low_act, high_act = action_space_bounds()
        self.observation_space = spaces.Box(
            low=low_obs,
            high=high_obs,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=low_act,
            high=high_act,
            shape=(ACTION_SIZE,),
            dtype=np.float32,
        )

        if seed is not None:
            self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs, _ = self._sample_obs()
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs, _ = self._sample_obs()
        # replace with real reward when wired to game
        reward = -0.01
        terminated = self._step_count >= self.max_episode_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _sample_obs(self):
        rng = np.random.default_rng()
        low, high = self.observation_space.low, self.observation_space.high
        obs = rng.uniform(low, high).astype(np.float32)
        return obs, {}


def make_rocket_league_env(max_episode_steps: int = 5000, seed: int | None = None) -> RocketLeagueEnv:
    """factory for RocketLeagueEnv (for SB3 and scripts)."""
    return RocketLeagueEnv(max_episode_steps=max_episode_steps, seed=seed)
