"""
Optional Gymnasium wrapper that injects diversity penalty into the reward signal.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple

from .diversity_manager import DiversityManager


class DiversityRewardWrapper(gym.Wrapper):
    """
    Wraps a TradingEnv to subtract a diversity penalty from each step's reward.
    """

    def __init__(self, env: gym.Env, manager: DiversityManager, expert_name: str):
        super().__init__(env)
        self.manager = manager
        self.expert_name = expert_name

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        step_return = info.get("step_return", 0.0)
        penalty = self.manager.compute_penalty(step_return, self.expert_name)

        info["diversity_penalty"] = penalty
        return obs, float(reward - penalty), terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.manager.reset_buffer()
        return obs, info