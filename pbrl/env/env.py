from abc import abstractmethod
from typing import Tuple, Any

import gym
import numpy as np


def reset_after_done(env, action):
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    return obs, reward, done, info


class VectorEnv:
    def __init__(
            self,
            env_num: int,
            observation_space: gym.Space,
            action_space: gym.Space
    ):
        self.env_num = env_num
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def seed(self, seed):
        pass

    def close(self):
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()
