import time
from abc import abstractmethod
from typing import Optional

import numpy as np
from pbrl.env.env import VectorEnv


class BaseRunner:
    def __init__(
            self,
            env: VectorEnv,
            max_episode_steps=np.inf,
            render: Optional[float] = None
    ):
        self.env = env
        self.env_num = env.env_num

        self.observations = None
        self.states_actor = None
        self.episode_rewards = np.zeros(self.env_num)
        self.returns = np.zeros(self.env_num)
        self.render = render

        self.max_episode_steps = max_episode_steps
        self.episode_steps = np.zeros(self.env_num, dtype=int)

    def reset(self):
        self.observations = self.env.reset()
        self.states_actor = None
        self.episode_rewards[:] = 0.
        self.returns[:] = 0.
        self.episode_steps[:] = 0

        if self.render is not None:
            self.env.render()
            time.sleep(self.render)

    @abstractmethod
    def run(self, **kwargs) -> dict:
        pass
