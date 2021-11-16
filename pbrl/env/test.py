from collections import deque

import gym
import numpy as np
from gym.envs.registration import register
from gym.spaces import Box, Discrete

register(
    id='Test-v0',
    entry_point='pbrl.env.test:TestEnv',
    max_episode_steps=100,
    reward_threshold=99.0
)


class TestEnv(gym.Env):
    def __init__(self):
        super(TestEnv, self).__init__()
        self.rs = np.random.RandomState()
        self.max_len = 3
        self.state = deque(maxlen=self.max_len)
        self.base = 3
        self.observation_space = Box(0, 1, (self.base,))
        self.action_space = Discrete(self.base ** 3)
        self.info_strs = ['Completely wrong.', 'Partially correct.', 'Partially correct.', 'Completely correct.']

    def reset(self):
        last = self.rs.randint(self.base)
        self.state.extend([last] * self.max_len)
        return self._get_obs()

    def step(self, action: int):
        reward = 0
        for last in self.state:
            if (action % self.base) == last:
                reward += 1
            action //= self.base
        info_str = self.info_strs[reward]
        reward /= self.base
        self.state.append(self.rs.randint(self.base))
        obs = self._get_obs()
        return obs, reward, False, {'str': info_str}

    def _get_obs(self):
        obs = np.zeros(self.base)
        obs[self.state[-1]] = 1.
        return obs

    def seed(self, seed=None):
        if seed is not None:
            self.rs.seed(seed)

    def render(self, mode='human'):
        print(self.state)
