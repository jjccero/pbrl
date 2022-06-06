from typing import List

import gym
import gym.spaces
import numpy as np

from pbrl.competitive.agent import Agent


class CompetitiveEnv:
    def __init__(self, env: gym.Env, index: int, indices):
        self.index = index
        assert isinstance(env.observation_space, gym.spaces.Tuple)
        self.observation_space = env.observation_space.spaces[self.index]
        self.action_space = env.action_space.spaces[self.index]
        self.env = env
        self.agent_num = len(env.action_space.spaces)

        self.observations = None
        self.rewards = None
        self.dones = None
        self.infos = None

        self.indices = indices
        self.agents: List[Agent] = []

    def step(self, action):
        action_list = [None] * self.agent_num
        action_list[self.index] = action

        # action can be None when evaluating
        observations = np.asarray(self.observations)

        for agent, index in zip(self.agents, self.indices):
            observations_ = observations[index]
            actions_ = agent.step(observations_).tolist()
            actions[index] = actions_
        results = self.env.step(tuple(actions))
        self.observations, self.rewards, self.dones, self.infos = results
        return tuple(arr[self.index] for arr in (self.observations, self.rewards, self.dones, self.infos))

    def reset(self):
        self.observations = self.env.reset()
        for agent in self.agents:
            agent.reset()
        return self.observations[self.index]

    def render(self):
        self.env.render()

    def seed(self, seed):
        self.env.seed(seed)
