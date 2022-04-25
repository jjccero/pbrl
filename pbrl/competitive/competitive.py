from typing import List

import gym
import gym.spaces
import numpy as np
from pbrl.competitive.agent import Agent


class CompetitiveEnv:
    def __init__(self, env: gym.Env, index=None, **kwargs):
        self.index = index
        assert isinstance(env.observation_space, gym.spaces.Tuple)
        if self.index is not None:
            self.observation_space = env.observation_space.spaces[self.index]
            self.action_space = env.action_space.spaces[self.index]
        else:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        self.role_num = len(env.observation_space.spaces)
        self.env = env

        self.observations = None
        self.rewards = None
        self.dones = None
        self.infos = None

        self.times_reset = 0
        self.random_state = np.random.RandomState()

        self.indices = []
        self.agents: List[Agent] = []
        self.state = dict()
        self.init(**kwargs)

    def init(self, **kwargs):
        pass

    def before_reset(self):
        pass

    def after_done(self):
        pass

    def step(self, action=None):
        if self.index is None and action is not None:
            actions = action
        else:
            actions = np.repeat(None, self.role_num)
            if self.index is not None:
                actions[self.index] = action

        # action can be None when evaluating
        observations = np.asarray(self.observations)

        for agent, index in zip(self.agents, self.indices):
            observations_ = observations[index]
            actions_ = agent.step(observations_).tolist()
            actions[index] = actions_
        results = self.env.step(tuple(actions))
        self.observations, self.rewards, self.dones, self.infos = results
        if True in self.dones:
            self.after_done()
        if self.index is not None:
            return (arr[self.index] for arr in (self.observations, self.rewards, self.dones, self.infos))
        else:
            return self.observations, self.rewards, self.dones, self.infos

    def reset(self):
        self.before_reset()
        self.times_reset += 1
        self.observations = self.env.reset()
        for agent in self.agents:
            agent.reset()
        if self.index is not None:
            return self.observations[self.index]
        else:
            return self.observations

    def render(self, mode="human"):
        self.env.render(mode)

    def seed(self, seed=None):
        self.random_state.seed(seed)
        self.env.seed(seed)

    def close(self):
        pass
