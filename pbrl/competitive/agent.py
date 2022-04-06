import os

import numpy as np
import torch

from pbrl.policy.policy import BasePolicy


class Agent:
    def __init__(self, policy: BasePolicy):
        self.policy = policy
        self.states_actor = None

    def reset(self):
        if self.policy.rnn:
            self.states_actor = None

    def load_from_dict(self, agent_dict):
        # load weights
        self.policy.actor.load_state_dict(agent_dict['actor'])
        # load RunningMeanStd
        if self.policy.obs_norm:
            self.policy.rms_obs.load(agent_dict['rms_obs'])
        self.policy.actor.eval()
        self.reset()

    def load_from_dir(self, filename_policy):
        if filename_policy is not None and os.path.exists(filename_policy):
            pkl = torch.load(filename_policy, map_location=self.policy.device)
            self.load_from_dict(pkl)

    def step(self, observations: np.ndarray) -> np.ndarray:
        actions, self.states_actor = self.policy.act(observations, self.states_actor)
        actions = self.policy.wrap_actions(actions)
        return actions
