import os

import numpy as np
import torch

from pbrl.policy.policy import PGPolicy


class Agent:
    def __init__(self, policy: PGPolicy):
        self.policy = policy
        self.states_actor = None

    def reset(self):
        if self.policy.use_rnn:
            self.states_actor = None

    def load_from_dir(self, filename_policy):
        if filename_policy is not None and os.path.exists(filename_policy):
            pkl = torch.load(filename_policy, map_location=self.policy.device)
            # load weights
            self.policy.actor.load_state_dict(pkl['actor'])
            # load RunningMeanStd
            if self.policy.obs_norm:
                self.policy.rms_obs.load(pkl['rms_obs'])

    def step(self, observations: np.ndarray) -> np.ndarray:
        observations = self.policy.normalize_observations(observations, update=False)
        actions, _, self.states_actor = self.policy.step(observations, self.states_actor)
        actions = self.policy.wrap_actions(actions)
        return actions
