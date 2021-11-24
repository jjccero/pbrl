import copy

import numpy as np
import torch

from pbrl.policy.net import DeterministicActor, DoubleQ
from pbrl.policy.policy import Policy


class TD3Policy(Policy):
    def __init__(
            self,
            noise_explore=0.1,
            noise_clip=0.5,
            critic=True,
            **kwargs
    ):
        super(TD3Policy, self).__init__(**kwargs)
        config_net = dict(
            obs_dim=self.observation_space.shape,
            action_dim=self.action_space.shape[0],
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            device=self.device
        )
        self.actor = DeterministicActor(rnn=None, **config_net)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()

        if critic:
            self.critic = DoubleQ(**config_net)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.eval()
        else:
            self.critic = None

        self.noise_explore = noise_explore
        self.noise_clip = noise_clip

    def eval(self):
        self.actor.eval()
        if self.critic:
            self.critic.eval()

    def train(self):
        self.actor.train()
        if self.critic:
            self.critic.train()

    @torch.no_grad()
    def step(
            self,
            observations: np.ndarray,
            states_actor
    ):
        actions, states_actor = self.act(observations, states_actor)
        eps = (self.noise_explore * np.random.randn(*actions.shape)).clip(-self.noise_clip, self.noise_clip)
        actions = (actions + eps).clip(-1.0, 1.0)
        return actions, states_actor

    @torch.no_grad()
    def act(
            self,
            observations: np.ndarray,
            states_actor
    ):
        observations = self.n2t(observations)
        actions, states_actor = self.actor.forward(observations, states_actor)
        actions = self.t2n(actions)
        return actions, states_actor
