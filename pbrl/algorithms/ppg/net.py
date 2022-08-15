from typing import Optional, List

import torch
from pbrl.algorithms.ppo.net import Actor
from pbrl.policy.base import Deterministic, orthogonal_init


class AuxActor(Actor):
    def __init__(
            self,
            obs_dim: tuple,
            action_dim: int,
            hidden_sizes: List,
            activation,
            rnn: Optional[str],
            continuous: bool,
            conditional_std: bool,
            orthogonal: bool
    ):
        super(AuxActor, self).__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            rnn=rnn,
            continuous=continuous,
            conditional_std=conditional_std,
            orthogonal=orthogonal
        )
        self.value = Deterministic(self.hidden_size, 1)
        if orthogonal:
            orthogonal_init(self.value)

    def aux(self, observations, states=None, dones: Optional[torch.Tensor] = None):
        x = self.f(observations)
        if self.rnn:
            x, states = self.f2(x, states, dones)
        dist = self.dist.forward(x)
        values = self.value.forward(x).squeeze(-1)
        return dist, values, states
