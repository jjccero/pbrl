from typing import Optional, List

import torch

from pbrl.policy.base import Deterministic, Cnn, Rnn, Continuous, Discrete, Mlp


class AuxActor(torch.nn.Module):
    def __init__(
            self,
            obs_dim: tuple,
            action_dim: int,
            hidden_sizes: List[int],
            activation,
            rnn: Optional[str],
            continuous: bool,
            device
    ):
        super(AuxActor, self).__init__()
        self.hidden_size = hidden_sizes[-1]
        if len(obs_dim) == 3:
            self.f = Cnn(obs_dim, hidden_sizes, activation)
        else:
            self.f = Mlp(obs_dim, hidden_sizes, activation)
        self.rnn = rnn
        self.continuous = continuous
        if self.rnn:
            self.f2 = Rnn(self.hidden_size, activation, self.rnn)
        if self.continuous:
            self.dist = Continuous(self.hidden_size, action_dim)
        else:
            self.dist = Discrete(self.hidden_size, action_dim)
        self.value = Deterministic(self.hidden_size, 1)
        self.device = device
        self.to(self.device)

    def forward(self, observations, states=None, dones: Optional[torch.Tensor] = None):
        x = self.f(observations)
        if self.rnn:
            x, states = self.f2(x, states, dones)
        dists = self.dist.forward(x)
        return dists, states

    def aux(self, observations, states=None, dones: Optional[torch.Tensor] = None):
        x = self.f(observations)
        if self.rnn:
            x, states = self.f2(x, states, dones)
        dist = self.dist.forward(x)
        values = self.value.forward(x).squeeze(-1)
        return dist, values, states
