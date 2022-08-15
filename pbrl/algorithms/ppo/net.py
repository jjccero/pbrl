from typing import Optional, List

import torch
import torch.nn as nn
from pbrl.policy.base import Mlp, Cnn, Rnn, Discrete, Continuous, Deterministic, orthogonal_init


class Actor(nn.Module):
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
        super(Actor, self).__init__()
        self.hidden_size = hidden_sizes[-1]
        self.rnn = rnn
        self.continuous = continuous
        if len(obs_dim) == 3:
            self.f = Cnn(obs_dim, hidden_sizes, activation)
        else:
            self.f = Mlp(obs_dim, hidden_sizes, activation)
        if self.rnn:
            self.f2 = Rnn(self.hidden_size, activation, self.rnn)
        if self.continuous:
            self.dist = Continuous(self.hidden_size, action_dim, conditional_std)
        else:
            self.dist = Discrete(self.hidden_size, action_dim)
        if orthogonal:
            orthogonal_init(self.f)
            if self.rnn:
                orthogonal_init(self.f2)
            orthogonal_init(self.dist)
            if self.continuous:
                self.dist.mean.weight.data.copy_(0.01 * self.dist.mean.weight.data)

    def forward(self, observations, states=None, dones: Optional[torch.Tensor] = None):
        x = self.f(observations)
        if self.rnn:
            x, states = self.f2(x, states, dones)
        dists = self.dist.forward(x)
        return dists, states


class Critic(nn.Module):
    def __init__(
            self,
            obs_dim: tuple,
            hidden_sizes: List,
            activation,
            rnn: Optional[str],
            orthogonal: bool
    ):
        super(Critic, self).__init__()
        self.hidden_size = hidden_sizes[-1]
        self.rnn = rnn
        if len(obs_dim) == 3:
            self.f = Cnn(obs_dim, hidden_sizes, activation)
        else:
            self.f = Mlp(obs_dim, hidden_sizes, activation)
        if self.rnn:
            self.f2 = Rnn(self.hidden_size, activation)
        self.value = Deterministic(self.hidden_size, 1)
        if orthogonal:
            orthogonal_init(self.f)
            if self.rnn:
                orthogonal_init(self.f2)
            orthogonal_init(self.value)

    def forward(self, observations, states=None, dones: Optional[torch.Tensor] = None):
        x = self.f.forward(observations)
        if self.rnn:
            x, states = self.f2.forward(x, states, dones)
        values = self.value.forward(x).squeeze(-1)
        return values, states
