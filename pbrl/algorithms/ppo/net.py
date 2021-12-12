from typing import Optional, List

import torch
import torch.nn as nn
from pbrl.policy.base import Mlp, Cnn, Rnn, Discrete, Continuous, Deterministic, init_weights


class Actor(nn.Module):
    def __init__(
            self,
            obs_dim: tuple,
            action_dim: int,
            hidden_sizes: List[int],
            activation,
            rnn: Optional[str],
            continuous: bool
    ):
        super(Actor, self).__init__()
        self.hidden_size = hidden_sizes[-1]
        self.rnn = rnn
        self.continuous = continuous
        if len(obs_dim) == 3:
            self.f = Cnn(obs_dim, hidden_sizes, activation)
        else:
            self.f = Mlp(obs_dim, hidden_sizes, activation)
        init_weights(self.f)
        if self.rnn:
            self.f2 = Rnn(self.hidden_size, activation, self.rnn)
            init_weights(self.f2)
        if self.continuous:
            self.dist = Continuous(self.hidden_size, action_dim)
            torch.nn.init.constant_(self.dist.logstd, -0.5)
        else:
            self.dist = Discrete(self.hidden_size, action_dim)
        init_weights(self.dist, 0.01)

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
            hidden_sizes: List[int],
            activation,
            rnn: Optional[str]
    ):
        super(Critic, self).__init__()
        self.hidden_size = hidden_sizes[-1]
        self.rnn = rnn
        if len(obs_dim) == 3:
            self.f = Cnn(obs_dim, hidden_sizes, activation)
        else:
            self.f = Mlp(obs_dim, hidden_sizes, activation)
        init_weights(self.f)
        if self.rnn:
            self.f2 = Rnn(self.hidden_size, activation)
            init_weights(self.f2)
        self.value = Deterministic(self.hidden_size, 1)
        init_weights(self.value, 1.0)

    def forward(self, observations, states=None, dones: Optional[torch.Tensor] = None):
        x = self.f.forward(observations)
        if self.rnn:
            x, states = self.f2.forward(x, states, dones)
        values = self.value.forward(x).squeeze(-1)
        return values, states
