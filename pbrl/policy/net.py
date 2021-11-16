from typing import Optional, List

import torch
import torch.nn as nn
from pbrl.policy.base import Mlp, Cnn, Rnn, Discrete, Continuous, Deterministic


def _init(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.orthogonal_(m.weight)
        elif isinstance(m, nn.GRU):
            torch.nn.init.zeros_(m.bias_ih_l0)
            torch.nn.init.zeros_(m.bias_hh_l0)
            torch.nn.init.orthogonal_(m.weight_ih_l0)
            torch.nn.init.orthogonal_(m.weight_hh_l0)


class Actor(nn.Module):
    def __init__(
            self,
            obs_dim: tuple,
            action_dim: int,
            continuous: bool,
            use_rnn: bool,
            hidden_sizes: List[int],
            activation,
            device
    ):
        super(Actor, self).__init__()
        self.continuous = continuous
        self.use_rnn = use_rnn
        self.hidden_size = hidden_sizes[-1]
        if len(obs_dim) == 3:
            self.f = Cnn(obs_dim, hidden_sizes, activation)
        else:
            self.f = Mlp(obs_dim, hidden_sizes, activation)
        if self.use_rnn:
            self.rnn = Rnn(self.hidden_size, activation)
        self.dist = Continuous(self.hidden_size, action_dim) if self.continuous \
            else Discrete(self.hidden_size, action_dim)
        _init(self)
        self.device = device
        self.to(self.device)

    def forward(
            self,
            observations,
            states: Optional[torch.Tensor] = None,
            dones: Optional[torch.Tensor] = None
    ):
        x = self.f(observations)
        if self.use_rnn:
            x, states = self.rnn(x, states, dones)
        dist = self.dist.forward(x)
        return dist, states


class Critic(nn.Module):
    def __init__(
            self,
            obs_dim: tuple,
            use_rnn: bool,
            hidden_sizes: List[int],
            activation,
            device
    ):
        super(Critic, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_sizes[-1]
        if len(obs_dim) == 3:
            self.f = Cnn(obs_dim, hidden_sizes, activation)
        else:
            self.f = Mlp(obs_dim, hidden_sizes, activation)
        if self.use_rnn:
            self.rnn = Rnn(self.hidden_size, activation)
        self.value = Deterministic(self.hidden_size, 1)
        _init(self)
        self.device = device
        self.to(self.device)

    def forward(self, observations, states, dones):
        x = self.f.forward(observations)
        if self.use_rnn:
            x, states = self.rnn.forward(x, states, dones)
        values = self.value.forward(x).squeeze(-1)
        return values, states
