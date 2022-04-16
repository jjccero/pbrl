from typing import List, Optional

import torch
import torch.nn as nn

from pbrl.policy.base import Mlp, Cnn, Deterministic, init_weights


class QNet(nn.Module):
    def __init__(
            self,
            obs_dim: tuple,
            action_dim: int,
            hidden_sizes: List,
            activation,
            rnn: Optional[str]
    ):
        super(QNet, self).__init__()
        self.hidden_size = hidden_sizes[-1]
        if len(obs_dim) == 3:
            self.f = Cnn(obs_dim, hidden_sizes, activation)
        else:
            self.f = Mlp(obs_dim, hidden_sizes, activation)
        init_weights(self.f)
        self.rnn = rnn
        if self.rnn:
            self.f2 = Rnn(self.hidden_size, activation, self.rnn)
            init_weights(self.f2)
        self.q = Deterministic(self.hidden_size, action_dim)

    def forward(self, observations, states=None, dones: Optional[torch.Tensor] = None):
        x = self.f(observations)
        if self.rnn:
            x, states = self.f2(x, states, dones)
        q = self.q(x)
        return q, states
