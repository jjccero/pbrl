import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


def orthogonal_init(module: nn.Module, gain=1.414):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.orthogonal_(m.weight, gain)
        if isinstance(m, (nn.GRU, nn.LSTM)):
            torch.nn.init.zeros_(m.bias_ih_l0)
            torch.nn.init.zeros_(m.bias_hh_l0)
            torch.nn.init.orthogonal_(m.weight_ih_l0)
            torch.nn.init.orthogonal_(m.weight_hh_l0)


class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation):
        super(Mlp, self).__init__()
        self.flat = len(input_dim) == 2
        last_size = input_dim[0] * input_dim[1] if self.flat else input_dim[0]
        mlp = []
        for hidden_size in hidden_sizes:
            mlp.append(nn.Linear(last_size, hidden_size))
            mlp.append(activation())
            last_size = hidden_size
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        if self.flat:
            x = torch.flatten(x, -2)
        x = self.mlp(x)
        return x


class Cnn(nn.Module):
    def __init__(self, shape, hidden_sizes, activation):
        super(Cnn, self).__init__()
        h, w, in_channels = shape
        cnn = []
        mlp_idx = 0
        for conv in hidden_sizes:
            if isinstance(conv, tuple):
                out_channels, kernel_size, pool_size = conv
                cnn.append(nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size)))
                cnn.append(nn.MaxPool2d(pool_size))
                cnn.append(activation())
                h = (h - kernel_size + 1) // pool_size
                w = (w - kernel_size + 1) // pool_size
                in_channels = out_channels
            else:
                break
            mlp_idx += 1

        self.cnn = nn.Sequential(*cnn)
        self.mlp = Mlp(
            (h * w * in_channels,),
            hidden_sizes[mlp_idx:],
            activation
        )
        self.activation = activation()

    def forward(self, x):
        x = x.transpose(-1, -3)
        if len(x.shape) == 5:
            l, b = x.shape[:2]
            x = x.flatten(0, 1)
            x = self.cnn(x)
            x = x.flatten(1)
            x = x.unflatten(0, (l, b))
        else:
            x = self.cnn(x)
            x = x.flatten(1)
        x = self.activation(self.mlp(x))
        return x


class Rnn(nn.Module):
    def __init__(self, hidden_size, activation, rnn='lstm'):
        super(Rnn, self).__init__()
        if rnn == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size)
        elif rnn == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size)
        else:
            raise NotImplementedError
        self.activation = activation()

    def forward(self, x, states, dones):
        if len(x.shape) == 3:
            # reshape to (chunk_len, batch_size, ...)
            x = x.transpose(0, 1)
            chunk_len = x.shape[0]
            xs = []
            for step in range(chunk_len):
                x_ = x[step:step + 1, :, :]
                x_, states = self.rnn(x_, states)
                done = dones[:, step]
                if isinstance(states, tuple):
                    for states_ in states:
                        states_[:, done, :] = 0.
                else:
                    states[:, done, :] = 0.
                xs.append(x_)
            # reshape to (1, batch_size, chunk_len, ...)
            x = torch.stack(xs, dim=2)
            # reshape to (batch_size, chunk_len, ...)
            x = x.squeeze(0)
        else:
            # reshape to (1, batch_size, ...)
            x = x.unsqueeze(0)
            x, states = self.rnn(x, states)
            # reshape to (batch_size, ...)
            x = x.squeeze(0)
        x = self.activation(x)
        return x, states


class Discrete(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super(Discrete, self).__init__()
        self.logits = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        logits = self.logits(x)
        return Categorical(logits=logits)


class Continuous(nn.Module):
    def __init__(self, hidden_size, action_dim, conditional_std):
        super(Continuous, self).__init__()
        self.mean = nn.Linear(hidden_size, action_dim)
        self.conditional_std = conditional_std
        if self.conditional_std:
            self.logstd = nn.Linear(hidden_size, action_dim)
        else:
            self.logstd_param = nn.Parameter(torch.zeros(action_dim))
            torch.nn.init.constant_(self.logstd_param, -0.5)

    def forward(self, x):
        mean = self.mean(x)
        if self.conditional_std:
            logstd = self.logstd(x)
            std = torch.clamp(logstd, -20., 2.).exp()
        else:
            std = self.logstd_param.exp().expand_as(mean)
        return Normal(mean, std)


class Deterministic(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super(Deterministic, self).__init__()
        self.x = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        return self.x(x)
