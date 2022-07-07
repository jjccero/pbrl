import copy
from typing import Optional, List, Type

import numpy as np
import torch
from gym.spaces import Space

from pbrl.algorithms.dqn.net import QNet
from pbrl.common.map import automap
from pbrl.policy.policy import BasePolicy


class Policy(BasePolicy):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            hidden_sizes: List,
            activation: Type[torch.nn.Module],
            rnn: Optional[str] = None,
            clip_fn='clip',
            obs_norm: bool = False,
            reward_norm: bool = False,
            gamma: float = 0.99,
            obs_clip: float = 10.0,
            reward_clip: float = 10.0,
            device=torch.device('cpu'),
            critic=True
    ):
        super(Policy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
            rnn=rnn,
            clip_fn=clip_fn,
            obs_norm=obs_norm,
            reward_norm=reward_norm,
            gamma=gamma,
            obs_clip=obs_clip,
            reward_clip=reward_clip,
            device=device
        )
        config_net = dict(
            obs_dim=self.observation_space.shape,
            action_dim=self.action_space.n,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            rnn=rnn
        )
        self.critic = QNet(**config_net).to(self.device)
        self.critic_target: Optional[QNet] = None
        if critic:
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.eval()

    @torch.no_grad()
    def step(
            self,
            observations: np.ndarray,
            states_actor,
            random=False
    ):
        observations = self.normalize_observations(observations, True)
        if random:
            actions = self.random_action(observations.shape[0])
        else:
            observations = automap(self.n2t, observations)
            q_values, states_actor = self.critic.forward(observations, states_actor)
            actions = torch.argmax(q_values, -1)
            actions = self.t2n(actions)
        return actions, states_actor

    @torch.no_grad()
    def act(
            self,
            observations: np.ndarray,
            states_actor
    ):
        observations = self.normalize_observations(observations)
        observations = automap(self.n2t, observations)
        q_values, states_actor = self.critic.forward(observations, states_actor)
        actions = torch.argmax(q_values, -1)
        actions = self.t2n(actions)
        return actions, states_actor
