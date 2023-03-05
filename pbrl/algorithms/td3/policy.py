import copy
from typing import Optional, List, Type

import numpy as np
import torch
from gym.spaces import Space

from pbrl.algorithms.td3.net import DeterministicActor, DoubleQ
from pbrl.common.map import auto_map, map_cpu
from pbrl.policy.policy import BasePolicy


class Policy(BasePolicy):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            hidden_sizes: List,
            activation: Type[torch.nn.Module],
            rnn: Optional[str] = None,
            clip_fn='',
            obs_norm: bool = False,
            reward_norm: bool = False,
            gamma: float = 0.99,
            obs_clip: float = 10.0,
            reward_clip: float = 10.0,
            device=torch.device('cpu'),
            noise_explore=0.1,
            actor_type=DeterministicActor,
            critic_type=DoubleQ
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
            action_dim=self.action_space.shape[0],
            hidden_sizes=self.hidden_sizes,
            activation=self.activation
        )
        self.actor = actor_type(rnn=None, **config_net).to(self.device)
        self.actor.eval()

        self.actor_target = None
        self.critic_target = None
        if critic_type is not None:
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_target.eval()
            # the critic may be centerQ
            self.critic = critic_type(**config_net).to(self.device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic.eval()
            self.critic_target.eval()

        self.noise_explore = noise_explore

    @torch.no_grad()
    def step(
            self,
            observations,
            states_actor,
            random,
            env_num
    ):
        observations = self.normalize_observations(observations, True)
        if random:
            actions = self.random_action(env_num)
        else:
            observations = auto_map(self.n2t, observations)
            actions, states_actor = self.actor.forward(observations, states_actor)
            actions = self.t2n(actions)
            eps = self.noise_explore * np.random.randn(*actions.shape)
            actions = (actions + eps).clip(-1.0, 1.0)
        return actions, states_actor

    @torch.no_grad()
    def act(
            self,
            observations,
            states_actor
    ):
        observations = self.normalize_observations(observations)
        observations = auto_map(self.n2t, observations)
        actions, states_actor = self.actor.forward(observations, states_actor)
        actions = self.t2n(actions)
        return actions, states_actor

    def to_pkl(self):
        pkl = super(Policy, self).to_pkl()
        pkl['actor'] = auto_map(map_cpu, self.actor.state_dict())
        pkl['critic'] = auto_map(map_cpu, self.critic.state_dict() if self.critic else None)
        return pkl

    def from_pkl(self, pkl):
        super(Policy, self).from_pkl(pkl)
        self.actor.load_state_dict(pkl['actor'])
        if self.critic:
            self.critic.load_state_dict(pkl['critic'])
            if self.critic_target:
                self.actor_target.load_state_dict(pkl['actor'])
                self.critic_target.load_state_dict(pkl['critic'])
