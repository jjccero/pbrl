from typing import Optional, List, Type

import torch
from gym.spaces import Box, Discrete, Space

from pbrl.algorithms.ppo.net import Actor, Critic
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
            clip_fn='clip',
            obs_norm: bool = False,
            reward_norm: bool = False,
            gamma: float = 0.99,
            obs_clip: float = 10.0,
            reward_clip: float = 10.0,
            device=torch.device('cpu'),
            conditional_std=False,
            orthogonal=True,
            deterministic=False,
            actor_type=Actor,
            critic_type=Critic
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
        self.deterministic = deterministic
        if isinstance(self.action_space, Box):
            continuous = True
            action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, Discrete):
            continuous = False
            action_dim = self.action_space.n
        else:
            raise NotImplementedError

        self.actor = actor_type(
            obs_dim=self.observation_space.shape,
            action_dim=action_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            rnn=self.rnn,
            continuous=continuous,
            conditional_std=conditional_std,
            orthogonal=orthogonal
        ).to(self.device)
        self.actor.eval()
        if critic_type is not None:
            self.critic = critic_type(
                obs_dim=self.observation_space.shape,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                rnn=self.rnn,
                orthogonal=orthogonal
            ).to(self.device)
            self.critic.eval()

    @torch.no_grad()
    def step(
            self,
            observations,
            states_actor
    ):
        observations = self.normalize_observations(observations, True)
        observations = auto_map(self.n2t, observations)
        dists, states_actor = self.actor.forward(observations, states_actor)
        actions = dists.sample()
        log_probs = dists.log_prob(actions)
        if self.actor.continuous:
            log_probs = log_probs.sum(-1)
        actions, log_probs = auto_map(self.t2n, (actions, log_probs))
        return actions, log_probs, states_actor

    @torch.no_grad()
    def act(
            self,
            observations,
            states_actor
    ):
        observations = self.normalize_observations(observations)
        observations = auto_map(self.n2t, observations)
        dists, states_actor = self.actor.forward(observations, states=states_actor)
        if self.deterministic:
            if self.actor.continuous:
                actions = dists.mean
            else:
                actions = torch.argmax(dists.logits, -1)
        else:
            actions = dists.sample()
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
