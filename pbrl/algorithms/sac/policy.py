import copy
from typing import Optional, List, Type

import numpy as np
import torch
from gym.spaces import Space

from pbrl.algorithms.ppo.net import Actor, Critic
from pbrl.algorithms.ppo.policy import Policy as PGPolicy
from pbrl.algorithms.td3.net import DoubleQ
from pbrl.common.map import auto_map, map_cpu


class Policy(PGPolicy):
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
            deterministic=True,
            actor_type=Actor,
            critic_type=Critic,
            q_type=DoubleQ,
            q_target=True
    ):
        if q_target:
            # arxiv 1812.05905v2
            critic_type = None

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
            device=device,
            conditional_std=True,
            orthogonal=False,
            deterministic=deterministic,
            actor_type=actor_type,
            critic_type=critic_type
        )
        self.critic_target = None
        self.q = None
        self.q_target = None

        if critic_type is not None:
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.eval()

        if q_type is not None:
            self.q = q_type(
                obs_dim=self.observation_space.shape,
                action_dim=self.action_space.shape[0],
                hidden_sizes=hidden_sizes,
                activation=activation
            ).to(self.device)
            self.q.eval()
            if q_target:
                self.q_target = copy.deepcopy(self.q)
                self.q_target.eval()

    def step(
            self,
            observations,
            states_actor,
            random=False,
            env_num=0
    ):
        observations = self.normalize_observations(observations, True)
        if random:
            actions = self.random_action(env_num)
        else:
            observations = auto_map(self.n2t, observations)
            dists, states_actor = self.actor.forward(observations, states_actor)
            actions = dists.sample()
            actions = self.t2n(actions)
            actions = np.tanh(actions)
        return actions, states_actor

    def act(
            self,
            observations,
            states_actor
    ):
        actions, states_actor = super(Policy, self).act(observations, states_actor)
        actions = np.tanh(actions)
        return actions, states_actor

    def squashing_action_log_prob(self, observations):
        dists, _ = self.actor.forward(observations)
        sampled_actions = dists.rsample()
        squashing_actions = torch.tanh(sampled_actions)

        log_probs = dists.log_prob(sampled_actions) - torch.log(1 - torch.square(squashing_actions) + 1e-8)
        log_probs = log_probs.sum(-1)
        return squashing_actions, log_probs

    def to_pkl(self):
        pkl = super(Policy, self).to_pkl()
        if self.q:
            pkl['q'] = auto_map(map_cpu, self.policy.q.state_dict())
        return pkl

    def from_pkl(self, pkl):
        super(Policy, self).from_pkl(pkl)
        if self.q:
            self.q.load_state_dict(pkl['q'])
            if self.q_target:
                self.q_target.load_state_dict(pkl['q'])
        if self.critic_target:
            self.critic_target.load_state_dict(pkl['critic'])
