import copy
from typing import Optional, List, Type

import torch
from gym.spaces import Space
from pbrl.algorithms.ppo.net import Actor, Critic
from pbrl.algorithms.ppo.policy import Policy as PGPolicy
from pbrl.algorithms.td3.net import DoubleQ


class Policy(PGPolicy):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            hidden_sizes: List,
            activation: Type[torch.nn.Module],
            rnn: Optional[str] = None,
            clip_fn='tanh',
            obs_norm: bool = False,
            reward_norm: bool = False,
            gamma: float = 0.99,
            obs_clip: float = 10.0,
            reward_clip: float = 10.0,
            device=torch.device('cpu'),
            conditional_std=True,
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
            conditional_std=conditional_std,
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

            if q_target:
                self.q_target = copy.deepcopy(self.q)
                self.q_target.eval()

    def step(
            self,
            observations,
            states_actor,
            **kwargs
    ):
        actions, _, states_actor = super(Policy, self).step(observations, states_actor)
        return actions, states_actor

    def squashing_action_log_prob(self, observations):
        dists, _ = self.actor.forward(observations)
        sampled_actions = dists.rsample()
        squashing_actions = torch.tanh(sampled_actions)

        log_probs = dists.log_prob(sampled_actions) - torch.log(1 - torch.square(squashing_actions) + 1e-8)
        log_probs = log_probs.sum(-1)
        return squashing_actions, log_probs
