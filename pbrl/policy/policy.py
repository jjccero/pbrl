from typing import Callable, Optional
from typing import List, Type

import numpy as np
import torch
from gym.spaces import Box
from pbrl.common.rms import RunningMeanStd
from pbrl.policy.net import Actor, Critic


def get_action_wrapper(action_space, clip_fn: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    action_wrapper = None
    if isinstance(action_space, Box):
        low = action_space.low
        high = action_space.high
        if clip_fn == 'tanh':
            def action_wrapper(x):
                return 0.5 * (high - low) * np.tanh(x) + 0.5 * (low + high)
        elif clip_fn == 'clip':
            def action_wrapper(x):
                return 0.5 * (high - low) * np.clip(x, -1.0, 1.0) + 0.5 * (low + high)
        else:
            raise NotImplementedError
    return action_wrapper


class PGPolicy:
    def __init__(
            self,
            observation_space,
            action_space,
            use_rnn: bool,
            hidden_sizes: List[int],
            activation: Type[torch.nn.Module],
            clip_fn='clip',
            obs_norm: bool = False,
            reward_norm: bool = False,
            gamma: float = 0.99,
            obs_clip: float = 10.0,
            reward_clip: float = 10.0,
            critic: bool = True,
            device=torch.device('cpu')
    ):
        obs_dim = observation_space.shape
        continuous = isinstance(action_space, Box)
        action_dim = action_space.shape[0] if continuous else action_space.n
        self.use_rnn = use_rnn
        self.device = device
        self.actor = Actor(
            obs_dim,
            action_dim,
            continuous,
            use_rnn,
            hidden_sizes,
            activation,
            device
        )
        self.critic = Critic(
            obs_dim,
            use_rnn,
            hidden_sizes,
            activation,
            device
        ) if critic else None
        self.obs_norm = obs_norm
        self.rms_obs = RunningMeanStd(
            np.zeros(observation_space.shape, dtype=np.float64),
            np.ones(observation_space.shape, dtype=np.float64)
        ) if self.obs_norm else None
        self.gamma = gamma
        self.obs_clip = obs_clip
        self.reward_norm = reward_norm
        self.rms_reward = RunningMeanStd(0.0, 1.0) if self.reward_norm else None
        self.reward_clip = reward_clip
        self._action_wrapper = get_action_wrapper(action_space, clip_fn)

    @staticmethod
    def t2n(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def n2t(self, n: np.ndarray) -> torch.Tensor:
        if n.dtype == np.float64:
            n = n.astype(np.float32)
        return torch.from_numpy(n).to(self.device)

    def train(self):
        self.actor.train()
        if self.critic:
            self.critic.train()

    def eval(self):
        self.actor.eval()
        if self.critic:
            self.critic.eval()

    @torch.no_grad()
    def step(
            self,
            observations: np.ndarray,
            states_actor
    ):
        observations = self.n2t(observations)
        action, log_prob, states_actor = self.get_actions(observations, states_actor)
        action = self.t2n(action)
        log_prob = self.t2n(log_prob)
        return action, log_prob, states_actor

    def get_values(
            self,
            observations: torch.Tensor,
            states_critic=None,
            dones: Optional[torch.Tensor] = None
    ):
        return self.critic.forward(observations, states_critic, dones)

    def get_actions(self, observations: torch.Tensor, states_actor):
        dist, states_actor = self.actor.forward(observations, states_actor)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if self.actor.continuous:
            log_probs = log_probs.sum(-1)
        return actions, log_probs, states_actor

    def evaluate_actions(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            dones: Optional[torch.Tensor]
    ):
        dist, _ = self.actor.forward(observations, dones=dones)
        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        if self.actor.continuous:
            log_probs = log_probs.sum(-1)
        return log_probs, dist_entropy

    def normalize_observations(self, observations: np.ndarray, update=False):
        if self.obs_norm:
            if update:
                self.rms_obs.update(observations)
            observations = (observations - self.rms_obs.mean) / np.sqrt(self.rms_obs.var + self.rms_obs.eps)
            observations = np.clip(observations, -self.obs_clip, self.obs_clip)
        return observations

    def normalize_rewards(self, returns: np.ndarray, rewards: np.ndarray, update=False):
        if self.reward_norm:
            if update:
                returns[:] = returns * self.gamma + rewards
                self.rms_reward.update(returns)
            rewards = rewards / np.sqrt(self.rms_reward.var + self.rms_reward.eps)
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        return rewards

    def wrap_actions(self, actions: np.ndarray):
        if self._action_wrapper:
            return self._action_wrapper(actions)
        else:
            return actions
