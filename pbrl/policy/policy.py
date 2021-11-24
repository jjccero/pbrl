from typing import Callable, Optional, Tuple, Any, List, Type

import numpy as np
import torch
from gym.spaces import Box

from pbrl.common.rms import RunningMeanStd


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


class Policy:
    def __init__(
            self,
            observation_space,
            action_space,
            hidden_sizes: List[int],
            activation: Type[torch.nn.Module],
            rnn: Optional[str],
            clip_fn='clip',
            obs_norm: bool = False,
            reward_norm: bool = False,
            gamma: float = 0.99,
            obs_clip: float = 10.0,
            reward_clip: float = 10.0,
            device=torch.device('cpu')
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if rnn is not None:
            rnn = rnn.lower()
            assert rnn in ('lstm', 'gru')
        self.rnn = rnn
        self.device = device
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

    def train(self):
        pass

    def eval(self):
        pass

    def act(
            self,
            observations: np.ndarray,
            states_actor
    ) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError

    @staticmethod
    def t2n(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def n2t(self, n: np.ndarray) -> torch.Tensor:
        if n.dtype == np.float64:
            n = n.astype(np.float32)
        return torch.from_numpy(n).to(self.device)

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

    def reset_state(self, states_actor, i):
        if self.rnn == 'lstm':
            for states_ in states_actor:
                states_[:, i, :] = 0.
        elif self.rnn == 'gru':
            states_actor[:, i, :] = 0.
