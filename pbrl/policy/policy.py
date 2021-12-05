from typing import Callable, Optional, Tuple, Any, List, Type

import numpy as np
import torch
from gym.spaces import Box, Discrete, Space
from pbrl.common.rms import RunningMeanStd
from pbrl.policy.wrapper import TanhWrapper, ClipWrapper


def get_action_wrapper(action_space, clip_fn: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    action_wrapper = None
    if isinstance(action_space, Box):
        low = action_space.low
        high = action_space.high
        if clip_fn == 'tanh':
            return TanhWrapper(low, high)
        elif clip_fn == 'clip':
            return ClipWrapper(low, high)
        else:
            raise NotImplementedError
    return action_wrapper


class BasePolicy:
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            hidden_sizes: List[int],
            activation: Type[torch.nn.Module],
            rnn: Optional[str],
            clip_fn: str,
            obs_norm: bool,
            reward_norm: bool,
            gamma: float,
            obs_clip: float,
            reward_clip: float,
            device: torch.device
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
        self.action_wrapper = get_action_wrapper(action_space, clip_fn)
        self.actor: Optional[torch.nn.Module] = None
        self.critic: Optional[torch.nn.Module] = None

    def step(
            self,
            observations: np.ndarray,
            states_actor
    ):
        raise NotImplementedError

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

    def normalize_rewards(
            self,
            rewards: np.ndarray,
            update=False,
            returns: np.ndarray = None,
            dones: np.ndarray = None
    ):
        if self.reward_norm:
            if update:
                returns[:] = returns * self.gamma + rewards
                self.rms_reward.update(returns)
                returns[dones] = 0.0
            rewards = rewards / np.sqrt(self.rms_reward.var + self.rms_reward.eps)
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)

        return rewards

    def wrap_actions(self, actions: np.ndarray):
        if self.action_wrapper:
            return self.action_wrapper(actions)
        else:
            return actions

    def reset_state(self, states_actor, i):
        if self.rnn == 'lstm':
            for states_ in states_actor:
                states_[:, i, :] = 0.
        elif self.rnn == 'gru':
            states_actor[:, i, :] = 0.

    def random_action(
            self,
            env_num: int
    ):
        if isinstance(self.action_space, Box):
            return np.random.uniform(-1.0, 1.0, size=(env_num, *self.action_space.shape))
        elif isinstance(self.action_space, Discrete):
            return np.random.randint(self.action_space.n, size=env_num)
        else:
            raise NotImplementedError
