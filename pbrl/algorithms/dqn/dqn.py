import os
from typing import Optional

import torch

from pbrl.algorithms.dqn.buffer import ReplayBuffer
from pbrl.algorithms.dqn.policy import Policy
from pbrl.algorithms.trainer import Trainer
from pbrl.common.map import auto_map


class DQN(Trainer):
    def __init__(
            self,
            policy: Policy,
            buffer_size: int = 20000,
            batch_size: int = 64,
            gamma: float = 0.99,
            repeat: int = 1,
            target_freq: int = 10,
            lr_critic: float = 1e-3,
            reward_scale: Optional[float] = None,
            optimizer=torch.optim.Adam,
            epsilon_scheduler=None
    ):
        super(DQN, self).__init__()
        self.policy = policy
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size=buffer_size)
        self.gamma = gamma
        self.repeat = repeat
        self.target_freq = target_freq
        self.lr_critic = lr_critic
        self.optimizer_critic = optimizer(
            self.policy.critic.parameters(),
            lr=self.lr_critic
        )
        self.reward_scale = reward_scale
        self.epsilon_scheduler = epsilon_scheduler

    def critic_loss(
            self,
            observations,
            actions: torch.Tensor,
            observations_next,
            rewards: torch.Tensor,
            dones: torch.Tensor
    ):
        with torch.no_grad():
            q_target, _ = self.policy.critic_target.forward(observations_next)
            q_target = q_target.max(-1)[0]
            td_target = rewards + ~dones * self.gamma * q_target

        q, _ = self.policy.critic.forward(observations)
        q = q.gather(1, actions.unsqueeze(-1)).squeeze()
        td_error = 0.5 * torch.square(td_target - q).mean()

        return td_error

    def train_loop(self, loss_info):
        observations, actions, observations_next, rewards, dones = self.buffer.sample(self.batch_size)
        observations = self.policy.normalize_observations(observations)
        observations_next = self.policy.normalize_observations(observations_next)
        if self.reward_scale is not None:
            rewards = rewards * self.reward_scale
        rewards = self.policy.normalize_rewards(rewards)
        observations, actions, observations_next, rewards, dones = auto_map(
            self.policy.n2t,
            (observations, actions, observations_next, rewards, dones)
        )

        td_error = self.critic_loss(observations, actions, observations_next, rewards, dones)

        self.optimizer_critic.zero_grad()
        td_error.backward()
        self.optimizer_critic.step()

        if self.iteration % self.target_freq == 0:
            self.policy.critic_target.load_state_dict(self.policy.critic.state_dict())
        loss_info['td'].append(td_error.item())

    def run_scheduler(self, loss_info):
        if self.epsilon_scheduler is not None:
            loss_info['epsilon'] = self.policy.epsilon
            self.policy.epsilon = self.epsilon_scheduler(self.timestep)

    def update(self):
        loss_info = dict(td=[])
        self.policy.critic.train()

        for _ in range(self.repeat):
            self.iteration += 1
            self.train_loop(loss_info)

        self.policy.critic.eval()
        self.run_scheduler(loss_info)
        return loss_info

    def save(self, filename: str):
        pkl = {
            'timestep': self.timestep,
            'iteration': self.iteration,
            'lr_critic': self.lr_critic,
            'critic': {k: v.cpu() for k, v in self.policy.critic.state_dict().items()},
            'rms_obs': self.policy.rms_obs,
            'rms_reward': self.policy.rms_reward,
            'optimizer_critic': self.optimizer_critic.state_dict()
        }
        torch.save(pkl, filename)

    @staticmethod
    def load(filename: str, policy: Policy, trainer=None):
        if os.path.exists(filename):
            pkl = torch.load(filename, map_location=policy.device)
            policy.critic.load_state_dict(pkl['critic'])
            if policy.critic:
                policy.critic_target.load_state_dict(pkl['critic'])
            if policy.obs_norm:
                policy.rms_obs.load(pkl['rms_obs'])
            if policy.reward_norm:
                policy.rms_reward.load(pkl['rms_reward'])
            if trainer:
                trainer.timestep = pkl['timestep']
                trainer.iteration = pkl['iteration']
                trainer.lr_critic = pkl['lr_critic']
                trainer.optimizer_critic.load_state_dict(pkl['optimizer_critic'])
