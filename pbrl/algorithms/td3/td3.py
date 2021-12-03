import os
from typing import Optional

import torch
from pbrl.algorithms.td3.buffer import ReplayBuffer
from pbrl.algorithms.td3.policy import Policy
from pbrl.common.trainer import Trainer


class TD3(Trainer):
    def __init__(
            self,
            policy: Policy,
            buffer_size: int = 1000000,
            batch_size: int = 256,
            gamma: float = 0.99,
            noise_target: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            double_q: bool = False,
            tau: float = 0.005,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            reward_scaling: Optional[float] = None
    ):
        super(TD3, self).__init__()
        self.policy = policy
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=policy.observation_space,
            action_space=policy.action_space
        )
        self.gamma = gamma
        self.noise_target = noise_target
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.double_q = double_q
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.optimizer_actor = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=self.lr_actor
        )
        self.optimizer_critic = torch.optim.Adam(
            self.policy.critic.parameters(),
            lr=self.lr_critic
        )
        self.reward_scaling = reward_scaling

    def policy_loss(self, observations: torch.Tensor) -> torch.Tensor:
        self.policy.critic.eval()
        actions, _ = self.policy.actor.forward(observations)
        if self.double_q:
            q1, q2 = self.policy.critic.forward(observations, actions)
            policy_loss = torch.min(q1, q2).mean()
        else:
            # origin TD3 only use Q1
            policy_loss = self.policy.critic.Q1(observations, actions).mean()
        return policy_loss

    def critic_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            observations_next: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor
    ):
        with torch.no_grad():
            actions_target, _ = self.policy.actor_target.forward(observations_next)
            noises_target = torch.clamp(
                torch.randn_like(actions_target) * self.noise_target,
                -self.noise_clip,
                self.noise_clip
            )
            actions_target = torch.clamp(actions_target + noises_target, -1.0, 1.0)

            q1_target, q2_target = self.policy.critic_target.forward(observations_next, actions_target)
            q_target = torch.min(q1_target, q2_target)
            y = rewards + (1.0 - dones) * self.gamma * q_target

        q1, q2 = self.policy.critic.forward(observations, actions)
        q1_loss = 0.5 * ((y - q1) ** 2).mean()
        q2_loss = 0.5 * ((y - q2) ** 2).mean()

        return q1_loss, q2_loss

    def update(self):
        loss_info = dict()
        self.policy.train()

        observations, actions, observations_next, rewards, dones = self.buffer.sample(self.batch_size)
        observations = self.policy.normalize_observations(observations)
        observations_next = self.policy.normalize_observations(observations_next)
        if self.reward_scaling:
            rewards = rewards / self.reward_scaling
        rewards = self.policy.normalize_rewards(rewards)
        observations, actions, observations_next, rewards, dones = map(
            self.policy.n2t,
            (observations, actions, observations_next, rewards, dones)
        )
        q1_loss, q2_loss = self.critic_loss(observations, actions, observations_next, rewards, dones)
        critic_loss = q1_loss + q2_loss
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        if self.iteration % self.policy_freq == 0:
            policy_loss = self.policy_loss(observations)
            actor_loss = -policy_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            Trainer.soft_update(self.policy.critic, self.policy.critic_target, self.tau)
            Trainer.soft_update(self.policy.actor, self.policy.actor_target, self.tau)

            loss_info['policy'] = policy_loss.item()
        loss_info['q1'] = q1_loss.item()
        loss_info['q2'] = q2_loss.item()
        return loss_info

    def save(self, filename: str):
        pkl = {
            'timestep': self.timestep,
            'iteration': self.iteration,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'actor': {k: v.cpu() for k, v in self.policy.actor.state_dict().items()},
            'critic': {k: v.cpu() for k, v in self.policy.critic.state_dict().items()},
            'rms_obs': self.policy.rms_obs,
            'rms_reward': self.policy.rms_reward,
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict()
        }
        torch.save(pkl, filename)

    @staticmethod
    def load(filename: str, policy: Policy, trainer=None):
        if os.path.exists(filename):
            pkl = torch.load(filename, map_location=policy.device)
            policy.actor.load_state_dict(pkl['actor'])
            policy.actor_target.load_state_dict(pkl['actor'])
            if policy.critic:
                policy.critic.load_state_dict(pkl['critic'])
                policy.critic_target.load_state_dict(pkl['critic'])
            if policy.obs_norm:
                policy.rms_obs.load(pkl['rms_obs'])
            if policy.reward_norm:
                policy.rms_reward.load(pkl['rms_reward'])
            if trainer:
                trainer.timestep = pkl['timestep']
                trainer.iteration = pkl['iteration']
                trainer.lr_actor = pkl['lr_actor']
                trainer.lr_critic = pkl['lr_critic']
                trainer.optimizer_actor.load_state_dict(pkl['optimizer_actor'])
                trainer.optimizer_critic.load_state_dict(pkl['optimizer_critic'])
