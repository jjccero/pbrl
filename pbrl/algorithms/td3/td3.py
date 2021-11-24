from typing import Optional

import torch

from pbrl.algorithms.td3.buffer import ReplayBuffer
from pbrl.algorithms.td3.policy import TD3Policy
from pbrl.common.trainer import Trainer


class TD3(Trainer):
    def __init__(
            self,
            policy: TD3Policy,
            buffer: ReplayBuffer,
            batch_size: int = 256,
            gamma: float = 0.99,
            noise_target: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            tau: float = 0.005,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4
    ):
        super(TD3, self).__init__()
        self.policy = policy
        self.batch_size = batch_size
        self.buffer = buffer
        self.gamma = gamma
        self.noise_target = noise_target
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
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

    def actor_loss(self, observations: torch.Tensor):
        actions, _ = self.policy.actor.forward(observations)
        q1, q2 = self.policy.critic.forward(observations, actions)
        policy_loss = (-torch.min(q1, q2)).mean()
        return policy_loss

    def critic_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            observations_next: torch.Tensor,
            rewards: Optional[torch.Tensor],
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
        batch = self.buffer.sample(self.batch_size)
        observations, actions, observations_next, rewards, dones = map(self.policy.n2t, batch)
        q1_loss, q2_loss = self.critic_loss(observations, actions, observations_next, rewards, dones)

        value_loss = q1_loss + q2_loss
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()

        loss_info['q1'] = q1_loss.item()
        loss_info['q2'] = q2_loss.item()

        if self.iteration % self.policy_freq == 0:
            policy_loss = self.actor_loss(observations)
            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            self.optimizer_actor.step()

            Trainer.soft_update(self.policy.critic, self.policy.critic_target, self.tau)
            Trainer.soft_update(self.policy.actor, self.policy.actor_target, self.tau)

            loss_info['policy'] = policy_loss.item()
        return loss_info
