from typing import Optional

import torch

from pbrl.algorithms.dqn.buffer import ReplayBuffer
from pbrl.algorithms.sac.policy import Policy
from pbrl.algorithms.trainer import Trainer
from pbrl.common.map import auto_map, map_cpu


class SAC(Trainer):
    def __init__(
            self,
            policy: Policy,
            buffer_size: int = 1000000,
            batch_size: int = 256,
            gamma: float = 0.99,
            target_freq: int = 1,
            tau: float = 0.005,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_q: float = 3e-4,
            repeat: int = 1,
            reward_scale: Optional[float] = None,
            optimizer=torch.optim.Adam,
            buffer=None
    ):
        super(SAC, self).__init__()
        self.policy = policy
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size=buffer_size) if buffer is None else buffer
        self.gamma = gamma
        self.target_freq = target_freq
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_q = lr_q
        self.optimizer_actor = optimizer(
            self.policy.actor.parameters(),
            lr=self.lr_actor
        )
        self.optimizer_critic = optimizer(
            self.policy.critic.parameters(),
            lr=self.lr_critic
        )
        self.optimizer_q = optimizer(
            self.policy.q.parameters(),
            lr=self.lr_q
        )
        self.repeat = repeat
        self.reward_scale = reward_scale

        assert self.policy.critic_target is not None

    def soft_value_loss(self, observations, q: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        soft_value_target = (q - log_probs).detach()
        soft_values, _ = self.policy.critic.forward(observations)
        soft_value_loss = 0.5 * torch.square(soft_values - soft_value_target).mean()
        return soft_value_loss

    def q_loss(
            self,
            observations,
            observations_next,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor
    ):
        with torch.no_grad():
            soft_values_target, _ = self.policy.critic_target.forward(observations_next)
            td_target = rewards + ~dones * self.gamma * soft_values_target
        q1, q2 = self.policy.q.forward(observations, actions)
        td_error1 = 0.5 * torch.square(td_target - q1).mean()
        td_error2 = 0.5 * torch.square(td_target - q2).mean()
        return td_error1, td_error2

    def train_loop(self, loss_info: dict):
        observations, actions, observations_next, rewards, dones = self.buffer.sample(self.batch_size)
        observations = self.policy.normalize_observations(observations)
        observations_next = self.policy.normalize_observations(observations_next)
        rewards = self.policy.normalize_rewards(rewards)
        if self.reward_scale is not None:
            rewards = rewards * self.reward_scale
        observations, actions, observations_next, rewards, dones = auto_map(
            self.policy.n2t,
            (observations, actions, observations_next, rewards, dones)
        )

        squashing_actions, log_probs = self.policy.squashing_action_log_prob(observations)

        q1, q2 = self.policy.q.forward(observations, squashing_actions)
        q = torch.min(q1, q2)

        # update actor before q for action reuse
        policy_loss = (log_probs - q).mean()
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()

        soft_value_loss = self.soft_value_loss(observations, q, log_probs)
        self.optimizer_critic.zero_grad()
        soft_value_loss.backward()
        self.optimizer_critic.step()

        td_error1, td_error2 = self.q_loss(observations, observations_next, actions, rewards, dones)
        q_loss = td_error1 + td_error2
        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step()

        if self.iteration % self.target_freq == 0:
            Trainer.soft_update(self.policy.critic, self.policy.critic_target, self.tau)

        loss_info['policy'].append(policy_loss.item())
        loss_info['value'].append(soft_value_loss.item())
        loss_info['td1'].append(td_error1.item())
        loss_info['td2'].append(td_error2.item())

    def update(self) -> dict:
        loss_info = dict(policy=[], value=[], td1=[], td2=[])

        self.policy.critic.train()
        self.policy.q.train()
        self.policy.actor.train()

        for _ in range(self.repeat):
            self.iteration += 1
            self.train_loop(loss_info)

        self.policy.critic.eval()
        self.policy.q.eval()
        self.policy.actor.eval()
        return loss_info

    def to_pkl(self):
        pkl = super(SAC, self).to_pkl()
        pkl['optimizer_actor'] = auto_map(map_cpu, self.optimizer_actor.state_dict())
        pkl['optimizer_critic'] = auto_map(map_cpu, self.optimizer_critic.state_dict())
        pkl['optimizer_q'] = auto_map(map_cpu, self.optimizer_q.state_dict())
        return pkl

    def from_pkl(self, pkl):
        super(SAC, self).from_pkl(pkl)
        self.optimizer_actor.load_state_dict(pkl['optimizer_actor'])
        self.optimizer_critic.load_state_dict(pkl['optimizer_critic'])
        self.optimizer_q.load_state_dict(pkl['optimizer_q'])
