import os
from typing import Tuple, Optional

import numpy as np
import torch

from pbrl.algorithms.ppo.buffer import PGBuffer
from pbrl.algorithms.ppo.policy import Policy
from pbrl.common.trainer import Trainer


class PPO(Trainer):
    def __init__(
            self,
            policy: Policy,
            batch_size: int = 64,
            chunk_len: Optional[int] = None,
            eps: float = 0.2,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            repeat: int = 10,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
            grad_norm: float = 0.5,
            entropy_coef: float = 0.0,
            vf_coef: float = 0.5,
            value_clip: bool = False,
            adv_norm: bool = False,
            recompute_adv: bool = True
    ):
        super(PPO, self).__init__()
        self.policy = policy
        # on-policy buffer for ppo
        self.buffer = PGBuffer(chunk_len)
        self.batch_size = batch_size
        self.eps = eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.repeat = repeat
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_norm = grad_norm
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.adv_norm = adv_norm
        self.recompute_adv = recompute_adv
        self.value_clip = value_clip
        self.optimizer = torch.optim.Adam(
            (
                {'params': self.policy.actor.parameters()},
                {'params': self.policy.critic.parameters()}
            ),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def gae(self):
        # reshape to (env_num, step, ...)
        # normalize obs and obs_next
        observations = self.policy.n2t(
            self.policy.normalize_observations(np.stack(self.buffer.observations, axis=1))
        )
        observations_next = self.policy.n2t(
            self.policy.normalize_observations(self.buffer.observations_next)
        )
        dones = None
        if self.policy.rnn:
            dones = self.policy.n2t(np.stack(self.buffer.dones, axis=1))
        with torch.no_grad():
            values, states_critic = self.policy.get_values(observations, dones=dones)
            values_next, _ = self.policy.get_values(observations_next, states_critic=states_critic)
        # reshape to (step, env_num, ...)
        values = self.policy.t2n(values).swapaxes(0, 1)
        values_next = self.policy.t2n(values_next)

        rewards, dones = map(np.asarray, (self.buffer.rewards, self.buffer.dones))
        rewards = self.policy.normalize_rewards(rewards)
        advantages = np.zeros_like(rewards)
        gae = np.zeros_like(values_next)
        masks = (1 - dones) * self.gamma
        for t in reversed(range(self.buffer.step)):
            delta = rewards[t] + masks[t] * values_next - values[t]
            values_next = values[t]
            gae = delta + masks[t] * self.gae_lambda * gae
            advantages[t] = gae
        returns = values + advantages

        self.buffer.advantages = advantages
        self.buffer.returns = returns

    def actor_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            advantages: torch.Tensor,
            log_probs_old: torch.Tensor,
            dones: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.adv_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_probs, dist_entropy = self.policy.evaluate_actions(observations, actions, dones)
        # calculate actor loss by clipping-PPO
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.eps, 1.0 + self.eps) * advantages
        policy_loss = torch.min(surr1, surr2).mean()
        entropy_loss = dist_entropy.mean()
        return policy_loss, entropy_loss

    def critic_loss(
            self,
            observations: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            dones: Optional[torch.Tensor]
    ) -> torch.Tensor:
        values, _ = self.policy.get_values(observations, dones=dones)
        # calculate critic loss by MSE
        if self.value_clip:
            values_old = returns - advantages
            vf_loss1 = (values - returns) ** 2
            vf_loss2 = ((values - values_old).clamp(-self.eps, self.eps) - advantages) ** 2
            # it makes delta close to advantages
            value_loss = torch.max(vf_loss1, vf_loss2).mean()
        else:
            value_loss = ((values - returns) ** 2).mean()
        return value_loss

    def update(self):
        loss_info = dict(critic=[], policy=[], entropy=[])
        self.policy.train()

        for i in range(self.repeat):
            if i == 0 or self.recompute_adv:
                self.gae()
            # sample batch from buffer
            for batch, batch_rnn in self.buffer.generator(self.batch_size):
                observations, actions, advantages, log_probs_old, returns = batch
                observations = self.policy.normalize_observations(observations)
                observations, actions, advantages, log_probs_old, returns = map(
                    self.policy.n2t,
                    (observations, actions, advantages, log_probs_old, returns)
                )
                dones = None
                if self.policy.rnn:
                    dones, = map(self.policy.n2t, batch_rnn)
                policy_loss, entropy_loss = self.actor_loss(observations, actions, advantages, log_probs_old, dones)
                critic_loss = self.critic_loss(observations, advantages, returns, dones)
                loss = critic_loss * self.vf_coef - policy_loss - entropy_loss * self.entropy_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_norm)
                self.optimizer.step()

                loss_info['critic'].append(critic_loss.item())
                loss_info['policy'].append(policy_loss.item())
                loss_info['entropy'].append(entropy_loss.item())

        # on-policy
        self.buffer.clear()
        return loss_info

    def save(self, filename: str):
        pkl = {
            'timestep': self.timestep,
            'iteration': self.iteration,
            'lr': self.lr,
            'actor': {k: v.cpu() for k, v in self.policy.actor.state_dict().items()},
            'critic': {k: v.cpu() for k, v in self.policy.critic.state_dict().items()},
            'rms_obs': self.policy.rms_obs,
            'rms_reward': self.policy.rms_reward,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(pkl, filename)

    @staticmethod
    def load(filename: str, policy: Policy, trainer=None):
        if os.path.exists(filename):
            pkl = torch.load(filename, map_location=policy.device)
            policy.actor.load_state_dict(pkl['actor'])
            if policy.critic:
                policy.critic.load_state_dict(pkl['critic'])
            if policy.obs_norm:
                policy.rms_obs.load(pkl['rms_obs'])
            if policy.reward_norm:
                policy.rms_reward.load(pkl['rms_reward'])
            if trainer:
                trainer.timestep = pkl['timestep']
                trainer.iteration = pkl['iteration']
                trainer.lr = pkl['lr']
                trainer.optimizer.load_state_dict(pkl['optimizer'])
