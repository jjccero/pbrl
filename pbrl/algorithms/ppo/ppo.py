from typing import Tuple, Optional

import numpy as np
import torch

from pbrl.algorithms.ppo.buffer import PGBuffer
from pbrl.algorithms.trainer import Trainer
from pbrl.common.map import auto_map, map_cpu


class PPO(Trainer):
    def __init__(
            self,
            policy,
            batch_size: int = 64,
            chunk_len: int = 0,
            eps: float = 0.2,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            repeat: int = 10,
            lr: float = 3e-4,
            weight_decay: float = 0.0,
            grad_norm: float = 0.5,
            entropy_coef: float = 0.0,
            vf_coef: float = 1.0,
            adv_norm: bool = True,
            recompute_adv: bool = False,
            optimizer=torch.optim.Adam,
            buffer=None
    ):
        super(PPO, self).__init__()
        self.policy = policy
        # on-policy buffer for ppo
        self.buffer = PGBuffer() if buffer is None else buffer
        self.batch_size = batch_size
        self.chunk_len = chunk_len
        self.eps = eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.repeat = repeat
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.adv_norm = adv_norm
        self.recompute_adv = recompute_adv
        self.grad_norm = grad_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer(
            (
                {'params': self.policy.actor.parameters()},
                {'params': self.policy.critic.parameters()}
            ),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.ks = ['observations', 'actions', 'advantages', 'log_probs_old', 'returns']
        if self.policy.rnn:
            self.ks.append('dones')

    def gae(self):
        # reshape to (env_num, step_num, ...)
        # normalize obs and obs_next if obs_norm
        observations, observations_next = auto_map(
            self.policy.n2t,
            (
                self.policy.normalize_observations(np.stack(self.buffer.observations, axis=1)),
                self.policy.normalize_observations(self.buffer.observations_next)
            )
        )
        dones = None
        if self.policy.rnn:
            dones = self.policy.n2t(np.stack(self.buffer.dones, axis=1))
        with torch.no_grad():
            values, states_critic = self.policy.critic.forward(observations, dones=dones)
            values_next, _ = self.policy.critic.forward(observations_next, states=states_critic)
        # reshape to (step_num, env_num, ...)
        values = self.policy.t2n(values).swapaxes(0, 1)
        values_next = self.policy.t2n(values_next)

        rewards, dones = map(np.asarray, (self.buffer.rewards, self.buffer.dones))
        rewards = self.policy.normalize_rewards(rewards)
        advantages = np.zeros_like(rewards)
        gae = np.zeros_like(values_next)
        masks = (1 - dones) * self.gamma
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + masks[t] * values_next - values[t]
            values_next = values[t]
            gae = delta + masks[t] * self.gae_lambda * gae
            advantages[t] = gae
        returns = values + advantages

        self.buffer.advantages = advantages
        self.buffer.returns = returns

    def actor_loss(
            self,
            observations,
            actions: torch.Tensor,
            advantages: torch.Tensor,
            log_probs_old: torch.Tensor,
            dones: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.adv_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dists, _ = self.policy.actor.forward(observations, dones=dones)
        log_probs = dists.log_prob(actions)
        if self.policy.actor.continuous:
            log_probs = log_probs.sum(-1)
        # calculate actor loss by clipping-PPO
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.eps, 1.0 + self.eps) * advantages
        policy_loss = torch.min(surr1, surr2).mean()
        entropy_loss = dists.entropy().mean()
        return policy_loss, entropy_loss

    def critic_loss(
            self,
            observations,
            returns: torch.Tensor,
            dones: Optional[torch.Tensor]
    ) -> torch.Tensor:
        values, _ = self.policy.critic.forward(observations, dones=dones)
        # value clipping is removed in the latest implementation (https://github.com/openai/phasic-policy-gradient)
        # see (https://github.com/openai/baselines/issues/445) for details
        # calculate critic loss by MSE
        value_loss = 0.5 * torch.square(values - returns).mean()
        return value_loss

    def train_pi_vf(self, loss_info):
        for mini_batch in self.buffer.generator(self.batch_size, self.chunk_len, self.ks):
            mini_batch['observations'] = self.policy.normalize_observations(mini_batch['observations'])
            mini_batch = auto_map(self.policy.n2t, mini_batch)
            observations = mini_batch['observations']
            actions = mini_batch['actions']
            advantages = mini_batch['advantages']
            log_probs_old = mini_batch['log_probs_old']
            returns = mini_batch['returns']
            dones = None
            if self.policy.rnn:
                dones = mini_batch['dones']
            policy_loss, entropy_loss = self.actor_loss(observations, actions, advantages, log_probs_old, dones)
            value_loss = self.critic_loss(observations, returns, dones)
            loss = self.vf_coef * value_loss - policy_loss - self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_norm)
            self.optimizer.step()

            loss_info['value'].append(value_loss.item())
            loss_info['policy'].append(policy_loss.item())
            loss_info['entropy'].append(entropy_loss.item())

    def update(self):
        self.iteration += 1
        loss_info = dict(value=[], policy=[], entropy=[])

        self.policy.actor.train()
        for i in range(self.repeat):
            if i == 0 or self.recompute_adv:
                self.policy.critic.eval()
                self.gae()
                self.policy.critic.train()
            # sample batch from buffer
            self.train_pi_vf(loss_info)
        self.policy.actor.eval()
        self.policy.critic.eval()
        # on-policy
        self.buffer.clear()
        return loss_info

    def to_pkl(self):
        pkl = super(PPO, self).to_pkl()
        pkl['optimizer'] = auto_map(map_cpu, self.optimizer.state_dict())
        return pkl

    def from_pkl(self, pkl):
        super(PPO, self).from_pkl(pkl)
        self.optimizer.load_state_dict(pkl['optimizer'])
