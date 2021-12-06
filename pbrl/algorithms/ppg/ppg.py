from typing import Optional

import numpy as np
import torch
from pbrl.algorithms.ppo import PPO, Policy
from torch.distributions import Normal, Categorical

from pbrl.algorithms.ppg.aux_buffer import AuxBuffer


class PPG(PPO):
    def __init__(
            self,
            policy: Optional[Policy] = None,
            batch_size: int = 64,
            chunk_len: Optional[int] = None,
            eps: float = 0.2,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            lr: float = 3e-4,
            grad_norm: float = 0.5,
            entropy_coef: float = 0.0,
            adv_norm: bool = False,
            n_pi: int = 32,
            epoch_pi: int = 1,
            epoch_vf: int = 1,
            epoch_aux: int = 6,
            beta_clone: float = 1.0,
            lr_aux: float = 5e-4
    ):
        super(PPG, self).__init__(
            policy=policy,
            batch_size=batch_size,
            chunk_len=chunk_len,
            eps=eps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            repeat=1,
            lr=lr,
            grad_norm=grad_norm,
            entropy_coef=entropy_coef,
            adv_norm=adv_norm
        )
        self.n_pi = n_pi
        self.epoch_pi = epoch_pi
        self.epoch_vf = epoch_vf
        self.epoch_aux = epoch_aux
        self.beta_clone = beta_clone
        self.lr_aux = lr_aux
        self.optimizer_aux = torch.optim.Adam(
            (
                {'params': self.policy.actor.parameters()},
                {'params': self.policy.critic.parameters()}
            ),
            lr=self.lr_aux
        )
        self.aux_buffer = AuxBuffer()
        self.ks_vf = ['observations', 'returns']
        self.ks_pi = ['observations', 'actions', 'advantages', 'log_probs_old']
        self.ks_aux = ['observations', 'vtargs', 'dists_old']
        if self.policy.rnn:
            self.ks_vf.append('dones')
            self.ks_pi.append('dones')
            self.ks_aux.append('dones')

    def train_vf(self, loss_info):
        for mini_batch in self.buffer.generator(self.batch_size, self.chunk_len, self.ks_vf):
            mini_batch['observations'] = self.policy.normalize_observations(mini_batch['observations'])
            mini_batch = {k: self.policy.n2t(v) for k, v in mini_batch.items()}
            observations = mini_batch['observations']
            returns = mini_batch['returns']
            dones = None
            if self.policy.rnn:
                dones = mini_batch['dones']
            value_loss = self.critic_loss(observations, returns, dones)

            self.optimizer.zero_grad()
            value_loss.backward()
            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_norm)
            self.optimizer.step()

            loss_info['value'].append(value_loss.item())

    def train_pi(self, loss_info):
        for mini_batch in self.buffer.generator(self.batch_size, self.chunk_len, self.ks_pi):
            mini_batch['observations'] = self.policy.normalize_observations(mini_batch['observations'])
            mini_batch = {k: self.policy.n2t(v) for k, v in mini_batch.items()}
            observations = mini_batch['observations']
            actions = mini_batch['actions']
            advantages = mini_batch['advantages']
            log_probs_old = mini_batch['log_probs_old']
            dones = None
            if self.policy.rnn:
                dones = mini_batch['dones']
            policy_loss, entropy_loss = self.actor_loss(observations, actions, advantages, log_probs_old, dones)
            loss = - policy_loss - self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_norm)
            self.optimizer.step()

            loss_info['policy'].append(policy_loss.item())
            loss_info['entropy'].append(entropy_loss.item())

    def auxiliary_phase(self, loss_info):
        for mini_batch in self.aux_buffer.generator(self.batch_size, self.chunk_len, self.ks_aux):
            mini_batch['observations'] = self.policy.normalize_observations(mini_batch['observations'])
            mini_batch = {k: self.policy.n2t(v) for k, v in mini_batch.items()}
            observations = mini_batch['observations']
            vtargs = mini_batch['vtargs']
            dists_old = mini_batch['dists_old']
            if self.policy.actor.continuous:
                dists_old = dists_old.permute(-1, *[i for i in range(len(dists_old.shape[:-1]))])
                dists_old = Normal(
                    loc=dists_old[0],
                    scale=dists_old[1]
                )
            else:
                dists_old = Categorical(logits=dists_old)
            dones = None
            if self.policy.rnn:
                dones = mini_batch['dones']
            dists, values, _ = self.policy.actor.aux(observations, dones=dones)
            aux_loss = 0.5 * torch.square(values - vtargs).mean()
            clone_loss = torch.distributions.kl_divergence(dists_old, dists).mean()

            value_loss = self.critic_loss(observations, vtargs, dones)
            self.optimizer_aux.zero_grad()
            (aux_loss + self.beta_clone * clone_loss + value_loss).backward()
            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_norm)
            self.optimizer_aux.step()

            loss_info['kl'].append(clone_loss.item())
            loss_info['aux_pi'].append(aux_loss.item())
            loss_info['aux_vf'].append(value_loss.item())

    @torch.no_grad()
    def compute_dists_old(self):
        states_actor = None
        for i in range(self.n_pi):
            observations = self.policy.n2t(
                self.policy.normalize_observations(self.aux_buffer.observations[i].swapaxes(0, 1))
            )
            dones = None
            if self.policy.rnn:
                dones = self.policy.n2t(self.aux_buffer.dones[i].swapaxes(0, 1))
            dists, states_actor = self.policy.actor.forward(observations, states_actor, dones)
            if isinstance(dists, Categorical):
                dists = self.policy.t2n(dists.logits)
            elif isinstance(dists, Normal):
                dists = np.stack(
                    (self.policy.t2n(dists.loc), self.policy.t2n(dists.scale)),
                    axis=-1
                )
            dists = dists.swapaxes(0, 1)
            self.aux_buffer.dists_old.append(dists)

    def update(self):
        loss_info = dict(value=[], policy=[], entropy=[])

        self.policy.critic.eval()
        self.gae()
        self.policy.actor.train()
        self.policy.critic.train()
        if self.epoch_pi == self.epoch_vf:
            for i in range(self.epoch_pi):
                # it is PPO
                self.train_pi_vf(loss_info)
        else:
            for i in range(self.epoch_vf):
                self.train_vf(loss_info)
            for i in range(self.epoch_pi):
                self.train_pi(loss_info)
        self.policy.actor.eval()
        self.aux_buffer.append(
            observations=self.buffer.observations,
            dones=self.buffer.dones,
            vtargs=self.buffer.returns
        )

        if (self.iteration + 1) % self.n_pi == 0:
            loss_info.update(dict(kl=[], aux_pi=[], aux_vf=[]))
            self.compute_dists_old()
            self.policy.actor.train()
            for i in range(self.epoch_aux):
                self.auxiliary_phase(loss_info)
            self.policy.actor.eval()
            self.aux_buffer.clear()
        self.policy.critic.eval()
        self.buffer.clear()
        return loss_info
