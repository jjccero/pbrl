import os
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from pbrl.common.logger import update_dict, Logger
from pbrl.core.buffer import PGBuffer
from pbrl.core.runner import Runner
from pbrl.policy.policy import PGPolicy


class PPO:
    def __init__(
            self,
            policy: PGPolicy,
            mini_batch_size: int = 64,
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
        self.policy = policy
        # on-policy buffer for ppo
        self.buffer = PGBuffer(mini_batch_size, chunk_len)
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
        self.timestep = 0
        self.iteration = 0
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None

    @staticmethod
    def load(
            filename: str,
            policy: PGPolicy,
            trainer=None
    ):
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

    def gae(self):
        # reshape to (env_num, step, ...)
        observations = self.policy.n2t(np.stack(self.buffer.observations, axis=1))
        observations_next = self.policy.n2t(self.buffer.observations_next)
        dones = None
        if self.policy.use_rnn:
            dones = self.policy.n2t(np.stack(self.buffer.dones, axis=1))
        with torch.no_grad():
            values, states_critic = self.policy.get_values(observations, dones=dones)
            values_next, _ = self.policy.get_values(observations_next, states_critic=states_critic)
        # reshape to (step, env_num, ...)
        values = self.policy.t2n(values).swapaxes(0, 1)
        values_next = self.policy.t2n(values_next)

        rewards, dones = map(np.asarray, (self.buffer.rewards, self.buffer.dones))
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

    def update(self, batch: Tuple[np.ndarray], batch_rnn: Optional[Tuple[np.ndarray, ...]]) -> Dict:
        observations, actions, advantages, log_probs_old, returns = map(self.policy.n2t, batch)
        if self.policy.use_rnn:
            dones, = map(self.policy.n2t, batch_rnn)
        else:
            dones = None
        policy_loss, entropy_loss = self.actor_loss(observations, actions, advantages, log_probs_old, dones)
        value_loss = self.critic_loss(observations, advantages, returns, dones)
        loss = value_loss * self.vf_coef - policy_loss - entropy_loss * self.entropy_coef

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_norm)
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_norm)
        self.optimizer.step()

        return dict(
            policy=policy_loss.item(),
            critic=value_loss.item(),
            entropy=entropy_loss.item()
        )

    def learn(
            self,
            timestep: int,
            runner_train: Runner,
            logger: Optional[Logger],
            log_interval: Optional[int],
            runner_test: Optional[Runner] = None,
            test_interval: Optional[int] = None
    ):
        timestep += self.timestep
        info = dict()
        runner_train.reset()

        if test_interval and log_interval and self.timestep == 0:
            runner_test.reset()
            test_info = runner_test.run()
            update_dict(info, test_info, 'test/')
            logger.log(self.timestep, info)

        while True:
            train_info = runner_train.run(self.buffer)
            self.timestep += train_info['timestep']
            for i in range(self.repeat):
                if i == 0 or self.recompute_adv:
                    self.gae()
                # sample batch from buffer
                for batch, batch_rnn in self.buffer.generator():
                    loss_info = self.update(batch, batch_rnn)
                    update_dict(info, loss_info, 'loss/')
            if self.scheduler:
                self.scheduler.step()
                train_info['lr'] = self.scheduler.get_last_lr()
            update_dict(info, train_info, 'train/')
            # on-policy
            self.buffer.clear()
            self.iteration += 1
            done = self.timestep >= timestep
            if test_interval and (self.iteration % test_interval == 0 or done):
                runner_test.reset()
                test_info = runner_test.run()
                update_dict(info, test_info, 'test/')
            if log_interval and (self.iteration % log_interval == 0 or done):
                logger.log(self.timestep, info)
            if done:
                break
