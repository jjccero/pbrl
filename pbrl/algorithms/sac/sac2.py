import torch

from pbrl.algorithms.dqn.buffer import ReplayBuffer
from pbrl.algorithms.sac.policy import Policy
from pbrl.algorithms.trainer import Trainer
from pbrl.common.map import auto_map, map_cpu


class SAC(Trainer):
    def __init__(
            self,
            policy: Policy,
            target_entropy: float,
            buffer_size: int = 1000000,
            batch_size: int = 256,
            gamma: float = 0.99,
            target_freq: int = 1,
            tau: float = 0.005,
            lr_actor: float = 3e-4,
            lr_q: float = 3e-4,
            lr_alpha: float = 3e-4,
            repeat: int = 1,
            init_alpha=1.0,
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
        self.lr_q = lr_q
        self.lr_alpha = lr_alpha
        self.optimizer_actor = optimizer(
            self.policy.actor.parameters(),
            lr=self.lr_actor
        )
        self.optimizer_q = optimizer(
            self.policy.q.parameters(),
            lr=self.lr_q
        )
        self.log_alpha = torch.log(torch.full((1,), init_alpha, device=self.policy.device)).requires_grad_(True)
        self.optimizer_alpha = optimizer(
            [self.log_alpha],
            lr=self.lr_alpha
        )
        self.target_entropy = target_entropy
        self.repeat = repeat

        assert self.policy.q_target is not None

    def q_loss(
            self,
            observations,
            observations_next,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            alpha
    ):
        with torch.no_grad():
            squashing_actions_next, log_probs_next = self.policy.squashing_action_log_prob(observations_next)
            q1_target, q2_target = self.policy.q_target.forward(observations_next, squashing_actions_next)
            q_target = torch.min(q1_target, q2_target)
            soft_values_target = q_target - alpha * log_probs_next
            td_target = rewards + ~dones * self.gamma * soft_values_target
        q1, q2 = self.policy.q.forward(observations, actions)
        td_error1 = 0.5 * torch.square(td_target - q1).mean()
        td_error2 = 0.5 * torch.square(td_target - q2).mean()
        return td_error1, td_error2

    def policy_loss(self, observations, alpha):
        squashing_actions, log_probs = self.policy.squashing_action_log_prob(observations)
        q1, q2 = self.policy.q.forward(observations, squashing_actions)
        q = torch.min(q1, q2)
        policy_loss = (alpha * log_probs - q).mean()
        return policy_loss, log_probs

    def train_loop(self, loss_info: dict):
        observations, actions, observations_next, rewards, dones = self.buffer.sample(self.batch_size)
        observations = self.policy.normalize_observations(observations)
        observations_next = self.policy.normalize_observations(observations_next)
        rewards = self.policy.normalize_rewards(rewards)

        observations, actions, observations_next, rewards, dones = auto_map(
            self.policy.n2t,
            (observations, actions, observations_next, rewards, dones)
        )

        alpha = torch.exp(self.log_alpha.detach())
        td_error1, td_error2 = self.q_loss(observations, observations_next, actions, rewards, dones, alpha)
        q_loss = td_error1 + td_error2
        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step()

        policy_loss, log_probs = self.policy_loss(observations, alpha)
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()

        alpha_loss = -self.log_alpha * (log_probs.detach().mean() + self.target_entropy)
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        if self.iteration % self.target_freq == 0:
            Trainer.soft_update(self.policy.q, self.policy.q_target, self.tau)

        loss_info['policy'].append(policy_loss.item())
        loss_info['td1'].append(td_error1.item())
        loss_info['td2'].append(td_error2.item())
        loss_info['alpha'].append(alpha.item())

    def update(self) -> dict:
        loss_info = dict(alpha=[], policy=[], td1=[], td2=[])

        self.policy.q.train()
        self.policy.actor.train()

        for _ in range(self.repeat):
            self.iteration += 1
            self.train_loop(loss_info)

        self.policy.q.eval()
        self.policy.actor.eval()
        return loss_info

    def to_pkl(self):
        pkl = super(SAC, self).to_pkl()
        pkl['log_alpha'] = self.log_alpha.item()
        pkl['optimizer_actor'] = auto_map(map_cpu, self.optimizer_actor.state_dict())
        pkl['optimizer_q'] = auto_map(map_cpu, self.optimizer_q.state_dict())
        return pkl

    def from_pkl(self, pkl):
        super(SAC, self).from_pkl(pkl)
        self.log_alpha.data[:] = pkl['log_alpha']
        self.optimizer_actor.load_state_dict(pkl['optimizer_actor'])
        self.optimizer_q.load_state_dict(pkl['optimizer_q'])
