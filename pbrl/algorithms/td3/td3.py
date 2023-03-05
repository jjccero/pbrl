import torch

from pbrl.algorithms.dqn.buffer import ReplayBuffer
from pbrl.algorithms.trainer import Trainer
from pbrl.common.map import auto_map, map_cpu


class TD3(Trainer):
    def __init__(
            self,
            policy,
            buffer_size: int = 1000000,
            batch_size: int = 256,
            gamma: float = 0.99,
            noise_target: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            double_q: bool = False,
            repeat: int = 1,
            tau: float = 0.005,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            optimizer=torch.optim.Adam,
            buffer=None
    ):
        super(TD3, self).__init__()
        self.policy = policy
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size=buffer_size) if buffer is None else buffer
        self.gamma = gamma
        self.noise_target = noise_target
        self.noise_clip = noise_clip
        self.repeat = repeat
        self.policy_freq = policy_freq
        self.double_q = double_q
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.optimizer_actor = optimizer(
            self.policy.actor.parameters(),
            lr=self.lr_actor
        )
        self.optimizer_critic = optimizer(
            self.policy.critic.parameters(),
            lr=self.lr_critic
        )

    def policy_loss(self, observations) -> torch.Tensor:
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
            observations,
            observations_next,
            actions: torch.Tensor,
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
            td_target = rewards + ~dones * self.gamma * q_target

        q1, q2 = self.policy.critic.forward(observations, actions)
        td_error1 = 0.5 * torch.square(td_target - q1).mean()
        td_error2 = 0.5 * torch.square(td_target - q2).mean()
        return td_error1, td_error2

    def train_loop(self, loss_info):
        self.policy.critic.train()

        observations, actions, observations_next, rewards, dones = self.buffer.sample(self.batch_size)
        observations = self.policy.normalize_observations(observations)
        observations_next = self.policy.normalize_observations(observations_next)

        rewards = self.policy.normalize_rewards(rewards)
        observations, actions, observations_next, rewards, dones = auto_map(
            self.policy.n2t,
            (observations, actions, observations_next, rewards, dones)
        )
        td_error1, td_error2 = self.critic_loss(observations, observations_next, actions, rewards, dones)
        critic_loss = td_error1 + td_error2
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.policy.critic.eval()
        if self.iteration % self.policy_freq == 0:
            self.policy.actor.train()

            policy_loss = self.policy_loss(observations)
            actor_loss = -policy_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.policy.actor.eval()
            Trainer.soft_update(self.policy.critic, self.policy.critic_target, self.tau)
            Trainer.soft_update(self.policy.actor, self.policy.actor_target, self.tau)

            loss_info['policy'].append(policy_loss.item())
        loss_info['td1'].append(td_error1.item())
        loss_info['td2'].append(td_error2.item())

    def update(self):
        loss_info = dict(policy=[], td1=[], td2=[])

        for _ in range(self.repeat):
            self.iteration += 1
            self.train_loop(loss_info)

        return loss_info

    def to_pkl(self):
        pkl = super(TD3, self).to_pkl()
        pkl['optimizer_actor'] = auto_map(map_cpu, self.optimizer_actor.state_dict())
        pkl['optimizer_critic'] = auto_map(map_cpu, self.optimizer_critic.state_dict())
        return pkl

    def from_pkl(self, pkl):
        super(TD3, self).from_pkl(pkl)
        self.optimizer_actor.load_state_dict(pkl['optimizer_actor'])
        self.optimizer_critic.load_state_dict(pkl['optimizer_critic'])
