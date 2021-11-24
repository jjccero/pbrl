import time
from typing import Optional, Dict

import numpy as np

from pbrl.algorithms.td3.buffer import ReplayBuffer
from pbrl.algorithms.td3.policy import TD3Policy
from pbrl.env.env import VectorEnv


class Runner:
    def __init__(
            self,
            env: VectorEnv,
            policy: TD3Policy,
            render: Optional[float] = None,
    ):
        self.env = env
        self.env_num = env.env_num
        self.policy = policy
        self.observations = None
        self.states_actor = None
        self.episode_rewards = np.zeros(self.env_num)
        self.returns = np.zeros(self.env_num)
        self.render = render

    def reset(self):
        self.observations = self.env.reset()
        if self.policy.rnn:
            self.states_actor = None
        self.episode_rewards[:] = 0.
        self.returns[:] = 0.

        if self.render is not None:
            self.env.render()
            time.sleep(self.render)

    def run(self, buffer_size=0, buffer: Optional[ReplayBuffer] = None, episode_num=0) -> Dict:
        timestep = 0
        episode = 0
        episode_rewards = []
        episode_infos = []
        update = buffer is not None

        while True:
            observations = self.policy.normalize_observations(self.observations, update)
            if update:
                actions, self.states_actor = self.policy.step(observations, self.states_actor)
            else:
                actions, self.states_actor = self.policy.act(observations, self.states_actor)
            actions_ = self.policy.wrap_actions(actions)
            self.observations, rewards, dones, infos = self.env.step(actions_)

            timestep += self.env_num
            self.episode_rewards += rewards

            if self.render is not None:
                self.env.render()
                time.sleep(self.render)

            if update:
                rewards = self.policy.normalize_rewards(self.returns, rewards, update=True)
                observations_next = self.policy.normalize_observations(self.observations, update=False)
                # add to buffer
                buffer.append(
                    observations,  # normalized obs
                    actions,  # raw action
                    observations_next,  # normalized obs_next
                    rewards,  # normalized reward
                    dones
                )

            for i in range(self.env_num):
                if dones[i]:
                    episode += 1
                    if self.policy.rnn:
                        self.policy.reset_state(self.states_actor, i)
                    episode_rewards.append(self.episode_rewards[i])
                    episode_infos.append(infos[i])
                    self.episode_rewards[i] = 0.

                    if update:
                        self.returns[i] = 0.0

            if (buffer_size and timestep >= buffer_size) or (episode_num and episode >= episode_num):
                break

        return dict(
            episode=episode,
            timestep=timestep,
            reward=episode_rewards,
            info=episode_infos
        )
