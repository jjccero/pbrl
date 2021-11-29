import time
from typing import Optional

import numpy as np

from pbrl.algorithms.td3.buffer import ReplayBuffer
from pbrl.algorithms.td3.policy import Policy
from pbrl.common.runner import BaseRunner
from pbrl.env.env import VectorEnv


class Runner(BaseRunner):
    def __init__(
            self,
            env: VectorEnv,
            max_episode_steps=np.inf,
            render: Optional[float] = None
    ):
        super(Runner, self).__init__(env, render)
        self.max_episode_steps = max_episode_steps
        self.episode_steps = np.zeros(self.env_num, dtype=int)

    def run(self, policy: Policy, buffer: Optional[ReplayBuffer] = None, timestep_num=0, episode_num=0, random=False):
        timestep = 0
        episode = 0
        episode_rewards = []
        episode_infos = []

        update = buffer is not None
        policy.eval()

        while True:
            observations = self.observations
            if update:
                actions, self.states_actor = policy.step(observations, self.states_actor, random)
            else:
                actions, self.states_actor = policy.act(observations, self.states_actor)
            self.observations, rewards, dones, infos = self.env.step(policy.wrap_actions(actions))

            timestep += self.env_num
            self.episode_rewards += rewards
            self.episode_steps += 1

            if self.render is not None:
                self.env.render()
                time.sleep(self.render)

            if update:
                policy.normalize_rewards(rewards, True, self.returns)
                # add to buffer
                buffer.append(
                    observations,  # raw obs
                    actions,
                    self.observations,  # raw obs_next
                    rewards,  # raw reward
                    dones & (self.episode_steps < self.max_episode_steps)  # TD3' trick
                )

            for i in range(self.env_num):
                if dones[i]:
                    episode += 1
                    if policy.rnn:
                        policy.reset_state(self.states_actor, i)
                    episode_rewards.append(self.episode_rewards[i])
                    episode_infos.append(infos[i])
                    self.episode_rewards[i] = 0.0
                    self.episode_steps[i] = 0
                    if update:
                        self.returns[i] = 0.0

            if (timestep_num and timestep >= timestep_num) or (episode_num and episode >= episode_num):
                break

        if episode:
            return dict(
                episode=episode,
                timestep=timestep,
                reward=episode_rewards,
                info=episode_infos
            )
        else:
            return dict(timestep=timestep)
