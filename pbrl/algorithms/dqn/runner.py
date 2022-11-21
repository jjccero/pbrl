import time
from typing import Optional

import numpy as np
from pbrl.algorithms.runner import BaseRunner
from pbrl.env.env import VectorEnv


class Runner(BaseRunner):
    def __init__(
            self,
            env: VectorEnv,
            max_episode_steps=np.inf,
            render: Optional[float] = None,
            start_timestep: Optional[int] = None
    ):
        super(Runner, self).__init__(
            env=env,
            max_episode_steps=max_episode_steps,
            render=render
        )
        self.start_timestep = start_timestep

    def run(self, policy, buffer=None, timestep_num=0, episode_num=0):
        timestep = 0
        episode = 0
        episode_rewards = []
        episode_infos = []

        update = buffer is not None
        random = False
        # TD3
        if self.start_timestep is not None:
            random = True
            timestep_num = self.start_timestep
            self.start_timestep = None

        while True:
            observations = self.observations
            if update:
                actions, self.states_actor = policy.step(
                    observations=observations,
                    states_actor=self.states_actor,
                    random=random,
                    env_num=self.env_num
                )
            else:
                actions, self.states_actor = policy.act(observations, self.states_actor)
            self.observations, rewards, dones, infos = self.env.step(policy.wrap_actions(actions))

            timestep += self.env_num
            self.episode_rewards += rewards
            self.episode_steps += 1

            if self.render is not None:
                self.env.render()
                time.sleep(self.render)

            dones_real = dones & (self.episode_steps < self.max_episode_steps)  # TD3' trick

            if update:
                policy.normalize_rewards(rewards, True, self.returns, dones_real)
                # add to buffer
                buffer.append(
                    observations,  # raw obs
                    actions,
                    self.observations,  # raw obs_next
                    rewards,  # raw reward
                    dones_real
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
