import time
from typing import Optional

import numpy as np
from pbrl.algorithms.runner import BaseRunner


class Runner(BaseRunner):
    def run(self, policy, buffer=None, timestep_num=0, episode_num=0):
        timestep = 0
        episode = 0
        episode_rewards = []
        episode_infos = []

        update = buffer is not None

        log_probs: Optional[np.ndarray] = None
        while True:
            observations = self.observations
            if update:
                actions, log_probs, self.states_actor = policy.step(observations, self.states_actor)
            else:
                actions, self.states_actor = policy.act(observations, self.states_actor)
            self.observations, rewards, dones, infos = self.env.step(policy.wrap_actions(actions))

            timestep += self.env_num
            self.episode_rewards += rewards

            if self.render is not None:
                self.env.render()
                time.sleep(self.render)

            if update:
                policy.normalize_rewards(rewards, True, self.returns, dones)
                # add to buffer
                buffer.append(
                    observations,  # raw obs
                    actions,
                    log_probs,
                    rewards,  # raw reward
                    dones
                )

            for i in range(self.env_num):
                if dones[i]:
                    episode += 1
                    if policy.rnn:
                        policy.reset_state(self.states_actor, i)
                    episode_rewards.append(self.episode_rewards[i])
                    episode_infos.append(infos[i])
                    self.episode_rewards[i] = 0.0

            if (timestep_num and timestep >= timestep_num) or (episode_num and episode >= episode_num):
                if update:
                    buffer.observations_next = self.observations
                break

        return dict(
            episode=episode,
            timestep=timestep,
            reward=episode_rewards,
            info=episode_infos
        )
