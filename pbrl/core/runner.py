import time
from typing import Optional, Dict

import numpy as np
from pbrl.core.buffer import PGBuffer
from pbrl.env.env import VectorEnv
from pbrl.policy.policy import PGPolicy


class Runner:
    def __init__(
            self,
            env: VectorEnv,
            policy: PGPolicy,
            buffer_size: Optional[int] = None,
            episode_num: Optional[int] = None,
            render: Optional[float] = None,
    ):
        self.env = env
        self.env_num = env.env_num
        self.policy = policy
        self.buffer_size = buffer_size
        self.episode_num = episode_num
        self.observations = None
        self.states_actor = None
        self.episode_rewards = np.zeros(self.env_num)
        self.returns = np.zeros(self.env_num)
        self.render = render

    def reset(self):
        self.observations = self.env.reset()
        if self.policy.use_rnn:
            self.states_actor = None
        self.episode_rewards[:] = 0.
        self.returns[:] = 0.

        if self.render is not None:
            self.env.render()
            time.sleep(self.render)

    def run(self, buffer: Optional[PGBuffer] = None) -> Dict:
        timestep = 0
        episode = 0
        episode_rewards = []
        episode_infos = []
        update = buffer is not None

        if update:
            self.policy.train()
        else:
            self.policy.eval()

        while True:
            observations = self.policy.normalize_observations(self.observations, update)
            actions, log_probs, self.states_actor = self.policy.step(
                observations,  # normalized obs
                self.states_actor
            )
            actions_ = self.policy.wrap_actions(actions)
            self.observations, rewards, dones, infos = self.env.step(actions_)

            timestep += self.env_num
            self.episode_rewards += rewards

            if self.render is not None:
                self.env.render()
                time.sleep(self.render)

            if update:
                rewards = self.policy.normalize_rewards(self.returns, rewards, update=True)
                # add to buffer
                buffer.append(
                    observations,  # normalized obs
                    actions,  # raw action
                    log_probs,
                    rewards,  # normalized reward
                    dones
                )

            for i in range(self.env_num):
                if dones[i]:
                    episode += 1
                    if self.policy.use_rnn:
                        if isinstance(self.states_actor, tuple):
                            # lstm
                            for states_ in self.states_actor:
                                states_[:, i, :] = 0.
                        else:
                            # gru
                            self.states_actor[:, i, :] = 0.
                    episode_rewards.append(self.episode_rewards[i])
                    episode_infos.append(infos[i])
                    self.episode_rewards[i] = 0.

                    if update:
                        self.returns[i] = 0.0

            if episode >= self.episode_num if self.episode_num else timestep >= self.buffer_size:
                if update:
                    buffer.observations_next = self.policy.normalize_observations(self.observations, update=False)
                break

        return dict(
            episode=episode,
            timestep=timestep,
            reward=episode_rewards,
            info=episode_infos
        )
