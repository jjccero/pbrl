import time
from typing import List

import numpy as np

from pbrl.policy import PGPolicy


class MultiPolicyRunner:
    def __init__(
            self,
            env,
            policy_num,
            episode_num,
            render=None
    ):
        self.env = env
        self.env_num = env.env_num
        self.policy_num = policy_num
        self.episode_num = episode_num
        self.observations = None
        self.states_actor = None
        self.episode_rewards = np.zeros((self.policy_num, self.env_num))
        self.render = render

    def reset(self):
        self.observations = self.env.reset()
        self.states_actor = tuple(None for _ in range(self.policy_num))
        self.episode_rewards[:, :] = 0.0

        if self.render is not None:
            self.env.render()
            time.sleep(self.render)

    def run(self, policies: List[PGPolicy]):
        timestep = 0
        episode = 0
        episode_rewards = tuple([] for _ in range(self.policy_num))
        episode_infos = tuple([] for _ in range(self.policy_num))

        for policy in policies:
            policy.eval()

        while True:
            observations = tuple(map(PGPolicy.normalize_observations, policies, self.observations))
            actions, log_probs, self.states_actor = zip(
                *map(PGPolicy.step, policies, observations, self.states_actor)
            )
            actions_ = tuple(map(PGPolicy.wrap_actions, policies, actions))
            self.observations, rewards, dones, infos = self.env.step(actions_)

            timestep += self.env_num
            self.episode_rewards += rewards

            if self.render is not None:
                self.env.render()
                time.sleep(self.render)

            for i in range(self.env_num):
                if dones[0][i]:
                    episode += 1

                    for index in range(self.policy_num):
                        states_actor = self.states_actor[index]
                        policy = policies[index]

                        if policy.use_rnn:
                            if isinstance(states_actor, tuple):
                                # lstm
                                for states_ in states_actor:
                                    states_[:, i, :] = 0.
                            else:
                                # gru
                                states_actor[:, i, :] = 0.
                        episode_rewards[index].append(self.episode_rewards[index, i])
                        episode_infos[index].append(infos[index][i])
                        self.episode_rewards[index, i] = 0.0

            if episode > self.episode_num:
                break

        return dict(
            episode=episode,
            timestep=timestep,
            reward=episode_rewards,
            info=episode_infos
        )
