import time
from typing import Optional

import gym
import numpy as np


def load_from_dict(policy, agent_dict):
    # load weights
    policy.actor.load_state_dict(agent_dict['actor'])
    # load RunningMeanStd
    if policy.obs_norm:
        policy.rms_obs.load(agent_dict['rms_obs'])
    policy.actor.eval()


def load_from_dir(policy, filename_policy):
    if filename_policy is not None and os.path.exists(filename_policy):
        pkl = torch.load(filename_policy, map_location=policy.device)
        self.load_from_dict(policy, pkl)


class CompetitiveRunner:
    def __init__(self, env: gym.Env, policies, collect_index=0, render=None, random_opponent=True):
        self.env = env
        self.agent_num = len(env.action_space.spaces)

        self.observations = None
        self.states_actor = None
        self.episode_rewards = np.zeros(1)
        self.returns = np.zeros(1)
        self.render = render
        self.random_opponent = random_opponent
        self.opponent_id = None
        self.collect_index = collect_index
        self.policies = policies
        self.pop = []

    def set_pop(self, pop):
        self.pop = pop

    def reset(self):
        self.observations = self.env.reset()
        self.states_actor = [None] * self.agent_num
        self.episode_rewards[:] = 0.
        self.returns[:] = 0.
        if self.render is not None:
            self.env.render()
            time.sleep(self.render)
        for i in range(self.agent_num):
            policy = self.policies[i]
            if i != self.collect_index and self.random_opponent:
                self.opponent_id = np.random.randint(len(self.pop))
                load_from_dict(policy, self.pop[self.opponent_id])

    def run(
            self,
            policy,
            buffer,
            timestep_num=0,
            episode_num=0
    ):
        timestep = 0
        episode = 0
        episode_rewards = []
        episode_infos = []

        update = buffer is not None

        log_probs: Optional[np.ndarray] = None
        while True:
            observations = None
            actions: Optional[np.ndarray] = None
            wrapped_actions = []
            for i in range(self.agent_num):
                policy = self.policies[i]
                observation = np.expand_dims(self.observations[i], 0)
                if self.collect_index == i and update:
                    action, log_probs, self.states_actor[i] = policy.step(observation, self.states_actor[i])
                    observations = observation
                    actions = action
                else:
                    action, self.states_actor[i] = policy.act(observation, self.states_actor[i])
                wrapped_actions.append(policy.wrap_actions(np.squeeze(action)))
            self.observations, rewards, dones, infos = self.env.step(wrapped_actions)
            rewards = np.expand_dims(rewards[self.collect_index], 0)
            dones = np.expand_dims(dones[self.collect_index], 0)

            timestep += 1
            self.episode_rewards += rewards

            if self.render is not None:
                self.env.render()
                time.sleep(self.render)

            if update:
                self.policies[self.collect_index].normalize_rewards(rewards, True, self.returns, dones)
                # add to buffer
                buffer.append(
                    observations,  # raw obs
                    actions,
                    log_probs,
                    rewards,  # raw reward
                    dones
                )

            if dones[0]:
                episode += 1
                info = dict(res=infos[self.collect_index]['res'], opponent_id=self.opponent_id)
                episode_infos.append(info)
                episode_rewards.append(self.episode_rewards[0])
                self.reset()

            if (timestep_num and timestep >= timestep_num) or (episode_num and episode >= episode_num):
                if update:
                    buffer.observations_next = np.expand_dims(self.observations[self.collect_index], 0)
                break

        return dict(
            episode=episode,
            timestep=timestep,
            reward=episode_rewards,
            info=episode_infos
        )
