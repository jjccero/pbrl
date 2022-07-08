import numpy as np
from pbrl.common.map import merge_map
from pbrl.env.env import VectorEnv, reset_after_done


class DummyVecEnv(VectorEnv):
    def __init__(self, make_env):
        self.envs = [env_fn() for env_fn in make_env]
        env = self.envs[0]
        super(DummyVecEnv, self).__init__(len(make_env), env.observation_space, env.action_space)

    def reset(self):
        return merge_map(np.asarray, tuple(env.reset() for env in self.envs))

    def step(self, actions):
        results = (reset_after_done(env, action) for env, action in zip(self.envs, actions))
        observations, rewards, dones, infos = zip(*results)
        observations = merge_map(np.asarray, observations)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        return observations, rewards, dones, infos

    def render(self):
        for env in self.envs:
            env.render()

    def seed(self, seed):
        for i, env in enumerate(self.envs):
            env.seed(seed + i)

    def close(self):
        super(DummyVecEnv, self).close()
        for env in self.envs:
            env.close()
