import numpy as np
from pbrl.env.dummy import DummyVecEnv


def transpose(xs):
    return tuple(np.asarray(x) for x in zip(*xs))


def reset_after_done(env, action):
    obs, reward, done, info = env.step(action)
    if True in done:
        obs = env.reset()
    return obs, reward, done, info


class MultiDummyEnv(DummyVecEnv):
    def reset(self):
        observations = transpose([env.reset() for env in self.envs])
        return observations

    def step(self, actions):
        actions = transpose(actions)
        results = [reset_after_done(env, action) for env, action in zip(self.envs, actions)]
        observations, rewards, dones, infos = zip(*results)
        return list(map(transpose, (observations, rewards, dones, infos)))
