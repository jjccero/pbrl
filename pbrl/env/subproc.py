import multiprocessing
from multiprocessing.connection import Connection

import numpy as np
from pbrl.common.pickle import CloudpickleWrapper
from pbrl.env.env import VectorEnv, reset_after_done

RESET = 0
STEP = 1
SPACE = 2
RENDER = 3
SEED = 4
CLOSE = 5


def worker_fn(env_fns: CloudpickleWrapper, remote: Connection, remote_parent: Connection):
    remote_parent.close()
    envs = [env_fn() for env_fn in env_fns.x]
    while True:
        cmd, data = remote.recv()
        if cmd == STEP:
            remote.send([reset_after_done(env, action) for env, action in zip(envs, data)])
        elif cmd == RENDER:
            for env in envs:
                env.render()
        elif cmd == RESET:
            remote.send([env.reset() for env in envs])
        elif cmd == SPACE:
            remote.send((envs[0].observation_space, envs[0].action_space))
        elif cmd == SEED:
            for i, env in enumerate(envs):
                env.seed(data + i)
        elif cmd == CLOSE:
            for env in envs:
                env.close()
            break


def flatten(x):
    return [x__ for x_ in x for x__ in x_]


class SubProcVecEnv(VectorEnv):
    def __init__(self, make_env, worker_num=4):
        self.remotes = []
        self.ps = []
        self.worker_num = worker_num
        self.env_nums = []
        ctx = multiprocessing.get_context('spawn')
        for env_fns in np.array_split(make_env, self.worker_num):
            remote, remote_worker = ctx.Pipe()
            p = ctx.Process(
                target=worker_fn,
                args=(
                    CloudpickleWrapper(env_fns),
                    remote_worker,
                    remote
                ),
                daemon=False
            )
            p.start()
            self.ps.append(p)
            self.remotes.append(remote)
            remote_worker.close()
            self.env_nums.append(len(env_fns))
        self.remotes[0].send((SPACE, None))
        observation_space, action_space = self.remotes[0].recv()
        super(SubProcVecEnv, self).__init__(len(make_env), observation_space, action_space)

    def reset(self):
        for remote in self.remotes:
            remote.send((RESET, None))
        observations = flatten([remote.recv() for remote in self.remotes])
        observations = np.asarray(observations)
        return observations

    def step(self, actions):
        actions = np.array_split(actions, self.worker_num)
        for remote, action in zip(self.remotes, actions):
            remote.send((STEP, action))
        results = flatten([remote.recv() for remote in self.remotes])
        observations, rewards, dones, infos = map(np.asarray, zip(*results))
        return observations, rewards, dones, infos

    def render(self):
        for remote in self.remotes:
            remote.send((RENDER, None))

    def seed(self, seed):
        index = 0
        for remote, env_num in zip(self.remotes, self.env_nums):
            remote.send((SEED, seed + index))
            index += env_num

    def close(self):
        super(SubProcVecEnv, self).close()
        for remote in self.remotes:
            remote.send((CLOSE, None))
        for p in self.ps:
            p.join()
