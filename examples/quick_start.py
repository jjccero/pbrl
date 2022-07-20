import os

import gym
import numpy as np
import torch

from pbrl.algorithms.dqn import DQN, Policy, Runner
from pbrl.common import Logger
from pbrl.env import DummyVecEnv


def main(env='CartPole-v1', seed=0):
    # define train and test environment
    env_train = DummyVecEnv([lambda: gym.make(env) for _ in range(10)])
    env_test = DummyVecEnv([lambda: gym.make(env) for _ in range(10)])
    # define train and test runner
    runner_train = Runner(env=env_train, start_timestep=10000, fill=True, epsilon=0.2)
    runner_test = Runner(env=env_test)

    env_train.seed(seed)
    env_test.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # define policy
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        hidden_sizes=[128, 128, 128],
        activation=torch.nn.ReLU,
        gamma=0.9,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )
    # define trainer for the task
    DQN(
        policy=policy,
        buffer_size=20000,
        batch_size=64,
        target_freq=10,
        gamma=0.9,
        lr_critic=3e-4
    ).learn(
        timestep=50000,
        runner_train=runner_train,
        timestep_update=10,
        logger=Logger('result/quick_start'),
        log_interval=5000,
        runner_test=runner_test,
        test_interval=5000,
        episode_test=10
    )
    os.system('tensorboard --logdir result/quick_start')


if __name__ == '__main__':
    main()
