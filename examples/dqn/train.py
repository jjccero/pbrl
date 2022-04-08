import argparse
import time

import gym
import numpy as np
import torch
from pbrl.algorithms.dqn import DQN, Policy
from pbrl.algorithms.td3 import Runner
from pbrl.common import Logger
from pbrl.env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--env_num', type=int, default=10)
    parser.add_argument('--env_num_test', type=int, default=10)
    parser.add_argument('--episode_num_test', type=int, default=10)
    parser.add_argument('--timestep', type=int, default=50000)
    parser.add_argument('--test_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--subproc', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--buffer_size', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--timestep_update', type=int, default=10)
    parser.add_argument('--target_freq', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.9)

    parser.add_argument('--lr_critic', type=float, default=3e-4)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    filename_log = 'result/{}-{}-{}'.format(args.env, args.seed, int(time.time()))
    filename_policy = '{}/policy.pkl'.format(filename_log)

    logger = Logger(filename_log)
    # define train and test environment
    env_train = DummyVecEnv([lambda: gym.make(args.env) for _ in range(args.env_num)])
    env_test = DummyVecEnv([lambda: gym.make(args.env) for _ in range(args.env_num_test)])
    env_train.seed(args.seed)
    env_test.seed(args.seed)
    # define policy
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        hidden_sizes=[128, 128, 128],
        activation=torch.nn.ReLU,
        gamma=args.gamma,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )
    # define trainer for the task
    trainer = DQN(
        policy=policy,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_freq=args.target_freq,
        gamma=args.gamma,
        lr_critic=args.lr_critic
    )

    # define train and test runner
    runner_train = Runner(
        env=env_train,
        start_timestep=10000,
        fill=True,
        epsilon=args.epsilon
    )
    runner_test = Runner(env_test)

    trainer.learn(
        timestep=args.timestep,
        runner_train=runner_train,
        timestep_update=args.timestep_update,
        logger=logger,
        log_interval=args.log_interval,
        runner_test=runner_test,
        test_interval=args.test_interval,
        episode_test=args.episode_num_test
    )
    trainer.save(filename_policy)
    print(filename_policy)


if __name__ == '__main__':
    main()
