import argparse
import time

import gym
import numpy as np
import torch

from pbrl.algorithms.dqn import Runner, DQN, Policy
from pbrl.common import Logger
from pbrl.env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--env_num', type=int, default=20)
    parser.add_argument('--env_num_test', type=int, default=2)
    parser.add_argument('--timestep', type=int, default=100000)
    parser.add_argument('--test_interval', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--start_timestep', type=int, default=1000)
    parser.add_argument('--buffer_size', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--timestep_update', type=int, default=500)
    parser.add_argument('--repeat', type=int, default=250)
    parser.add_argument('--target_freq', type=int, default=10)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.2)

    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--reward_norm', action='store_true')

    parser.add_argument('--lr', type=float, default=2.3e-3)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    filename_log = 'result/{}-{}-{}'.format(args.env, args.seed, int(time.time()))
    filename_policy = '{}/policy.pkl'.format(filename_log)

    env_train = DummyVecEnv([lambda: gym.make(args.env) for _ in range(args.env_num)])
    env_test = DummyVecEnv([lambda: gym.make(args.env) for _ in range(args.env_num_test)])
    env_train.seed(args.seed)
    env_test.seed(args.seed)

    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        hidden_sizes=[256, 256],
        activation=torch.nn.ReLU,
        obs_norm=args.obs_norm,
        reward_norm=args.reward_norm,
        gamma=args.gamma,
        epsilon=args.epsilon,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )
    # define trainer for the task
    trainer = DQN(
        policy=policy,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_freq=args.target_freq,
        repeat=args.repeat,
        lr=args.lr
    )

    runner_train = Runner(
        env=env_train,
        start_timestep=args.start_timestep
    )
    runner_test = Runner(env_test)

    logger = Logger(filename_log)
    trainer.learn(
        timestep=args.timestep,
        runner_train=runner_train,
        timestep_update=args.timestep_update,
        logger=logger,
        log_interval=args.log_interval,
        runner_test=runner_test,
        test_interval=args.test_interval,
        episode_test=args.env_num_test
    )
    trainer.save(filename_policy)
    print(filename_policy)


if __name__ == '__main__':
    main()
