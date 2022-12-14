import argparse
import time

import gym
import numpy as np
import torch

from pbrl.algorithms.dqn import Runner
from pbrl.algorithms.sac import SAC1, Policy
from pbrl.common import Logger
from pbrl.env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--episode_num_test', type=int, default=10)
    parser.add_argument('--timestep', type=int, default=1000000)
    parser.add_argument('--test_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--start_timestep', type=int, default=1000)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--timestep_update', type=int, default=1)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--reward_norm', action='store_true')
    parser.add_argument('--reward_scale', type=float, default=5.0)

    args = parser.parse_args()
    # define train and test environment
    env_train = DummyVecEnv([lambda: gym.make(args.env)])
    env_test = DummyVecEnv([lambda: gym.make(args.env)])
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env_train.seed(args.seed)
    env_test.seed(args.seed)
    # define policy
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        hidden_sizes=[256, 256],
        activation=torch.nn.ReLU,
        gamma=args.gamma,
        obs_norm=args.obs_norm,
        reward_norm=args.reward_norm,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
        q_target=False
    )
    # define trainer for the task
    trainer = SAC1(
        policy=policy,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
    )
    # define train and test runner
    runner_train = Runner(
        env=env_train,
        max_episode_steps=gym.make(args.env).spec.max_episode_steps,
        start_timestep=args.start_timestep
    )
    runner_test = Runner(env_test)

    filename_log = 'result/{}-{}-{}-sac1'.format(args.env, args.seed, int(time.time()))
    filename_policy = '{}/policy.pkl'.format(filename_log)
    logger = Logger(filename_log)
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
