import argparse
import time

import gym
import numpy as np
import torch

from pbrl.algorithms.dqn import Runner
from pbrl.algorithms.td3 import TD3, Policy
from pbrl.common import Logger
from pbrl.env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--env_num', type=int, default=1)
    parser.add_argument('--env_num_test', type=int, default=1)
    parser.add_argument('--episode_num_test', type=int, default=2)
    parser.add_argument('--timestep', type=int, default=1000000)
    parser.add_argument('--test_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--subproc', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--start_timestep', type=int, default=5000)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--timestep_update', type=int, default=1)
    parser.add_argument('--policy_freq', type=int, default=2)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--noise_explore', type=float, default=0.1)
    parser.add_argument('--noise_target', type=float, default=0.2)
    parser.add_argument('--double_q', action='store_true')  # whether min(Q1,Q2) when updating actor

    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--reward_norm', action='store_true')

    parser.add_argument('--lr_actor', type=float, default=3e-4)
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
        hidden_sizes=[256, 256],
        activation=torch.nn.ReLU,
        obs_norm=args.obs_norm,
        reward_norm=args.reward_norm,
        gamma=args.gamma,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
        noise_explore=args.noise_explore
    )
    # define trainer for the task
    trainer = TD3(
        policy=policy,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        noise_target=args.noise_target,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        double_q=args.double_q,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic
    )

    # define train and test runner
    runner_train = Runner(
        env=env_train,
        max_episode_steps=gym.make(args.env).spec.max_episode_steps,
        start_timestep=args.start_timestep
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
