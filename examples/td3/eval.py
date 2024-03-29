import argparse

import gym
import torch

from pbrl.algorithms.dqn import Runner
from pbrl.algorithms.td3 import TD3, Policy
from pbrl.env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--obs_norm', action='store_true')

    parser.add_argument('--env_num_test', type=int, default=1)
    parser.add_argument('--episode_num_test', type=int, default=1)
    parser.add_argument('--render', type=float, default=0.005)

    args = parser.parse_args()
    # define test environment
    env_test = DummyVecEnv([lambda: gym.make(args.env) for _ in range(args.env_num_test)])
    torch.manual_seed(args.seed)
    env_test.seed(args.seed)
    # define policy
    policy = Policy(
        observation_space=env_test.observation_space,
        action_space=env_test.action_space,
        hidden_sizes=[256, 256],
        activation=torch.nn.ReLU,
        obs_norm=args.obs_norm,
        critic_type=None,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )
    # load policy from disk
    TD3.load(args.filename, policy)
    # define test runner
    runner_test = Runner(env=env_test, render=args.render)
    runner_test.reset()
    while True:
        try:
            test_info = runner_test.run(policy, episode_num=args.episode_num_test)
            print(test_info)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
