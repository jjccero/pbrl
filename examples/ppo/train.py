import argparse

import gym
import numpy as np
import torch

from pbrl.algorithms.ppo import PPO, Runner, Policy
from pbrl.common import Logger
from pbrl.env import SubProcVecEnv, DummyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--test_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--subproc', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--env_num', type=int, default=16)
    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--chunk_len', type=int, default=None)
    parser.add_argument('--rnn', type=str, default=None)
    parser.add_argument('--env_num_test', type=int, default=10)
    parser.add_argument('--episode_num_test', type=int, default=10)
    parser.add_argument('--timestep', type=int, default=100000)

    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.0)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--adv_norm', action='store_true')
    parser.add_argument('--recompute_adv', action='store_true')
    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--reward_norm', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--grad_norm', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = args.log_dir if args.log_dir is not None else '{}-{}'.format(args.env, args.seed)
    filename_log = 'result/{}'.format(log_dir)
    filename_policy = 'result/{}/policy.pkl'.format(log_dir)

    logger = Logger(filename_log)
    # define train and test environment
    env_class = SubProcVecEnv if args.subproc else DummyVecEnv
    env_train = env_class([lambda: gym.make(args.env) for _ in range(args.env_num)])
    env_test = env_class([lambda: gym.make(args.env) for _ in range(args.env_num_test)])
    env_train.seed(args.seed)
    env_test.seed(args.seed)
    # define policy
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        rnn=args.rnn,
        hidden_sizes=[64, 64],
        activation=torch.nn.Tanh,
        obs_norm=args.obs_norm,
        reward_norm=args.reward_norm,
        gamma=args.gamma,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )
    # define trainer for the task
    trainer = PPO(
        policy=policy,
        batch_size=args.batch_size,
        chunk_len=args.chunk_len,
        eps=args.eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        repeat=args.repeat,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_norm=args.grad_norm,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef,
        adv_norm=args.adv_norm,
        recompute_adv=args.recompute_adv
    )
    # lr scheduler
    if args.lr_decay:
        total_update = args.timestep / args.buffer_size
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer,
            lambda update: 1.0 - update / total_update
        )
    if args.resume:
        PPO.load(filename_policy, policy, trainer)
    # define train and test runner
    runner_train = Runner(env=env_train)
    runner_test = Runner(env=env_test)
    trainer.learn(
        timestep=args.timestep,
        runner_train=runner_train,
        timestep_update=args.buffer_size,
        logger=logger,
        log_interval=args.log_interval,
        runner_test=runner_test,
        test_interval=args.test_interval,
        episode_test=args.episode_num_test
    )
    # save result
    trainer.save(filename_policy)


if __name__ == '__main__':
    main()
