import argparse
import time

import gym
import numpy as np
import torch
from pbrl.algorithms.ppg import AuxActor, PPG
from pbrl.algorithms.ppo import Runner, Policy
from pbrl.common import Logger
from pbrl.env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v3')
    parser.add_argument('--test_interval', type=int, default=20480)
    parser.add_argument('--log_interval', type=int, default=20480)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=1024000)
    parser.add_argument('--env_num_test', type=int, default=2)
    parser.add_argument('--episode_num_test', type=int, default=10)

    # PPO hyperparameters
    parser.add_argument('--env_num', type=int, default=2)
    parser.add_argument('--chunk_len', type=int, default=None)
    parser.add_argument('--rnn', type=str, default=None)
    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--entropy_coef', type=float, default=0.0)
    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--reward_norm', action='store_true')
    parser.add_argument('--grad_norm', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-4)
    # PPG hyperparameters
    parser.add_argument('--aux_batch_size', type=int, default=256)
    parser.add_argument('--lr_aux', type=float, default=3e-4)
    parser.add_argument('--beta_clone', type=float, default=1.0)
    parser.add_argument('--epoch_pi', type=int, default=4)
    parser.add_argument('--epoch_vf', type=int, default=4)
    parser.add_argument('--epoch_aux', type=int, default=6)
    parser.add_argument('--n_pi', type=int, default=10)

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
        rnn=args.rnn,
        hidden_sizes=[64, 64],
        activation=torch.nn.Tanh,
        obs_norm=args.obs_norm,
        reward_norm=args.reward_norm,
        gamma=args.gamma,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
        actor_type=AuxActor,
        deterministic=True
    )
    # define trainer for the task
    trainer = PPG(
        policy=policy,
        batch_size=args.batch_size,
        chunk_len=args.chunk_len,
        eps=args.eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        grad_norm=args.grad_norm,
        entropy_coef=args.entropy_coef,
        aux_batch_size=args.aux_batch_size,
        lr_aux=args.lr_aux,
        beta_clone=args.beta_clone,
        epoch_aux=args.epoch_aux,
        epoch_pi=args.epoch_pi,
        epoch_vf=args.epoch_vf,
        n_pi=args.n_pi
    )
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
    print(filename_policy)


if __name__ == '__main__':
    main()
