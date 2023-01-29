import argparse
import time
from multiprocessing.connection import Connection

import gym
import numpy as np
import torch

from pbrl.algorithms.ppo import Runner, Policy, PPO
from pbrl.common import Logger, update_dict
from pbrl.common.map import auto_map, map_cpu
from pbrl.env import DummyVecEnv
from pbrl.pbt import PBT


def test(runner_test, policy, episode_num_test, info):
    runner_test.reset()
    eval_info = runner_test.run(policy=policy, episode_num=episode_num_test)
    update_dict(info, eval_info, 'test/')
    return np.mean(eval_info['reward'])


def worker_fn(
        worker_num: int, worker_id: int, remote: Connection, remote_parent: Connection,
        trainer_config: dict,
        policy_config: dict,
        env,
        seed,
        env_num,
        env_num_test,
        timestep: int,
        ready_timestep: int,
        log_interval,
        buffer_size,
        episode_num_test,
        log_dir: str
):
    remote_parent.close()

    seed_worker = seed + worker_id
    torch.manual_seed(seed_worker)
    np.random.seed(seed_worker)
    env_train = DummyVecEnv([lambda: gym.make(env) for _ in range(env_num)])
    env_test = DummyVecEnv([lambda: gym.make(env) for _ in range(env_num_test)])
    env_train.seed(seed_worker)
    env_test.seed(seed_worker)

    filename_log = '{}/{}'.format(log_dir, worker_id)
    filename_policy = '{}/policy.pkl'.format(filename_log)
    logger = Logger(filename_log)

    # define policy
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        **policy_config
    )
    # define trainer for the task
    trainer = PPO(
        policy,
        **trainer_config
    )
    # define train and test runner
    runner_train = Runner(env_train)
    runner_test = Runner(env_test)
    info = dict()
    test(runner_test, policy, episode_num_test, info)
    logger.log(trainer.timestep, info)
    while trainer.timestep < timestep:
        trainer.learn(
            timestep=ready_timestep,
            runner_train=runner_train,
            timestep_update=buffer_size,
            logger=logger,
            log_interval=log_interval
        )
        hyperparameter = dict(lr=trainer.lr)
        update_dict(info, hyperparameter, 'hyperparameter/')
        x = auto_map(
            map_cpu,
            dict(
                actor=policy.actor.state_dict(),
                critic=policy.critic.state_dict(),
                optimizer=trainer.optimizer.state_dict(),
                lr=trainer.lr,
                rms_obs=policy.rms_obs,
                rms_reward=policy.rms_reward
            )
        )

        # evaluate
        score = test(runner_test, policy, episode_num_test, info)
        remote.send(('exploit', (trainer.iteration, score, x)))

        exploit, y = remote.recv()
        if exploit is not None:
            policy.actor.load_state_dict(y['actor'])
            policy.critic.load_state_dict(y['critic'])
            trainer.optimizer.load_state_dict(y['optimizer'])
            for param in trainer.optimizer.param_groups:
                param['lr'] = y['lr']
            if policy.obs_norm:
                policy.rms_obs.load(y['rms_obs'])
            if policy.reward_norm:
                policy.rms_reward.load(y['rms_reward'])
        # log
        logger.log(trainer.timestep, info)
    # save
    trainer.save(filename_policy)
    remote.send(('close', None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Walker2d-v3')
    parser.add_argument('--log_interval', type=int, default=20480)
    parser.add_argument('--ready_timestep', type=int, default=40960)
    parser.add_argument('--worker_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--env_num', type=int, default=16)
    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--chunk_len', type=int, default=None)
    parser.add_argument('--rnn', type=str, default=None)
    parser.add_argument('--env_num_test', type=int, default=2)
    parser.add_argument('--episode_num_test', type=int, default=10)
    parser.add_argument('--timestep', type=int, default=1024000)

    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
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
    policy_config = dict(
        rnn=None,
        hidden_sizes=[64, 64],
        activation=torch.nn.Tanh,
        obs_norm=args.obs_norm,
        reward_norm=args.reward_norm,
        gamma=args.gamma,
        device=torch.device('cuda:0'),
    )
    trainer_config = dict(
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
        adv_norm=args.adv_norm,
        recompute_adv=args.recompute_adv
    )
    pbt = PBT(
        worker_fn=worker_fn,
        worker_num=args.worker_num,
        policy_config=policy_config,
        trainer_config=trainer_config,
        log_dir='result/{}/{}-{}'.format(args.env, args.seed, int(time.time())),
        env=args.env,
        seed=args.seed,
        env_num=args.env_num,
        env_num_test=args.env_num_test,
        timestep=args.timestep,
        ready_timestep=args.ready_timestep,
        log_interval=args.log_interval,
        buffer_size=args.buffer_size,
        episode_num_test=args.episode_num_test
    )
    pbt.seed(args.seed)
    pbt.run()


if __name__ == '__main__':
    main()
