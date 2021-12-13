import argparse
import time
from multiprocessing.connection import Connection

import gym
import numpy as np
import torch
from pbrl.algorithms.ppo import PPO, Runner, Policy
from pbrl.common import Logger, update_dict
from pbrl.env import DummyVecEnv
from pbrl.pbt import PBT


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
    PPO.load(filename_policy, policy, trainer)
    # define train and test runner
    runner_train = Runner(env_train)
    runner_test = Runner(env_test)
    info = dict()
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

        x = dict(
            actor={k: v.cpu() for k, v in policy.actor.state_dict().items()},
            critic={k: v.cpu() for k, v in policy.critic.state_dict().items()},
            lr=trainer.lr,
            rms_obs=policy.rms_obs,
            rms_reward=policy.rms_reward
        )
        # evaluate
        runner_test.reset()
        eval_info = runner_test.run(policy=policy, episode_num=episode_num_test)
        update_dict(info, eval_info, 'test/')
        score = np.mean(eval_info['reward'])
        remote.send((trainer.iteration, score, x))

        exploit, _, x = remote.recv()
        if exploit:
            policy.actor.load_state_dict(x['actor'])
            policy.critic.load_state_dict(x['critic'])
            trainer.lr = x['lr']
            trainer.optimizer = torch.optim.Adam(
                (
                    {'params': policy.actor.parameters()},
                    {'params': policy.critic.parameters()}
                ),
                lr=trainer.lr,
                weight_decay=trainer.weight_decay,
            )
            if policy.obs_norm:
                policy.rms_obs.load(x['rms_obs'])
            if policy.reward_norm:
                policy.rms_reward.load(x['rms_reward'])
        # log
        logger.log(trainer.timestep, info)
    # save
    trainer.save(filename_policy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Walker2d-v3')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--worker_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--env_num', type=int, default=16)
    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--env_num_test', type=int, default=2)
    parser.add_argument('--episode_num_test', type=int, default=10)
    parser.add_argument('--ready_timestep', type=int, default=204800)
    parser.add_argument('--timestep', type=int, default=3000000)

    args = parser.parse_args()
    policy_config = dict(
        rnn=None,
        hidden_sizes=[64, 64],
        activation=torch.nn.Tanh,
        obs_norm=True,
        reward_norm=True,
        gamma=0.99,
        device=torch.device('cuda:0')
    )
    trainer_config = dict(
        batch_size=64,
        chunk_len=None,
        eps=0.2,
        gamma=policy_config['gamma'],
        gae_lambda=0.95,
        repeat=10,
        lr=3e-4,
        entropy_coef=0.0,
        adv_norm=False,
        recompute_adv=True
    )
    pbt = PBT(
        worker_fn=worker_fn,
        policy_config=policy_config,
        trainer_config=trainer_config,
        log_dir='result/{}/{}-{}'.format(args.env, args.seed, int(time.time())),
        **vars(args)
    )
    pbt.seed(args.seed)
    pbt.run()


if __name__ == '__main__':
    main()
