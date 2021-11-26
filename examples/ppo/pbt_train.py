import argparse
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
        env,
        seed,
        env_num,
        env_num_test,
        obs_norm,
        reward_norm,
        gamma,
        chunk_len,
        rnn,
        batch_size,
        eps,
        gae_lambda,
        repeat,
        lr,
        weight_decay,
        grad_norm,
        entropy_coef,
        vf_coef,
        value_clip,
        adv_norm,
        recompute_adv,
        ready_timestep,
        log_interval,
        buffer_size,
        episode_num_test
):
    remote_parent.close()

    seed_worker = seed + worker_id
    torch.manual_seed(seed_worker)
    np.random.seed(seed_worker)

    log_dir = '{}-{}-PBT-{}'.format(env, seed, worker_id)
    filename_log = 'result/{}'.format(log_dir)
    filename_policy = 'result/{}/policy.pkl'.format(log_dir)
    logger = Logger(filename_log)
    # define train and test environment
    env_train = DummyVecEnv([lambda: gym.make(env) for _ in range(env_num)])
    env_test = DummyVecEnv([lambda: gym.make(env) for _ in range(env_num_test)])

    env_train.seed(seed_worker)
    env_test.seed(seed_worker)
    # define policy
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        rnn=rnn,
        hidden_sizes=[64, 64],
        activation=torch.nn.Tanh,
        obs_norm=obs_norm,
        reward_norm=reward_norm,
        gamma=gamma,
        device=torch.device('cuda:0')
    )
    # define trainer for the task
    trainer = PPO(
        policy,
        batch_size=batch_size,
        chunk_len=chunk_len,
        eps=eps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        repeat=repeat,
        lr=lr,
        weight_decay=weight_decay,
        grad_norm=grad_norm,
        entropy_coef=entropy_coef,
        vf_coef=vf_coef,
        value_clip=value_clip,
        adv_norm=adv_norm,
        recompute_adv=recompute_adv
    )
    PPO.load(filename_policy, policy, trainer)
    # define train and test runner
    runner_train = Runner(env_train)
    runner_test = Runner(env_test)
    info = dict()
    while True:
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--chunk_len', type=int, default=None)
    parser.add_argument('--rnn', type=str, default=None)
    parser.add_argument('--env_num_test', type=int, default=20)
    parser.add_argument('--episode_num_test', type=int, default=100)
    parser.add_argument('--ready_timestep', type=int, default=100000)

    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.0)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--value_clip', action='store_true')
    parser.add_argument('--adv_norm', action='store_true')
    parser.add_argument('--recompute_adv', action='store_true')
    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--reward_norm', action='store_true')

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--grad_norm', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()

    pbt = PBT(
        worker_fn=worker_fn,
        **vars(args)
    )
    pbt.seed(args.seed)
    pbt.run()


if __name__ == '__main__':
    main()
