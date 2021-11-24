import os
from multiprocessing.connection import Connection

import gym
import numpy as np
import torch

from pbrl.algorithms.ppo import PPO, Runner, PGPolicy
from pbrl.common import Logger, update_dict
from pbrl.competitive import CompetitiveEnv, Agent, MultiDummyEnv, MultiPolicyRunner, CompetitivePBT
from pbrl.env import DummyVecEnv


class TrainEnv(CompetitiveEnv):
    try:
        import gym_compete
    except ModuleNotFoundError:
        raise ModuleNotFoundError

    def init(self, config_policy, **kwargs):
        agent = Agent(
            PGPolicy(
                observation_space=self.observation_space,
                action_space=self.action_space,
                **config_policy
            )
        )
        self.agents.append(agent)
        self.state.update(**kwargs)

    def before_reset(self):
        indices = np.arange(self.role_num)
        self.rs.shuffle(indices)
        self.index = indices[0]
        self.indices = [indices[1:]]
        if self.times_reset % 8 == 0:
            opponent_dir = None
            if self.rs.random() < 0.3:
                opponent_filenames = os.listdir(self.state['history_dir'])
                if len(opponent_filenames) > 0:
                    opponent_filename = self.rs.choice(opponent_filenames)
                    opponent_dir = os.path.join(self.state['history_dir'], opponent_filename)
            else:
                opponent_id = self.rs.choice(self.state['worker_num'])
                opponent_dir = os.path.join(self.state['current_dir'], '{}.pkl'.format(opponent_id))
            load_success = False
            while load_success:
                try:
                    self.agents[0].load_from_dir(opponent_dir)
                    load_success = True
                except:
                    pass


def worker_fn(
        worker_num: int, worker_id: int, remote: Connection, remote_parent: Connection,
        env_name: str,
        history_dir: str,
        current_dir: str,
        seed: int
):
    remote_parent.close()
    env_num = 16
    env_num_test = 4
    episode_num_test = 8
    buffer_size = 409600
    ready_iteration = 10
    config_ppo = dict(
        chunk_len=10,
        lr=1e-3,
        batch_size=5120,
        repeat=3,
        eps=0.2,
        gamma=0.995,
        gae_lambda=0.95
    )
    config_policy = dict(
        rnn='lstm',
        hidden_sizes=[128, 128],
        activation=torch.nn.ReLU,
        obs_norm=True,
        gamma=0.995,
        device=torch.device('cuda:0')
    )
    seed_worker = seed + worker_id
    torch.manual_seed(seed_worker)
    np.random.seed(seed_worker)

    logger = Logger('result/{}-{}-PBT-{}'.format(env_name, seed, worker_id))
    filename_policy = '{}/{}.pkl'.format(current_dir, worker_id)

    env_train = DummyVecEnv(
        [
            lambda: TrainEnv(
                gym.make(env_name),
                index=0,  # the observation_space and action_space will are Tuples if index is None
                config_policy=config_policy,
                history_dir=history_dir,
                worker_num=worker_num,
                current_dir=current_dir
            ) for _ in range(env_num)
        ]
    )
    eval_env = MultiDummyEnv([lambda: gym.make(env_name) for _ in range(env_num_test)])
    env_train.seed(seed_worker)
    eval_env.seed(seed_worker)

    policy = PGPolicy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        **config_policy
    )
    trainer = PPO(policy, **config_ppo)
    PPO.load(filename_policy, policy, trainer)
    trainer.save(filename_policy)
    trainer.save('{}/{}-{}.pkl'.format(history_dir, trainer.iteration, worker_id))
    # define opponents' policies
    policy_opponent = PGPolicy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        critic=False,
        **config_policy
    )
    policies = [policy, policy_opponent]

    runner_train = Runner(env_train, policy)
    runner_eval = MultiPolicyRunner(eval_env, policy_num=2, episode_num=episode_num_test)
    info = dict()
    while True:
        # learning one iteration
        trainer.learn(
            timestep=buffer_size,
            runner_train=runner_train,
            buffer_size=buffer_size,
            logger=logger,
            log_interval=1
        )

        x = dict(
            actor={k: v.cpu() for k, v in policy.actor.state_dict().items()},
            critic={k: v.cpu() for k, v in policy.critic.state_dict().items()},
            lr=trainer.lr,
            rms_obs=policy.rms_obs,
            rms_reward=policy.rms_reward
        )

        # save history
        trainer.save('{}/{}-{}.pkl'.format(history_dir, trainer.iteration, worker_id))
        remote.send((trainer.iteration, x))
        # evaluate
        _ = remote.recv()
        eval_infos = []
        total_win = 0
        total_lose = 0
        total_episode = 0
        episode_rewards = []
        for i in range(worker_num):
            if i == worker_id:
                continue
            PPO.load('{}/{}-{}.pkl'.format(history_dir, trainer.iteration, i), policy_opponent)
            runner_eval.reset()
            eval_dict = runner_eval.run(policies)
            episode = eval_dict['episode']
            win = 0
            lose = 0
            for eval_info in eval_dict['info'][0]:
                if 'winner' in eval_info:
                    win += 1
            for eval_info in eval_dict['info'][1]:
                if 'winner' in eval_info:
                    lose += 1
            eval_infos.append((worker_id, i, episode, win, lose, eval_dict['reward']))
            total_episode += episode
            total_win += win
            total_lose += lose
            episode_rewards += eval_dict['reward'][0]

        ready = trainer.iteration % ready_iteration == 0
        remote.send((ready, eval_infos))
        exploit, score, x = remote.recv()
        rate_info = dict(
            win=total_win / total_episode,
            lose=total_lose / total_episode,
            tie=(total_episode - total_win - total_lose) / total_episode,
            elo=score,
            reward=episode_rewards
        )
        update_dict(info, rate_info, 'test/')
        if exploit:
            hyperparameter_dict = dict(lr=trainer.lr)
            update_dict(info, hyperparameter_dict, 'hyperparameter/')
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
        # save current
        trainer.save(filename_policy)


def main():
    worker_num = 5
    seed = 0
    env_name = 'sumo-humans-v0'
    history_dir = os.path.join('result', '{}-{}-history'.format(env_name, seed))
    current_dir = os.path.join('result', '{}-{}-current'.format(env_name, seed))
    for package_dir in ['result/', history_dir, current_dir]:
        if not os.path.exists(package_dir):
            os.mkdir(package_dir)

    pbt = CompetitivePBT(
        worker_num=worker_num,
        worker_fn=worker_fn,
        exploit=True,
        env_name=env_name,
        history_dir=history_dir,
        current_dir=current_dir,
        seed=seed
    )
    pbt.seed(seed)
    pbt.run()


if __name__ == '__main__':
    main()
