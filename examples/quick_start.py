import gym
import numpy as np
import torch

from pbrl.algorithms.dqn import DQN, Policy, Runner
from pbrl.common import Logger
from pbrl.env import DummyVecEnv


def main(env='CartPole-v1', reward_threshold=495., seed=0):
    # define train and test environment
    env_train = DummyVecEnv([lambda: gym.make(env) for _ in range(20)])
    env_test = DummyVecEnv([lambda: gym.make(env) for _ in range(2)])
    # define train and test runner
    runner_train = Runner(env=env_train, start_timestep=1000)
    runner_test = Runner(env=env_test)

    env_train.seed(seed)
    env_test.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # define policy
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        hidden_sizes=[256, 256],
        activation=torch.nn.ReLU,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )
    # define trainer for the task
    logger = Logger('result/quick_start')
    trainer = DQN(
        policy=policy,
        target_freq=10,
        lr=2.3e-3,
        repeat=250
    )
    while True:
        info = trainer.learn(
            timestep=2000,
            runner_train=runner_train,
            timestep_update=500,
            runner_test=runner_test,
            test_interval=2000,
            episode_test=10
        )
        test_reward = np.mean(info['test/reward'])
        logger.log(trainer.timestep, info)
        if test_reward > reward_threshold:
            break

    runner_test.render = 0.0
    runner_test.run(policy, episode_num=10)


if __name__ == '__main__':
    main()
