import os

import gym
import torch
from pbrl.algorithms.ppo import PPO, Runner, Policy
from pbrl.common import Logger
from pbrl.env import DummyVecEnv

env = DummyVecEnv([lambda: gym.make('CartPole-v0') for _ in range(16)])
policy = Policy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    hidden_sizes=[64, 64],
    activation=torch.nn.ReLU,
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
)
PPO(policy, lr=1e-3).learn(20480, Runner(env), 2048, Logger('quick_start'), 1)
os.system('tensorboard --logdir quick_start')
