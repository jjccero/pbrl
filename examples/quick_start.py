import os

import gym
import numpy as np
import torch
from pbrl.algorithms.ppo import PPO, Runner, Policy
from pbrl.common import Logger
from pbrl.env import DummyVecEnv

env = DummyVecEnv([lambda: gym.make('CartPole-v0') for _ in range(16)])
env.seed(0)
np.random.seed(0)
torch.manual_seed(0)
policy = Policy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    hidden_sizes=[64, 64],
    activation=torch.nn.ReLU,
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
)
PPO(policy, lr=1e-3).learn(20480, Runner(env), 2048, Logger('result/quick_start'), 1)
os.system('tensorboard --logdir .')
