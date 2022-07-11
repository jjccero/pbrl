# PBRL: A Population Based Reinforcement Learning Library based on PyTorch

![Python](https://img.shields.io/badge/language-python-green.svg)

* [Introduction](#introduction)
* [Algorithm](#algorithm)
* [Custom Tasks](#custom-tasks)

## Introduction

Small, fast and <font color=#FF0000>reproducible</font> implementation of reinforcement learning algorithms.  
Support [OpenAI Gym](https://gym.openai.com/) (Atari, [MuJoCo](http://www.mujoco.org/) and Box2D) and custom tasks.

### Installation

Make sure your Conda environment is activated before installing following requirements:  
[Pytorch](https://pytorch.org/)

```
git clone https://github.com/jjccero/pbrl.git
cd pbrl
pip install -e .
```

### Examples

Train and evaluate CartPole-v0 agent:

```
python examples/quick_start.py
```

Try replacing CartPole-v0 above with MountainCar-v0 (rl-baselines3-zoo):

```
cd examples/ppo
python train.py --obs_norm --reward_norm --adv_norm --gae_lambda 0.98 --repeat 4 --buffer_size 256 --env MountainCar-v0
```

MuJoCo:

```
cd examples/ppo
python train.py --obs_norm --reward_norm --recompute_adv --lr_decay --subproc
```

Use Population Based Training:  
`python pbt_train.py`

Open a new terminal:  
`tensorboard --logdir result`

Then you can access the training information by visiting http://localhost:6006/ in browser.

### Structure

* [examples/](/examples)
* [pbrl/](/pbrl)
    * [algorithms/](/pbrl/algorithms)
        * [dqn/](/pbrl/algorithms/dqn) Deep Q Network
        * [ppg/](/pbrl/algorithms/ppg) Phasic Policy Gradient
        * [ppo/](/pbrl/algorithms/ppo) Proximal Policy Optimization
        * [sac/](/pbrl/algorithms/sac) Soft Actor Critic
        * [td3/](/pbrl/algorithms/td3) Twin Delayed Deep Deterministic Policy Gradient
    * [competitive/](/pbrl/competitive) Multi-agent support
    * [env/](/pbrl/env)
        * [env.py](/pbrl/env/env.py) wrapped vector environment
        * [test.py](/pbrl/env/test/rnn.py) test
    * [pbt/](/pbrl/pbt) Population Based Training
    * [policy/](/pbrl/policy)
        * [base.py](/pbrl/policy/base.py) MLP, CNN and RNN
        * [policy.py](/pbrl/policy/policy.py) action wrapper and policy

## Algorithm

### PPO's Tricks

* Parallel collecting
* Learning rate decay
* Generalized Advantage Estimation (GAE)
* Observation Normalization and Reward Scaling (RunningMeanStd)

### TD3 and SAC's Tricks

* Infinite MDPs (no time limit)

### Population Based Training (PBT)

* **_select()_** Ranked by mean episodic rewards. Agents in the bottom 20% copy the top 20%.
* **_explore()_** Each hyperparameter is randomly perturbed by a factor of 1.2 or 0.8.

## Custom Tasks

Refer to the [rnn.py](/pbrl/env/test/rnn.py) to customize your own environment.

```
cd examples/ppo
python train.py --env RnnTest-v0 --chunk_len 8 --rnn gru --gamma 0.0 --lr 1e-3 --log_interval 1
```

* General RL algorithms will achieve an average reward of 55.5.
* Because of the state memory unit, RNN based RL algorithms can reach the goal of 100.0.

#### Multi-agent Environment

The types of **_obs_**, **_reward_**, **_done_** and **_info_** should be tuples.

```
>>> import numpy as np
>>> from gym.spaces import Box, Tuple
>>> Tuple([Discrete(3), Box(-np.inf, np.inf, (2,))])
Tuple(Discrete(3), Box([-inf -inf], [inf inf], (2,), float32))
>>> 
```

---
*2021, ICCD Lab, Dalian University of Technology. Author: Jingcheng Jiang.*  
