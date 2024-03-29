# PBRL: A Population Based Reinforcement Learning Library based on PyTorch

![Python](https://img.shields.io/badge/language-python-green.svg)

* [Introduction](#introduction)
* [Algorithm](#algorithm)
* [Custom Tasks](#custom-tasks)

## Introduction

Small, fast and <font color=#FF0000>reproducible</font> implementation of reinforcement learning algorithms.  
Support [OpenAI Gym](https://gym.openai.com/) (Atari, [MuJoCo](http://www.mujoco.org/) and Box2D) and custom tasks.

In general, the default hyperparameters of each algorithm are consistent with those in the original paper. PBRL provides
default training scripts in the `./examples` folder. These scripts can change the hyperparameters of the algorithm by
command line parameters.

PBRL provides a base class for PBT so that developers can quickly implement asynchronous training of the model by
rewriting the parent and child process working functions.

### Installation

Ubuntu is recommended.

Make sure your Conda environment is activated before installing following requirements:  
[Pytorch](https://pytorch.org/)

```
git clone https://github.com/jjccero/pbrl.git
cd pbrl
pip install -e .
```

### Examples

Train and evaluate CartPole agent:

```
python examples/quick_start.py
```

Try replacing CartPole above with MountainCar-v0 (rl-baselines3-zoo):

```
cd examples/ppo
python train.py --obs_norm --reward_norm --adv_norm --gae_lambda 0.98 --repeat 4 --buffer_size 256 --env MountainCar-v0
```

MuJoCo:

PPG for Humanoid-v3

```
cd examples/ppg
python train.py --obs_norm --reward_norm
```

Use Population Based Training:  
`python pbt_train.py`

PPO for Walker2d-v3

```
cd examples/ppo
python train.py --obs_norm --reward_norm --recompute_adv --lr_decay --subproc
```

TD3 and SAC for HalfCheetah-v3

```
cd examples/td3
python train.py
```

```
cd examples/sac
python train_sac2.py
```

Open a new terminal (`./result` will be automatically created when the training starts):  
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
    * [env/](/pbrl/env)
        * [env.py](/pbrl/env/env.py) wrapped vector environment
        * [test.py](/pbrl/env/test/rnn.py) test
    * [pbt/](/pbrl/pbt) Population Based Training
    * [policy/](/pbrl/policy)
        * [base.py](/pbrl/policy/base.py) MLP, CNN and RNN
        * [policy.py](/pbrl/policy/policy.py) action wrapper and policy

## Algorithm

### PPO's Tricks

* Orthogonal Initialize
* Learning rate decay
* Generalized Advantage Estimation (GAE)
* Observation Normalization and Reward Scaling (RunningMeanStd)

### Off-policy algorithms' Tricks

* Infinite MDPs (`done_real = done & (episode_steps < max_episode_steps)`)
* DistributionalReplayBuffer (A distributed experience replay buffer is implemented in the `pbrl.algorithms.dqn.buffer` module, which
  allows some off-policy algorithms to collect samples through sub processes.)

### Population Based Training (PBT)

PBT implements communication between multiple processes by creating pipelines of multiple parent-child processes. This
needs to pass the working function of the child process to the constructor of PBT. The parameters of the working
function can also be passed through the constructor, and then the run method of PBT executes the listening of the child
process commands. It is strongly recommended that you call `seed()` after `PBT()`.

PBT can be inherited, and the `run()`
method is rewritten to handle the logic of the corresponding work functions, which means that some methods of PBT will
not be used if unnecessary.

* `select()` Ranked by mean episodic rewards. Agent in the bottom copies the top.
* `explore()` Each hyperparameter is randomly perturbed by a factor of 1.2 or 0.8.

## Custom Tasks

Each environment needs to rewrite the `step()`, `reset()` methods, and define the corresponding `action_space`
and `observation_space`. Before creating an environment through `gym.make()`, you need to register the corresponding
environment in the module that can be imported.

Refer to the [rnn.py](/pbrl/env/test/rnn.py) to customize your own environment.

```
cd examples/ppo
python train.py --env RnnTest-v0 --chunk_len 8 --rnn gru --gamma 0.0 --lr 1e-3 --log_interval 2048
```

* General RL algorithms will achieve an average reward of 55.5.
* Because of the state memory unit, RNN based RL algorithms can reach the goal of 100.0.

---
*2021, ICCD Lab, Dalian University of Technology. Author: Jingcheng Jiang.*  
