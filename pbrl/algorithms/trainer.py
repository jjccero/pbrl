import os
from typing import Optional

import torch

from pbrl.common.logger import update_dict
from pbrl.policy.policy import BasePolicy


class Trainer:
    def __init__(self):
        self.policy: Optional[BasePolicy] = None
        self.buffer = None
        self.timestep = 0
        self.iteration = 0
        self.scheduler = None

    @staticmethod
    def soft_update(net: torch.nn.Module, net_target: torch.nn.Module, tau: float):
        for param, param_target in zip(net.parameters(), net_target.parameters()):
            param_target.data.copy_((1.0 - tau) * param_target.data + tau * param.data)

    def update(self) -> dict:
        pass

    def learn(
            self,
            timestep: int,
            runner_train,
            timestep_update: int,
            logger=None,
            log_interval=0,
            runner_test=None,
            test_interval=0,
            episode_test=0
    ):
        assert log_interval % timestep_update == 0 and test_interval % timestep_update == 0
        target_timestep = self.timestep + timestep
        info = dict()
        if runner_train.observations is None:
            runner_train.reset()

        if log_interval and test_interval and self.timestep == 0:
            runner_test.reset()
            test_info = runner_test.run(policy=self.policy, episode_num=episode_test)
            update_dict(info, test_info, 'test/')
            logger.log(self.timestep, info)

        while True:
            if self.scheduler is not None:
                schedule_info = self.scheduler(self)
                update_dict(info, schedule_info)

            train_info = runner_train.run(policy=self.policy, buffer=self.buffer, timestep_num=timestep_update)
            update_dict(info, train_info, 'train/')
            self.timestep += train_info['timestep']

            loss_info = self.update()
            update_dict(info, loss_info, 'loss/')

            done = self.timestep >= target_timestep

            if test_interval and (self.timestep % test_interval == 0 or done):
                runner_test.reset()
                test_info = runner_test.run(policy=self.policy, episode_num=episode_test)
                update_dict(info, test_info, 'test/')
            if log_interval and (self.timestep % log_interval == 0 or done):
                logger.log(self.timestep, info)
            if done:
                break
        return info

    def to_pkl(self):
        pkl = self.policy.to_pkl()
        return pkl

    def from_pkl(self, pkl):
        self.policy.from_pkl(pkl)

    def save(self, filename: str):
        torch.save(self.to_pkl(), filename)

    @staticmethod
    def load(filename: str, policy, trainer=None):
        if os.path.exists(filename):
            pkl = torch.load(filename, map_location=policy.device)
            if trainer is not None:
                trainer.from_pkl(pkl)
            else:
                policy.from_pkl(pkl)
