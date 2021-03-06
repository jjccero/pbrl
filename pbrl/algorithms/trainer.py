from typing import Optional

import torch

from pbrl.common.logger import update_dict, Logger
from pbrl.policy.policy import BasePolicy


class Trainer:
    def __init__(self):
        self.policy: Optional[BasePolicy] = None
        self.buffer = None
        self.timestep = 0
        self.iteration = 0
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None

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
            logger: Optional[Logger] = None,
            log_interval=0,
            runner_test: Optional = None,
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
            train_info = runner_train.run(policy=self.policy, buffer=self.buffer, timestep_num=timestep_update)
            self.timestep += train_info['timestep']
            loss_info = self.update()
            update_dict(info, loss_info, 'loss/')
            if self.scheduler:
                self.scheduler.step()
                train_info['lr'] = self.scheduler.get_last_lr()
            update_dict(info, train_info, 'train/')
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
