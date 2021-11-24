from abc import abstractmethod
from typing import Optional

import torch

from pbrl.common.logger import update_dict, Logger


class Trainer:
    def __init__(self):
        self.policy = None
        self.buffer = None
        self.timestep = 0
        self.iteration = 0
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None

    @staticmethod
    def soft_update(net: torch.nn.Module, net_target: torch.nn.Module, tau: float):
        for param, param_target in zip(net.parameters(), net_target.parameters()):
            param_target.data.copy_((1.0 - tau) * param_target.data + tau * param.data)

    @abstractmethod
    def update(self) -> dict:
        pass

    def learn(
            self,
            timestep: int,
            runner_train,
            buffer_size: int,
            logger: Logger,
            log_interval: int,
            runner_test: Optional = None,
            test_interval=0,
            episode_test=0
    ):
        timestep += self.timestep
        info = dict()
        runner_train.reset()

        if log_interval and test_interval and self.timestep == 0:
            self.policy.eval()
            runner_test.reset()
            test_info = runner_test.run(episode_num=episode_test)
            update_dict(info, test_info, 'test/')
            logger.log(self.timestep, info)

        self.policy.train()
        while True:
            train_info = runner_train.run(buffer_size=buffer_size, buffer=self.buffer)
            self.timestep += train_info['timestep']
            loss_info = self.update()
            update_dict(info, loss_info, 'loss/')

            if self.scheduler:
                self.scheduler.step()
                train_info['lr'] = self.scheduler.get_last_lr()

            update_dict(info, train_info, 'train/')
            self.iteration += 1
            done = self.timestep >= timestep
            if test_interval and (self.iteration % test_interval == 0 or done):
                runner_test.reset()
                self.policy.eval()
                test_info = runner_test.run(episode_num=episode_test)
                self.policy.train()
                update_dict(info, test_info, 'test/')
            if log_interval and (self.iteration % log_interval == 0 or done):
                logger.log(self.timestep, info)
            if done:
                break
