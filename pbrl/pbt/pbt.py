import logging
import multiprocessing
from multiprocessing.connection import Connection
from typing import List, Callable

import numpy as np

from pbrl.pbt.data import Data


class PBT:
    def __init__(
            self,
            worker_num: int,
            worker_fn: Callable,
            exploit=True,
            **kwargs
    ):
        assert worker_num > 0
        self.worker_num = worker_num
        self.remotes: List[Connection] = []
        self.ps = []
        self.closed = False
        self.datas: List[Data] = []
        ctx = multiprocessing.get_context('spawn')
        for worker_id in range(self.worker_num):
            remote, remote_worker = ctx.Pipe()
            self.remotes.append(remote)
            p = ctx.Process(
                target=worker_fn,
                args=(
                    worker_num,
                    worker_id,
                    remote_worker,
                    remote
                ),
                kwargs=kwargs,
                daemon=False
            )
            p.start()
            self.ps.append(p)
            self.datas.append(Data(worker_id))
            remote_worker.close()
        self.exploits = [False] * self.worker_num
        self.exploit = exploit
        self.rs = np.random.RandomState()
        self.state = dict()

    def eval(self):
        for remote, data in zip(self.remotes, self.datas):
            iteration, score, x = remote.recv()
            data.iteration = iteration
            data.score = score
            data.x = x

    def select(self):
        sorted_data = sorted(self.datas, reverse=True)
        top_index = round(self.worker_num * 0.2)
        for i in range(self.worker_num):
            worker_id = sorted_data[i].worker_id
            self.exploits[worker_id] = False
            # condition 1: bottom 20%
            if i + top_index >= self.worker_num:
                # top 20%
                parent_worker_id = sorted_data[self.rs.choice(top_index)].worker_id
                data_parent = self.datas[parent_worker_id]
                data = self.datas[worker_id]
                self.exploits[worker_id] = True
                data.x = data_parent.x.copy()
                logging.info('{}->{}'.format(worker_id, parent_worker_id))

    def explore(self):
        for worker_id in range(self.worker_num):
            if self.exploits[worker_id]:
                x = self.datas[worker_id].x
                if self.rs.random() > 0.5:
                    x['lr'] = x['lr'] * 1.2
                else:
                    x['lr'] = x['lr'] * 0.8

    def seed(self, seed):
        self.rs.seed(seed)

    def send(self):
        for worker_id in range(self.worker_num):
            exploit = self.exploits[worker_id]
            x = self.datas[worker_id].x
            score = self.datas[worker_id].score
            self.remotes[worker_id].send(
                (exploit, score, x if exploit else None)
            )

    def run(self):
        try:
            # after ready_timestep
            while True:
                # receive hyperparameters and weights from workers
                # evaluate new policies
                self.eval()
                if self.exploit:
                    # copy hyperparameters and weights
                    self.select()
                    # perturb hyperparameters
                    self.explore()
                # send new data to workers
                self.send()
        except KeyboardInterrupt:
            self.close()

    def close(self):
        self.closed = True
        for p in self.ps:
            p.join()

    def __del__(self):
        if not self.closed:
            self.close()
