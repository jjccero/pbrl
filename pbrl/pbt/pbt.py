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
        self.exploit = exploit
        self.random_state = np.random.RandomState()
        self.state = dict()

    def eval(self):
        for remote, data in zip(self.remotes, self.datas):
            iteration, score, x = remote.recv()
            data.iteration = iteration
            data.score = score
            data.x = x
            data.y.clear()

    def select(self):
        sorted_datas = sorted(self.datas, reverse=True)
        for i in range(self.worker_num):
            worker_id = sorted_datas[i].worker_id
            data = self.datas[worker_id]
            data.order = i
            data.exploit = None
            # worst agent
            if i == self.worker_num - 1:
                # best agent
                parent_worker_id = sorted_datas[0].worker_id
                data_parent = self.datas[parent_worker_id]
                data.exploit = parent_worker_id
                for k, v in data_parent.x.items():
                    data.y[k] = v
                logging.info('{}->{}'.format(worker_id, parent_worker_id))

    def explore(self):
        for worker_id in range(self.worker_num):
            if self.datas[worker_id].exploit is not None:
                y = self.datas[worker_id].y
                if self.random_state.random() > 0.5:
                    y['lr'] = y['lr'] * 1.2
                else:
                    y['lr'] = y['lr'] * 0.8

    def seed(self, seed):
        self.random_state.seed(seed)

    def send(self):
        for worker_id in range(self.worker_num):
            exploit = self.datas[worker_id].exploit
            y = self.datas[worker_id].y
            score = self.datas[worker_id].score
            self.remotes[worker_id].send(
                (exploit, score, y)
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
        except:
            self.close()

    def close(self):
        self.closed = True
        for p in self.ps:
            p.join()

    def __del__(self):
        if not self.closed:
            self.close()
