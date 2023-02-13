import logging
import multiprocessing

import numpy as np

from pbrl.common import Logger
from pbrl.pbt.data import Data


class PBT:
    def __init__(
            self,
            worker_num: int,
            worker_fn: callable,
            worker_params: dict,
            log_dir=None,
            exploit=True
    ):
        self.worker_num = worker_num
        self.remotes = []
        self.processes = []
        self.closed = False
        self.datas = []
        self.objs = []
        ctx = multiprocessing.get_context('spawn')
        for worker_id in range(self.worker_num):
            remote, remote_worker = ctx.Pipe()
            self.remotes.append(remote)
            p = ctx.Process(
                target=worker_fn,
                args=(
                    worker_num,
                    worker_id,
                    log_dir,
                    remote_worker,
                ),
                kwargs=worker_params,
                daemon=False
            )
            p.start()
            self.processes.append(p)
            self.datas.append(Data(worker_id))
            remote_worker.close()
        self.exploit = exploit
        self.log_dir = log_dir
        self.logger = Logger(log_dir) if log_dir is not None else None
        self.random_state = np.random.RandomState()

    def select(self):
        sorted_datas = sorted(self.datas, reverse=True)
        for i in range(self.worker_num):
            worker_id = sorted_datas[i].worker_id
            data = self.datas[worker_id]
            data.order = i
            data.exploit = None
            data.y.clear()
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
            self.remotes[worker_id].send((exploit, y))

    def recv(self):
        self.objs.clear()
        cmd = None
        for worker_id in range(self.worker_num):
            cmd, obj = self.remotes[worker_id].recv()
            self.objs.append(obj)
        return cmd

    def close(self):
        self.closed = True
        self.datas.clear()
        self.objs.clear()
        for worker_id in range(self.worker_num):
            self.processes[worker_id].join()

    def run(self):
        while True:
            # receive hyperparameters and weights from workers
            cmd = self.recv()
            # evaluate new policies
            if cmd == 'exploit':
                if self.exploit:
                    for worker_id in range(self.worker_num):
                        (iteration, score, x) = self.objs[worker_id]
                        data = self.datas[worker_id]
                        data.iteration = iteration
                        data.score = score
                        data.x = x
                    # copy hyperparameters and weights
                    self.select()
                    # perturb hyperparameters
                    self.explore()
                # send new data to workers
                self.send()
            elif cmd == 'close':
                break

    def __del__(self):
        if not self.closed:
            self.close()
