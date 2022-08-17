import numpy as np
from pbrl.algorithms.dqn.buffer import ReplayBuffer
from pbrl.pbt.pbt import PBT


class CompetitivePBT(PBT):
    def __init__(self, k=8, buffer=None, **kwargs):
        super(CompetitivePBT, self).__init__(**kwargs)
        self.k = k
        self.buffer = buffer

    def expectation(self, a, b):
        return 1 / (1 + 10 ** ((self.datas[b].score - self.datas[a].score) / 400))

    def run(self):
        while True:
            cmd = self.recv()
            if cmd == 'syn':
                deltas = np.zeros(self.worker_num)
                for worker_id in range(self.worker_num):
                    iteration, x, eval_results = self.objs[worker_id]
                    data = self.datas[worker_id]
                    data.iteration = iteration
                    data.x = x
                    for a, b, res in eval_results:
                        ea = self.expectation(a, b)
                        delta = self.k * (res - ea)
                        deltas[a] += delta
                        deltas[b] -= delta
                pop = []
                for worker_id in range(self.worker_num):
                    x = self.datas[worker_id].x
                    pop.append(
                        {k: x[k] for k in ('actor', 'rms_obs')}
                    )
                for worker_id in range(self.worker_num):
                    self.datas[worker_id].score += deltas[worker_id]
                    self.remotes[worker_id].send((self.datas[worker_id].score, pop))
            elif cmd == 'exploit':
                if self.exploit:
                    self.select()
                self.send()
            elif cmd == 'append':
                increment = 0
                for sample in self.objs:
                    increment += self.buffer.append(*sample)
                for worker_id in range(self.worker_num):
                    self.remotes[worker_id].send(increment)
            elif cmd == 'sample':
                for worker_id in range(self.worker_num):
                    batch_size = self.objs[worker_id]
                    self.remotes[worker_id].send(self.buffer.sample(batch_size))
            elif cmd == 'close':
                break
