import numpy as np

from pbrl.pbt.pbt import PBT


class CompetitivePBT(PBT):
    def __init__(self, k=8, **kwargs):
        super(CompetitivePBT, self).__init__(**kwargs)
        self.k = k

    def expectation(self, a, b):
        return 1 / (1 + 10 ** ((self.datas[b].score - self.datas[a].score) / 400))

    def run(self):
        while True:
            cmd = self.recv()
            if cmd == 'up':
                deltas = np.zeros(self.worker_num)
                for worker_id in range(self.worker_num):
                    iteration, x, eval_results = self.objs[worker_id]
                    data = self.datas[worker_id]
                    data.iteration = iteration
                    data.x = x
                    for a, b, episode, win, lose in eval_results:
                        tie = episode - win - lose
                        ea = self.expectation(a, b)
                        delta = self.k * (win + 0.5 * tie - episode * ea)
                        deltas[a] += delta
                        deltas[b] -= delta
                for worker_id in range(self.worker_num):
                    self.datas[worker_id].score += deltas[worker_id]
                    self.remotes[worker_id].send(self.datas[worker_id].score)
            elif cmd == 'down':
                pop = []
                for worker_id in range(self.worker_num):
                    x = self.datas[worker_id].x
                    pop.append(
                        {k: x[v] for k in ('actor', 'rms_obs')}
                    )
                for worker_id in range(self.worker_num):
                    self.remotes[worker_id].send(pop)
            elif cmd == 'exploit':
                if self.exploit:
                    self.select()
                self.send()
            elif cmd == 'close':
                break
