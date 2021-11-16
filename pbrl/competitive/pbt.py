import logging

import numpy as np

from pbrl.pbt.pbt import PBT


class CompetitivePBT(PBT):
    def __init__(self, k=8, **kwargs):
        super(CompetitivePBT, self).__init__(**kwargs)
        self.k = k

    @staticmethod
    def _expectation(score_a, score_b):
        return 1 / (1 + 10 ** ((score_b - score_a) / 400))

    def eval(self):
        while True:
            ready = False
            for remote, data in zip(self.remotes, self.datas):
                iteration, x = remote.recv()
                data.iteration = iteration
                data.x = x
            # worker start evaluating
            for remote in self.remotes:
                remote.send(True)
            # receive evaluation result
            deltas = np.zeros(self.worker_num)
            rewards = [[] for _ in range(self.worker_num)]
            for remote in self.remotes:
                ready, eval_infos = remote.recv()
                logging.info(eval_infos)
                for a, b, episode, win, lose, episode_rewards in eval_infos:
                    rewards_a, rewards_b = episode_rewards
                    rewards[a] += rewards_a
                    rewards[b] += rewards_b
                    tie = episode - win - lose
                    data_a, data_b = self.datas[a], self.datas[b]
                    ea = CompetitivePBT._expectation(data_a.score, data_b.score)
                    delta = self.k * (win + 0.5 * tie - episode * ea)
                    deltas[a] += delta
                    deltas[b] -= delta
            # update elo
            for i in range(self.worker_num):
                self.datas[i].score += deltas[i]
                print(i, self.datas[i].iteration, self.datas[i].score, deltas[i], np.mean(rewards[i]))
            if ready:
                break
            for remote, data in zip(self.remotes, self.datas):
                remote.send((False, data.score, None))

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
                # condition 2:
                if data.score + 20.0 < data_parent.score:
                    self.exploits[worker_id] = True
                    data.x = data_parent.x.copy()
                    print('{}->{}'.format(worker_id, parent_worker_id))
