from typing import Union

import numpy as np


class RunningMeanStd:
    def __init__(self, mean: Union[float, np.ndarray], var: Union[float, np.ndarray]):
        self.mean = mean
        self.var = var
        self.n = 0
        self.eps = 1e-8

    def load(self, o):
        if o is None:
            return
        self.mean = o.mean
        self.var = o.var
        self.n = o.n
        self.eps = o.eps

    def update(self, x: np.ndarray):
        n = x.shape[0]
        n_new = self.n + n
        mean = x.mean(axis=0)
        delta = mean - self.mean
        mean_new = self.mean + delta * n / n_new
        var = x.var(axis=0)
        m_a = self.var * self.n
        m_b = var * n
        m_2 = m_a + m_b + delta ** 2 * self.n * n / n_new
        var_new = m_2 / n_new
        self.mean = mean_new
        self.var = var_new
        self.n = n_new
