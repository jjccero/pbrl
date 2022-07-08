import numpy as np

from pbrl.common.map import map_space, merge_map


def normalize(data, rms):
    x, mean, var = data
    if rms.reduce_mean:
        x = x - mean
    y = x / np.sqrt(var + rms.eps)
    if rms.clip is not None:
        y = np.clip(y, -rms.clip, rms.clip)
    return y


def update(data, rms):
    x, mean, var = data
    n = x.shape[0]
    n_new = rms.n + n
    local_mean = x.mean(axis=0)
    delta = local_mean - mean
    mean_new = mean + delta * n / n_new
    local_var = x.var(axis=0)
    m_a = var * rms.n
    m_b = local_var * n
    m_2 = m_a + m_b + delta ** 2 * rms.n * n / n_new
    var_new = m_2 / n_new

    mean[:] = mean_new
    var[:] = var_new
    rms.n_new = n_new


class RunningMeanStd:
    def __init__(self, space, clip, reduce_mean):
        if space is None:
            self.mean = np.zeros(1, np.float64)
            self.var = np.ones(1, np.float64)
        else:
            self.mean = map_space(lambda x: np.zeros(x.shape, np.float64), space)
            self.var = map_space(lambda x: np.ones(x.shape, np.float64), space)
        self.n = 0
        self.n_new = 0
        self.eps = 1e-8
        self.clip = clip
        self.reduce_mean = reduce_mean

    def load(self, o):
        if o is None:
            return
        self.mean = o.mean
        self.var = o.var
        self.n = o.n
        self.eps = o.eps
        self.clip = clip
        self.reduce_mean = reduce_mean

    def update(self, x):
        merge_map(update, (x, self.mean, self.var), rms=self)
        self.n = self.n_new

    def normalize(self, x):
        return merge_map(normalize, (x, self.mean, self.var), rms=self)
