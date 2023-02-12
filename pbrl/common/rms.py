import gym.spaces
import numpy as np

from pbrl.common.map import merge_map


class RunningMeanStd:
    def __init__(self, space, clip, reduce_mean):
        if space is None:
            self.mean = np.zeros(1, np.float64)
            self.std = np.ones(1, np.float64)
        else:
            self.mean = map_space(lambda x: np.zeros(x.shape, np.float64), space)
            self.std = map_space(lambda x: np.ones(x.shape, np.float64), space)
        self.n = 0
        self.n_new = 0
        self.eps = 1e-8
        self.clip = clip
        self.reduce_mean = reduce_mean

    def load(self, o):
        if o is None:
            return
        self.mean = o.mean
        self.std = o.std
        self.n = o.n
        self.eps = o.eps
        self.clip = o.clip
        self.reduce_mean = o.reduce_mean

    def update(self, x):
        merge_map(update, (x, self.mean, self.std), rms=self)
        self.n = self.n_new

    def extend(self, o):
        merge_map(extend, (self.mean, self.std, o.mean, o.std), rms1=self, rms2=o)
        self.n = self.n_new

    def normalize(self, x):
        return merge_map(normalize, (x, self.mean, self.std), rms=self)


def normalize(data, rms):
    x, mean, std = data
    if rms.reduce_mean:
        x = x - mean
    y = x / (std + rms.eps)
    if rms.clip is not None:
        y = np.clip(y, -rms.clip, rms.clip)
    return y


def update(data, rms):
    x, mean, std = data
    n = x.shape[0]
    n_new = rms.n + n
    local_mean = x.mean(axis=0)
    delta = local_mean - mean
    mean_new = mean + delta * n / n_new
    local_var = x.var(axis=0)
    m_a = std ** 2 * rms.n
    m_b = local_var * n
    m_2 = m_a + m_b + delta ** 2 * rms.n * n / n_new
    var_new = m_2 / n_new

    mean[:] = mean_new
    std[:] = np.sqrt(var_new)
    rms.n_new = n_new


def extend(data, rms1, rms2):
    mean1, std1, mean2, std2 = data
    n_new = rms1.n + rms2.n
    delta = mean2 - mean1
    mean_new = mean1 + delta * rms2.n / n_new
    m_a = std1 ** 2 * rms1.n
    m_b = std2 ** 2 * rms2.n
    m_2 = m_a + m_b + delta ** 2 * rms1.n * rms2.n / n_new
    var_new = m_2 / n_new

    mean1[:] = mean_new
    std1[:] = np.sqrt(var_new)
    rms1.n_new = n_new


def map_space(f, x):
    if isinstance(x, gym.spaces.Dict):
        return {k: map_space(f, v) for k, v in x.spaces.items()}
    elif isinstance(x, gym.spaces.Tuple):
        return tuple(map_space(f, e) for e in x.spaces)
    else:
        return f(x)
