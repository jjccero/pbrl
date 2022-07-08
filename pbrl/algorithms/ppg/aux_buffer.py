from typing import List

import numpy as np


class AuxBuffer:
    def __init__(self):
        self.observations = []
        self.dones = []
        self.vtargs = []
        self.dists_old = []

    def append(
            self,
            observations: List,
            dones: List[np.ndarray],
            vtargs: np.ndarray
    ):
        self.observations.append(np.stack(observations))
        self.dones.append(np.stack(dones))
        self.vtargs.append(vtargs)

    def generator(self, batch_size: int, chunk_len: int, ks):
        n_pi = len(self.vtargs)
        step_num, env_num = self.vtargs[0].shape

        if chunk_len:
            chunk_size = step_num // chunk_len
            buffer_size = chunk_size * n_pi * env_num
            batch_size = batch_size // chunk_len

            # process RNN chunk
            def map_f(arr):
                # arr's shape is (n_pi, chunk_size * chunk_len, env_num, ...)
                # because chunk_size * chunk_len = step_num
                # reshape to (n_pi * env_num, chunk_size * chunk_len, ...)
                arr = np.concatenate(arr, axis=1)
                # reshape to (chunk_size * chunk_len, n_pi * env_num, ...)
                arr = arr.swapaxes(0, 1)
                # len(arr.shape) may more than 2, use * operator
                arr = arr.reshape((n_pi * env_num, chunk_size, chunk_len, *arr.shape[2:]))
                # reshape to (batch_size, chunk_len, ...)
                # because batch_size = env_num * chunk_size
                arr = np.concatenate(arr)
                return arr

        else:
            buffer_size = step_num * n_pi * env_num

            def map_f(arr):
                return np.concatenate(np.concatenate(arr, axis=1))

        batch = {key: map_f(self.__getattribute__(key)) for key in ks}

        indices = np.arange(buffer_size)
        np.random.shuffle(indices)
        start = 0
        while start < buffer_size:
            if start + 2 * batch_size <= buffer_size:
                index = indices[start:start + batch_size]
                start += batch_size
            else:
                index = indices[start:]
                start = buffer_size
            mini_batch = {k: v[index] for k, v in batch.items()}
            yield mini_batch

    def clear(self):
        self.observations.clear()
        self.dones.clear()
        self.vtargs.clear()
        self.dists_old.clear()
