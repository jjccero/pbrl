from typing import Optional

import numpy as np


class PGBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs_old = []
        self.rewards = []
        self.dones = []

        self.observations_next = None
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def append(
            self,
            observations,
            actions: np.ndarray,
            log_probs_old: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray
    ):
        self.observations.append(observations)
        self.actions.append(actions)
        self.log_probs_old.append(log_probs_old)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def generator(self, batch_size: int, chunk_len: int, ks, shuffle=True):
        env_num = self.observations_next.shape[0]
        step_num = len(self.rewards)

        if chunk_len:
            assert step_num % chunk_len == 0
            assert batch_size % chunk_len == 0
            chunk_size = step_num // chunk_len
            buffer_size = chunk_size * env_num
            batch_size = batch_size // chunk_len

            def map_f(arr):
                # arr's shape is (step_num, env_num, ...)
                # because chunk_size * chunk_len = step_num
                # reshape to (env_num, chunk_size * chunk_len, ...)
                arr = np.stack(arr, axis=1)
                # len(arr.shape) may more than 2, use * operator
                arr = arr.reshape((env_num, chunk_size, chunk_len, *arr.shape[2:]))
                # reshape to (batch_size, chunk_len, ...)
                # because batch_size = env_num * chunk_size
                arr = np.concatenate(arr)
                return arr

        else:
            buffer_size = step_num * env_num
            map_f = np.concatenate

        batch = {key: map_f(self.__getattribute__(key)) for key in ks}
        indices = np.arange(buffer_size)
        if shuffle:
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
        self.actions.clear()
        self.log_probs_old.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
