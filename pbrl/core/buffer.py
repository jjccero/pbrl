from typing import Optional

import numpy as np


class PGBuffer:
    def __init__(
            self,
            mini_batch_size: int,
            chunk_len: Optional[int]
    ):
        self.mini_batch_size = mini_batch_size
        self.chunk_len = chunk_len
        self.step = 0
        self.observations = []
        self.actions = []
        self.log_probs_old = []
        self.rewards = []
        self.dones = []

        self.observations_next: Optional[np.ndarray] = None
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def append(
            self,
            observations: np.ndarray,
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
        self.step += 1

    def generator(self):
        env_num = self.observations_next.shape[0]
        if self.chunk_len:
            assert self.step % self.chunk_len == 0
            assert self.mini_batch_size % self.chunk_len == 0
            chunk_size = self.step // self.chunk_len
            batch_size = chunk_size * env_num
            mini_batch_size = self.mini_batch_size // self.chunk_len

            # process RNN chunk
            def to_rnn_chunk(arr):
                # arr's shape is (chunk_size * chunk_len, env_num, ...)
                # because chunk_size * chunk_len = step_num
                # reshape to (env_num, chunk_size * chunk_len, ...)
                arr = np.stack(arr, axis=1)
                # reshape to (env_num, chunk_size, chunk_len, ...)
                # len(arr.shape) may more than 2, use * operator
                arr = arr.reshape((env_num, chunk_size, self.chunk_len, *arr.shape[2:]))
                # reshape to (batch_size, chunk_len, ...)
                # because batch_size = env_num * chunk_size
                arr = np.concatenate(arr)
                return arr

            observations, actions, advantages, log_probs_old, returns, dones = map(
                to_rnn_chunk,
                (self.observations, self.actions, self.advantages, self.log_probs_old, self.returns, self.dones)
            )
        else:
            batch_size = self.step * env_num
            mini_batch_size = self.mini_batch_size
            observations, actions, advantages, log_probs_old, returns = map(
                np.concatenate,
                (self.observations, self.actions, self.advantages, self.log_probs_old, self.returns)
            )
            dones = None

        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        start = 0
        while start < batch_size:
            if start + 2 * mini_batch_size <= batch_size:
                index = indices[start:start + mini_batch_size]
                start += mini_batch_size
            else:
                index = indices[start:]
                start = batch_size
            batch_rnn = (dones[index],) if self.chunk_len else None
            yield (arr[index] for arr in (observations, actions, advantages, log_probs_old, returns)), batch_rnn

    def clear(self):
        self.step = 0
        self.observations.clear()
        self.actions.clear()
        self.log_probs_old.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
