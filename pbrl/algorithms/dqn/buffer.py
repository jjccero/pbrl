import numpy as np

from pbrl.common.map import merge_map


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.data = np.zeros(buffer_size, object)
        self.ptr = 0
        self.len = 0

    def append(
            self,
            observations,
            actions: np.ndarray,
            observations_next,
            rewards: np.ndarray,
            dones: np.ndarray
    ):
        env_num = rewards.shape[0]
        for i in range(env_num):
            index = (self.ptr + i) % self.buffer_size
            self.data[index] = (
                observations[i],
                actions[i],
                observations_next[i],
                rewards[i],
                dones[i]
            )
        self.ptr = (self.ptr + env_num) % self.buffer_size
        self.len = min(self.len + env_num, self.buffer_size)
        return env_num

    def sample(self, batch_size: int):
        indices = np.random.randint(self.len, size=batch_size)
        # sampling with replacement may take more time
        # indices = np.random.choice(self.len, size=batch_size, replace=False)
        return merge_map(np.asarray, self.data[indices])

    def clear(self):
        self.ptr = 0
        self.len = 0
