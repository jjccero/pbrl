from multiprocessing.connection import Connection

from pbrl.algorithms.dqn.buffer import ReplayBuffer


class DistReplayBuffer(ReplayBuffer):
    def __init__(self, remote: Connection):
        super(DistReplayBuffer, self).__init__(buffer_size=0)
        self.remote = remote

    def append(
            self,
            *args
    ):
        # send samples to server
        self.remote.send(('append', args))
        # return increment
        return self.remote.recv()

    def sample(self, batch_size: int):
        # receive samples from server
        self.remote.send(('sample', batch_size))
        return self.remote.recv()
