from multiprocessing.connection import Connection


class DistReplayBuffer:
    def __init__(self, remote: Connection):
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
