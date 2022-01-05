from typing import Optional, Dict


class Data:
    def __init__(
            self,
            worker_id: int
    ):
        self.worker_id = worker_id
        self.score: float = 0.0
        self.order = None
        self.exploit: Optional[int] = None
        self.iteration: Optional[int] = None
        self.x: Optional[Dict] = None
        self.y: Optional[Dict] = {}
        self.policy = None

    def __lt__(self, other):
        return self.score < other.score
