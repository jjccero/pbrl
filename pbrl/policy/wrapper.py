import numpy as np


class ActionWrapper:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        return 0.5 * (self.high - self.low) * x + 0.5 * (self.low + self.high)


class TanhWrapper(ActionWrapper):
    def __call__(self, x):
        return super(TanhWrapper, self).__call__(np.tanh(x))


class ClipWrapper(ActionWrapper):
    def __call__(self, x):
        return super(ClipWrapper, self).__call__(np.clip(x, -1.0, 1.0))
