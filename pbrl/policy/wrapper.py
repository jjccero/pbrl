import numpy as np


class TanhWrapper:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        return 0.5 * (self.high - self.low) * np.tanh(x) + 0.5 * (self.low + self.high)


class ClipWrapper:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        return 0.5 * (self.high - self.low) * np.clip(x, -1.0, 1.0) + 0.5 * (self.low + self.high)
