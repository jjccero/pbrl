from typing import List

import numpy as np


class Bandits:
    def __init__(self, arms=List[float], random_choice=5):
        self.arms = arms
        self.arm_num = len(arms)
        self.alpha = np.ones(self.arm_num, dtype=int)
        self.beta = np.ones(self.arm_num, dtype=int)
        self.reward = -np.inf
        self.arm = 0
        self.choices = 0
        self.random_choice = random_choice

    @property
    def value(self):
        return self.arms[self.arm]

    def update(self, reward):
        self.choices += 1
        if reward > self.reward:
            self.alpha[self.arm] += 1
        else:
            self.beta[self.arm] += 1
        self.reward = reward

    def sample(self):
        if self.choices > self.random_choice:
            self.arm = np.argmax(np.random.beta(self.alpha, self.beta))
        else:
            self.arm = np.random.choice(self.arm_num)
        return self.value
