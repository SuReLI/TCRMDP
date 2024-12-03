import numpy as np


class MockAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs: np.ndarray, use_random: bool = False):
        if use_random:
            return self.action_space.sample()
        else:
            return self.action_space.low
