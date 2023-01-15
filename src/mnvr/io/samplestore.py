import numpy as np


class StoredSample:
    def __init__(self, actions, states, penalty:float) -> None:
        self.actions = np.array(actions)
        self.states = np.array(states)
        self.penalty = penalty