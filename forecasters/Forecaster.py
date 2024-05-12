from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    def __init__(self, horizon):
        self.horizon = horizon

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, history):
        pass