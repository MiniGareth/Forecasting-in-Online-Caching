from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """
    Forecasters predict the future requests.
    They take in vectors and output vectors.
    """
    def __init__(self, horizon):
        self.horizon = horizon

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, latest_req:np.ndarray):
        pass