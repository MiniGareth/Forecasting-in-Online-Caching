from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, history):
        pass