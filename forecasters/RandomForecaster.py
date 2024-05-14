import random

import numpy as np

from forecasters.Forecaster import Forecaster


class RandomForecaster(Forecaster):
    """
    This class represents a forecaster that does nothing except predict the first file
    """
    def __init__(self, cache_size, library_size, horizon=1):
        super().__init__(horizon)
        self.cache_size = cache_size
        self.library_size = library_size

    def predict(self):
        idx = random.randint(0, self.library_size - 1)
        return np.array([1 if i == idx else 0 for i in range(self.library_size)])

    def update(self, history):
        pass