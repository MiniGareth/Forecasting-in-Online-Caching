import numpy as np

from forecasters.Forecaster import Forecaster


class RecommenderForecaster(Forecaster):
    def __init__(self, cache_size: int, library_size: int, recommender):
        self.cache_size = cache_size
        self.library_size = library_size

    def predict(self):
        return np.array([1 if i == 0 else 0 for i in range(self.library_size)])

    def update(self, history):
        pass