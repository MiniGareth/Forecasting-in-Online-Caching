import numpy as np

from forecasters.Forecaster import Forecaster

class ZeroForecaster(Forecaster):
    """
    This class represents a forecaster that does nothing and returns zero predictions
    """
    def __init__(self, library_size, horizon=1):
        super().__init__(horizon)
        self.library_size = library_size

    def predict(self):
        return np.zeros(self.library_size)

    def update(self, history):
        pass