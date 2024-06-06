import numpy as np

from forecasters.Forecaster import Forecaster


class NaiveForecaster(Forecaster):
    """
    Naive Forecaster naively predicts by predicting the last seen event/item
    """
    def __init__(self, library_size: int, horizon=1):
        super().__init__(horizon)
        # Start with a random "last seen event"
        self.last_seen_event = np.eye(library_size)[np.random.choice(library_size)]
        self.library_size = library_size

    def predict(self) -> np.ndarray:
        return self.last_seen_event

    def update(self, latest_req:np.ndarray):
        self.last_seen_event = latest_req