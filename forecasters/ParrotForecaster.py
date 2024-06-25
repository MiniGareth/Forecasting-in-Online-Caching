import random

import numpy as np

import utils
from forecasters.Forecaster import Forecaster

class ParrotForecaster(Forecaster):
    """
    The ParrotForecaster is a forecaster that predicts the future exactly equal to the list of requests it receives.
    For future beyond the given request list, the latest request on the list is returned.
    """
    def __init__(self, future_request_vectors: np.ndarray, start_position=0, horizon=1, accuracy=1, seed=None):
        super().__init__(horizon)
        self.request_list = list(future_request_vectors)     #Request list of vectors
        self.position = start_position   # Pointer of which request in the request list is the next prediction
        self.accuracy = accuracy

        if seed is not None:
            np.random.seed(seed)

    def predict(self) -> np.ndarray:
        """
        Predict the future request by returning the request at index self.position.
        If self.position pointer is beyond the available requests, return the last request
        :return:
        """
        if self.position >= len(self.request_list):
            self.position = len(self.request_list) - 1
        prediction = self.request_list[self.position]

        # There is a chance to return the correct prediction and a chance to return a random file
        if np.random.random() < self.accuracy:
            return prediction
        else:
            v = np.zeros(len(prediction))
            v[np.random.randint(0, len(prediction) - 1)] = 1
            return v

    def update(self, latest_req: np.ndarray):
        if (self.request_list[self.position] != latest_req).all():
            raise ValueError(f"The latest request {latest_req} does not match the request known {self.request_list[self.position]} in this Forecaster")

        self.position += 1
