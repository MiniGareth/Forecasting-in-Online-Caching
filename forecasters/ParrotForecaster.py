import random

import numpy as np

from forecasters.Forecaster import Forecaster

class ParrotForecaster(Forecaster):
    """
    The ParrotForecaster is a forecaster that predicts the future exactly equal to the list of requests it receives.
    For future beyond the given request list, the latest request on the list is returned.
    """
    def __init__(self, future_requests: list, horizon=1, accuracy=1):
        super().__init__(horizon)
        self.request_list = future_requests
        self.position = 0   # Pointer of which request in the request list is the next prediction
        self.accuracy = accuracy
    def predict(self) -> np.ndarray:
        """
        Predict the future request by returning the request at index self.position.
        If self.position pointer is beyond the available requests, return the last request
        :return:
        """
        if self.position >= len(self.request_list):
            self.position = len(self.request_list) - 1
        prediction = self.request_list[self.position]
        self.position += 1

        # There is a chance to return the correct prediction and a chance to return a random file
        if random.random() < self.accuracy:
            return prediction
        else:
            v = np.zeros(len(prediction))
            v[random.randint(0, len(prediction) - 1)] = 1
            return v

    def update(self, history: list[np.ndarray]) -> None:
        if len(history) == 0:
            return

        for i in range(min(len(history), len(self.request_list))):
            if (history[i] != self.request_list[i]).all():
                raise ValueError(f"The history of requests {history} does not match the future requests {self.request_list} in this Forecaster")

        # Update the position
        self.position = i + 1