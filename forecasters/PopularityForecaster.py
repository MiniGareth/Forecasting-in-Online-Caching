import numpy as np

from forecasters.Forecaster import Forecaster

# 1450 and 20
class PopularityForecaster(Forecaster):
    def __init__(self, history, horizon=1450, one_hot=False):
        super().__init__(horizon)
        self.history = list(history)
        self.library_size = len(history[0])
        self.one_hot = one_hot


        # Basically an array with indices that keep track of how many times the file has been requested in a particular horizon
        self.counter = np.zeros(self.library_size)

        # Initialize frequency dict based on history
        # Only look back as far into the past as the horizon specified.
        for i in np.arange(np.maximum(len(self.history) - horizon, 0), len(self.history)):
            req = self.history[i]
            idx = np.where(req == 1)[0][0]
            self.counter[idx] += 1

    def predict(self) -> np.ndarray:
        prediction = self.counter / np.sum(self.counter)

        if self.one_hot is True:
            return (prediction == np.max(prediction)).astype(np.float_)
        return prediction

    def update(self, latest_req: np.ndarray):
        self.history.append(latest_req)
        begin_idx = np.maximum(len(self.history) - self.horizon, 0)
        end_idx = len(self.history) - 1

        earliest_req = self.history[begin_idx - 1]
        self.counter[np.where(latest_req == 1)[0][0]] += 1
        if begin_idx > 0:
            self.counter[np.where(earliest_req == 1)[0][0]] -= 1

        if len(self.history) >= self.horizon:
            assert np.sum(self.counter) == self.horizon
        else:
            assert np.sum(self.counter) < self.horizon
