import numpy as np

from forecasters.Forecaster import Forecaster

class MFRForecaster(Forecaster):
    """
    Most Frequently Requested Forecaster returns a prediction based on what was most frequently requested.
    """

    def __init__(self, history, horizon: int = None):
        super().__init__(horizon)
        self.history = list(history)
        self.library_size = len(history[0])

        self.frequency_dict = dict(zip(np.arange(self.library_size), np.zeros(self.library_size)))
        for v in history:
            v_id = np.where(v == 1)[0][0]
            self.frequency_dict[v_id] += 1

    def predict(self) -> np.ndarray:
        # Sort key value pairs based on frequency count (value)
        sorted_freq = sorted(self.frequency_dict.items(), key=lambda x: x[1], reverse=True)
        v = np.zeros(self.library_size)
        v[int(sorted_freq[0][0])] = 1
        return v

    def update(self, latest_req: np.ndarray):
        self.history.append(latest_req)
        # If horizon is not specified, use whole history to count frequency
        if self.horizon is None:
            v_id = np.where(latest_req == 1)[0][0]
            self.frequency_dict[v_id] += 1

        else:
            # Reset frequency counts to zero
            for key in self.frequency_dict.keys():
                self.frequency_dict[key] = 0
            # Count frequency of each file request based on horizon
            for v in self.history[-self.horizon:]:
                v_id = np.where(v == 1)[0][0]
                self.frequency_dict[v_id] += 1
