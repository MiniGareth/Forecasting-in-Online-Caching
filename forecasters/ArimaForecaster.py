import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from forecasters.Forecaster import Forecaster


class ArimaForecaster(Forecaster):
    def __init__(self, history_vec, library_size, frequency=50, horizon=1, p=2, d=0, q=2):
        super().__init__(horizon)
        self.library_size = library_size
        self.arima_params = (p, d, q)

        # Convert history of vectors into history of file ids.
        self.history = [np.where(v == 1)[0][0] for v in history_vec]
        self.arima_model = ARIMA(self.history, order=self.arima_params)

        self.arima_results = self.arima_model.fit()
        self.prediction_nr = 0
        self.frequency = frequency
        self.forecasts = self.arima_results.forecast(frequency)

    def predict(self) -> np.ndarray:
        if self.prediction_nr >= self.frequency:
            raise ValueError("End of available predictions, please update the forecaster before calling predict again.")
        prediction = int(max(0, min(self.forecasts[self.prediction_nr], self.library_size - 1)))
        self.prediction_nr += 1
        prediction_vec = np.zeros(self.library_size)
        prediction_vec[prediction] = 1
        return prediction_vec


    def update(self, req):
        if self.prediction_nr >= self.frequency:
            # Convert history of vectors into history of file ids
            self.history.append(np.where(req == 1)[0][0])
            self.arima_model = ARIMA(self.history, order=self.arima_params)
            self.arima_results = self.arima_model.fit()
            self.forecasts = self.arima_results.forecast(self.frequency)
            self.prediction_nr = 0