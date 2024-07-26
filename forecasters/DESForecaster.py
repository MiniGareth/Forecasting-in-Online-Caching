import numpy as np

from forecasters.Forecaster import Forecaster


#horizon 1200, 1300
class DESForecaster(Forecaster):
    def __init__(self, history, horizon, one_hot=False):
        self.history = list(history)
        self.library_size = len(self.history[0])
        self.horizon = horizon
        self.one_hot = one_hot

        self.alpha = None

        self.x = np.sum(history[np.maximum(-horizon, -len(history)):], axis=0)
        self.s1 = self.x
        self.s2 = self.x
        self._fit()

    def predict(self) -> np.ndarray:
        a = 2 * self.s1 - self.s2
        b = self.alpha / (1 - self.alpha) * (self.s1 - self.s2)
        # Get prediction and calculate loss
        prediction = (a + b)/np.sum(a + b)

        if self.one_hot is True:
            # In case there are multiple max numbers
            result = np.zeros(self.library_size)
            rand_idx = np.random.choice(np.where(prediction == np.max(prediction))[0])
            result[rand_idx] = 1
            assert np.sum(result) == 1
            return result

        return prediction

    def update(self, latest_req: np.ndarray):
        self.history.append(latest_req)

        self.x += latest_req
        if len(self.history) > self.horizon:
            self.x -= self.history[-self.horizon]

    def _fit(self):
        best_loss = float("inf")
        best_alpha = 0
        # Try different alphas
        for alpha in np.arange(0, 1, 0.1):
            total_loss = 0
            # x is the popularity value at time t (the sum), s1 is smoothed value 1, s2 is smoothed value 2
            x, s1, s2 = None, None, None
            # For every request in the history, calculate the popularity, try predicting and measure MSE loss
            for idx in range(0, len(self.history) - 1):
                # Case for 1st request
                if idx == 0:
                    x = self.history[0]
                    s1 = x
                    s2 = x
                else:
                    x = np.sum(self.history[np.maximum(0, idx - self.horizon): idx])
                    s1 = alpha * x + (1 - alpha) * s1
                    s2 = alpha * s1 + (1 - alpha) * s2
                # The two variables used for prediction
                a = 2 * s1 - s2
                b = alpha / (1 - alpha) * (s1 - s2)
                # Get prediction and calculate loss
                popularity_pred = a + b
                total_loss += np.square(popularity_pred - x)

            # Replace best_loss and best_alpha if better
            total_loss /= len(self.history)
            total_loss = np.mean(total_loss)
            if total_loss < best_loss:
                best_loss = total_loss
                best_alpha = alpha

        self.alpha = best_alpha
