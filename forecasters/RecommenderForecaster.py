import sys

import numpy as np
import pandas as pd

from forecasters.Forecaster import Forecaster
from recommender import Recommender
from recommender.kNNRecommender import kNNRecommender


class RecommenderForecaster(Forecaster):
    """
    This class uses a recommendation system to predict the future.
    """
    def __init__(self, library_size: int, history: np.ndarray, horizon=1, k=None, one_hot=False):
        super().__init__(horizon)
        self.k = library_size - 1 if k is None else k
        self.recommender = kNNRecommender(self.k)
        # List of vectors
        self.history = list(history)
        self.library_size = library_size
        # Train KNN
        self._train(history)
        self.one_hot = one_hot


    def predict(self) -> np.ndarray:
        """
        Returns a predicted request that is the top 1 from the top N results returned by the recommender when given
        the last request.
        If no requests have been made yet, return a prediction of the first file being requested
        :return: a vector (numpy array)
        """
        # If no requests have been made yet
        if len(np.unique(self.history)) < 2:
            result = np.zeros(self.library_size)
            result[0] = 1

        else:
            # Get top N recommendataions based on the last request made.
            req_idx = np.where(self.history[-1] == 1)[0][0]
            recommendations = self.recommender.recommend(req_idx)
            result = np.zeros(self.library_size)
            # Get the weighted average of the top N recommendations. Top recommendation is weighted highest
            for i, rec in enumerate(recommendations):
                result[int(rec)] += 1 / (i + 1)
            result = result/np.sum(result)  # Normalize the result vector

        if self.one_hot is True:
            return (np.max(result) == result).astype(np.float_)

        return result


    def update(self, latest_req:np.ndarray):
        """
        Updates the state of the forecaster based on the current history of requests.
        The history of requests is treated as a time series before being converted into a utility matrix for the
        recommender. This conversion is based on Alvaro Gomez Time Series Forecasting by Recommendation: An Empirical Analysis on Amazon Marketplace (2019)
        :param history: List of vectors (requests) that have been made.
        """
        self.history.append(latest_req)



    def _train(self, train_data:np.ndarray):
        if len(train_data) == 0:
            return

        train_scalar = [list(v).index(1) for v in train_data]
        ts0 = pd.DataFrame(np.round(train_scalar))
        ts1 = ts0.shift(self.horizon)

        # Remove first h items from ts1 and last h items from ts0 to make them the same length
        ts0 = pd.Series(ts0.head(max(0, ts0.size - self.horizon))[0])
        ts1 = pd.Series(ts1.tail(max(0, ts1.size - self.horizon))[0])
        # print(ts0, ts1)

        users = pd.Series(ts0.unique())
        items = pd.Series(ts1.unique())

        matrix = pd.crosstab(ts0, ts1, rownames=['u'], colnames=['i'])
        # print("====================================================================")
        # print(matrix)
        df = []

        for i in matrix.axes[0]:
            for j in matrix.axes[1]:
                df.append([i, j, matrix.at[i, j]])
        # This is to make sure the data frame has all users and items at least once.
        for i in range(0, self.library_size):
            if i not in matrix.axes[0] or i not in matrix.axes[1]:
                # df.append([i, i, -1*sys.maxsize])
                df.append([i, i, 0])

        # a = np.array(df).nnz / np.array(df).shape[0]
        # print(a)

        df = pd.DataFrame(df, columns=["users", "items", "rating"])

        self.recommender.train(df, users="users", items="items", ratings="rating")

