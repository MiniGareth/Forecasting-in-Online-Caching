import sys

import numpy as np
import pandas as pd

from forecasters.Forecaster import Forecaster
from recommender import Recommender


class RecommenderForecaster(Forecaster):
    def __init__(self, recommender: Recommender, library_size: int, horizon=10):
        self.recommender = recommender
        self.horizon = horizon
        self.latest_history = []
        self.library_size = library_size


    def predict(self) -> np.ndarray:
        if len(np.unique(self.latest_history)) < 2:
            result = np.zeros(self.library_size)
            result[0] = 1
        else:
            recommendations = self.recommender.recommend(self.latest_history[-1])
            result = np.zeros(self.library_size)
            result[int(recommendations[0])] = 1
        return result


    def update(self, history: list):
        self.latest_history = [list(v).index(1) for v in history]
        ts0 = pd.DataFrame(np.round(self.latest_history))
        ts1 = ts0.shift(self.horizon)     # TS1 is time series shifted by a certain horizon

        # Remove first h items from ts1 and last h items from ts0 to make them the same length
        ts0 = ts0.head(max(0, ts0.size - self.horizon))
        ts1 = ts1.tail(max(0, ts1.size - self.horizon))
        # print(ts0, ts1)

        users = pd.Series(ts0[0].unique())
        items = pd.Series(ts1[0].unique())

        matrix = pd.crosstab(users, items, rownames=['u'], colnames=['i'])
        # print("====================================================================")
        # print(matrix)
        df = []

        for i in matrix.axes[0]:
            for j in matrix.axes[1]:
                df.append([i, j, matrix.at[i, j]])
        # This is to make sure the data frame has all users and items at least once.
        for i in range(0, self.library_size):
            if i not in matrix.axes[0] or i not in matrix.axes[1]:
                df.append([i, i, -1*sys.maxsize])

        # a = np.array(df).nnz / np.array(df).shape[0]
        # print(a)

        df = pd.DataFrame(df, columns=["users", "items", "rating"])

        self.recommender.train(df, users="users", items="items", ratings="rating")
