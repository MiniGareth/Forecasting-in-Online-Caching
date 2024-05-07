import random

import numpy as np

from forecasters.RecommenderForecaster import RecommenderForecaster
from recommender.kNNRecommender import kNNRecommender

def convert_to_one_hot_vector(n, L):
    x = np.zeros(L)
    x[n] = 1
    return x

if __name__ == "__main__":
    library_size = 100
    history_num = 50
    req_num = 500
    forecaster = RecommenderForecaster(kNNRecommender(1), library_size)

    history = [random.randint(0,library_size -1) for i in range(history_num)]
    requests = [random.randint(0,library_size - 1) for i in range(req_num)]

    history_vec = [convert_to_one_hot_vector(i, library_size) for i in history]
    requests_vec = [convert_to_one_hot_vector(i, library_size) for i in requests]
    forecaster.update(history_vec)
    for req in requests_vec:
        print(forecaster.predict())
        history_vec.append(req)
        forecaster.update(history_vec)
