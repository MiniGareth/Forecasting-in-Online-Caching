import random
from math import isclose

import numpy as np

from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster
from recommender.kNNRecommender import kNNRecommender

def convert_to_one_hot_vector(n, L):
    x = np.zeros(L)
    x[n] = 1
    return x

def test_forecaster_distribution(forecaster, requests, history, library_size, history_num, req_num) -> float:
    predictions_vec = []
    # Convert to vectors
    history_vec = [convert_to_one_hot_vector(i, library_size) for i in history]
    requests_vec = [convert_to_one_hot_vector(i, library_size) for i in requests]
    forecaster.update(history_vec)
    for req in requests_vec:
        predictions_vec.append(forecaster.predict())
        # print(predictions[-1])
        history_vec.append(req)
        forecaster.update(history_vec)

    assert len(predictions_vec) == len(requests_vec)
    assert len(history_vec) == history_num + len(predictions_vec)
    # Calculate Score: Nr. of cache hits
    score = 0
    for i in range(len(predictions_vec)):
        score += np.dot(predictions_vec[i], requests_vec[i])

    return score

def test_recommender_uniform(library_size, history_num, req_num):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]
    forecaster = RecommenderForecaster(kNNRecommender(10), library_size, horizon=1)

    score = test_forecaster_distribution(forecaster, requests, history, library_size, history_num, req_num)

    print("================================================================================")
    print("KNN Recommender:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Accuracy: {score / req_num * 100:.2f}%")
    assert isclose(score / req_num, 1 / library_size, abs_tol=0.1)


def test_parrot(library_size, history_num, req_num, accuracy:float =1):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]

    # Convert to vectors
    history_vec = [convert_to_one_hot_vector(i, library_size) for i in history]
    requests_vec = [convert_to_one_hot_vector(i, library_size) for i in requests]
    forecaster = ParrotForecaster(history_vec + requests_vec, accuracy=accuracy)

    score = test_forecaster_distribution(forecaster, requests, history, library_size, history_num, req_num)

    print("================================================================================")
    print("Parrot Forecaster:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Accuracy: {score / req_num * 100:.2f}%")
    assert isclose(score / req_num, accuracy, abs_tol=0.1)



if __name__ == "__main__":
    test_recommender_uniform(100, 50, 500)
    test_parrot(100, 50, 500, accuracy=1)
    test_parrot(100, 50, 500, accuracy=0.8)
    test_parrot(100, 50, 500, accuracy=0.5)
