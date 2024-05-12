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


def test_recommender_uniform(library_size, history_num, req_num):
    forecaster = RecommenderForecaster(kNNRecommender(1), 100, horizon=1)
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]
    predictions = []
    # Convert to vectors
    history_vec = [convert_to_one_hot_vector(i, library_size) for i in history]
    requests_vec = [convert_to_one_hot_vector(i, library_size) for i in requests]
    forecaster.update(history_vec)
    for req in requests_vec:
        predictions.append(list(forecaster.predict()).index(1))
        # print(predictions[-1])
        history_vec.append(req)
        forecaster.update(history_vec)

    assert len(predictions) == len(history)
    # Calculate accuracy
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == requests[i]:
            count += 1

    print("================================================================================")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Accuracy: {count / len(predictions)}")
    assert count/len(predictions) == 1/library_size


def test_parrot(library_size, history_num, req_num, accuracy:float =1):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]
    predictions = []

    # Convert to vectors
    history_vec = [convert_to_one_hot_vector(i, library_size) for i in history]
    requests_vec = [convert_to_one_hot_vector(i, library_size) for i in requests]
    forecaster = ParrotForecaster(history_vec + requests_vec, accuracy=accuracy)
    forecaster.update(history_vec)
    for req in requests_vec:
        predictions.append(list(forecaster.predict()).index(1))
        # print(predictions[-1])
        history_vec.append(req)
        forecaster.update(history_vec)

    assert len(predictions) == len(requests)
    assert len(history_vec) == history_num + len(predictions)
    # Calculate accuracy
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == requests[i]:
            count += 1

    print("================================================================================")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Accuracy: {count / len(predictions)}")
    assert isclose(count / len(predictions), accuracy, rel_tol=0.05)

    if accuracy >= 1:
        assert predictions == requests


if __name__ == "__main__":
    # test_recommender_uniform(100, 500, 500)
    test_parrot(100, 10, 500, accuracy=1)
    test_parrot(100, 10, 500, accuracy=0.8)
    test_parrot(100, 10, 500, accuracy=0.5)
