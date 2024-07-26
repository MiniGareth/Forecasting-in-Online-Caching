import random
from math import isclose
from pathlib import Path
import numpy as np
import torch
from darts import TimeSeries
from darts.models import TCNModel
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sys

root_dir = Path(".").resolve()
sys.path.append(str(root_dir.absolute()))

import utils.utils as utils
from forecasters.ArimaForecaster import ArimaForecaster
from forecasters.DESForecaster import DESForecaster
from forecasters.MFRForecaster import MFRForecaster
from forecasters.NaiveForecaster import NaiveForecaster
from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RandomForecaster import RandomForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster
from forecasters.TCNForecaster import TCNForecaster

def convert_to_one_hot_vector(n, L):
    x = np.zeros(L)
    x[n] = 1
    return x


def test_forecaster_score(forecaster, requests_vec) -> tuple[int, list[np.ndarray]]:
    predictions_vec = []
    for i, req in enumerate(requests_vec):
        predictions_vec.append(forecaster.predict())
        forecaster.update(req)
        # print(f"{i}, ", end="")
        # print(predictions_vec[-1])
    # print()

    assert len(predictions_vec) == len(requests_vec)
    # Calculate Score: Nr. of cache hits
    score = 0
    for i in range(len(predictions_vec)):
        score += np.dot(predictions_vec[i], requests_vec[i])

    return score, predictions_vec


def test_random(library_size, history_num, req_num):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]

    # Convert to vectors
    requests_vec = utils.convert_to_vectors(requests, library_size)
    history_vec = utils.convert_to_vectors(history, library_size)

    forecaster = RandomForecaster(library_size)

    score, predictions = test_forecaster_score(forecaster, requests_vec)
    print()
    print("================================================================================")
    print("Random Forecaster:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num * 100:.2f}")


def test_random_movielens():
    # Draw requests from uniform distribution
    train_requests, train_library = utils.get_movie_lens_train(str(root_dir / "ml-latest-small" / "ml-latest-small"))
    test_requests, test_library = utils.get_movie_lens_test(str(root_dir / "ml-latest-small" / "ml-latest-small"))

    library_size = train_library
    assert train_library == test_library
    history_num = len(train_requests)
    req_num = len(test_requests)
    # Convert to vectors
    test_vec = utils.convert_to_vectors(test_requests, library_size)
    train_vec = utils.convert_to_vectors(train_requests, library_size)

    forecaster = RandomForecaster(library_size)

    score, predictions = test_forecaster_score(forecaster, test_vec)
    print()
    print("================================================================================")
    print("Random Forecaster on MovieLens:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num * 100:.2f}")


def test_recommender_uniform(library_size, history_num, req_num):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]
    requests_vec = utils.convert_to_vectors(requests, library_size)
    history_vec = utils.convert_to_vectors(history, library_size)

    forecaster = RecommenderForecaster(library_size, history_vec, horizon=1)

    score, predictions = test_forecaster_score(forecaster, requests_vec)
    print()
    print("================================================================================")
    print("KNN Recommender:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num * 100:.2f}")
    assert isclose(score / req_num, 1 / library_size, abs_tol=0.1)


def test_parrot(library_size, history_num, req_num, accuracy: float = 1):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]

    # Convert to vectors
    requests_vec = utils.convert_to_vectors(requests, library_size)
    history_vec = utils.convert_to_vectors(history, library_size)

    forecaster = ParrotForecaster(requests_vec, accuracy=accuracy)

    score, predictions = test_forecaster_score(forecaster, requests_vec)
    print()
    print("================================================================================")
    print("Parrot Forecaster:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num * 100:.2f}")
    assert isclose(score / req_num, accuracy, abs_tol=0.1)


def test_naive(library_size, history_num, req_num):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]
    requests2 = []
    for req in requests:
        requests2.append(req)
        requests2.append(req)

    # Convert to vectors

    requests_vec = utils.convert_to_vectors(requests, library_size)
    history_vec = utils.convert_to_vectors(history, library_size)
    requests_vec2 = utils.convert_to_vectors(requests2, library_size)
    history_vec2 = utils.convert_to_vectors(history, library_size)

    forecaster = NaiveForecaster(library_size)

    score, predictions = test_forecaster_score(forecaster, requests_vec)
    score2, predictions = test_forecaster_score(forecaster, requests_vec2)

    print()
    print("================================================================================")
    print("Naive Forecaster:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num:.2f}")
    print(f"Utility 2: {score2 / req_num * 2:.2f}")


def test_naive_movielens():
    # Draw requests from uniform distribution
    train_requests, train_library = utils.get_movie_lens_train(str(root_dir / "ml-latest-small" / "ml-latest-small"))
    test_requests, test_library = utils.get_movie_lens_test(str(root_dir / "ml-latest-small" / "ml-latest-small"))

    library_size = train_library
    assert train_library == test_library
    history_num = len(train_requests)
    req_num = len(test_requests)
    # Convert to vectors
    test_vec = utils.convert_to_vectors(test_requests, library_size)
    train_vec = utils.convert_to_vectors(train_requests, library_size)

    forecaster = NaiveForecaster(library_size)

    score, predictions = test_forecaster_score(forecaster, test_vec)
    print()
    print("================================================================================")
    print("Naive Forecaster on MovieLens:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num:.2f}")


def test_arima_movielens():
    # Draw requests from uniform distribution
    train_requests, train_library = utils.get_movie_lens_train(str(root_dir / "ml-latest-small" / "ml-latest-small"))
    test_requests, test_library = utils.get_movie_lens_test(str(root_dir / "ml-latest-small" / "ml-latest-small"))

    library_size = train_library
    assert train_library == test_library
    history_num = len(train_requests)
    req_num = len(test_requests)
    # Convert to vectors
    test_vec = utils.convert_to_vectors(test_requests, library_size)
    train_vec = utils.convert_to_vectors(train_requests, library_size)

    forecaster = ArimaForecaster(train_vec, library_size, frequency=req_num)

    score, predictions = test_forecaster_score(forecaster, test_vec,)
    print()
    print("================================================================================")
    print("Arima Forecaster on MovieLens:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num:.2f}")


def generate_loaders(X, val_X, y, val_y, batch_size=1):
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.FloatTensor)
    print(X.size(), y.size())
    train_loader = DataLoader(TensorDataset(X, y),
                              batch_size=batch_size, shuffle=True)

    if val_X is not None:
        val_X = torch.from_numpy(val_X).type(torch.FloatTensor)
        val_y = torch.from_numpy(val_y).type(torch.FloatTensor)
        val_loader = DataLoader(TensorDataset(val_X, val_y),
                                batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_mfr_forecaster(library_size=100, requests_num=200, history_num=1000):
    requests = utils.get_requests_from_distribution("zipf", library_size, requests_num)
    history = utils.get_requests_from_distribution("zipf", library_size, history_num)
    # Shuffle indices
    shuffled_ids = np.arange(library_size)
    np.random.shuffle(shuffled_ids)
    mapper = dict(zip(np.arange(library_size), shuffled_ids))
    requests_vec = utils.convert_to_vectors(requests, library_size, idx_mapper=mapper)
    history_vec = utils.convert_to_vectors(history, library_size, idx_mapper=mapper)

    forecaster = MFRForecaster(history_vec)

    score, predictions = test_forecaster_score(forecaster, requests_vec)

    # Since this is zipf the most frequent request should be the original id 0 or new id mapper[0]
    v = np.zeros(library_size)
    v[mapper[0]] = 1
    assert (np.array(predictions) == v).all()
    print("================================================================================")
    print("Most Frequently Requested Forecaster:")
    print(f"Library size: {library_size}, Request num: {requests_num}, History num: {history_num}")
    print(f"Average Utility: {score / requests_num:.5f}")

def test_tcn_forecaster(library_size=100):
    train, val, test = utils.get_movie_lens_split(str(root_dir / "ml-latest-small" / "ml-latest-small"), library_limit=library_size)


    train_vec = utils.convert_to_vectors(train, library_size)
    val_vec = utils.convert_to_vectors(val, library_size)
    test_vec = utils.convert_to_vectors(test, library_size)

    forecaster = TCNForecaster(model_path=str(root_dir / "tcn" / "tcn_best"), history=np.concatenate((train_vec, val_vec)))

    score, predictions = test_forecaster_score(forecaster, test_vec)

    print("================================================================================")
    print("TCN Forecaster:")
    print(f"Library size: {library_size}, Request num: {len(test)}, History num: {len(train) + len(val)}")
    print(f"Average Utility: {score / len(test):.5f}")

def test_des_movielens(library_size=100):
    train, val, test = utils.get_movie_lens_split(str(root_dir / "ml-latest-small" / "ml-latest-small"), library_limit=library_size)

    train_vec = utils.convert_to_vectors(train, library_size)
    val_vec = utils.convert_to_vectors(val, library_size)
    test_vec = utils.convert_to_vectors(test, library_size)

    forecaster = DESForecaster(history=np.concatenate((train_vec, val_vec)), horizon=1200, one_hot=True)

    score, predictions = test_forecaster_score(forecaster, test_vec)

    print("================================================================================")
    print("Double Exponential Smoothing  Forecaster:")
    print(f"Library size: {library_size}, Request num: {len(test)}, History num: {len(train) + len(val)}")
    print(f"Average Utility: {score / len(test):.5f}")


if __name__ == "__main__":
    test_recommender_uniform(100, 50, 500)
    test_parrot(100, 50, 500, accuracy=1)
    test_parrot(100, 50, 500, accuracy=0.8)
    test_parrot(100, 50, 500, accuracy=0.5)
    test_random(100, 50, 500)
    test_random_movielens()
    test_naive(100, 50, 500)
    test_naive_movielens()
    test_arima_movielens()
    test_mfr_forecaster()
    test_tcn_forecaster()
    test_des_movielens(library_size=100)
    pass
