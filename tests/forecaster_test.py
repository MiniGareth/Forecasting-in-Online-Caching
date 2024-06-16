import os
import random
from math import isclose

import numpy as np
import darts
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
from scipy.ndimage import shift
from torch import nn
from torch.autograd import Variable
from torch.nn import NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor, transforms

import utils
from dataset.MovieLensDataset import MovieLensDataset
from forecasters.ArimaForecaster import ArimaForecaster
from forecasters.DESForecaster import DESForecaster
from forecasters.MFRForecaster import MFRForecaster
from forecasters.NaiveForecaster import NaiveForecaster
from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RandomForecaster import RandomForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster
from forecasters.TCNForecaster import TCNForecaster
from recommender.kNNRecommender import kNNRecommender
from tcn.models import TemporalConvNet


def convert_to_one_hot_vector(n, L):
    x = np.zeros(L)
    x[n] = 1
    return x


def test_forecaster_score(forecaster, requests_vec) -> float:
    predictions_vec = []
    for i, req in enumerate(requests_vec):
        print(f"{i}, ", end="")
        predictions_vec.append(forecaster.predict())
        print(predictions_vec[-1])
        forecaster.update(req)
    print()

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
    requests_vec, history_vec = utils.convert_to_vectors(requests, history, library_size)

    forecaster = RandomForecaster(library_size)

    score, predictions = test_forecaster_score(forecaster, requests_vec, history_vec, library_size, history_num,
                                               req_num)
    print()
    print("================================================================================")
    print("Random Forecaster:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num * 100:.2f}")


def test_random_movielens():
    # Draw requests from uniform distribution
    train_requests, train_library = utils.get_movie_lens_train("../ml-latest-small/ml-latest-small")
    test_requests, test_library = utils.get_movie_lens_test("../ml-latest-small/ml-latest-small")

    library_size = train_library
    assert train_library == test_library
    history_num = len(train_requests)
    req_num = len(test_requests)
    # Convert to vectors
    test_vec, train_vec = utils.convert_to_vectors(test_requests, train_requests, library_size)

    forecaster = RandomForecaster(library_size)

    score, predictions = test_forecaster_score(forecaster, test_vec, train_vec, library_size, history_num, req_num)
    print()
    print("================================================================================")
    print("Random Forecaster on MovieLens:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num * 100:.2f}")


def test_recommender_uniform(library_size, history_num, req_num):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]
    requests_vec, history_vec = utils.convert_to_vectors(requests, history, library_size)

    forecaster = RecommenderForecaster(kNNRecommender(10), library_size, horizon=1)

    score, predictions = test_forecaster_score(forecaster, requests_vec, history_vec, library_size, history_num,
                                               req_num)
    print()
    print("================================================================================")
    print("KNN Recommender:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num * 100:.2f}")
    assert isclose(score / req_num, 1 / library_size, abs_tol=0.1)


def test_parrot(library_size, history_num, req_num, Utility: float = 1):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]

    # Convert to vectors
    requests_vec, history_vec = utils.convert_to_vectors(requests, history, library_size)

    forecaster = ParrotForecaster(history_vec + requests_vec, Utility=Utility)

    score, predictions = test_forecaster_score(forecaster, requests_vec, history_vec, library_size, history_num,
                                               req_num)
    print()
    print("================================================================================")
    print("Parrot Forecaster:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num * 100:.2f}")
    assert isclose(score / req_num, Utility, abs_tol=0.1)


def test_naive(library_size, history_num, req_num):
    # Draw requests from uniform distribution
    history = [random.randint(0, library_size - 1) for i in range(history_num)]
    requests = [random.randint(0, library_size - 1) for i in range(req_num)]
    requests2 = []
    for req in requests:
        requests2.append(req)
        requests2.append(req)

    # Convert to vectors
    requests_vec, history_vec = utils.convert_to_vectors(requests, history, library_size)
    requests_vec2, history_vec2 = utils.convert_to_vectors(requests2, history, library_size)

    forecaster = NaiveForecaster(library_size)

    score, predictions = test_forecaster_score(forecaster, requests_vec, history_vec, library_size, history_num,
                                               req_num)
    score2, predictions = test_forecaster_score(forecaster, requests_vec2, history_vec2, library_size, history_num,
                                                req_num * 2)
    print()
    print("================================================================================")
    print("Naive Forecaster:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num:.2f}")
    print(f"Utility 2: {score2 / req_num * 2:.2f}")


def test_naive_movielens():
    # Draw requests from uniform distribution
    train_requests, train_library = utils.get_movie_lens_train("../ml-latest-small/ml-latest-small")
    test_requests, test_library = utils.get_movie_lens_test("../ml-latest-small/ml-latest-small")

    library_size = train_library
    assert train_library == test_library
    history_num = len(train_requests)
    req_num = len(test_requests)
    # Convert to vectors
    test_vec, train_vec = utils.convert_to_vectors(test_requests, train_requests, library_size)

    forecaster = NaiveForecaster(library_size)

    score, predictions = test_forecaster_score(forecaster, test_vec, train_vec, library_size, history_num, req_num)
    print()
    print("================================================================================")
    print("Naive Forecaster on MovieLens:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num:.2f}")


def test_arima_movielens():
    # Draw requests from uniform distribution
    train_requests, train_library = utils.get_movie_lens_train("../ml-latest-small/ml-latest-small")
    test_requests, test_library = utils.get_movie_lens_test("../ml-latest-small/ml-latest-small")

    library_size = train_library
    assert train_library == test_library
    history_num = len(train_requests)
    req_num = len(test_requests)
    # Convert to vectors
    test_vec, train_vec = utils.convert_to_vectors(test_requests, train_requests, library_size)

    forecaster = ArimaForecaster(train_vec, library_size, frequency=req_num)

    score, predictions = test_forecaster_score(forecaster, test_vec, train_vec, library_size, history_num, req_num)
    print()
    print("================================================================================")
    print("Arima Forecaster on MovieLens:")
    print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
    print(f"Utility: {score / req_num:.2f}")


def test_tcn_movielens_darts():
    # train_requests, val_requests, test_requests = utils.get_movie_lens_split("../ml-latest-small/ml-latest-small")
    # library_size = len(np.unique(np.concatenate((train_requests, val_requests, test_requests))))
    test_requests, train_requests, library_size = utils.get_requests_from_movielens(
        "../ml-latest-small/ml-latest-small", history_percentage=0.9, request_limit=2560, library_limit=100)
    val_requests = train_requests[2560 - 512:2560 - 256]
    train_requests = train_requests[:2560 - 512]
    req_num = len(test_requests)
    print(train_requests.shape, test_requests.shape, val_requests.shape)

    train_vec, history = utils.convert_to_vectors(train_requests, [], library_size)
    val_vec, history = utils.convert_to_vectors(val_requests, [], library_size)
    test_vec, history = utils.convert_to_vectors(test_requests, [], library_size)
    train_ts = TimeSeries.from_values(np.array(train_vec))
    val_ts = TimeSeries.from_values(np.array(val_vec))
    test_ts = TimeSeries.from_values(np.array(test_vec))
    # scaler = Scaler()
    # train_scaled = scaler.fit_transform(train_ts)
    # val_scaled = scaler.fit_transform(val_ts)
    # test_scaled = scaler.transform(test_ts)

    print(np.array(train_vec).shape)
    print(np.where(np.sum(train_vec, axis=0) == 0))
    print(np.where(np.sum(test_vec, axis=0) == 0))
    print(np.where(np.sum(val_vec, axis=0) == 0))

    model_name = "TCN MovieLens"
    model = TCNModel(
        input_chunk_length=128,
        output_chunk_length=1,
        n_epochs=20,
        dropout=0.2,
        dilation_base=2,
        kernel_size=8,
        num_filters=50,
        save_checkpoints=True,
        model_name=model_name,
        force_reset=True,
        loss_fn=nn.CrossEntropyLoss()
    )
    model.fit(
        series=train_ts,
        val_series=val_ts
    )
    best_model = model.load_from_checkpoint(model_name=model_name, best=True)
    best_model.save("TCN MovieLens.pt")
    predictions_ts = best_model.predict(n=req_num)
    predictions_vec = predictions_ts.values()
    print(predictions_vec.shape)

    # Normalize prediction values
    predictions_vec = predictions_vec - np.minimum(np.min(predictions_vec, axis=1, keepdims=True), 0) / np.sum(
        predictions_vec, axis=1, keepdims=True)

    # Calculate Score: Nr. of cache hits
    score = 0
    for i in range(req_num):
        print(predictions_vec[i])
        score += np.dot(predictions_vec[i], test_vec[i])

    # score, predictions = test_forecaster_score(forecaster, test_vec, train_vec, library_size, history_num, req_num)
    print("================================================================================")
    print("TCN Forecaster on MovieLens:")
    # print(f"Library size: {library_size}, Request num: {req_num}, History num: {history_num}")
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


def test_mfr_foreccaster(library_size=100, requests_num=200, history_num=1000):
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
    assert (np.array(predictions) == v).all() is True
    print("================================================================================")
    print("Most Frequently Requested Forecaster:")
    print(f"Library size: {library_size}, Request num: {requests_num}, History num: {history_num}")
    print(f"Average Utility: {score / requests_num:.5f}")

def test_tcn_forecaster(library_size=100):
    train, val, test = utils.get_movie_lens_split("../ml-latest-small/ml-latest-small", library_limit=library_size)


    train_vec = utils.convert_to_vectors(train, library_size)
    val_vec = utils.convert_to_vectors(val, library_size)
    test_vec = utils.convert_to_vectors(test, library_size)

    forecaster = TCNForecaster(model_path="../tcn/tcn_best", history=np.concatenate((train_vec, val_vec)))

    score, predictions = test_forecaster_score(forecaster, test_vec)

    print("================================================================================")
    print("Most Frequently Requested Forecaster:")
    print(f"Library size: {library_size}, Request num: {len(test)}, History num: {len(train) + len(val)}")
    print(f"Average Utility: {score / len(test):.5f}")

def test_des_movielens(library_size=100):
    train, val, test = utils.get_movie_lens_split("../ml-latest-small/ml-latest-small", library_limit=library_size)

    train_vec = utils.convert_to_vectors(train, library_size)
    val_vec = utils.convert_to_vectors(val, library_size)
    test_vec = utils.convert_to_vectors(test, library_size)

    forecaster = DESForecaster(history=np.concatenate((train_vec, val_vec)), horizon=100, one_hot=True)

    score, predictions = test_forecaster_score(forecaster, test_vec)

    print("================================================================================")
    print("Most Frequently Requested Forecaster:")
    print(f"Library size: {library_size}, Request num: {len(test)}, History num: {len(train) + len(val)}")
    print(f"Average Utility: {score / len(test):.5f}")


if __name__ == "__main__":
    # test_recommender_uniform(100, 50, 500)
    # test_parrot(100, 50, 500, Utility=1)
    # test_parrot(100, 50, 500, Utility=0.8)
    # test_parrot(100, 50, 500, Utility=0.5)
    # test_random(100, 50, 500)
    # test_random_movielens()
    # test_naive(100, 50, 500)
    # test_naive_movielens()
    # test_arima_movielens()
    # test_tcn_movielens_darts()
    # test_mfr_foreccaster()
    # test_tcn_forecaster()
    test_des_movielens()
    pass
