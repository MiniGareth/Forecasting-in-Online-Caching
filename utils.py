import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
def get_requests_from_distribution(distribution: str, library_size, num_of_requests, history_size):
    history = None
    requests = None
    # Create history and requests based on the chosen distribution
    if distribution == "uniform":
        requests = np.random.uniform(0, library_size, size=num_of_requests)
        history = np.random.uniform(0, library_size, size=history_size)
    elif distribution == "zipf":
        zipf_param = 1.1
        history = np.random.zipf(zipf_param, size=history_size)
        history = history - 1
        history = history[history < library_size]
        requests = np.random.zipf(zipf_param, size=num_of_requests)
        requests = requests - 1
        requests = requests[requests < library_size]
    elif distribution == "normal":
        history = np.random.normal(library_size / 2, library_size / 6, size=history_size)
        history = history[history < library_size]
        history = history[history >= 0]
        requests = np.random.normal(library_size / 2, library_size / 6, size=num_of_requests)
        requests = requests[requests < library_size]
        requests = requests[requests >= 0]
    elif distribution == "arima":
        history = np.random.normal(library_size / 2, library_size / 6, size=max(100, history_size))
        history = history[history < library_size][history >= 0]
        arima = ARIMA(history, order=(1, 0, 1))
        arima_res = arima.fit()
        requests = arima_res.forecast(num_of_requests)
        requests = requests[requests < library_size]
        requests = requests[requests >= 0]

    else:
        raise ValueError("Distribution must be uniform, zipf, normal or arima.")

    return requests, history

def convert_to_vectors(requests, library_size, idx_mapper: dict=None):
    if idx_mapper is None:
        idx_mapper = dict(zip(np.arange(0, library_size), np.arange(0, library_size)))

    request_vectors = []

    for req in requests:
        idx = idx_mapper[int(req)]
        vector = np.zeros(library_size)
        vector[idx] = 1
        request_vectors.append(vector)

    return np.array(request_vectors)

def get_requests_from_movielens(path: str, history_percentage=0, request_limit=None, library_limit=None):
    ratings = pd.read_csv(path +"/ratings.csv")
    library = np.array(ratings["movieId"].value_counts().reset_index())[:library_limit, 0]

    # Get movie requests based on time
    sorted_ratings = ratings.sort_values(by="timestamp")
    temp_requests = np.array(sorted_ratings["movieId"])
    # only keep requests of a movie if it is in our "library"
    requests = []
    for req in temp_requests:
        if req in library:
            requests.append(req)

    # Map movieIds to new indices
    movie_mapper = dict(zip(np.unique(requests), list(range(len(library)))))
    for i in range(len(requests)):
        requests[i] = movie_mapper[requests[i]]

    # Limit history and requests size based on parameters
    if request_limit is not None:
        requests = requests[-request_limit:]

    history = requests[:int(len(requests) * history_percentage)]
    requests = requests[int(len(requests) * history_percentage):]

    return np.array(requests), np.array(history), len(library)

def get_movie_lens_train(path: str, n_train=0.8):
    all_requests, history, library_size = get_requests_from_movielens(path, library_limit=1000)
    nr_requests = len(all_requests)
    return np.array(all_requests[:int(nr_requests * n_train)]), library_size

def get_movie_lens_test(path: str, n_test=0.1):
    all_requests, history, library_size = get_requests_from_movielens(path, library_limit=1000)
    nr_requests = len(all_requests)
    return np.array(all_requests[int(nr_requests * (1- n_test)):]), library_size

def get_movie_lens_split(path: str, n_train=0.8, n_val=0.1, n_test=0.1):
    all_requests, history, library_size = get_requests_from_movielens(path, library_limit=1000)
    nr_requests = len(all_requests)

    train = np.array(all_requests[:int(nr_requests * n_train)])
    validation = np.array(all_requests[int(nr_requests * n_train):int(nr_requests *(n_train + n_val))])
    test = np.array(all_requests[int(nr_requests * (1- n_test)):])

    return train, validation, test