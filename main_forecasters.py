import time
import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

import utils
from forecasters.TCNForecaster import TCNForecaster
from forecasters.MFRForecaster import MFRForecaster
from forecasters.NaiveForecaster import NaiveForecaster
from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RandomForecaster import RandomForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster
from plotters import plot_utility_bar
from recommender.kNNRecommender import kNNRecommender

forecaster_names = ["Random Forecaster", "Naive Forecaster", "Most Frequently Requested", "KNN Recommender",
                    "TCN Forecaster", "Parrot 50"]

def all_forecasters_all_distributions(cache_size, library_size, num_of_requests, history_size):
    utilities_per_distribution = []
    distributions = ["uniform", "zipf", "normal"]
    # distributions = ["arima"]
    for distribution in distributions:
        print(f"Distribution: {distribution}")
        requests = utils.get_requests_from_distribution(distribution, library_size, num_of_requests)
        history = utils.get_requests_from_distribution(distribution, library_size, history_size)
        # Shuffling so it is not always guaranteed that the file 1 will have the highest chance of getting requested.
        a = np.arange(0, library_size)
        np.random.shuffle(a)
        mapper = dict(zip(a, np.arange(0, library_size)))
        # Convert to vectors with shuffled mapper
        request_vecs = utils.convert_to_vectors(requests, library_size, mapper)
        history_vecs = utils.convert_to_vectors(history, library_size, mapper)
        print(len(request_vecs), len(history_vecs))

        forecasters = [RandomForecaster(library_size),
                       NaiveForecaster(library_size),
                       MFRForecaster(history_vecs),
                       RecommenderForecaster(library_size, history_vecs),
                       TCNForecaster(model_path="tcn/tcn_best", history=history_vecs),
                       ParrotForecaster(np.concatenate((history_vecs, request_vecs), axis=0), accuracy=0.5, start_position=len(history_vecs))]
        utilities = []

        # For every forecaster we get their predictions and store the utilities
        for forecaster in forecasters:
            forecaster_history = list(history_vecs).copy()
            predictions = []
            for req in request_vecs:
                predictions.append(forecaster.predict())
                forecaster_history.append(req)
                forecaster.update(req)
            # Calculate Score: Nr. of cache hits
            utility = 0
            for i in range(len(predictions)):
                utility += np.dot(predictions[i], request_vecs[i])
            utilities.append(utility / len(predictions))

        utilities_per_distribution.append(utilities)

    df = pd.DataFrame(utilities_per_distribution, columns=forecaster_names, index=distributions)
    df.to_csv("tables/forecaster utilities per distribution.csv")
    print(df)
    return df

def all_forecasters_movielens(cache_size, library_size):
    utilities_per_distribution = []
    # distributions = ["arima"]
    print("MovieLens")
    train, val, test = utils.get_movie_lens_split("ml-latest-small/ml-latest-small", library_limit=library_size)

    train_vecs = utils.convert_to_vectors(train, library_size)
    val_vecs = utils.convert_to_vectors(val, library_size)
    test_vecs = utils.convert_to_vectors(test, library_size)
    print(len(train_vecs), len(val_vecs), len(test_vecs))

    forecasters = [RandomForecaster(library_size),
                   NaiveForecaster(library_size),
                   MFRForecaster(np.concatenate((train_vecs, val_vecs))),
                   RecommenderForecaster(library_size, np.concatenate((train_vecs, val_vecs))),
                   TCNForecaster(model_path="tcn/tcn_best", history=np.concatenate((train_vecs, val_vecs))),
                   ParrotForecaster(np.concatenate((train_vecs, val_vecs, test_vecs), axis=0), accuracy=0.5,
                                    start_position=len(train_vecs) + len(val_vecs))
                   ]
    utilities = []

    # For every forecaster we get their predictions and store the utilities
    for forecaster in forecasters:
        forecaster_history = list(np.concatenate((train_vecs, val_vecs))).copy()
        predictions = []
        for req in test_vecs:
            predictions.append(forecaster.predict())
            forecaster_history.append(req)
            forecaster.update(req)
        # Calculate Score: Nr. of cache hits
        utility = 0
        for i in range(len(predictions)):
            utility += np.dot(predictions[i], test_vecs[i])
        utilities.append(utility / len(predictions))

    utilities_per_distribution.append(utilities)

    df = pd.DataFrame(utilities_per_distribution, columns=forecaster_names)
    df.to_csv(f"tables/forecaster utilities for MovieLens {library_size}.csv")
    print(df)


if __name__ == "__main__":
    all_forecasters_all_distributions(5, 100, 1000, 9000)
    # all_forecasters_all_distributions(15, 1000, 1000, 20)
    # all_forecasters_all_distributions(15, 1000, 1000, 100)
    all_forecasters_movielens(5, 100)
    # all_forecasters_movielens(5, 200)
    # all_forecasters_movielens(5, 300)
    # all_forecasters_movielens(5, 400)