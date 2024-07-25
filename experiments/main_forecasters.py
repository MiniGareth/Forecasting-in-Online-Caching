from pathlib import Path
import numpy as np
import pandas as pd
import sys

root_dir = Path(".").resolve()
sys.path.append(str(root_dir.absolute()))

import utils
from forecasters.DESForecaster import DESForecaster
from forecasters.PopularityForecaster import PopularityForecaster
from forecasters.TCNForecaster import TCNForecaster
from forecasters.MFRForecaster import MFRForecaster
from forecasters.NaiveForecaster import NaiveForecaster
from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RandomForecaster import RandomForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster

forecaster_seed = 10
forecaster_names = ["random", "naive", "mfr", "recommender", "recommender one-hot", "popularity", "popularity one-hot",
                    "tcn", "tcn one-hot", "des", "des on-hot", "Parrot 50"]

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
                       RecommenderForecaster(library_size, history_vecs, one_hot=True),
                       TCNForecaster(model_path=str(root_dir / "tcn" / "tcn_best"), history=history_vecs),
                       TCNForecaster(model_path=str(root_dir / "tcn" / "tcn_best"), history=history_vecs, one_hot=True),
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
    df.to_csv(root_dir / "tables" / "forecaster utilities per distribution.csv")
    print(df)
    return df

def all_forecasters_movielens(cache_size, library_size):
    print("MovieLens")
    train, val, test = utils.get_movie_lens_split(str(root_dir / "ml-latest-small" / "ml-latest-small"), library_limit=library_size)

    train_vecs = utils.convert_to_vectors(train, library_size)
    val_vecs = utils.convert_to_vectors(val, library_size)
    test_vecs = utils.convert_to_vectors(test, library_size)
    print(len(train_vecs), len(val_vecs), len(test_vecs))

    forecasters = [RandomForecaster(library_size, seed=forecaster_seed),
                   NaiveForecaster(library_size),
                   MFRForecaster(np.concatenate((train_vecs, val_vecs))),
                   RecommenderForecaster(library_size, np.concatenate((train_vecs, val_vecs))),
                   RecommenderForecaster(library_size, np.concatenate((train_vecs, val_vecs)), one_hot=True),
                   PopularityForecaster(np.concatenate((train_vecs, val_vecs)), 1450),
                   PopularityForecaster(np.concatenate((train_vecs, val_vecs)), 1580, one_hot=True),
                   TCNForecaster(model_path=str(root_dir / "tcn" / "tcn_best"), history=np.concatenate((train_vecs, val_vecs))),
                   TCNForecaster(model_path=str(root_dir / "tcn" / "tcn_best"), history=np.concatenate((train_vecs, val_vecs)), one_hot=True),
                   DESForecaster(np.concatenate((train_vecs, val_vecs)), 1200),
                   DESForecaster(np.concatenate((train_vecs, val_vecs)), 1300, one_hot=True),
                   ParrotForecaster(np.concatenate((train_vecs, val_vecs, test_vecs), axis=0), accuracy=0.5,
                                    start_position=len(train_vecs) + len(val_vecs), seed=forecaster_seed)
                   ]
    utilities = []
    accuracies = []
    prediction_errs = []
    # For every forecaster we get their predictions and store the utilities, accuracies, and prediction errors
    for forecaster in forecasters:
        forecaster_history = list(np.concatenate((train_vecs, val_vecs))).copy()
        predictions = []
        for req in test_vecs:
            predictions.append(forecaster.predict())
            forecaster_history.append(req)
            forecaster.update(req)
        # Calculate Score: Nr. of cache hits
        utility_list = []
        accuracy_list = []
        prediction_err_list = []
        for i in range(len(predictions)):
            utility_list.append(np.dot(predictions[i], test_vecs[i]))
            accuracy_list.append(int(((predictions[i] == np.max(predictions[i])) == test_vecs[i]).all()))
            prediction_err_list.append(np.linalg.norm((predictions[i] - test_vecs[i]), ord=2) ** 2)
        utilities.append(utility_list)
        accuracies.append(accuracy_list)
        prediction_errs.append(prediction_err_list)


    df_all_prediction_errs = pd.DataFrame(np.array(utilities).T, columns=forecaster_names)
    df_all_prediction_errs.to_csv(str(root_dir / "tables" / f"forecaster utilities for MovieLens {library_size}.csv"))
    print(df_all_prediction_errs)

    df_all_accuracies = pd.DataFrame(np.array(accuracies).T*100, columns=forecaster_names)
    df_all_accuracies.to_csv(str(root_dir / "tables" / f"forecaster accuracies for MovieLens {library_size}.csv"))
    print(df_all_accuracies)

    df_all_prediction_errs = pd.DataFrame(np.array(prediction_errs).T, columns=forecaster_names)
    df_all_prediction_errs.to_csv(str(root_dir / "tables" / f"forecaster prediction_errs for MovieLens {library_size}.csv"))
    print(df_all_prediction_errs)

    stats = [np.sum(utilities, axis=1)/len(predictions), np.sum(accuracies, axis=1)/len(predictions) * 100, np.sum(prediction_errs, axis=1)/len(predictions)]
    df_stats = pd.DataFrame(np.array(stats).T, columns=["Utility", "Accuracy", "Prediction Error"], index=forecaster_names)
    df_stats.to_csv(str(root_dir / "tables" / f"forecaster stats for MovieLens {library_size}.csv"))
    print(df_stats)


if __name__ == "__main__":
    # all_forecasters_all_distributions(5, 100, 1600, 16000 - 1600)
    # all_forecasters_all_distributions(15, 1000, 1000, 20)
    # all_forecasters_all_distributions(15, 1000, 1000, 100)
    all_forecasters_movielens(5, 100)
    # all_forecasters_movielens(5, 200)
    # all_forecasters_movielens(5, 300)
    # all_forecasters_movielens(5, 400)