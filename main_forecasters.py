import time
import datetime

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

import utils
from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RandomForecaster import RandomForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster
from plotters import plot_accuracy_bar
from recommender.kNNRecommender import kNNRecommender


def all_forecasters_all_distributions(cache_size, library_size, num_of_requests, history_size):
    forecaster_names = ["Random Forecaster", "KNN Recommender", "Parrot 50"]
    accuracies_per_distribution = []
    distributions = ["uniform", "zipf", "normal"]
    for distribution in distributions:
        requests, history = utils.get_requests_from_distribution(distribution, library_size, num_of_requests, history_size)
        request_vectors, history_vectors = utils.convert_to_vectors(requests, history, library_size)

        forecasters = [RandomForecaster(cache_size, library_size),
                       RecommenderForecaster(kNNRecommender(10), library_size),
                       ParrotForecaster(history_vectors + request_vectors, accuracy=0.5)]
        accuracies = []

        # For every forecaster we get their predictions and store the accuracies
        for forecaster in forecasters:
            forecaster.update(history_vectors)
            forecaster_history = history_vectors.copy()
            predictions = []
            for req in request_vectors:
                predictions.append(forecaster.predict())
                forecaster_history.append(req)
                forecaster.update(history_vectors)
            # Calculate Score: Nr. of cache hits
            score = 0
            for i in range(len(predictions)):
                score += np.dot(predictions[i], request_vectors[i])
            accuracies.append(score / len(predictions))

        accuracies_per_distribution.append(accuracies)


    plot_accuracy_bar(accuracies_per_distribution, forecaster_names, distributions,
                      title=f"Forecaster Accuracies, L={library_size}, N={num_of_requests}, H={history_size}")
    plt.savefig(f"new_plots/{datetime.datetime.now().strftime('%d%b%y%H%M')}_forecaster_accuracies_L-{library_size}_H-{history_size}_N-{num_of_requests}.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    all_forecasters_all_distributions(5, 100, 1000, 0)
    all_forecasters_all_distributions(5, 100, 1000, 20)
    all_forecasters_all_distributions(5, 100, 1000, 100)