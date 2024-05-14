# Press the green button in the gutter to run the script.
import datetime
import random
import time
from statsmodels.tsa.arima.model import ARIMA

import numpy as np
from matplotlib import pyplot as plt

from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster
from forecasters.RandomForecaster import RandomForecaster
from oftrl import OFTRL
from plotters import plot_cummulative_regret, plot_average_regret
from recommender.kNNRecommender import kNNRecommender


def alphabet_test():
    cache_size = 5
    library = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
               "v", "w", "x", "y", "z"]
    # forecaster = UselessForecaster(cache_size, len(library))
    forecaster = RecommenderForecaster(kNNRecommender(1), library_size=len(library))
    oftrl = OFTRL(forecaster, cache_size, len(library))

    requests = [library[random.randint(0, len(library) - 1)] for i in range(100)]
    request_vectors = []
    # Convert requests into vectors
    for req in requests:
        idx = library.index(req)
        vector = np.zeros(len(library))
        vector[idx] = 1
        request_vectors.append(vector)

    regret_list = []
    regret_t_list = []
    for i, req in enumerate(request_vectors):
        # print(f"Request {req}")
        oftrl.get_next(req)
        regret = oftrl.regret()
        regret_list.append(regret)
        regret_t_list.append(regret / i)
    # print(oftrl.prediction_err_log)
    print(regret_list)
    print(regret_t_list)

    plot_cummulative_regret(regret_list, title="Alphabets with C = 5")
    plot_average_regret(regret_t_list, title="Alphabets with C = 5")

def arbitrary_random_test(cache_size, library_size, num_of_requests):
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")
    # Create random requests
    request_vectors = []
    for i in range(num_of_requests):
        idx = random.randint(0, library_size - 1)
        vector = np.zeros(library_size)
        vector[idx] = 1
        request_vectors.append(vector)

    # Initialize OFTRL
    predictor = RandomForecaster(cache_size, library_size)
    # predictor = RecommenderForecaster(kNNRecommender(1), library_size)
    oftrl = OFTRL(predictor, cache_size, library_size)
    regret_list = []

    # Calculate regret for every request
    get_next_time = 0
    regret_time = 0
    for i, req in enumerate(request_vectors):
        print(i)
        start = time.time() * 1000
        oftrl.get_next(req)
        get_next_time += int(time.time() * 1000 - start)

        start = time.time() * 1000
        regret = oftrl.regret()
        regret_time += int(time.time() * 1000 - start)
        regret_list.append(regret)

    print("Cache assignment time: " + str(get_next_time) + "ms")
    print("Regret calculation time: " + str(regret_time) + "ms")
    plot_cummulative_regret(regret_list, title=f"Random requests with C = {cache_size}, L = {library_size} ")
    plt.show()
    plot_average_regret(regret_list, title=f"Random requests with C = {cache_size}, L = {library_size}")
    plt.show()


def oftrl_diff_predict_acc(cache_size, library_size, num_of_requests):
    print("========================================================================")
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")

    # Create random requests from uniform distribution
    request_vectors = []
    for i in range(num_of_requests):
        idx = random.randint(0, library_size - 1)
        vector = np.zeros(library_size)
        vector[idx] = 1
        request_vectors.append(vector)

    # Initialize OFTRL
    accuracies = np.arange(0, 1, step=0.2)
    for acc in accuracies:
        print(f"Accuracy: {acc}")
        predictor = ParrotForecaster(request_vectors, accuracy=acc)
        oftrl = OFTRL(predictor, cache_size, library_size)

        # Calculate regret for every request
        regret_list = []
        get_next_time = 0
        regret_time = 0
        for i, req in enumerate(request_vectors):
            print(i, end=", ")
            start = time.time() * 1000
            oftrl.get_next(req)
            get_next_time += int(time.time() * 1000 - start)

            start = time.time() * 1000
            regret = oftrl.regret()
            regret_time += int(time.time() * 1000 - start)
            regret_list.append(regret)

        print("")
        # plot_cummulative_regret(regret_list, title=f"Random requests with C = {cache_size}, L = {library_size} ")
        plot_average_regret(regret_list, title=f"Random requests with C = {cache_size}, L = {library_size}", label=f"Accuracy {acc}")
    plt.legend(loc="upper right")
    plt.show()


def oftrl_regret_list(oftrl, request_vectors):
    regret_list = []
    # Calculate regret for every request
    get_next_time = 0
    regret_time = 0
    # Run OFTRL
    for i, req in enumerate(request_vectors):
        print(i, end=", ")
        start = time.time() * 1000
        oftrl.get_next(req)
        get_next_time += int(time.time() * 1000 - start)

        start = time.time() * 1000
        regret_list.append(oftrl.regret())
        regret_time += int(time.time() * 1000 - start)

    print("")
    print("Cache assignment time: " + str(get_next_time) + "ms")
    print("Regret calculation time: " + str(regret_time) + "ms")

    print(regret_list)
    return regret_list

def run_oftrl_regret(requests, history, forecaster, cache_size, library_size):
    # Shuffling so it is not always guaranteed that the file 1 will have the highest chance of getting requested.
    shuffled_range = np.arange(0, library_size)
    np.random.shuffle(shuffled_range)
    request_vectors = []
    history_vectors = []
    for req in history:
        idx = shuffled_range[int(req)]
        vector = np.zeros(library_size)
        vector[idx] = 1
        history_vectors.append(vector)

    for req in requests:
        idx = shuffled_range[int(req)]
        vector = np.zeros(library_size)
        vector[idx] = 1
        request_vectors.append(vector)

    # print(request_vectors)

    # Initialize OFTRL
    forecaster.update(history_vectors)
    oftrl = OFTRL(forecaster, cache_size, library_size)

    regret_list = oftrl_regret_list(oftrl, request_vectors)
    print(regret_list)
    return regret_list

def graph_oftrl_regret(forecasters_options, distribution_options, cache_size, library_size, num_of_requests, history_size=50):
    forecaster_list = []
    for f in forecasters_options:
        if f == "random":
            forecaster = RandomForecaster(cache_size, library_size)
            forecaster_list.append(forecaster)
        if f == "recommender":
            forecaster = RecommenderForecaster(kNNRecommender(10), library_size)
            forecaster_list.append(forecaster)

    history = None
    requests = None
    # Create history and requests based on the chosen distribution
    if distribution_options == "uniform":
        requests = np.random.uniform(0, library_size, size=num_of_requests)
        history = np.random.uniform(0, library_size, size=history_size)
    elif distribution_options == "zipf":
        zipf_param = 1.1
        history = np.random.zipf(zipf_param, size=history_size)
        history = history - 1
        history = history[history < library_size]
        requests = np.random.zipf(zipf_param, size=num_of_requests)
        requests = requests - 1
        requests = requests[requests < library_size]
    elif distribution_options == "normal":
        history = np.random.normal(library_size / 2, library_size / 6, size=history_size)
        history = history[history < library_size][history >= 0]
        requests = np.random.normal(library_size / 2, library_size / 6, size=num_of_requests)
        requests = requests[requests < library_size]
        requests = requests[requests >= 0]
    elif distribution_options == "arima":
        history = np.random.normal(library_size / 2, library_size / 6, size=history_size)
        history = history[history < library_size][history >= 0]
        arima = ARIMA(history, order=(1, 0, 1))
        arima_res = arima.fit()
        requests = arima_res.forecast(num_of_requests)
        requests = requests[requests < library_size]
        requests = requests[requests >= 0]
    else:
        raise ValueError("Distribution must be uniform, zipf, normal or arima.")

    # Collect regrets of different OFTRL forecaster combinations.
    regret_list_list = []
    for predictor in forecaster_list:
        regret_list = run_oftrl_regret(requests, history, predictor, cache_size, library_size)
        regret_list_list.append(regret_list)

   # Plot the results on one graph
    for regret_list, forecaster_name in zip(regret_list_list, forecasters_options):
        plot_average_regret(regret_list, label=f"{forecaster_name} Forecaster")

    plt.title(f"{distribution_options} requests with C = {cache_size}, L = {library_size}, H ={history_size}")
    plt.legend(loc="upper right")
    plt.savefig(f"new_plots/C-{cache_size}_L-{library_size}_H-{history_size}_N-{num_of_requests}_{distribution_options}_{datetime.datetime.now().strftime("%d%b%y%H%M")}.png")
    plt.show()


if __name__ == '__main__':
    start_time = time.time() * 1000
    # alphabet_test()
    # arbitrary_random_test(10, 100, 500)
    # arbitrary_random_test(75, 5000, 20)
    # oftrl_diff_predict_acc(75, 5000, 100)
    # oftrl_recommender_uniform(75, 5000, 100, history_size=0)
    # oftr_recommender_zipf(75, 5000, 100, history_size=0)
    # oftrl_recommender_normal(75, 5000, 100, history_size=0)
    # oftrl_recommender_uniform(75, 5000, 100, history_size=50)
    # oftr_recommender_zipf(75, 5000, 100, history_size=50)
    # oftrl_recommender_normal(75, 5000, 100, history_size=50)
    # oftrl_recommender_uniform(5, 250, 1000, history_size=50)
    # oftr_recommender_zipf(5, 250, 1000, history_size=50)
    # oftrl_recommender_normal(5, 250, 1000, history_size=50)
    # oftrl_recommender_arma(5, 250, 200, history_size=50)

    graph_oftrl_regret(("random", "recommender"), "uniform",
                       5, 250, 200, 50)
    graph_oftrl_regret(("random", "recommender"), "zipf",
                       5, 250, 200, 50)
    graph_oftrl_regret(("random", "recommender"), "normal",
                       5, 250, 200, 50)
    graph_oftrl_regret(("random", "recommender"), "arima",
                       5, 250, 200, 50)

    graph_oftrl_regret(("random", "recommender"), "uniform",
                       150, 10000, 10000, 50)
    graph_oftrl_regret(("random", "recommender"), "zipf",
                       150, 10000, 10000, 50)
    graph_oftrl_regret(("random", "recommender"), "normal",
                       150, 10000, 10000, 50)
    graph_oftrl_regret(("random", "recommender"), "arima",
                       150, 10000, 10000, 50)
    plt.show()
    print("Total time taken: " + str(int(time.time() * 1000 - start_time)) + "ms")
