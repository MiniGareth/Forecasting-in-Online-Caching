# Press the green button in the gutter to run the script.
import random
import time

import numpy as np
from matplotlib import pyplot as plt

from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster
from forecasters.RandomForecaster import RandomForecaster
from oftrl import OFTRL
from plotters import plot_average_utility
from recommender.kNNRecommender import kNNRecommender


def alphabet_test():
    cache_size = 5
    library = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
               "v", "w", "x", "y", "z"]
    forecaster = RandomForecaster(cache_size, len(library))
    # forecaster = RecommenderForecaster(kNNRecommender(1), library_size=len(library))
    oftrl = OFTRL(forecaster, cache_size, len(library))

    requests = [library[random.randint(0, len(library) - 1)] for i in range(100)]
    request_vectors = []
    # Convert requests into vectors
    for req in requests:
        idx = library.index(req)
        vector = np.zeros(len(library))
        vector[idx] = 1
        request_vectors.append(vector)

    utility_list = []
    for i, req in enumerate(request_vectors):
        # print(f"Request {req}")
        oftrl.get_next(req)
        utility = oftrl.utility()
        utility_list.append(utility)
    # print(oftrl.prediction_err_log)
    print(utility_list)

    plot_average_utility(utility_list, title="Alphabets with C = 5")
    plt.show()

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
    utility_list = []

    # Calculate utility for every request
    get_next_time = 0
    utility_time = 0
    for i, req in enumerate(request_vectors):
        print(i)
        start = time.time() * 1000
        oftrl.get_next(req)
        get_next_time += int(time.time() * 1000 - start)

        start = time.time() * 1000
        utility = oftrl.utility()
        utility_time += int(time.time() * 1000 - start)
        utility_list.append(utility)

    print("Cache assignment time: " + str(get_next_time) + "ms")
    print("utility calculation time: " + str(utility_time) + "ms")
    plot_average_utility(utility_list, title=f"Random requests with C = {cache_size}, L = {library_size}")
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
    accuracies = np.arange(0, 1.1, step=0.2)
    for acc in accuracies:
        print(f"Accuracy: {acc}")
        predictor = ParrotForecaster(request_vectors, accuracy=acc)
        oftrl = OFTRL(predictor, cache_size, library_size)

        # Calculate utility for every request
        utility_list = []
        get_next_time = 0
        utility_time = 0
        for i, req in enumerate(request_vectors):
            print(i, end=", ")
            start = time.time() * 1000
            oftrl.get_next(req)
            get_next_time += int(time.time() * 1000 - start)

            start = time.time() * 1000
            utility = oftrl.utility()
            utility_time += int(time.time() * 1000 - start)
            utility_list.append(utility)

        print("")
        # plot_cummulative_utility(utility_list, title=f"Random requests with C = {cache_size}, L = {library_size} ")
        plot_average_utility(utility_list, title=f"Random requests with C = {cache_size}, L = {library_size}", label=f"Accuracy {acc}")
    plt.legend(loc="upper right")
    plt.show()


def oftrl_utility_list(oftrl, request_vectors):
    utility_list = []
    # Calculate utility for every request
    get_next_time = 0
    utility_time = 0
    # Run OFTRL
    for i, req in enumerate(request_vectors):
        print(i, end=", ")
        start = time.time() * 1000
        oftrl.get_next(req)
        get_next_time += int(time.time() * 1000 - start)

        start = time.time() * 1000
        utility_list.append(oftrl.utility())
        utility_time += int(time.time() * 1000 - start)

    print("")
    print("Cache assignment time: " + str(get_next_time) + "ms")
    print("utility calculation time: " + str(utility_time) + "ms")

    print(utility_list)
    return utility_list


def oftrl_compare_two(requests, history, forecaster1, forecaster2, cache_size, library_size):
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

    print(request_vectors)

    # Initialize OFTRL
    forecaster1.update(history_vectors)
    forecaster2.update(history_vectors)
    oftrl1 = OFTRL(forecaster1, cache_size, library_size)
    oftrl2 = OFTRL(forecaster2, cache_size, library_size)

    utility_list1 = oftrl_utility_list(oftrl1, request_vectors)
    utility_list2 = oftrl_utility_list(oftrl2, request_vectors)

    return utility_list1, utility_list2

def oftrl_recommender_uniform(cache_size, library_size, num_of_requests, history_size=50):
    print("========================================================================")
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")
    requests = np.random.uniform(0, library_size, size=num_of_requests)
    history = np.random.uniform(0, library_size, size=history_size)
    print(requests)
    print(history)

    # The two forecasters to compare
    predictor1 = RandomForecaster(cache_size, library_size)
    predictor2 = RecommenderForecaster(kNNRecommender(1), library_size)

    # Get the two utility lists from 2 different forecasters applied to OFTRL
    utility_list1, utility_list2 = oftrl_compare_two(requests, history, predictor1, predictor2, cache_size,
                                                   library_size)

    plot_average_utility(utility_list1, label=f"Useless Forecaster")
    plot_average_utility(utility_list2, label=f"Recommender Forecaster")
    plt.title(f"Uniform requests with C = {cache_size}, L = {library_size}, H ={history_size}")
    plt.legend(loc="upper right")
    plt.show()

def oftr_recommender_zipf(cache_size, library_size, num_of_requests, history_size=50, zipf_param=1.5):
    print("========================================================================")
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")
    # Create random requests from Zipf distribution
    history = np.random.zipf(zipf_param, size=history_size)
    history = history - 1
    history = history[history < library_size]
    requests = np.random.zipf(zipf_param, size=num_of_requests)
    requests = requests - 1
    requests = requests[requests < library_size]
    print(requests)
    print(history)

    # The two forecasters to compare
    predictor1 = RandomForecaster(cache_size, library_size)
    predictor2 = RecommenderForecaster(kNNRecommender(1), library_size)

    # Get the two utility lists from 2 different forecasters applied to OFTRL
    utility_list1, utility_list2 = oftrl_compare_two(requests, history, predictor1, predictor2, cache_size, library_size)

    plot_average_utility(utility_list1, label=f"Useless Forecaster")
    plot_average_utility(utility_list2, label=f"Recommender Forecaster")
    plt.title(f"Zipf requests with C = {cache_size}, L = {library_size}, H ={history_size}")
    plt.legend(loc="upper right")
    plt.show()

def oftrl_recommender_normal(cache_size, library_size, num_of_requests, history_size=50):
    print("========================================================================")
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")
    # Create random requests from Normal distribution
    history = np.random.normal(library_size/2, library_size/6, size=history_size)
    history = history[history < library_size][history >= 0]
    requests = np.random.normal(library_size/2, library_size/6, size=num_of_requests)
    requests = requests[requests < library_size][requests >= 0]
    print(requests)
    print(history)

    # The two forecasters to compare
    predictor1 = RandomForecaster(cache_size, library_size)
    predictor2 = RecommenderForecaster(kNNRecommender(1), library_size)

    # Get the two utility lists from 2 different forecasters applied to OFTRL
    utility_list1, utility_list2 = oftrl_compare_two(requests, history, predictor1, predictor2, cache_size,
                                                   library_size)

    plot_average_utility(utility_list1, label=f"Useless Forecaster")
    plot_average_utility(utility_list2, label=f"Recommender Forecaster")
    plt.title(f"Normal requests with C = {cache_size}, L = {library_size}, H ={history_size}")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    start_time = time.time() * 1000
    # alphabet_test()
    # arbitrary_random_test(10, 100, 500)
    # arbitrary_random_test(75, 5000, 100)
    # oftrl_diff_predict_acc(75, 5000, 100)
    # oftrl_recommender_uniform(75, 5000, 100, history_size=0)
    # oftr_recommender_zipf(75, 5000, 100, history_size=0)
    # oftr_recommender_normal(75, 5000, 100, history_size=0)
    # oftrl_recommender_uniform(75, 5000, 100, history_size=0)
    # oftr_recommender_zipf(75, 5000, 100, history_size=50)
    # oftr_recommender_normal(75, 5000, 100, history_size=50)
    oftrl_recommender_uniform(5, 250, 200, history_size=50)
    oftr_recommender_zipf(5, 250, 200, history_size=50)
    oftrl_recommender_normal(5, 250, 200, history_size=50)
    # oftrl_recommender_arma(5, 250, 200, history_size=50)

    plt.show()
    print("Total time taken: " + str(int(time.time() * 1000 - start_time)) + "ms")
