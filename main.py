# Press the green button in the gutter to run the script.
import datetime
import random
import time
from statsmodels.tsa.arima.model import ARIMA

import numpy as np
from matplotlib import pyplot as plt

import utils
from forecasters.ArimaForecaster import ArimaForecaster
from forecasters.NaiveForecaster import NaiveForecaster
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


def oftrl_diff_predict_acc(cache_size, library_size, num_of_requests, distribution="uniform"):
    print("========================================================================")
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")

    # Create random requests from uniform distribution
    requests, history = utils.get_requests_from_distribution(distribution, library_size, num_of_requests, 0)
    request_vec, history_vec = utils.convert_to_vectors(requests, history, library_size)

    # Initialize OFTRL
    accuracies = np.arange(0, 1.1, step=0.25)
    for acc in accuracies:
        print(f"Accuracy: {acc}")
        predictor = ParrotForecaster(request_vec, accuracy=acc)
        oftrl = OFTRL(predictor, cache_size, library_size)

        # Calculate regret for every request
        regret_list = []
        get_next_time = 0
        regret_time = 0
        for i, req in enumerate(request_vec):
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
        print(regret_list)
        plot_average_regret(regret_list, title=f"Random requests with C = {cache_size}, L = {library_size}", label=f"Accuracy {acc}")
    plt.legend(loc="upper right")
    plt.savefig(f"new_plots/{datetime.datetime.now().strftime('%d%b%y%H%M')}_Regret per accuracy_C-{cache_size}_L-{library_size}_H-{0}_N-{num_of_requests}_{distribution}.png")
    plt.show()
    plt.close()


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

def run_oftrl_regret(request_vectors, history_vectors, forecaster, cache_size, library_size):
    # print(request_vectors)

    # Initialize OFTRL
    oftrl = OFTRL(forecaster, cache_size, library_size)

    regret_list = oftrl_regret_list(oftrl, request_vectors)
    print(regret_list)
    return regret_list

def graph_oftrl_regret(forecasters_options, distribution_options, cache_size, library_size, num_of_requests, history_size=50):
    print("========================================================================")
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")
    requests, history = utils.get_requests_from_distribution(distribution_options, library_size, num_of_requests, history_size)
    # Shuffling so it is not always guaranteed that the file 1 will have the highest chance of getting requested.
    request_vectors, history_vectors = utils.convert_to_vectors(requests, history, library_size)

    forecaster_list = []
    for f in forecasters_options:
        if f == "random":
            forecaster = RandomForecaster(cache_size, library_size)
            forecaster_list.append(forecaster)
        if f == "recommender":
            forecaster = RecommenderForecaster(kNNRecommender(10), library_size)
            forecaster_list.append(forecaster)
        if f == "naive":
            forecaster = NaiveForecaster(library_size)
            forecaster_list.append(forecaster)
        if f == "arima":
            forecaster = ArimaForecaster(history_vectors, library_size)
            forecaster_list.append(forecaster)
        if f == "parrot":
            forecaster = ParrotForecaster(request_vectors)
            forecaster_list.append(forecaster)


    # Collect regrets of different OFTRL forecaster combinations.
    regret_list_list = []
    for predictor in forecaster_list:
        regret_list = run_oftrl_regret(request_vectors, history_vectors, predictor, cache_size, library_size)
        regret_list_list.append(regret_list)

   # Plot the results on one graph
    for regret_list, forecaster_name in zip(regret_list_list, forecasters_options):
        plot_average_regret(regret_list, label=f"{forecaster_name} Forecaster")

    plt.title(f"{distribution_options} requests with C = {cache_size}, L = {library_size}, H ={history_size}")
    plt.legend(loc="upper right")
    plt.savefig(f"new_plots/{datetime.datetime.now().strftime('%d%b%y%H%M')}_C-{cache_size}_L-{library_size}_H-{history_size}_N-{num_of_requests}_{distribution_options}.png")
    plt.show()
    plt.close()

def graph_oftrl_regret_movielens(forecasters_options, path, cache_size, library_limit=None, num_of_requests=None, history_percentage=None):
    print("========================================================================")
    requests, history, library_size = utils.get_requests_from_movielens(path, history_percentage=history_percentage,
                                                                        library_limit=library_limit, request_limit=num_of_requests)
    # Shuffling so it is not always guaranteed that the file 1 will have the highest chance of getting requested.
    request_vectors, history_vectors = utils.convert_to_vectors(requests, history, library_size)

    forecaster_list = []
    for f in forecasters_options:
        if f == "random":
            forecaster = RandomForecaster(cache_size, library_size)
            forecaster_list.append(forecaster)
        if f == "recommender":
            forecaster = RecommenderForecaster(kNNRecommender(10), library_size)
            forecaster_list.append(forecaster)
        if f == "naive":
            forecaster = NaiveForecaster(library_size)
            forecaster_list.append(forecaster)
        if f == "arima":
            forecaster = ArimaForecaster(history_vectors, library_size)
            forecaster_list.append(forecaster)
        if f == "parrot":
            forecaster = ParrotForecaster(request_vectors)
            forecaster_list.append(forecaster)

    # Collect regrets of different OFTRL forecaster combinations.
    regret_list_list = []
    for predictor in forecaster_list:
        regret_list = run_oftrl_regret(request_vectors, history_vectors, predictor, cache_size, library_size)
        regret_list_list.append(regret_list)

    # Plot the results on one graph
    for regret_list, forecaster_name in zip(regret_list_list, forecasters_options):
        plot_average_regret(regret_list, label=f"{forecaster_name} Forecaster")

    plt.title(f"Movielens requests with C = {cache_size}, L = {library_size}, H ={history_percentage}")
    plt.legend(loc="upper right")
    plt.savefig(
        f"new_plots/{datetime.datetime.now().strftime('%d%b%y%H%M')}_C-{cache_size}_L-{library_size}_H-{history_percentage}_N-{num_of_requests}_movielens.png")
    plt.show()
    plt.close()



if __name__ == '__main__':
    start_time = time.time() * 1000
    # alphabet_test()
    # arbitrary_random_test(10, 100, 500)
    # arbitrary_random_test(75, 5000, 20)
    # oftrl_diff_predict_acc(5, 300, 300)
    # oftrl_diff_predict_acc(5, 300, 300, "zipf")
    # oftrl_diff_predict_acc(5, 300, 300, "normal")
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

    # graph_oftrl_regret(("random", "recommender"), "uniform",
    #                    5, 250, 200, 50)
    # graph_oftrl_regret(("random", "recommender"), "zipf",
    #                    5, 250, 200, 50)
    # graph_oftrl_regret(("random", "recommender"), "normal",
    #                    5, 250, 200, 50)
    # graph_oftrl_regret(("random", "recommender"), "arima",
    #                    5, 250, 200, 50)

    # graph_oftrl_regret(("random", "recommender"), "uniform",
    #                    30, 2000, 2000, 0)
    # graph_oftrl_regret(("random", "recommender"), "zipf",
    #                    30, 2000, 2000, 0)
    # graph_oftrl_regret(("random", "recommender"), "normal",
    #                    30, 2000, 2000, 0)
    # graph_oftrl_regret(("random", "recommender"), "arima",
    #                    30, 2000, 2000, 0)

    # graph_oftrl_regret(("random", "recommender"), "uniform",
    #                    5, 300, 300, 0)
    # graph_oftrl_regret(("random", "recommender"), "zipf",
    #                    5, 300, 300, 0)
    # graph_oftrl_regret(("random", "recommender"), "normal",
    #                    5, 300, 300, 0)
    # graph_oftrl_regret(("random", "recommender"), "arima",
    #                    5, 300, 300, 0)

    # graph_oftrl_regret_movielens(["random"], "ml-latest-small/ml-latest-small", 75, library_limit=5000, num_of_requests=1000)
    graph_oftrl_regret(["random", "naive", "arima", "parrot"], "zipf",
                       5, 300, 3000, 30)
    graph_oftrl_regret_movielens(["random", "naive", "arima", "parrot"], "ml-latest-small/ml-latest-small", 5, library_limit=300, num_of_requests=3000, history_percentage=0.01)
    train, train_library = utils.get_movie_lens_train("ml-latest-small/ml-latest-small")
    test, test_library = utils.get_movie_lens_test("ml-latest-small/ml-latest-small")
    print(len(train), train_library)
    print(len(test), test_library)
    print("Total time taken: " + str(int(time.time() * 1000 - start_time)) + "ms")
