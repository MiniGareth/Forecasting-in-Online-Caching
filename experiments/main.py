# Press the green button in the gutter to run the script.
from matplotlib import pyplot as plt
from pathlib import Path
import datetime
import random
import time
import pandas as pd
import numpy as np

import sys

root_dir = Path(".").resolve()
sys.path.append(str(root_dir.absolute()))

import utils.utils as utils
from forecasters.MFRForecaster import MFRForecaster
from forecasters.NaiveForecaster import NaiveForecaster
from forecasters.ParrotForecaster import ParrotForecaster
from forecasters.RecommenderForecaster import RecommenderForecaster
from forecasters.RandomForecaster import RandomForecaster
from forecasters.TCNForecaster import TCNForecaster
from forecasters.ZeroForecaster import ZeroForecaster
from oftrl import OFTRL
from utils.plotters import plot_cummulative_regret, plot_average_regret

tables_folder = root_dir / "tables"
new_plots_folder = root_dir / "new_plots"
forecaster_seed = 10
oftrl_seed = 11

def alphabet_test():
    cache_size = 5
    library = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
               "v", "w", "x", "y", "z"]

    requests = [library[random.randint(0, len(library) - 1)] for i in range(100)]
    history = [library[random.randint(0, len(library) - 1)] for i in range(100)]
    request_vecs = utils.convert_to_vectors(requests, len(library), idx_mapper=dict(zip(library, np.arange(len(library)))))
    history_vecs = utils.convert_to_vectors(history, len(library),
                                               idx_mapper=dict(zip(library, np.arange(len(library)))))

    forecaster = RecommenderForecaster(len(library), history_vecs)
    oftrl = OFTRL(forecaster, cache_size, len(library), seed=oftrl_seed)

    regret_list = []
    regret_t_list = []
    for i, req in enumerate(request_vecs):
        # print(f"Request {req}")
        oftrl.get_next(req)
        regret = oftrl.regret()
        regret_list.append(regret)
        regret_t_list.append(regret / i)
    # print(oftrl.prediction_err_log)
    print(regret_list)
    print(regret_t_list)

    plot_cummulative_regret(regret_list, title="Alphabets with C = 5")
    plt.show()
    plot_average_regret(regret_t_list, title="Alphabets with C = 5")
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
    oftrl = OFTRL(predictor, cache_size, library_size, seed=oftrl_seed)
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


def oftrl_diff_predict_acc(cache_size, library_size, num_of_requests, history_size, distribution="uniform"):
    print("========================================================================")
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")

    # Create random requests from uniform distribution
    requests = utils.get_requests_from_distribution(distribution, library_size, num_of_requests)
    history = utils.get_requests_from_distribution(distribution, library_size, history_size)
    request_vec = utils.convert_to_vectors(requests, library_size)
    history_vec = utils.convert_to_vectors(history, library_size)

    # Initialize OFTRL
    accuracies = np.arange(0, 1.1, step=0.25)
    for acc in accuracies:
        print(f"Accuracy: {acc}")
        predictor = ParrotForecaster(np.concatenate((history_vec, request_vec)), accuracy=acc, start_position=history_size,
                                     seed=forecaster_seed)
        oftrl = OFTRL(predictor, cache_size, library_size, seed=oftrl_seed)

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
        plot_average_regret(regret_list, title=f"{distribution[0].upper()+distribution[1:]} requests with C = {cache_size}, L = {library_size}", label=f"Accuracy {acc}")
    plt.legend(loc="upper right")
    plt.savefig(str(root_dir / f"{new_plots_folder}" / f"{datetime.datetime.now().strftime('%d%b%y%H%M')}_Regret per accuracy_C-{cache_size}_L-{library_size}_H-{0}_N-{num_of_requests}_{distribution}.png"))
    plt.show()
    plt.close()


def oftrl_regret_list(request_vectors, history_vectors, forecaster, cache_size, library_size):
    # Initialize OFTRL
    oftrl = OFTRL(forecaster, cache_size, library_size, seed=oftrl_seed)

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

def graph_oftrl_regret(forecasters_options, distribution_options, cache_size, library_size, num_of_requests, history_size=50):
    print("========================================================================")
    print(f"Cache size {cache_size}, Library size {library_size}, {num_of_requests} requests")
    requests = utils.get_requests_from_distribution(distribution_options, library_size, num_of_requests)
    history = utils.get_requests_from_distribution(distribution_options, library_size, history_size)
    # Shuffling so it is not always guaranteed that the file 1 will have the highest chance of getting requested.
    a = np.arange(0, library_size)
    np.random.shuffle(a)
    mapper = dict(zip(a, np.arange(0, library_size)))
    # Convert to vectors with shuffled mapper
    request_vecs = utils.convert_to_vectors(requests, library_size, mapper)
    history_vecs = utils.convert_to_vectors(history, library_size, mapper)

    forecaster_list = []
    for f in forecasters_options:
        if f == "random":
            forecaster = RandomForecaster(library_size)
            forecaster_list.append(forecaster)
        if f == "recommender":
            forecaster = RecommenderForecaster(library_size, history_vecs)
            forecaster_list.append(forecaster)
        if f == "naive":
            forecaster = NaiveForecaster(library_size)
            forecaster_list.append(forecaster)
        if f == "parrot":
            forecaster = ParrotForecaster(np.concatenate((history_vecs, request_vecs)), start_position=history_size,
                                          accuracy=0.5)
            forecaster_list.append(forecaster)
        if f == "mfr" or f == "most frequently requested":
            forecaster = MFRForecaster(history_vecs)
            forecaster_list.append(forecaster)


    # Collect regrets of different OFTRL forecaster combinations.
    regret_list_list = []
    for predictor in forecaster_list:
        regret_list = oftrl_regret_list(request_vecs, history_vecs, predictor, cache_size, library_size)
        regret_list_list.append(regret_list)

   # Plot the results on one graph
    for regret_list, forecaster_name in zip(regret_list_list, forecasters_options):
        plot_average_regret(regret_list, label=f"{forecaster_name} Forecaster")

    filename = f"{datetime.datetime.now().strftime('%d%b%y%H%M')}_C-{cache_size}_L-{library_size}_H-{history_size}_N-{num_of_requests}_{distribution_options}"
    plt.title(f"{distribution_options} requests with C = {cache_size}, L = {library_size}, H ={history_size}")
    plt.legend(loc="upper right")
    plt.savefig(str(root_dir / f"{new_plots_folder}" / f"{filename}.png"))
    plt.show()
    plt.close()
    # Save data into csv
    df = pd.DataFrame(regret_list_list, index=forecasters_options)
    df = df.T
    print(df)
    df.to_csv(str(root_dir / f"{tables_folder}" / f"{filename}.csv"))

def graph_oftrl_regret_movielens(forecasters_options, path, cache_size, library_limit=None, num_of_requests=None, history_percentage=None):
    print("========================================================================")
    train, val, test = utils.get_movie_lens_split("ml-latest-small/ml-latest-small", library_limit=library_limit)

    train_vecs = utils.convert_to_vectors(train, library_limit)
    val_vecs = utils.convert_to_vectors(val, library_limit)
    test_vecs = utils.convert_to_vectors(test, library_limit)

    # Create the forecasters to calculate regret with
    forecaster_list = []
    for f in forecasters_options:
        if f == "zero":
            forecaster = ZeroForecaster(library_limit)
            forecaster_list.append(forecaster)
        if f == "random":
            forecaster = RandomForecaster(library_limit, seed=forecaster_seed)
            forecaster_list.append(forecaster)
        if f == "recommender":
            forecaster = RecommenderForecaster(library_limit, np.concatenate((train_vecs, val_vecs)))
            forecaster_list.append(forecaster)
        if f == "recommender one-hot":
            forecaster = RecommenderForecaster(library_limit, np.concatenate((train_vecs, val_vecs)), one_hot=True)
            forecaster_list.append(forecaster)
        if f == "naive":
            forecaster = NaiveForecaster(library_limit)
            forecaster_list.append(forecaster)
        if f == "parrot":
            forecaster = ParrotForecaster(np.concatenate((train_vecs, val_vecs, test_vecs)), start_position=len(train) + len(val),
                                          accuracy=0.5, seed=forecaster_seed)
            forecaster_list.append(forecaster)
        if f == "mfr" or f == "most frequently requested":
            forecaster = MFRForecaster(np.concatenate((train_vecs, val_vecs)))
            forecaster_list.append(forecaster)
        if f == "tcn":
            forecaster = TCNForecaster(model_path="tcn/tcn_best", history=np.concatenate((train_vecs, val_vecs)))
            forecaster_list.append(forecaster)
        if f == "tcn one-hot":
            forecaster = TCNForecaster(model_path="tcn/tcn_best", history=np.concatenate((train_vecs, val_vecs)), one_hot=True)
            forecaster_list.append(forecaster)

    # Collect regrets of different OFTRL forecaster combinations.
    regret_list_list = []
    for predictor in forecaster_list:
        regret_list = oftrl_regret_list(np.array(test_vecs), np.concatenate((train_vecs, val_vecs)), predictor, cache_size, library_limit)
        regret_list_list.append(regret_list)

    # Plot the results on one graph
    for regret_list, forecaster_name in zip(regret_list_list, forecasters_options):
        plot_average_regret(regret_list, label=f"{forecaster_name} Forecaster")

    filename = f"{datetime.datetime.now().strftime('%d%b%y%H%M')}_C-{cache_size}_L-{library_limit}_H-{len(train) +len(val)}_N-{len(test)}_MovieLens"
    # Save data into csv
    df = pd.DataFrame(regret_list_list, index=forecasters_options)
    df = df.T
    print(df)
    df.to_csv(str(root_dir / f"{tables_folder}" / f"{filename}.csv"))

    plt.title(f"Movielens requests with C = {cache_size}, L = {library_limit}, H ={history_percentage}")
    plt.legend(loc="upper right")
    plt.savefig(str(root_dir / f"{new_plots_folder}" / f"{filename}.png"))
    plt.show()
    plt.close()


if __name__ == '__main__':
    start_time = time.time() * 1000
    # alphabet_test()

    # oftrl_diff_predict_acc(5, 100, 300, 100)
    # oftrl_diff_predict_acc(5, 100, 300, 100, "zipf")
    # oftrl_diff_predict_acc(5, 100, 300, 100,"normal")


    # graph_oftrl_regret(["random", "naive", "mfr", "recommender", "parrot"], "uniform",
    #                    5, 100, 300, 1700)
    # graph_oftrl_regret(["random", "naive", "mfr", "recommender", "parrot"], "zipf",
    #                    5, 100, 300, 1700)
    # graph_oftrl_regret(["random", "naive", "mfr", "recommender", "parrot"], "normal",
    #                    5, 100, 300, 1700)

    graph_oftrl_regret_movielens(["random", "naive", "mfr","recommender", "recommender one-hot", "tcn", "tcn one-hot", "zero", "parrot"], str(root_dir / "ml-latest-small"), 5, 100)

    print("Total time taken: " + str(int(time.time() * 1000 - start_time)) + "ms")
