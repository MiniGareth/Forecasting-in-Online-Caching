# Press the green button in the gutter to run the script.
import random
import time

import numpy as np

from forecasters.UselessForecaster import UselessForecaster
from oftrl import OFTRL
from plotters import plot_cummulative_regret, plot_average_regret


def alphabet_test():
    cache_size = 5
    library = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
               "v", "w", "x", "y", "z"]
    forecaster = UselessForecaster(cache_size, len(library))
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
    predictor = UselessForecaster(cache_size, library_size)
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
        regret = oftrl.regret()
        regret_time += int(time.time() * 1000 - start)
        regret_list.append(regret)

    print("Cache assignment time: " + str(get_next_time) + "ms")
    print("Regret calculation time: " + str(regret_time) + "ms")
    plot_cummulative_regret(regret_list, title=f"Random requests with C = {cache_size}, L = {library_size} ")
    plot_average_regret(regret_list, title=f"Random requests with C = {cache_size}, L = {library_size}")


if __name__ == '__main__':
    start_time = time.time() * 1000
    # alphabet_test()
    arbitrary_random_test(10, 100, 500)
    arbitrary_random_test(75, 5000, 100)
    print("Total time taken: " + str(int(time.time() * 1000 - start_time)) + "ms")
