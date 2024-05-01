# Press the green button in the gutter to run the script.
import random

import numpy as np

from forecasters.UselessForecaster import UselessForecaster
from oftrl import OFTRL

if __name__ == '__main__':
    cache_size = 5
    library = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n","o", "p", "q", "r", "s", "t", "u", "v" ,"w", "x", "y", "z"]
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
        print(f"Request {req}")
        oftrl.get_next(req)
        regret = oftrl.regret()
        regret_list.append(regret)
        regret_t_list.append(regret/ i)
    # print(oftrl.prediction_err_log)
    print(regret_list)
    print(regret_t_list)