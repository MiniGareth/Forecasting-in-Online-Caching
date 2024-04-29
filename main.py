# Press the green button in the gutter to run the script.
import numpy as np

from forecasters.UselessForecaster import UselessForecaster
from oftrl import OFTRL

if __name__ == '__main__':
    cache_size = 5
    library = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n","o", "p", "q", "r", "s", "t", "u", "v" ,"w", "x", "y", "z"]
    forecaster = UselessForecaster(cache_size, len(library))
    oftrl = OFTRL(forecaster, cache_size, len(library))

    requests = ["t", "z", "m", "a", "e", "k", "q"]
    request_vectors = []
    # Convert requests into vectors
    for req in requests:
        idx = library.index(req)
        vector = np.zeros(len(library))
        vector[idx] = 1
        request_vectors.append(vector)


    print(oftrl.get_all(request_vectors))
    print(oftrl.prediction_err_log)
    print(oftrl.regret())