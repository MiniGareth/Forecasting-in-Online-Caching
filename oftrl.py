import numpy as np
import itertools
from forecasters.Forecaster import Forecaster
class OFTRL:

    def __init__(self, predictor: Forecaster, cache_size: int, library_size:int):
        self.predictor = predictor
        self.prediction_log = []
        self.prediction_err_log = []
        self.cache = None
        self.cache_size = cache_size
        self.library_size = library_size
        self.cache_log = []
        self.sigma = 1/np.sqrt(self.cache_size)
        self.reg_params = []
        self.request_log = []


    def get_next(self):
        pass

    def get_all(self, requests: list):
        prev_gradient = 0
        gradient = 0
        for i in range(0, len(requests)):
            prediction = self.predictor.predict()
            if i == 0:
                self.cache = np.zeros(self.library_size)
                for j in range(self.cache_size):
                    self.cache[i] = 1
                np.random.shuffle(self.cache)
                self.cache_log.append(self.cache)

            else:
                self.assign_cache(prediction, gradient)

            request = requests[i]
            self.request_log.append(request)
            self.predictor.update(self.request_log)

            prediction_err = np.square(np.linalg.norm(request - prediction))
            self.prediction_err_log.append(prediction_err)

            if i == 0:
                self.reg_params.append(self.sigma * np.sqrt(prediction_err))

            else:
                self.reg_params.append(self.sigma * (
                        np.sqrt(np.sum(self.prediction_err_log)) - np.sqrt(np.sum(self.prediction_err_log[:-1]))
                ))
                prev_gradient = gradient
                gradient = prev_gradient + request

            assert len(self.reg_params) == len(self.cache_log)

    def assign_cache(self, prediction: np.ndarray, gradient: np.ndarray) -> None:
        max_reward = 0
        max_cache = 0
        # Take the cache configuration that gives the highest reward
        for comb in itertools.combinations(range(self.library_size), self.cache_size):
            # Initializing cache vector from combination tuple
            cache = np.zeros(self.library_size)
            for file_idx in comb:
                cache[file_idx] = 1
            # print(comb)
            # print(cache)
            # print("=======================================================================")

            # Calculate regularizer
            regularizer = 0
            for j in range(len(self.reg_params)):
                regularizer += self.reg_params[j]/2 * np.square(np.linalg.norm(cache - self.cache_log[j]))

            # Calculate reward
            reward = self.reward(cache, prediction + gradient) - regularizer
            if reward > max_reward:
                max_reward = reward
                max_cache = cache

        self.cache = max_cache
        self.cache_log.append(self.cache)

    def reward(self, cache: np.ndarray, request: np.ndarray) -> np.ndarray:
        return np.dot(cache, request)
    def results(self):
        return np.array(self.cache_log)

