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


    def get_next(self, request: np.ndarray) -> np.ndarray:
        pass

    def get_all(self, requests: list) -> list:
        """
        Returns the list of cache states this OFTRL policy would have chosen given the list of requests
        :param requests: List of requests in one-hot vector form.
        :return:  A list of cache vectors
        """
        prev_gradient = 0
        gradient = 0
        # Iterate over all file requests
        for req_idx in range(0, len(requests)):
            # Get prediction
            prediction = self.predictor.predict()
            # For first request simply use a random cache
            if req_idx == 0:
                self.cache = np.zeros(self.library_size)
                for j in range(self.cache_size):
                    self.cache[j] = 1
                np.random.shuffle(self.cache)
                self.cache_log.append(self.cache)
            # Otherwise assign cache based on prediction
            else:
                self.assign_cache(prediction, gradient)

            # Get and store request info
            request = requests[req_idx]
            self.request_log.append(request)
            self.predictor.update(self.request_log)

            # Calculate and store prediction error.
            prediction_err = np.square(np.linalg.norm(request - prediction))
            self.prediction_err_log.append(prediction_err)

            # Calculate regularizer parameters differently depending if this is first request
            if req_idx == 0:
                self.reg_params.append(self.sigma * np.sqrt(prediction_err))

            else:
                self.reg_params.append(self.sigma * (
                        np.sqrt(np.sum(self.prediction_err_log)) - np.sqrt(np.sum(self.prediction_err_log[:-1]))
                ))
                prev_gradient = gradient
                gradient = prev_gradient + request

            # The number of regularizer parameters should be the same as the cache log
            assert len(self.reg_params) == len(self.cache_log)

        return self.cache_log

    def assign_cache(self, prediction: np.ndarray, gradient: np.ndarray) -> None:
        """
        Assigns the cache based on the current prediction and the gradient of previous file requests.
        Assignment is done by trying to find a cache vector that maximizes reward
        :param prediction: Prediction vector. A discrete probability distribution
        :param gradient: Gradient vector. A sum of all previous request vectors.
        """
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

            # Calculate sum of previous regularizers
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

    def reward(self, cache: np.ndarray, request: np.ndarray) -> float:
        """
        The reward function is simply the dot product of the cache vector and the request vector.
        If the request vector is a one-hot vector a cache hit will result in reward 1 and a miss with reward 0
        If the request vector is a discrete probability distribution, the reward will be between 0 and 1
        :param cache: The cache vector of 0s and 1s.
        :param request: The request vector that is either a one-hot vector or a discrete probability distribution.
        :return: a number between 0 and 1
        """
        return np.dot(cache, request)

    def regret(self) -> float:
        static_best_cache = None
        best_reward = 0
        # Get static best cache by finding the cache configuration with highest reward
        for comb in itertools.combinations(range(self.library_size), self.cache_size):
            # Initializing cache vector from combination tuple
            cache = np.zeros(self.library_size)
            for file_idx in comb:
                cache[file_idx] = 1

            # Get sum of reward with this cache vector and save it if it is the best one yet
            reward = 0
            for req in self.request_log:
                reward += self.reward(cache, req)
            if reward > best_reward:
                best_reward = reward
                static_best_cache = cache


        regret = 0
        for cache, req in zip(self.cache_log, self.request_log):
             regret += self.reward(static_best_cache, req) - self.reward(cache, req)

        return regret