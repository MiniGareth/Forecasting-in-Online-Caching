import numpy as np
import itertools
import cvxpy as cp
from forecasters.Forecaster import Forecaster
class OFTRL:

    def __init__(self, predictor: Forecaster, cache_size: int, library_size:int):
        # Constants
        self.cache_size = cache_size
        self.library_size = library_size
        self.sigma = 1/np.sqrt(self.cache_size)

        # Parameters
        self.predictor = predictor
        self.reg_params = []     # Regularizer parameters
        self.prev_gradient = None
        self.gradient = None

        # Logs
        self.prediction_log = []
        self.prediction_err_log = []
        self.cache_log = []
        self.request_log = []

        # Initialize cache state
        self.cache = None
        self._initialize_cache()

    def _initialize_cache(self):
        """
        Initialize the cache to store random files.
        """
        # Store first prediction to calculate prediction error later
        self.prediction_log.append(self.predictor.predict())

        self.cache = np.zeros(self.library_size)
        for j in range(self.cache_size):
            self.cache[j] = 1
        np.random.shuffle(self.cache)
        self.cache_log.append(self.cache)



    def get_next(self, request: np.ndarray) -> np.ndarray:
        """
        Get the next cache state after receiving a request.
        For request r_t this function returns the cache state x_t+1
        :param request: A one-hot vector representing the file requested.
        :return: A cache vector anticipating the next request.
        """
        # Get and store request info
        self.request_log.append(request)
        self.predictor.update(self.request_log)

        # Calculate and store prediction error.
        prediction = self.prediction_log[-1]
        prediction_err = np.square(np.linalg.norm(request - prediction))
        self.prediction_err_log.append(prediction_err)

        # Calculate regularizer parameters differently depending if this is first request
        if len(self.request_log) == 1:
            self.reg_params.append(self.sigma * np.sqrt(prediction_err))
            self.gradient = request

        else:
            self.reg_params.append(self.sigma * (
                    np.sqrt(np.sum(self.prediction_err_log)) - np.sqrt(np.sum(self.prediction_err_log[:-1]))
            ))
            self.prev_gradient = self.gradient
            self.gradient = self.prev_gradient + request

        # The number of regularizer parameters should be the same as the cache log
        assert len(self.reg_params) == len(self.cache_log)

        # New prediction and next cache state
        new_prediction = self.predictor.predict()
        self.prediction_log.append(new_prediction)
        self.assign_cache(new_prediction, self.gradient)

        return self.cache

    def get_all(self, requests: list) -> list:
        """
        Returns the list of cache states this OFTRL policy would have chosen given the list of requests
        :param requests: List of requests in one-hot vector form.
        :return:  A list of cache vectors
        """
        for req in requests:
            # get_next already stores cache in cache log using the assign_cache function
            self.get_next(req)

        return self.cache_log

    def assign_cache(self, prediction: np.ndarray, gradient: np.ndarray) -> None:
        """
        Assigns the cache based on the current prediction and the gradient of previous file requests.
        Assignment is done by trying to find a cache vector that maximizes reward
        :param prediction: Prediction vector. A discrete probability distribution
        :param gradient: Gradient vector. A sum of all previous request vectors.
        """

        x = cp.Variable(self.library_size)
        regularizer = cp.sum(np.array(self.reg_params) / 2 @
                             cp.square(
                                 cp.norm(
                                     cp.vstack([x for i in range(len(self.cache_log))]) - np.array(self.cache_log),
                                     axis=1
                                 )))
        # Reward of a hit is simply the dot product
        reward = x @ (prediction + gradient)
        objective = cp.Maximize(reward - regularizer)
        constraints = [cp.sum(x) == self.cache_size, 0 <= x, x <= 1]

        prob = cp.Problem(objective, constraints)
        prob.solve()
        max_cache = x.value

        # Store and log the new assigned cache
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
        for req in self.request_log:
            # Finds best cache for a particular request
            x = cp.Variable(self.library_size)
            regularizer = cp.sum(np.array(self.reg_params) / 2 @
                                 cp.square(
                                     cp.norm(
                                         cp.vstack([x for i in range(len(self.cache_log) - 1)]) - np.array(self.cache_log[:-1]),
                                         axis=1
                                     )))
            # Reward of a hit is simply the dot product
            reward_expr = x @ np.array(req)
            objective = cp.Maximize(reward_expr)
            constraints = [cp.sum(x) == self.cache_size, 0 <= x, x <= 1]

            prob = cp.Problem(objective, constraints)
            reward = prob.solve()

            if reward > best_reward:
                static_best_cache = x.value


        regret = 0
        for cache, req in zip(self.cache_log, self.request_log):
             regret += self.reward(static_best_cache, req) - self.reward(cache, req)

        return regret