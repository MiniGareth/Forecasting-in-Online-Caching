import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from recommender.Recommender import CollaborativeFilteringRecommender

class kNNRecommender(CollaborativeFilteringRecommender):
    def __init__(self, dataframe: pd.DataFrame, users: str, items: str, ratings: str):
        self.dataframe: pd.DataFrame = dataframe
        self.users: str = users
        self.items: str = items
        self.ratings: str = ratings

        # Create utility matrix and the the mappers of this matrix
        self.utility_matrix = None
        self._user_mapper: dict = None
        self._item_mapper: dict = None
        self._user_inv_mapper: dict = None
        self._item_inv_mapper: dict = None
        self._create_utility_matrix()

    def _create_utility_matrix(self):
        """
        Creates the utility matrix using the given dataframe.
        """
        U = self.dataframe[self.users].nunique()
        I = self.dataframe[self.items].nunique()

        # Mappers to map the id of the user/item with the index of the utility matrix
        self._user_mapper = dict(zip(np.unique(self.dataframe[self.users]), list(range(U))))
        self._item_mapper = dict(zip(np.unique(self.dataframe[self.items]), list(range(I))))

        # Inverse mappers that maps indices to user/item id
        self._user_inv_mapper = dict(zip(list(range(U)), np.unique(self.dataframe[self.users])))
        self._item_inv_mapper = dict(zip(list(range(I)), np.unique(self.dataframe[self.items])))

        user_index = [self._user_mapper[i] for i in self.dataframe[self.users]]
        item_index = [self._item_mapper[i] for i in self.dataframe[self.items]]

        # Create the sparse utility matrix
        self.utility_matrix = csr_matrix((self.dataframe[self.ratings], (user_index, item_index)), shape=(U, I))

    def recommend(self, input, k=10, type="item", metric="cosine") -> list:
        """
        Recommends k nearest users/items based on the input depending on the type of input user/item pairs specified
        :param input: The user/item id the active user wants to find recommendations for
        :param k: The number of recommendations
        :param type: The type of the input. Whether users or items should be returned
        :return: list of k recommendtations
        """
        if type == "item":
            utility_matrix = self.utility_matrix.T
            mapper = self._item_mapper
            inv_mapper = self._item_inv_mapper
        else:
            utility_matrix = self.utility_matrix
            mapper = self._user_mapper
            inv_mapper = self._user_inv_mapper

        index = mapper[input]
        vector = utility_matrix[index]

        # k + 1 is used for n_neigbours here because the algorithm includes the input as one of the kNNs
        kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
        kNN.fit(utility_matrix)
        neighbours = kNN.kneighbors(vector, return_distance=False)

        nearest_neighbours = []
        for i in range(0,k):
            n = neighbours.item(i)
            nearest_neighbours.append(inv_mapper[n])
        nearest_neighbours.pop(0)

        return nearest_neighbours

