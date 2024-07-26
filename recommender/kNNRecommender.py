import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from recommender.Recommender import CollaborativeFilteringRecommender

class kNNRecommender(CollaborativeFilteringRecommender):
    """
    A collaborative filtering recommender using the KNN algorithm.
    """
    def __init__(self, k:int, metric="cosine"):
        self.k = k
        self.metric = metric
        # Train method needs to be called to fill in the attributes
        self.dataframe: pd.DataFrame = None
        self.ratings: str = ""
        self.users: str = ""
        self.items: str = ""

        # Create utility matrix and the the mappers of this matrix
        self.utility_matrix = None
        self._user_mapper: dict = None
        self._item_mapper: dict = None
        self._user_inv_mapper: dict = None
        self._item_inv_mapper: dict = None

        # The two kNN algorithms
        self._item_kNN = None
        self._user_kNN = None

    def train(self, dataframe: pd.DataFrame, users: str, items: str, ratings: str):
        """
        Method that trains the recommender given the dataframe and the relevant columns
        :param dataframe: Dataframe of the data
        :param users: Column to be considered users
        :param items: Column to be considered items
        :param ratings: Column to be considered ratings
        :return:
        """
        self.dataframe = dataframe
        self.users = users
        self.items = items
        self.ratings = ratings
        self._create_utility_matrix()
        # Train kNN algorithms
        # k + 1 is used for n_neigbours here because the algorithm includes the input as one of the kNNs
        self._user_kNN = NearestNeighbors(n_neighbors=self.k + 1, algorithm="brute", metric=self.metric)
        self._user_kNN.fit(self.utility_matrix)
        # k + 1 is used for n_neigbours here because the algorithm includes the input as one of the kNNs
        self._item_kNN = NearestNeighbors(n_neighbors=self.k + 1, algorithm="brute", metric=self.metric)
        self._item_kNN.fit(self.utility_matrix.T)


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

    def recommend(self, input, type="item") -> list:
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
            kNN = self._item_kNN
        else:
            utility_matrix = self.utility_matrix
            mapper = self._user_mapper
            inv_mapper = self._user_inv_mapper
            kNN = self._user_kNN

        index = mapper[input]
        vector = utility_matrix[index]

        neighbours = kNN.kneighbors(vector, return_distance=False)

        nearest_neighbours = []
        for i in range(0, self.k+1):
            n = neighbours.item(i)
            nearest_neighbours.append(inv_mapper[n])
        nearest_neighbours.pop(0)

        return nearest_neighbours

