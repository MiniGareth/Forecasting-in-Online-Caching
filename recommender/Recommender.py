from abc import ABC, abstractmethod

class Recommender(ABC):

    @abstractmethod
    def recommend(self, input):
        pass

class CollaborativeFilteringRecommender(Recommender):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def recommend(self, input):
        pass