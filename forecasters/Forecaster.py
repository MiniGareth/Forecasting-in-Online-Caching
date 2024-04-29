from abc import ABC, abstractmethod


class Forecaster(ABC):
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, history):
        pass