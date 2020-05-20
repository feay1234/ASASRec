from abc import ABC, abstractmethod

class Recommender(ABC):

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def load_pre_train(self, pre):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def train(self, x_train, y_train, batch_size):
        pass

    @abstractmethod
    def rank(self, users, items):
        pass

    @abstractmethod
    def get_train_instances(self, train):
        pass
