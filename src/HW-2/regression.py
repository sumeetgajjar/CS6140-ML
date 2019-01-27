import numpy as np


class LinearRegression(object):

    def __init__(self) -> None:
        self.weights = None

    def train(self, features, labels):
        x_t = features.copy().transpose()
        temp = np.matmul(x_t, features)
        inverse = np.linalg.pinv(temp)
        temp = np.matmul(inverse, x_t)
        self.weights = np.matmul(temp, labels)

    def predict(self, features):
        return np.matmul(features, self.weights)


class RidgeRegression(LinearRegression):

    def __init__(self, tuning_parameter) -> None:
        super().__init__()
        self.tuning_parameter = tuning_parameter

    def train(self, features, labels):
        x_t = features.copy().transpose()
        temp = np.matmul(x_t, features)
        temp = temp + (self.tuning_parameter * np.identity(temp.shape[0]))
        inverse = np.linalg.pinv(temp)
        temp = np.matmul(inverse, x_t)
        self.weights = np.matmul(temp, labels)
