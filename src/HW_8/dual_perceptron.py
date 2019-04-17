import numpy as np


class DualPerceptron(object):

    def __init__(self, learning_rate, kernel) -> None:
        self.learning_rate = learning_rate
        self.m = None
        self.kernel = kernel
        self.training_features = None
        self.training_labels = None

    def train(self, features, labels):
        self.training_features = features
        self.training_labels = labels
        self.m = np.zeros(features.shape[0])

        iteration = 1
        while True:

            misclassified_x = []
            for ix, x in enumerate(features):
                temp = labels[ix] * self.__get_summation(x)
                if temp <= 0:
                    self.m[ix] += labels[ix]
                    misclassified_x.append(ix)

            misclassified_x = np.array(misclassified_x)
            print("Iteration: {}, Misclassified Point: {}".format(iteration, misclassified_x.shape[0]))
            if misclassified_x.size == 0:
                break

            iteration += 1

    def __get_summation(self, x):
        return np.sum(self.m * self.kernel(x, self.training_features))

    def predict(self, features):
        y_pred = np.ones(features.shape[0])
        for ix, x in enumerate(features):
            temp = self.__get_summation(x)
            if temp <= 0:
                y_pred[ix] = -1

        return y_pred
