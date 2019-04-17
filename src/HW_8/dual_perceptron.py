import numpy as np


class DualPerceptron(object):

    def __init__(self, kernel) -> None:
        self.m = None
        self.kernel = kernel
        self.training_features = None
        self.training_labels = None

    def train(self, features, labels, epochs=100, display_step=5):
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

            if iteration == 1 or iteration % display_step == 0:
                print("Iteration: {}, Misclassified Point: {}".format(iteration, misclassified_x.shape[0]))

            if misclassified_x.size == 0 or iteration > epochs:
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
