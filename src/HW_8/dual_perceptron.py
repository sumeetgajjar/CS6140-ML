import numpy as np
from sklearn.metrics import accuracy_score

from HW_2 import utils


class DualPerceptron(object):

    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate
        self.weights = None
        self.normalized_weights = None
        self.m = None

    def train(self, features, labels):
        self.weights = np.random.random((features.shape[1], 1))
        self.m = np.zeros(features.shape[0])

        iteration = 1
        while True:
            y_pred_label = self.predict(features)
            misclassified_x = []
            i = 0
            for y_pred in y_pred_label:
                if y_pred < 0:
                    misclassified_x.append(features[i])
                i += 1

            misclassified_x = np.array(misclassified_x)

            print("Iteration: {}, Misclassified Point: {}".format(iteration, misclassified_x.shape[0]))

            if misclassified_x.size == 0:
                break

            x_t = np.transpose(misclassified_x)
            diff = (self.learning_rate * np.sum(x_t, axis=1))
            self.weights = self.weights + np.reshape(diff, (self.weights.shape[0], 1))

            iteration += 1

        self.normalized_weights = self.weights.ravel()[1:] / (-self.weights[0])

    def predict(self, features):
        y_pred = np.matmul(features, self.weights).flatten()
        # y_pred_label = np.fromiter(map(lambda x: 1 if x > 0 else -1, y_pred), dtype=np.int)
        y_pred_label = np.ones(features.shape[0])
        y_pred_label[y_pred < 0] = -1
        return y_pred_label


def demo_perceptron():
    data = utils.get_perceptron_data()
    perceptron = DualPerceptron(0.02)
    features = data['features']
    features = utils.normalize_data_using_zero_mean_unit_variance(features)
    features = utils.prepend_one_to_feature_vectors(features)

    labels = data['labels']
    perceptron.train(features, labels)
    print("Raw Weights: ", perceptron.weights.ravel().tolist())
    print("Normalized Weights: ", perceptron.normalized_weights.ravel().tolist())

    predicted_labels = perceptron.predict(features)
    print("Accuracy: ", accuracy_score(labels, predicted_labels))


if __name__ == '__main__':
    np.random.seed(11)
    demo_perceptron()
