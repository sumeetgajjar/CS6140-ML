import numpy as np
from sklearn.metrics import accuracy_score

from HW_2 import utils


class Perceptron(object):

    def __init__(self, learning_rate, seed) -> None:
        self.seed = seed
        self.learning_rate = learning_rate
        self.weights = None
        self.normalized_weights = None
        np.random.seed(self.seed)

    def pre_processing(self, features, labels):

        i = 0
        n = labels.shape[0]
        while i < n:
            if labels[i] == -1:
                features[i] = -features[i]
                labels[i] = -labels[i]

            i += 1

        return features, labels

    def train(self, features, labels):
        self.weights = np.random.random((features.shape[1], 1))
        features, labels = self.pre_processing(features, labels)

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
    perceptron = Perceptron(0.02, 11)
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
    demo_perceptron()
