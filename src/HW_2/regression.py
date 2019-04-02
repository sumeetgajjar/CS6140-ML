import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from HW_2 import utils


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


class LogisticRegression(LinearRegression):

    def predict(self, features):
        wx = super().predict(features)
        g_x = 1 / (1 + np.exp(-wx))
        return g_x


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


class SGDLinearRegression(LinearRegression):

    def __init__(self, learning_parameter, epochs, seed=1, debug=False) -> None:
        super().__init__()
        self.learning_rate = learning_parameter
        self.epochs = epochs
        self.debug = debug
        self.seed = seed
        np.random.seed(self.seed)

    def train(self, features, labels):
        self.weights = np.random.random(features.shape[1])

        for i in range(self.epochs):
            for t in range(features.shape[0]):
                x_t = features[t]
                y_t = labels[t]
                h_x = self.predict(x_t)
                diff = h_x - y_t
                self.weights = self.weights - (x_t * (self.learning_rate * diff))

            if self.debug and i % 20 == 0:
                print("Step=>{}".format(i), np.transpose(self.weights).tolist())


class SGDLogisticRegression(SGDLinearRegression):

    def predict(self, features):
        wx = super().predict(features)
        g_x = 1 / (1 + np.exp(-wx))
        return g_x


class BGDLinearRegression(SGDLinearRegression):

    def train(self, features, labels):
        self.weights = np.random.random((features.shape[1]))

        total_features = self.weights.shape[0]
        for i in range(self.epochs):
            for j in range(total_features):
                h_x_t = self.predict(features)
                y_t_t = labels
                x_t_j = features[:, j]
                temp_sum = np.sum((h_x_t - y_t_t) * x_t_j)

                self.weights[j] = self.weights[j] - (self.learning_rate * temp_sum)

            if self.debug and i % 5 == 0:
                print("Step=>{}".format(i), np.transpose(self.weights).tolist())


class BGDLogisticRegression(BGDLinearRegression):

    def predict(self, features):
        wx = super().predict(features)
        g_x = 1 / (1 + np.exp(-wx))
        return g_x


class NewtonMethodLogisticRegression(LinearRegression):

    def __init__(self, learning_rate, epochs, seed) -> None:
        super().__init__()
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate
        np.random.seed(self.seed)

    def train(self, features, labels):
        self.weights = np.random.random((features.shape[1]))

        for k in range(self.epochs):
            x_t = np.transpose(features)
            pie_k = self.predict(features)
            one_minus_pie_k = 1 - pie_k
            temp = pie_k * one_minus_pie_k

            diag = np.identity(features.shape[0])
            np.fill_diagonal(diag, temp)

            m1 = np.matmul(x_t, diag)
            m2 = np.matmul(m1, features)

            inverse = np.linalg.pinv(m2)
            g_k = np.matmul(x_t, pie_k - labels)

            self.weights = self.weights - self.learning_rate * np.matmul(inverse, g_k)

            if k % 20 == 0:
                print(np.transpose(self.weights).tolist())

    def predict(self, features):
        wx = super().predict(features)
        g_x = 1 / (1 + np.exp(-wx))
        return g_x


def linear_regression_on_housing_data():
    data = utils.get_housing_data()
    training_features = data['training']['features']
    testing_features = data['testing']['features']

    combined_features = np.concatenate((training_features, testing_features))
    normalized_features = utils.normalize_data_using_zero_mean_unit_variance(combined_features)

    training_features = normalized_features[:training_features.shape[0]]
    testing_features = normalized_features[training_features.shape[0]:]

    training_features = utils.prepend_one_to_feature_vectors(training_features)
    testing_features = utils.prepend_one_to_feature_vectors(testing_features)

    model = SGDLinearRegression(0.0002, 800, 11)
    # model = BGDLinearRegression(0.0002, 800, 1)
    model.train(training_features, data['training']['prices'])

    training_predictions = model.predict(training_features)
    training_mse = np.square(data['training']['prices'] - training_predictions).mean()
    print('Training MSE for housing prices', training_mse)

    testing_predictions = model.predict(testing_features)
    testing_mse = np.square(data['testing']['prices'] - testing_predictions).mean()
    print('Testing MSE for housing prices', testing_mse)


def linear_regression_on_spambase_data():
    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])
    data['features'] = utils.prepend_one_to_feature_vectors(data['features'])

    k = 4
    splits = utils.k_fold_split(k, data, shuffle=True)

    training_accuracy = []
    testing_accuracy = []
    label_threshold = 0.413

    for split in splits[:1]:
        # model = BGDLinearRegression(0.0005, 1000, 1)
        model = NewtonMethodLogisticRegression(0.1, 30, 11)
        # model = SGDLogisticRegression(0.0005, 200, 1)
        model.train(split['training']['features'], split['training']['labels'])
        training_predictions = model.predict(split['training']['features'])
        training_predictions = [1 if t >= label_threshold else 0 for t in training_predictions]

        training_accuracy.append(accuracy_score(split['training']['labels'], training_predictions))

        testing_predictions = model.predict(split['testing']['features'])
        testing_predictions = [1 if t >= label_threshold else 0 for t in testing_predictions]

        testing_accuracy.append(accuracy_score(split['testing']['labels'], testing_predictions))

        confusion_matrix(split['testing']['labels'], testing_predictions)

    print('\n')

    print('Training Accuracy for spam labels', training_accuracy)
    print('Mean Training Accuracy for spam labels', np.mean(training_accuracy))

    print('Testing Accuracy for spam labels', testing_accuracy)
    print('Mean Testing Accuracy for spam labels', np.mean(testing_accuracy))


if __name__ == '__main__':
    # linear_regression_on_housing_data()
    linear_regression_on_spambase_data()
