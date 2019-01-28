import numpy as np

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

    def __init__(self, learning_parameter, epochs) -> None:
        super().__init__()
        self.learning_rate = learning_parameter
        self.epochs = epochs
        self.list = []

    def train(self, features, labels):
        np.random.seed(123)
        self.weights = np.random.random((features.shape[1], 1))

        total_features = self.weights.shape[0]
        for i in range(self.epochs):
            for t in range(features.shape[0]):
                x_t = features[t]
                y_t = labels[t]
                for j in range(total_features):
                    h_x = self.predict(x_t)
                    self.weights[j] = self.weights[j] - (self.learning_rate * (h_x - y_t) * x_t[j])

            if i % 200 == 0:
                print(np.transpose(self.weights).tolist())
                self.list.append(self.weights.tolist())


class BGDLinearRegression(SGDLinearRegression):

    def train(self, features, labels):
        self.weights = np.random.random((features.shape[1], 1))

        total_features = self.weights.shape[0]
        for i in range(self.epochs):
            for j in range(total_features):
                temp_sum = 0.0
                for t in range(features.shape[0]):
                    x_t = features[t]
                    y_t = labels[t]
                    h_x = self.predict(x_t)
                    temp_sum = temp_sum + ((h_x - y_t) * x_t[j])

                self.weights[j] = self.weights[j] - (self.learning_rate * temp_sum)


if __name__ == '__main__':
    data = utils.get_housing_data()
    # data = utils.get_housing_data_for_regression()

    training_features = data['training']['features']
    testing_features = data['testing']['features']
    combined_features = {
        'features': np.concatenate((training_features, testing_features))
    }

    # normalized_features = utils.normalize_data_using_zero_mean_unit_variance(combined_features)
    # training_features = normalized_features['features'][:training_features.shape[0]]
    # testing_features = normalized_features['features'][training_features.shape[0]:]

    # training_features = np.concatenate((np.ones((training_features.shape[0], 1)), training_features), axis=1)
    # testing_features = np.concatenate((np.ones((testing_features.shape[0], 1)), testing_features), axis=1)

    for i in range(1000, 1001):
        model = SGDLinearRegression(0.00001, 100000)
        model.train(training_features, data['training']['prices'])

        training_predictions = model.predict(training_features)
        training_mse = np.square(data['training']['prices'] - training_predictions).mean()
        print('Training MSE for housing prices', training_mse)

        testing_predictions = model.predict(testing_features)
        testing_mse = np.square(data['testing']['prices'] - testing_predictions).mean()
        print('Testing MSE for housing prices', testing_mse)
