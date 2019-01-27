import numpy as np
from sklearn.metrics import accuracy_score

import utils


class LinearRegressor(object):

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


def predict_housing_prices():
    data = utils.get_housing_data_for_regression()
    model = LinearRegressor()
    model.train(data['training']['features'], data['training']['prices'])

    training_predictions = model.predict(data['training']['features'])
    training_mse = np.square(data['training']['prices'] - training_predictions).mean()
    print('Training MSE for housing prices', training_mse)

    testing_predictions = model.predict(data['testing']['features'])
    testing_mse = np.square(data['testing']['prices'] - testing_predictions).mean()
    print('Testing MSE for housing prices', testing_mse)


def predict_spam_labels():
    _data = utils.get_spam_data_for_regression()
    # _data = utils.normalize_data_using_shift_and_scale(_data)
    _k = 4
    _splits = utils.k_fold_split(_k, _data, shuffle=True)

    training_accuracy = []
    testing_accuracy = []
    label_threshold = 0.413

    for split in _splits:
        model = LinearRegressor()
        model.train(split['training']['features'], split['training']['labels'])
        training_predictions = model.predict(split['training']['features'])
        training_predictions = [1 if t >= label_threshold else 0 for t in training_predictions]

        training_accuracy.append(accuracy_score(split['training']['labels'], training_predictions))

        testing_predictions = model.predict(split['testing']['features'])
        testing_predictions = [1 if t >= label_threshold else 0 for t in testing_predictions]

        testing_accuracy.append(accuracy_score(split['testing']['labels'], testing_predictions))

    print('\n')

    print('Training Accuracy for spam labels', training_accuracy)
    print('Mean Training Accuracy for spam labels', np.mean(training_accuracy))

    print('Testing Accuracy for spam labels', testing_accuracy)
    print('Mean Testing Accuracy for spam labels', np.mean(testing_accuracy))


if __name__ == '__main__':
    predict_housing_prices()
    predict_spam_labels()
