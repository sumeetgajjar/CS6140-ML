import numpy as np
from sklearn.metrics import accuracy_score

from HW_2 import regression
from HW_2 import utils


def predict_housing_prices(regressor):
    data = utils.get_housing_data()
    training_features = utils.prepend_one_to_feature_vectors(data['training']['features'])
    testing_features = utils.prepend_one_to_feature_vectors(data['testing']['features'])

    model = regressor()
    model.train(training_features, testing_features)

    training_predictions = model.predict(training_features)
    training_mse = np.square(data['training']['prices'] - training_predictions).mean()
    print('Training MSE for housing prices', training_mse)

    testing_predictions = model.predict(testing_features)
    testing_mse = np.square(data['testing']['prices'] - testing_predictions).mean()
    print('Testing MSE for housing prices', testing_mse)


def predict_spam_labels(regressor):
    data = utils.get_spam_data()
    data['features'] = utils.prepend_one_to_feature_vectors(data['features'])

    # _data = utils.normalize_data_using_shift_and_scale(_data)
    _k = 4
    _splits = utils.k_fold_split(_k, data, shuffle=True)

    training_accuracy = []
    testing_accuracy = []
    label_threshold = 0.413

    for split in _splits:
        model = regressor()
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
    print("Linear Regression\n")
    producer = lambda: regression.LinearRegression()
    predict_housing_prices(producer)
    predict_spam_labels(producer)

    print("\n")
    print("=" * 100, '\n')
    print("Ridge Regression\n")
    producer = lambda: regression.RidgeRegression(0.034)
    predict_housing_prices(producer)
    predict_spam_labels(producer)
