import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


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


def get_housing_data():
    training_data = pd.read_csv('data/housing/housing_train.txt', delimiter='\\s+', header=None)
    training_features = np.array(training_data.iloc[:, 0:13])
    ones = np.ones((training_features.shape[0], 1))
    training_features = np.concatenate((ones, training_features), axis=1)

    training_labels = np.array(training_data.iloc[:, 13])

    testing_data = pd.read_csv('data/housing/housing_test.txt', delimiter='\\s+', header=None)
    testing_features = np.array(testing_data.iloc[:, 0:13])
    ones = np.ones((testing_features.shape[0], 1))
    testing_features = np.concatenate((ones, testing_features), axis=1)

    testing_labels = np.array(testing_data.iloc[:, 13])

    if training_features.shape[0] != training_labels.shape[0]:
        raise Exception("Mismatch in Training Feature Tuples(%s) and Label Tuples(%s)" % (
            training_features.shape, training_labels.shape))

    if testing_features.shape[0] != testing_labels.shape[0]:
        raise Exception("Mismatch in Testing Feature Tuples(%s) and Label Tuples(%s)" % (
            testing_features.shape, testing_labels.shape))

    return {
        'training': {
            'features': training_features,
            'prices': training_labels
        },
        'testing': {
            'features': testing_features,
            'prices': testing_labels
        }
    }


def predict_housing_prices():
    data = get_housing_data()
    model = LinearRegressor()
    model.train(data['training']['features'], data['training']['prices'])

    training_predictions = model.predict(data['training']['features'])
    training_mse = np.square(data['training']['prices'] - training_predictions).mean()
    print('Training MSE for housing prices', training_mse)

    testing_predictions = model.predict(data['testing']['features'])
    testing_mse = np.square(data['testing']['prices'] - testing_predictions).mean()
    print('Testing MSE for housing prices', testing_mse)


def get_spam_data():
    data = pd.read_csv('data/spam-email/spambase.data', header=None)
    features = np.array(data.iloc[:, 0:57])

    ones = np.ones((features.shape[0], 1))
    features = np.concatenate((ones, features), axis=1)

    labels = np.array(data.iloc[:, 57])

    if features.shape[0] != labels.shape[0]:
        raise Exception("Mismatch in Feature Tuples(%d) and Label Tuples(%d)" % (features.size, labels.size))

    return {
        'features': features,
        'labels': labels
    }


def normalize_data(data):
    feature_vectors = data['features']

    total_features = len(feature_vectors[0])
    i = 1
    while i < total_features:
        feature_values = feature_vectors[:, i]
        mean = feature_values.mean()
        std = feature_values.std()
        normalized_values = (feature_values - mean) / std
        feature_vectors[:, i] = normalized_values

        # feature_values = feature_vectors[:, i]
        # f_min = feature_values.min()
        # f_max = feature_values.max()
        # normalized_values = (feature_values - f_min) / (f_max - f_min)
        # feature_vectors[:, i] = normalized_values

        i += 1

    data['features'] = feature_vectors
    return data


def k_fold_split(k, data, shuffle=False):
    sample_size = data['features'].shape[0]

    indices = np.arange(0, sample_size)

    if shuffle:
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    testing_fold_index = 0
    final_data = []

    for i in range(0, k):
        training_folds = [folds[j] for j in range(0, k) if j != testing_fold_index]
        training_data_indices = np.concatenate(training_folds)
        training_data_features = data['features'][training_data_indices]
        training_data_labels = data['labels'][training_data_indices]

        testing_data_indices = folds[testing_fold_index]
        testing_data_features = data['features'][testing_data_indices]
        testing_data_labels = data['labels'][testing_data_indices]

        temp = {
            'training': {
                'features': training_data_features,
                'labels': training_data_labels
            },
            'testing': {
                'features': testing_data_features,
                'labels': testing_data_labels
            }
        }

        final_data.append(temp)
        testing_fold_index += 1

    return final_data


def predict_spam_labels():
    _data = get_spam_data()
    _data = normalize_data(_data)
    _k = 4
    _splits = k_fold_split(_k, _data, shuffle=True)

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
