import numpy as np
import pandas as pd


def get_spam_data():
    data = pd.read_csv('data/spam-email/spambase.data', header=None)
    features = np.array(data.iloc[:, 0:57])
    labels = np.array(data.iloc[:, 57])

    if features.shape[0] != labels.shape[0]:
        raise Exception("Mismatch in Feature Tuples(%d) and Label Tuples(%d)" % (features.size, labels.size))

    return {
        'features': features,
        'labels': labels
    }


def get_housing_data():
    training_data = pd.read_csv('data/housing/housing_train.txt', delimiter='\\s+', header=None)
    training_features = np.array(training_data.iloc[:, 0:13])
    training_labels = np.array(training_data.iloc[:, 13])

    testing_data = pd.read_csv('data/housing/housing_test.txt', delimiter='\\s+', header=None)
    testing_features = np.array(testing_data.iloc[:, 0:13])
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


def normalize_data_using_zero_mean_unit_variance(data):
    feature_vectors = data['features']

    total_features = len(feature_vectors[0])
    i = 0
    while i < total_features:
        #     for i in [26]:
        feature_values = feature_vectors[:, i]
        mean = feature_values.mean()
        std = feature_values.std()
        normalized_values = (feature_values - mean) / std
        feature_vectors[:, i] = normalized_values

        #         feature_values = feature_vectors[:,i]
        #         f_min = feature_values.min()
        #         f_max = feature_values.max()
        #         normalized_values = (feature_values - f_min)/(f_max - f_min)
        #         feature_vectors[:,i] = normalized_values

        i += 1

    data['features'] = feature_vectors
    return data


def normalize_data_using_shift_and_scale(data):
    feature_vectors = data['features']

    total_features = len(feature_vectors[0])
    i = 0
    while i < total_features:
        feature_values = feature_vectors[:, i]
        f_min = feature_values.min()
        f_max = feature_values.max()
        normalized_values = (feature_values - f_min) / (f_max - f_min)
        feature_vectors[:, i] = normalized_values

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


def prepend_one_to_feature_vectors(data):
    features = data['features']
    ones = np.ones((features.shape[0], 1))
    features = np.concatenate((ones, features), axis=1)
    data['features'] = features
    return data


def get_spam_data_for_regression():
    data = get_spam_data()
    data['features'] = prepend_one_to_feature_vectors(data['features'])
    return data


def get_housing_data_for_regression():
    data = get_housing_data()
    training_data = data['training']
    training_data = prepend_one_to_feature_vectors(training_data)

    testing_data = data['testing']
    testing_data = prepend_one_to_feature_vectors(testing_data)

    data['training'] = training_data
    data['testing'] = testing_data
    return data
