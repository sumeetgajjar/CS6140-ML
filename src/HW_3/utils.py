import numpy as np
import pandas as pd

ROOT = '../../'


def read_wine_data():
    training_data = pd.read_csv('{}data/wine/train_wine.csv'.format(ROOT), header=None)
    training_labels = training_data.iloc[:, 0]
    training_features = training_data.iloc[:, 1:]

    testing_data = pd.read_csv('{}data/wine/test_wine.csv'.format(ROOT), header=None)
    testing_labels = testing_data.iloc[:, 0]
    testing_features = testing_data.iloc[:, 1:]

    return {
        'training': {
            'features': training_features,
            'labels': training_labels
        },
        'testing': {
            'features': testing_features,
            'labels': testing_labels
        }
    }


def normalize_data_using_zero_mean_unit_variance(feature_vectors):
    total_features = len(feature_vectors[0])
    i = 0
    while i < total_features:
        feature_values = feature_vectors[:, i]
        mean = feature_values.mean()
        std = feature_values.std()
        normalized_values = (feature_values - mean) / std
        feature_vectors[:, i] = normalized_values

        i += 1

    return feature_vectors


def normalize_data_using_shift_and_scale(feature_vectors):
    total_features = len(feature_vectors[0])
    i = 0
    while i < total_features:
        feature_values = feature_vectors[:, i]
        f_min = feature_values.min()
        f_max = feature_values.max()
        normalized_values = (feature_values - f_min) / (f_max - f_min)
        feature_vectors[:, i] = normalized_values

        i += 1

    return feature_vectors


def get_input_for_encoder(dim):
    features = np.repeat(np.zeros((1, dim), dtype=np.int), dim, axis=0)
    for i in range(0, 8):
        features[i][i] = 1
    return features


def one_hot_encode_wine_classifier_labels(label):
    label = label.copy()
    label = label - 1
    unique_values = np.unique(label).shape[0]
    label_size = label.shape[0]
    hot_encoded_vector = np.zeros((label_size, unique_values), dtype=int)
    hot_encoded_vector[np.arange(label_size), label] = 1
    return hot_encoded_vector


def prepend_one_to_feature_vectors(features_vectors):
    ones = np.ones((features_vectors.shape[0], 1))
    features_vectors = np.concatenate((ones, features_vectors), axis=1)
    return features_vectors
