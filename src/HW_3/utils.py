import numpy as np
import pandas as pd

ROOT = '../../'


def read_wine_data():
    data = pd.read_csv('{}data/wine/wine.txt'.format(ROOT), header=None)
    labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    return {
        'labels': labels,
        'features': features
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
