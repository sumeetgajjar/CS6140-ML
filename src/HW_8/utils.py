import numpy as np
import pandas as pd
from mnist import MNIST

ROOT = '../../'


def get_spam_data():
    data = pd.read_csv('%sdata/spam-email/spambase.data' % ROOT, header=None)
    features = np.array(data.iloc[:, :-1])
    labels = np.array(data.iloc[:, -1])

    if features.shape[0] != labels.shape[0]:
        raise Exception("Mismatch in Feature Tuples(%d) and Label Tuples(%d)" % (features.size, labels.size))

    return {
        'features': features,
        'labels': labels
    }


def k_fold_split(k, data, seed=11, shuffle=False):
    sample_size = data['features'].shape[0]

    indices = np.arange(0, sample_size)

    if shuffle:
        np.random.seed(seed)
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


def get_mnist_data():
    mnist = MNIST(path='%sdata/mnist' % ROOT, gz=True)
    training_images, training_labels = mnist.load_training()
    testing_images, testing_labels = mnist.load_testing()

    return {
        'training': {
            'images': np.array(training_images),
            'labels': np.array(training_labels)
        },
        'testing': {
            'images': np.array(testing_images),
            'labels': np.array(testing_labels)
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


def get_perceptron_data():
    data = pd.read_csv('%sdata/perceptron/perceptronData.txt' % ROOT, delimiter='\\s+', header=None)
    features = np.array(data.iloc[:, :-1])
    labels = np.array(data.iloc[:, -1])

    if features.shape[0] != labels.shape[0]:
        raise Exception("Mismatch in Feature Tuples(%d) and Label Tuples(%d)" % (features.size, labels.size))

    return {
        'features': features,
        'labels': labels
    }


def get_perceptron_spiral_data():
    data = pd.read_csv('%sdata/perceptron/twoSpirals.txt' % ROOT, delimiter='\\s+', header=None)
    features = np.array(data.iloc[:, :-1])
    labels = np.array(data.iloc[:, -1])

    if features.shape[0] != labels.shape[0]:
        raise Exception("Mismatch in Feature Tuples(%d) and Label Tuples(%d)" % (features.size, labels.size))

    return {
        'features': features,
        'labels': labels
    }


def prepend_one_to_feature_vectors(features_vectors):
    ones = np.ones((features_vectors.shape[0], 1))
    features_vectors = np.concatenate((ones, features_vectors), axis=1)
    return features_vectors
