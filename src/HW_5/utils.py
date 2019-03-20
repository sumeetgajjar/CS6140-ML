import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

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


def get_spam_data_for_ada_boost():
    data = get_spam_data()
    labels = data['labels']
    labels[labels == 0] = -1
    data['labels'] = labels
    return data


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


class UciDataParser:

    def __init__(self, file) -> None:
        super().__init__()
        self.file = file
        self.config = self.__parse_config()

    def __parse_config(self):
        config = {}
        with open(self.__get_config_file_path(), mode='r') as handle:
            header = handle.readline().strip()
            splits = re.split("\\s+", header)
            if len(splits) != 3:
                raise Exception("Invalid Config File")

            config['data'] = {
                'size': int(splits[0])
            }
            config['features'] = {
                'size': {
                    'discrete': int(splits[1]),
                    'continuous': int(splits[2]),
                    'all': int(splits[1]) + int(splits[2])
                }
            }

            categorical = set()
            numeric = set()
            for index in range(config['features']['size']['all']):
                line = handle.readline().strip()
                if not line.startswith("-1000"):
                    splits = re.split("\\s+", line)
                    config['features'][index] = {
                        'distinct_values': int(splits[0]),
                        'mapping': {value: i for i, value in enumerate(sorted(splits[1:]))}
                    }

                    if config['features'][index]['distinct_values'] != len(config['features'][index]['mapping']):
                        raise Exception("Invalid Config File")

                    categorical.add(index)
                else:
                    numeric.add(index)

            config['features']['type'] = {
                'categorical': categorical,
                'numeric': numeric
            }

            splits = re.split("\\s+", handle.readline().strip())
            config['labels'] = {
                'distinct_values': int(splits[0]),
                'mapping': {value: i for i, value in enumerate(sorted(splits[1:]))}
            }

        return config

    def __get_config_file_path(self):
        return '%sdata/%s/%s.config' % (ROOT, self.file, self.file)

    def __get_data_file_path(self):
        return '%sdata/%s/%s.data' % (ROOT, self.file, self.file)

    def parse_data(self):
        no_of_features = self.config['features']['size']['all']
        data = pd.read_csv(self.__get_data_file_path(), delimiter="\\s+", header=None)
        if data.shape[1] != no_of_features + 1:
            raise Exception('Data Inconsistent with the config')

        for index in range(no_of_features):
            if index in self.config['features']:
                data[index].replace(to_replace=self.config['features'][index]['mapping'], inplace=True)

            data[index].replace('?', np.nan, inplace=True)

        data[no_of_features].replace(to_replace=self.config['labels']['mapping'], inplace=True)
        data[no_of_features].replace(0, -1, inplace=True)

        features = np.array(data.iloc[:, :-1])
        labels = np.array(data.iloc[:, -1])

        if features.shape[0] != labels.shape[0]:
            raise Exception("Mismatch in Feature Tuples(%d) and Label Tuples(%d)" % (features.size, labels.size))

        features.astype(np.float, copy=False)

        return {
            'features': features,
            'labels': labels
        }


def plot_roc_curve(true_labels, predictions):
    fpr, tpr, threshold = roc_curve(true_labels, predictions)
    testing_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="SpamBase Data Set, auc={}".format(testing_auc))
    plt.legend(loc=4)
    plt.show()


def convert_predictions_to_labels(true_labels, predictions, negative_label_value=-1):
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    best_predicted_labels = None
    max_accuracy = 0
    for threshold in thresholds:
        predicted_labels = np.ones(true_labels.shape[0])
        predicted_labels[predictions <= threshold] = negative_label_value
        current_accuracy = accuracy_score(true_labels, predicted_labels)
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_predicted_labels = predicted_labels

    return max_accuracy, best_predicted_labels
