import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc

ROOT = '../../'


def get_spam_missing_data():
    data = pd.read_csv('%sdata/spam_20_percent_missing/20_percent_missing_train.txt' % ROOT, delimiter=',',
                       header=None)
    training_features = np.array(data.iloc[:, :-1])
    training_labels = np.array(data.iloc[:, -1])

    data = pd.read_csv('%sdata/spam_20_percent_missing/20_percent_missing_test.txt' % ROOT, delimiter=',',
                       header=None)
    testing_features = np.array(data.iloc[:, :-1])
    testing_labels = np.array(data.iloc[:, -1])

    if training_features.shape[0] != training_labels.shape[0]:
        raise Exception("Mismatch in Training Feature Tuples(%s) and Label Tuples(%s)" % (
            training_features.shape, training_labels.shape))

    if testing_features.shape[0] != testing_labels.shape[0]:
        raise Exception("Mismatch in Testing Feature Tuples(%s) and Label Tuples(%s)" % (
            testing_features.shape, testing_labels.shape))

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


def get_spam_polluted_data():
    training_features = pd.read_csv('%sdata/spam_polluted/train_feature.txt' % ROOT, delimiter='\\s+', header=None)
    training_features = np.array(training_features.iloc[:, :])

    training_labels = pd.read_csv('%sdata/spam_polluted/train_label.txt' % ROOT, delimiter='\\s+', header=None)
    training_labels = np.array(training_labels.iloc[:, :]).flatten()

    testing_features = pd.read_csv('%sdata/spam_polluted/test_feature.txt' % ROOT, delimiter='\\s+', header=None)
    testing_features = np.array(testing_features.iloc[:, :])

    testing_labels = pd.read_csv('%sdata/spam_polluted/test_label.txt' % ROOT, delimiter='\\s+', header=None)
    testing_labels = np.array(testing_labels.iloc[:, :]).flatten()

    if training_features.shape[0] != training_labels.shape[0]:
        raise Exception("Mismatch in Training Feature Tuples(%s) and Label Tuples(%s)" % (
            training_features.shape, training_labels.shape))

    if testing_features.shape[0] != testing_labels.shape[0]:
        raise Exception("Mismatch in Testing Feature Tuples(%s) and Label Tuples(%s)" % (
            testing_features.shape, testing_labels.shape))

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


def get_polluted_spam_data_for_ada_boost():
    data = get_spam_polluted_data()
    for s in ['training', 'testing']:
        labels = data[s]['labels']
        labels[labels == 0] = -1
        data[s]['labels'] = labels
    return data


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


def plot_roc_curve(true_labels, predictions):
    fpr, tpr, threshold = roc_curve(true_labels, predictions)
    testing_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="SpamBase Data Set, auc={}".format(testing_auc))
    plt.legend(loc=4)
    plt.show()


def log(x):
    logs = np.zeros(x.shape[0])
    logs[x != 0] = np.log(x[x != 0])
    return logs


def convert_predictions_to_labels(true_labels, predictions, negative_label_value=-1):
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    best_predicted_labels = None
    max_accuracy = 0
    best_threshold = None
    for threshold in thresholds:
        predicted_labels = np.ones(true_labels.shape[0])
        predicted_labels[predictions <= threshold] = negative_label_value
        current_accuracy = accuracy_score(true_labels, predicted_labels)
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_predicted_labels = predicted_labels
            best_threshold = threshold

    return max_accuracy, best_predicted_labels, best_threshold


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
