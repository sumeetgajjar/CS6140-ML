import numpy as np
import pandas as pd

ROOT = '../../'


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
