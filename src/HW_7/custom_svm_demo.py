from math import ceil

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from HW_6.image_feature_extraction import get_mnist_images_features
from HW_7 import utils
from HW_7.custom_svm import SVM, MultiClassSVM


def display_2d_data(X, y):
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'pink', 5: 'black'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()


def demo_custom_svm_on_2d_data():
    print("+" * 40, "Custom SVM on 2d Data", "+" * 40)
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=3)
    y[y == 0] = -1

    partition = ceil(X.shape[0] * 0.8)

    training_features, training_labels = X[:partition], y[:partition]
    testing_features, testing_labels = X[partition:], y[partition:]

    classifier = SVM(1, 1e-3, 100)
    classifier.train(training_features, training_labels)

    pred_training_labels = classifier.predict(training_features)
    acc = accuracy_score(training_labels, pred_training_labels)
    print("Training Accuracy:", acc)

    pred_testing_labels = classifier.predict(testing_features)
    acc = accuracy_score(testing_labels, pred_testing_labels)
    print("Testing Accuracy:", acc)

    display_2d_data(X, y)
    print("+" * 40, "Custom SVM on 2d Data", "+" * 40)
    print()


def wrapper_for_spam_data(args):
    data, _id = args

    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier = SVM(0.01, 1e-2, max_passes=100, max_iterations=100, _id=_id)
    classifier.train(training_features, training_labels)

    pred_training_labels = classifier.predict(training_features)
    training_acc = accuracy_score(training_labels, pred_training_labels)

    pred_testing_labels = classifier.predict(testing_features)
    testing_acc = accuracy_score(testing_labels, pred_testing_labels)
    return training_acc, testing_acc


def demo_custom_svm_on_spam_data():
    print("+" * 40, "Custom SVM on Spam Data", "+" * 40)

    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])

    labels = data['labels']
    labels[labels == 0] = -1
    data['labels'] = labels

    arg_list = [(fold, ix + 1) for ix, fold in enumerate(utils.k_fold_split(10, data, seed=11, shuffle=True))]
    result = Parallel(n_jobs=10, backend="threading", verbose=49)(map(delayed(wrapper_for_spam_data), arg_list))
    result = np.array(result)

    print("Training Accuracy:", np.mean(result[:, 0]))
    print("Testing Accuracy:", np.mean(result[:, 1]))
    print("+" * 40, "Custom SVM on Spam Data", "+" * 40)


def demo_custom_svm_on_multiclass_2d_data():
    print("+" * 40, "Custom Multi class SVM on 2d Data", "+" * 40)
    X, y = make_blobs(n_samples=1200, centers=6, n_features=2, cluster_std=1)

    partition = ceil(X.shape[0] * 0.8)

    training_features, training_labels = X[:partition], y[:partition]
    testing_features, testing_labels = X[partition:], y[partition:]

    classifier = MultiClassSVM(1, 1e-3, 60, 60, display=True, no_of_jobs=24)
    classifier.train(training_features, training_labels)

    pred_training_labels = classifier.predict(training_features)
    acc = accuracy_score(training_labels, pred_training_labels)
    print("Training Accuracy:", acc)

    pred_testing_labels = classifier.predict(testing_features)
    acc = accuracy_score(testing_labels, pred_testing_labels)
    print("Testing Accuracy:", acc)

    display_2d_data(X, y)
    print("+" * 40, "Custom Multi class  SVM on 2d Data", "+" * 40)
    print()


def demo_custom_svm_on_mnist_data():
    print("+" * 40, "Custom SVM on MNIST Data", "+" * 40)

    data = get_mnist_images_features(percentage=2)

    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier = MultiClassSVM(0.1, 1e-3, 40, 40, display=True, no_of_jobs=24)
    classifier.train(training_features, training_labels)

    pred_training_labels = classifier.predict(training_features)
    acc = accuracy_score(training_labels, pred_training_labels)
    print("Training Accuracy:", acc)

    pred_testing_labels = classifier.predict(testing_features)
    acc = accuracy_score(testing_labels, pred_testing_labels)
    print("Testing Accuracy:", acc)

    print("+" * 40, "Custom SVM on MNIST Data", "+" * 40)


if __name__ == '__main__':
    np.random.seed(11)
    # demo_custom_svm_on_2d_data()
    # demo_custom_svm_on_spam_data()
    demo_custom_svm_on_multiclass_2d_data()
    # demo_custom_svm_on_mnist_data()
