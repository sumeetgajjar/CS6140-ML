from math import ceil

import numpy as np
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from HW_7 import utils
from HW_7.custom_svm import SVM


def display_2d_data(X, y):
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {-1: 'red', 1: 'blue'}
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


def demo_custom_svm_on_spam_data():
    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])

    labels = data['labels']
    labels[labels == 0] = -1
    data['labels'] = labels

    training_acc = []
    testing_acc = []
    for fold in utils.k_fold_split(10, data, seed=11, shuffle=True)[:1]:
        training_features = fold['training']['features']
        training_labels = fold['training']['labels']

        testing_features = fold['testing']['features']
        testing_labels = fold['testing']['labels']

        classifier = SVM(0.01, 1e-2, max_passes=100, max_iterations=100)
        classifier.train(training_features, training_labels)

        pred_training_labels = classifier.predict(training_features)
        acc = accuracy_score(training_labels, pred_training_labels)
        training_acc.append(acc)

        pred_testing_labels = classifier.predict(testing_features)
        acc = accuracy_score(testing_labels, pred_testing_labels)
        testing_acc.append(acc)

    print("Training Accuracy:", np.mean(training_acc))
    print("Testing Accuracy:", np.mean(testing_acc))


def demo_custom_svm_on_mnist_data():
    pass


if __name__ == '__main__':
    np.random.seed(11)
    # demo_custom_svm_on_2d_data()
    # demo_custom_svm_on_spam_data()
    demo_custom_svm_on_mnist_data()
