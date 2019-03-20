import numpy as np
from matplotlib import pylab as plt
from sklearn.metrics import accuracy_score

from HW_1 import DecisionClassificationTree
from HW_5 import utils


def random_sample_data(features, labels, N):
    random_indexes = np.random.random_integers(0, features.shape[0] - 1, N)
    return features[random_indexes], labels[random_indexes]


def demo_bagging_on_spam():
    data = utils.get_spam_data()
    bags, N, k = 50, 100, 10
    folds = utils.k_fold_split(k, data, seed=11, shuffle=True)
    training_features = folds[0]['training']['features']
    testing_features = folds[0]['testing']['features']
    training_labels = folds[0]['training']['labels']
    testing_labels = folds[0]['testing']['labels']

    training_acc = []
    testing_acc = []
    classifiers = []
    testing_predictions = []
    for t in range(1, bags + 1):
        sampled_training_features, sampled_training_labels = random_sample_data(training_features, training_labels, N)

        tree = DecisionClassificationTree.create_tree(sampled_training_features, sampled_training_labels, 2, 4)
        classifiers.append(tree)

        current_training_predictions = tree.predict(sampled_training_features)
        current_training_acc = accuracy_score(sampled_training_labels, current_training_predictions)
        training_acc.append(current_training_acc)

        current_testing_predictions = tree.predict(testing_features)
        current_testing_accuracy = accuracy_score(testing_labels, current_testing_predictions)
        testing_acc.append(current_testing_accuracy)
        testing_predictions.append(current_testing_predictions)

        print("Bag:{}, Training Acc:{}, Testing Acc:{}".format(t, current_training_acc, current_testing_accuracy))

    acc = accuracy_score(testing_labels, testing_predictions[0])
    print("Testing Accuracy without Bagging: ", acc)

    testing_predictions = np.mean(np.array(testing_predictions), axis=0)
    acc, labels = utils.convert_predictions_to_labels(testing_labels, testing_predictions, 0)
    print("Testing Accuracy with Bagging: ", acc)

    plt.plot(range(bags), training_acc, label="Training Acc", c="Orange")
    plt.plot(range(bags), testing_acc, label="Testing Acc", c="Blue")
    plt.legend(loc=4)
    plt.show()

    utils.plot_roc_curve(testing_labels, testing_predictions)


if __name__ == '__main__':
    np.random.seed(11)
    demo_bagging_on_spam()
