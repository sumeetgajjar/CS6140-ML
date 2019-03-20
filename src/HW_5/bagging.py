import numpy as np
from sklearn.metrics import accuracy_score

from HW_1 import DecisionClassificationTree
from HW_5 import utils


class Bagging:

    def __init__(self, bags) -> None:
        super().__init__()
        self.bags = bags
        self.classifiers = []

    @staticmethod
    def __random_sample_data(features, labels, N):
        random_indexes = np.random.random_integers(0, features.shape[0] - 1, N)
        return features[random_indexes], labels[random_indexes]

    def train(self, features, labels, N):

        for t in range(self.bags):
            sampled_training_features, sampled_training_labels = self.__random_sample_data(features, labels, N)

            tree = DecisionClassificationTree.create_tree(sampled_training_features, sampled_training_labels, 2, 4)
            self.classifiers.append(tree)

            current_training_predictions = tree.predict(sampled_training_features)
            current_training_acc = accuracy_score(sampled_training_labels, current_training_predictions)

            print("Bag:{}, Training Acc:{}".format(t + 1, current_training_acc))

    def predict(self, features):
        predicted = []
        for classifier in self.classifiers:
            predicted.append(classifier.predict(features))

        return np.mean(np.array(predicted), axis=0)


def demo_bagging_on_spam():
    data = utils.get_spam_data()
    bags, N, k = 50, 100, 10
    folds = utils.k_fold_split(k, data, seed=11, shuffle=True)
    training_features = folds[0]['training']['features']
    testing_features = folds[0]['testing']['features']
    training_labels = folds[0]['training']['labels']
    testing_labels = folds[0]['testing']['labels']

    classifier = Bagging(bags)
    classifier.train(training_features, training_labels, N)

    acc = accuracy_score(training_labels, classifier.classifiers[0].predict(training_features))
    print("Training Accuracy without Bagging:", acc)

    training_predictions = classifier.predict(training_features)
    acc, labels = utils.convert_predictions_to_labels(training_labels, training_predictions)
    print("Training Accuracy with Bagging:", acc)

    acc = accuracy_score(testing_labels, classifier.classifiers[0].predict(testing_features))
    print("Testing Accuracy without Bagging:", acc)

    testing_predictions = classifier.predict(testing_features)
    acc, labels = utils.convert_predictions_to_labels(testing_labels, testing_predictions, 0)
    print("Testing Accuracy with Bagging:", acc)

    utils.plot_roc_curve(testing_labels, testing_predictions)


if __name__ == '__main__':
    np.random.seed(11)
    demo_bagging_on_spam()
