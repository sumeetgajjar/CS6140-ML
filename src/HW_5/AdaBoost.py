from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_curve

from HW_5 import utils


class Predictor:

    def __init__(self, feature_index, threshold) -> None:
        super().__init__()
        self.feature_index = feature_index
        self.threshold = threshold

    def predict(self, features):
        y_predicted = np.ones(features.shape[0])
        y_predicted[features[:, self.feature_index] <= self.threshold] = -1
        return y_predicted


class DecisionStumpType(Enum):
    OPTIMAL = 1
    RANDOM = 2


class DecisionStump:

    def __init__(self, decision_stump_type, features, labels, d_t) -> None:
        super().__init__()
        self.decision_stump_type = decision_stump_type
        self.features = features
        self.labels = labels
        self.d_t = d_t
        self.predictor = None
        self.__initialize()

    def __initialize(self):
        if self.decision_stump_type is DecisionStumpType.OPTIMAL:
            self.find_optimal_decision_stump()
        elif self.decision_stump_type is DecisionStumpType.RANDOM:
            self.find_random_decision_stump()
        else:
            raise Exception("Invalid Decision Stump Type")

    def __get_unique_thresholds_for_feature(self, feature_index):
        all_values = self.features[:, feature_index]
        return np.append(np.unique(all_values), [np.min(all_values) - 1, np.max(all_values) + 1])

    def find_optimal_decision_stump(self):
        optimal_goal = 0
        best_predictor = None

        for i in range(self.features.shape[1]):
            unique_thresholds = self.__get_unique_thresholds_for_feature(i)
            sorted_unique_thresholds = np.sort(unique_thresholds)

            for threshold in sorted_unique_thresholds:
                predictor = Predictor(i, threshold)
                y_predicted = predictor.predict(self.features)

                error = np.sum(np.square(self.labels - y_predicted) * self.d_t)
                goal = abs(0.5 - error)

                if goal > optimal_goal:
                    optimal_goal = goal
                    best_predictor = predictor

        self.predictor = best_predictor

    def find_random_decision_stump(self):
        feature_index = np.random.randint(0, self.features.shape[1])

        unique_thresholds = self.__get_unique_thresholds_for_feature(feature_index)
        feature_threshold = unique_thresholds[np.random.randint(0, unique_thresholds.shape[0])]

        self.predictor = Predictor(feature_index, feature_threshold)

    def predict(self, features):
        return self.predictor.predict(features)


class AdaBoost:

    def __init__(self, decision_stump_type) -> None:
        super().__init__()
        self.decision_stump_type = decision_stump_type
        self.alpha = None
        self.weak_learners = []
        self.local_round_error = []
        self.running_training_error = []
        self.running_testing_error = []
        self.test_auc = []

    def get_weak_learner(self, features, true_labels, d_t):
        return DecisionStump(self.decision_stump_type, features, true_labels, d_t)

    def train(self, training_features, training_labels, testing_features, testing_labels, no_of_weak_learners,
              delta=0.000001):
        d_t = np.repeat(1 / training_features.shape[0], training_features.shape[0])

        alpha = []
        running_training_predictions = np.zeros(training_features.shape[0])
        running_testing_predictions = np.zeros(testing_features.shape[0])

        for t in range(1, no_of_weak_learners + 1):
            training_features = training_features.copy()
            # for i in range(training_features.shape[1]):
            #     training_features[:, i] = training_features[:, i] * d_t

            weak_learner = self.get_weak_learner(training_features, training_labels, d_t)
            self.weak_learners.append(weak_learner)

            training_predictions = weak_learner.predict(training_features)
            epsilon_error = d_t[training_labels != training_predictions].sum()
            self.local_round_error.append(epsilon_error)

            alpha_t = 0.5 * (np.log(1 - epsilon_error) - np.log(epsilon_error))
            alpha.append(alpha_t)

            gamma_t = 0.5 - epsilon_error
            z_t = np.sqrt(1 - (4 * np.square(gamma_t)))

            d_t = d_t * (np.exp(-alpha_t * training_labels * training_predictions)) / z_t

            # calculating the running training error
            running_training_predictions += (training_predictions * alpha_t)
            running_training_prediction_labels = np.ones(training_features.shape[0])
            running_training_prediction_labels[running_training_predictions < 0] = -1

            running_training_error = 1 - accuracy_score(training_labels, running_training_prediction_labels)
            self.running_training_error.append(running_training_error)

            # calculating the running testing error
            testing_predictions = weak_learner.predict(testing_features)
            running_testing_predictions += (testing_predictions * alpha_t)
            running_testing_prediction_labels = np.ones(testing_features.shape[0])
            running_testing_prediction_labels[running_testing_predictions < 0] = -1

            running_testing_error = 1 - accuracy_score(testing_labels, running_testing_prediction_labels)
            self.running_testing_error.append(running_testing_error)

            # calculating the testing auc
            fpr, tpr, thresholds = roc_curve(testing_labels, running_testing_prediction_labels)
            testing_auc = auc(fpr, tpr)
            self.test_auc.append(testing_auc)

            print(
                "Round {}, Feature:{}, Threshold:{} Round Err:{:.8f}, Training Err:{:.8f}, Testing Err:{:.8f}, Testing AUC:{:.8f}"
                    .format(t, weak_learner.predictor.feature_index,
                            weak_learner.predictor.threshold,
                            epsilon_error,
                            running_training_error,
                            running_testing_error,
                            testing_auc))

        self.alpha = np.array(alpha)

    def predict(self, features):
        h_t = []
        for t in range(self.alpha.shape[0]):
            predicted_labels = self.weak_learners[t].predict(features)
            h_t.append(predicted_labels)

        return np.sum(np.transpose(np.array(h_t)) * self.alpha, axis=1)

    def plot_metrics(self):
        x = range(self.alpha.shape[0])
        lw = 2
        plt.plot(x, self.local_round_error, label="Local Round Err", lw=lw, color=np.random.rand(3))
        plt.legend(loc=4)
        plt.show()

        plt.plot(x, self.running_training_error, label="Running Training Err", lw=lw, color="red")
        plt.plot(x, self.running_testing_error, label="Running Testing Err", lw=lw, color="black")
        plt.plot(x, self.test_auc, label="Test AUC", lw=lw, color="Orange")
        plt.legend(loc=4)
        plt.show()


def plot_information(testing_labels, testing_predictions):
    fpr, tpr, threshold = roc_curve(testing_labels, testing_predictions)
    testing_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="SpamBase Data Set, auc={}".format(testing_auc))
    plt.legend(loc=4)
    plt.show()


def demo_ada_boost_with_optimal_decision_stump():
    data = utils.get_spam_data_for_ada_boost()
    k = 10
    folds = utils.k_fold_split(k, data, seed=11, shuffle=True)

    training_accuracy = []
    testing_accuracy = []
    i = 0
    for data in folds[:1]:
        training_features = data['training']['features']
        training_labels = data['training']['labels']
        testing_features = data['testing']['features']
        testing_labels = data['testing']['labels']

        # ada_boost = AdaBoost(DecisionStumpType.OPTIMAL)
        ada_boost = AdaBoost(DecisionStumpType.RANDOM)
        ada_boost.train(training_features, training_labels, testing_features, testing_labels, 200)

        # training_predictions = ada_boost.predict(training_features)
        # training_accuracy.append(accuracy_score(training_labels, training_predictions))

        testing_predictions = ada_boost.predict(testing_features)
        ada_boost.plot_metrics()
        plot_information(testing_labels, testing_predictions)
        # testing_accuracy.append(accuracy_score(testing_labels, testing_predictions))

    # print("Training Accuracy:", np.mean(training_accuracy))
    # print("Testing Accuracy:", np.mean(testing_accuracy))


if __name__ == '__main__':
    np.random.seed(2)
    demo_ada_boost_with_optimal_decision_stump()
