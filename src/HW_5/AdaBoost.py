from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_curve


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
        self.d_t = d_t
        self.predictor = None
        self.__initialize(features, labels)

    def __initialize(self, features, labels):
        if self.decision_stump_type is DecisionStumpType.OPTIMAL:
            self.find_optimal_decision_stump(features, labels)
        elif self.decision_stump_type is DecisionStumpType.RANDOM:
            self.find_random_decision_stump(features)
        else:
            raise Exception("Invalid Decision Stump Type")

    @staticmethod
    def __get_unique_thresholds_for_feature(features, feature_index):
        all_values = features[:, feature_index]
        return np.append(np.unique(all_values), [np.min(all_values) - 1, np.max(all_values) + 1])

    def find_optimal_decision_stump(self, features, labels):
        optimal_goal = 0
        best_predictor = None

        for i in range(features.shape[1]):
            unique_thresholds = self.__get_unique_thresholds_for_feature(features, i)
            sorted_unique_thresholds = np.sort(unique_thresholds)

            for threshold in sorted_unique_thresholds:
                predictor = Predictor(i, threshold)
                y_predicted = predictor.predict(features)

                error = np.mean(np.square(labels - y_predicted) * self.d_t)
                goal = abs(0.5 - error)

                if goal > optimal_goal:
                    optimal_goal = goal
                    best_predictor = predictor

        self.predictor = best_predictor

    def find_random_decision_stump(self, features):
        feature_index = np.random.randint(0, features.shape[1])

        unique_thresholds = self.__get_unique_thresholds_for_feature(features, feature_index)
        feature_threshold = unique_thresholds[np.random.randint(0, unique_thresholds.shape[0])]

        self.predictor = Predictor(feature_index, feature_threshold)

    def predict(self, features):
        return self.predictor.predict(features)


class AdaBoost:

    def __init__(self, decision_stump_type) -> None:
        super().__init__()
        self.decision_stump_type = decision_stump_type
        self.alpha = None
        self.feature_confidence = None
        self.weak_learners = []
        self.local_round_error = []
        self.running_training_error = []
        self.running_testing_error = []
        self.test_auc = []

    def get_weak_learner(self, features, true_labels, d_t):
        return DecisionStump(self.decision_stump_type, features, true_labels, d_t)

    def train(self, training_features, training_labels, testing_features, testing_labels, no_of_weak_learners,
              display_step=100, display=True, calculate_running_error=True):

        d_t = np.repeat(1 / training_features.shape[0], training_features.shape[0])

        alpha = []
        running_training_predictions = np.zeros(training_features.shape[0])
        running_testing_predictions = np.zeros(testing_features.shape[0])

        for t in range(1, no_of_weak_learners + 1):
            weak_learner = self.get_weak_learner(training_features, training_labels, d_t)
            self.weak_learners.append(weak_learner)

            training_predictions = weak_learner.predict(training_features)
            epsilon_error = d_t[training_labels != training_predictions].sum()
            self.local_round_error.append(epsilon_error)

            log = 0 if epsilon_error == 1 else np.log(1 - epsilon_error)
            alpha_t = 0.5 * (log - np.log(epsilon_error))
            alpha.append(alpha_t)

            gamma_t = 0.5 - epsilon_error
            z_t = np.sqrt(1 - (4 * np.square(gamma_t)))

            d_t = d_t * (np.exp(-alpha_t * training_labels * training_predictions)) / z_t

            running_training_error, running_testing_error, testing_auc = None, None, None
            if calculate_running_error:
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

            if display and (t % display_step == 0 or t == 1):
                print(
                    "Round {}, Feature:{}, Threshold:{} Round Err:{}, Training Err:{}, Testing Err:{}, Testing AUC:{}"
                        .format(t, weak_learner.predictor.feature_index,
                                weak_learner.predictor.threshold,
                                epsilon_error,
                                running_training_error,
                                running_testing_error,
                                testing_auc))

        self.alpha = np.array(alpha)
        self.feature_confidence = self.__compute_feature_confidence()

    def __compute_feature_confidence(self):
        feature_confidence = {}

        total_weight = 0
        for i in range(len(self.weak_learners)):
            feature_index = self.weak_learners[i].predictor.feature_index

            if feature_index not in feature_confidence:
                feature_confidence[feature_index] = {
                    'weight': 0
                }

            feature_confidence[feature_index]['weight'] += abs(self.alpha[i])
            total_weight += abs(self.alpha[i])

        for key, value in feature_confidence.items():
            value['confidence'] = value['weight'] / total_weight

        return feature_confidence

    def predict(self, features):
        h_t = np.zeros(features.shape[0])
        for t in range(self.alpha.shape[0]):
            predicted_labels = self.weak_learners[t].predict(features)
            h_t += (predicted_labels * self.alpha[t])

        return h_t

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
