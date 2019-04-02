import numpy as np
from sklearn.linear_model import LogisticRegression

from HW_2 import regression
from HW_2.regression import SGDRidgeLogisticRegression
from HW_6 import utils


def demo_regressor(regressor, data):
    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    regressor.fit(training_features, training_labels)

    training_predictions = regressor.predict(training_features)
    acc, predicted_labels, threshold = utils.convert_predictions_to_labels(training_labels, training_predictions, 0)
    print("Training Acc:", acc)

    testing_predictions = regressor.predict(testing_features)
    acc, predicted_labels, threshold = utils.convert_predictions_to_labels(testing_labels, testing_predictions, 0)
    print("Testing Acc:", acc)


def demo_regularized_regression_ridge():
    data = utils.get_spam_polluted_data()
    print("+" * 40, "Ridge Regression On Polluted Spam", "+" * 40)
    demo_regressor(LogisticRegression(C=0.2, penalty='l2', random_state=11, solver='liblinear', max_iter=800), data)
    print("+" * 40, "Ridge Regression On Polluted Spam", "+" * 40)
    print()


def demo_regularized_regression_lasso():
    print("+" * 40, "Lasso Regression On Polluted Spam", "+" * 40)
    data = utils.get_spam_polluted_data()
    demo_regressor(LogisticRegression(C=0.2, penalty='l1', random_state=11, solver='liblinear'), data)
    print("+" * 40, "Lasso Regression On Polluted Spam", "+" * 40)
    print()


class RegressorWrapper:

    def __init__(self, regressor) -> None:
        super().__init__()
        self.regressor = regressor

    def fit(self, features, labels):
        self.regressor.train(features, labels)

    def predict(self, features):
        return self.regressor.predict(features)


def demo_logistic_regression():
    print("+" * 40, "Custom Logistic Regression On Polluted Spam", "+" * 40)
    data = utils.get_spam_polluted_data()
    demo_regressor(RegressorWrapper(regression.LogisticRegression()), data)
    print("+" * 40, "Custom Logistic Regression On Polluted Spam", "+" * 40)
    print()


def demo_regularized_regression_custom_ridge():
    data = utils.get_spam_polluted_data()

    training_features = data['training']['features']
    testing_features = data['testing']['features']

    combined_features = np.concatenate((training_features, testing_features))
    normalized_features = utils.normalize_data_using_zero_mean_unit_variance(combined_features)

    training_features = normalized_features[:training_features.shape[0]]
    testing_features = normalized_features[training_features.shape[0]:]

    data['training']['features'] = training_features
    data['testing']['features'] = testing_features

    print("+" * 40, "Custom Ridge Regression On Spam", "+" * 40)
    demo_regressor(RegressorWrapper(SGDRidgeLogisticRegression(0.00002, 1200, 10, True)), data)
    print("+" * 40, "Custom Ridge Regression On Spam", "+" * 40)
    print()


if __name__ == '__main__':
    np.random.seed(11)
    demo_regularized_regression_ridge()
    demo_regularized_regression_lasso()
    demo_logistic_regression()
    demo_regularized_regression_custom_ridge()
