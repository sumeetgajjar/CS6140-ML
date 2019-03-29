from sklearn.linear_model import LogisticRegression

from HW_2.regression import SGDLinearRegression, BGDLogisticRegression
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


class LogisticRegressionWrapper(BGDLogisticRegression):

    def fit(self, features, labels):
        self.train(features, labels)


class ALogisticRegressionWrapper(SGDLinearRegression):

    def fit(self, features, labels):
        self.train(features, labels)


def demo_logistic_regression():
    print("+" * 40, "Custom Logistic Regression On Polluted Spam", "+" * 40)
    data = utils.get_spam_polluted_data()
    demo_regressor(LogisticRegressionWrapper(0.0000001, 20, 121, debug=True), data)
    # demo_regressor(ALogisticRegressionWrapper(0.0000003, 8000, 11, debug=True), data)
    print("+" * 40, "Custom Logistic Regression On Polluted Spam", "+" * 40)
    print()


if __name__ == '__main__':
    # demo_logistic_regression_on_polluted_spam()
    # demo_regularized_regression_ridge()
    # demo_regularized_regression_lasso()
    demo_logistic_regression()
