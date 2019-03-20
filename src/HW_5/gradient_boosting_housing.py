import numpy as np
from sklearn.metrics import mean_squared_error

from HW_1 import DecisionRegressionTree
from HW_5 import utils


class GradientBoosting:

    def __init__(self, iteration) -> None:
        super().__init__()
        self.iteration = iteration
        self.regressors = []

    def train(self, features, prices):
        current_training_prices = prices.copy()
        for i in range(0, self.iteration):
            tree = DecisionRegressionTree.create_tree(features, current_training_prices, 2, 2)
            self.regressors.append(tree)

            current_training_predicted_prices = tree.predict(features)
            current_training_prices = current_training_prices - current_training_predicted_prices

            mse = mean_squared_error(current_training_prices, current_training_predicted_prices)

            print("Round:{}, Training MSE:{}".format(i + 1, mse))

    def predict(self, features):
        predicted_prices = []
        for regressor in self.regressors:
            predicted_prices.append(regressor.predict(features))

        return np.sum(np.array(predicted_prices), axis=0)


def demo_gradient_boosted_trees_on_housing():
    data = utils.get_housing_data()
    training_features = data['training']['features']
    training_prices = data['training']['prices']
    testing_features = data['testing']['features']
    testing_prices = data['testing']['prices']

    regressor = GradientBoosting(6)
    regressor.train(training_features, training_prices)

    predicted_training_prices = regressor.predict(training_features)

    print()
    print("Training MSE without Boosting:",
          mean_squared_error(training_prices, regressor.regressors[0].predict(training_features)))
    print("Training MSE with Boosting:", mean_squared_error(training_prices, predicted_training_prices))

    predicted_testing_prices = regressor.predict(testing_features)
    print("Testing MSE without Boosting:",
          mean_squared_error(testing_prices, regressor.regressors[0].predict(testing_features)))
    print("Testing MSE with Boosting:", mean_squared_error(testing_prices, predicted_testing_prices))


if __name__ == '__main__':
    demo_gradient_boosted_trees_on_housing()
