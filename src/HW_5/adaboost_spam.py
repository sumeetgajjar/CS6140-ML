import numpy as np

from HW_5 import utils
from HW_5.AdaBoost import AdaBoost, DecisionStumpType


def demo_ada_boost_with_optimal_decision_stump():
    print("+" * 40, "Optimal Decision Stump", "+" * 40)
    data = utils.get_spam_data_for_ada_boost()
    k = 10
    folds = utils.k_fold_split(k, data, seed=11, shuffle=True)

    for data in folds[:1]:
        training_features = data['training']['features']
        training_labels = data['training']['labels']
        testing_features = data['testing']['features']
        testing_labels = data['testing']['labels']

        ada_boost = AdaBoost(DecisionStumpType.OPTIMAL)
        ada_boost.train(training_features, training_labels, testing_features, testing_labels, 200, 10)

        testing_predictions = ada_boost.predict(testing_features)
        ada_boost.plot_metrics()
        utils.plot_roc_curve(testing_labels, testing_predictions)
    print("+" * 40, "Optimal Decision Stump", "+" * 40)
    print()


def demo_ada_boost_with_random_decision_stump():
    print("+" * 40, "Random Decision Stump", "+" * 40)
    data = utils.get_spam_data_for_ada_boost()
    k = 10
    folds = utils.k_fold_split(k, data, seed=11, shuffle=True)

    for data in folds[:1]:
        training_features = data['training']['features']
        training_labels = data['training']['labels']
        testing_features = data['testing']['features']
        testing_labels = data['testing']['labels']

        ada_boost = AdaBoost(DecisionStumpType.RANDOM)
        ada_boost.train(training_features, training_labels, testing_features, testing_labels, 2000)

        testing_predictions = ada_boost.predict(testing_features)
        ada_boost.plot_metrics()
        utils.plot_roc_curve(testing_labels, testing_predictions)

        acc, labels, thr = utils.convert_predictions_to_labels(testing_labels, testing_predictions)
        print("Testing Accuracy:", acc)

    print("+" * 40, "Random Decision Stump", "+" * 40)
    print()


if __name__ == '__main__':
    np.random.seed(2)
    demo_ada_boost_with_random_decision_stump()
    # demo_ada_boost_with_optimal_decision_stump()
