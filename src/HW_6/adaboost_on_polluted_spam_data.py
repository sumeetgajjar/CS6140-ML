from HW_5.AdaBoost import AdaBoost, DecisionStumpType
from HW_6 import utils


def feature_analysis_for_adaboost_on_spam():
    print("+" * 40, "Optimal Decision Stump", "+" * 40)
    data = utils.get_spam_data_for_ada_boost()
    k = 10
    folds = utils.k_fold_split(k, data, seed=11, shuffle=True)

    data = folds[:1]
    training_features = data['training']['features']
    training_labels = data['training']['labels']
    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    ada_boost = AdaBoost(DecisionStumpType.OPTIMAL)
    ada_boost.train(training_features, training_labels, testing_features, testing_labels, 300, 10)

    testing_predictions = ada_boost.predict(testing_features)
    ada_boost.plot_metrics()
    utils.plot_roc_curve(testing_labels, testing_predictions)
    print("+" * 40, "Optimal Decision Stump", "+" * 40)
    print()


def demo_ada_boost_on_polluted_spam():
    print("+" * 40, "ADA Boost on Polluted Spam", "+" * 40)
    data = utils.get_polluted_spam_data_for_ada_boost()
    training_features = data['training']['features']
    training_labels = data['training']['labels']
    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    ada_boost = AdaBoost(DecisionStumpType.RANDOM)
    ada_boost.train(training_features, training_labels, testing_features, testing_labels, 13500, 200)

    testing_predictions = ada_boost.predict(testing_features)
    ada_boost.plot_metrics()

    acc, labels, thr = utils.convert_predictions_to_labels(testing_labels, testing_predictions)
    print("Testing Accuracy:", acc)

    utils.plot_roc_curve(testing_labels, testing_predictions)
    print("+" * 40, "ADA Boost on Polluted Spam", "+" * 40)
    print()


if __name__ == '__main__':
    # feature_analysis_for_adaboost_on_spam()
    demo_ada_boost_on_polluted_spam()
