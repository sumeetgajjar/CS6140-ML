import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer

from HW_5 import utils
from HW_5.AdaBoost import AdaBoost, DecisionStumpType
from HW_5.utils import UciDataParser


def fill_missing_data(data, config):
    numeric_features = config['features']['type']['numeric']
    features = data['features']
    for index in numeric_features:
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean", copy=False)
        features[:, index] = np.reshape(imputer.fit_transform(np.reshape(features[:, index], (-1, 1))), (-1,))

    categorical_features = config['features']['type']['categorical']
    for index in categorical_features:
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent", copy=False)
        features[:, index] = np.reshape(imputer.fit_transform(np.reshape(features[:, index], (-1, 1))), (-1,))

    data['features'] = features
    return data


def demo_uci_crx_optimal_decision_stump():
    print("+" * 40, "CRX", "+" * 40)
    parser = UciDataParser('crx')
    data = parser.parse_data()
    config = parser.config
    data = fill_missing_data(data, config)

    k = 10
    folds = utils.k_fold_split(k, data, seed=1, shuffle=True)

    for data in folds[:1]:
        training_features = data['training']['features']
        training_labels = data['training']['labels']
        testing_features = data['testing']['features']
        testing_labels = data['testing']['labels']

        ada_boost = AdaBoost(DecisionStumpType.RANDOM)
        ada_boost.train(training_features, training_labels, testing_features, testing_labels, 200, 20)

        testing_predictions = ada_boost.predict(testing_features)
        ada_boost.plot_metrics()
        utils.plot_roc_curve(testing_labels, testing_predictions)

        acc, labels, thr = utils.convert_predictions_to_labels(testing_labels, testing_predictions)
        print("Testing Accuracy: ", acc)

    print("+" * 40, "CRX", "+" * 40)
    print()


def demo_uci_vote_optimal_decision_stump():
    print("+" * 40, "VOTE", "+" * 40)
    parser = UciDataParser('vote')
    data = parser.parse_data()
    config = parser.config
    data = fill_missing_data(data, config)

    k = 10
    folds = utils.k_fold_split(k, data, seed=11, shuffle=True)

    for data in folds[:1]:
        training_features = data['training']['features']
        training_labels = data['training']['labels']
        testing_features = data['testing']['features']
        testing_labels = data['testing']['labels']

        ada_boost = AdaBoost(DecisionStumpType.OPTIMAL)
        ada_boost.train(training_features, training_labels, testing_features, testing_labels, 12, 2)

        testing_predictions = ada_boost.predict(testing_features)
        ada_boost.plot_metrics()
        utils.plot_roc_curve(testing_labels, testing_predictions)

        acc, labels, thr = utils.convert_predictions_to_labels(testing_labels, testing_predictions)
        print("Testing Accuracy: ", acc)

    print("+" * 40, "VOTE", "+" * 40)
    print()


def demo_uci_vote_optimal_decision_stump_for_c_percent_data():
    parser = UciDataParser('vote')
    data = parser.parse_data()
    config = parser.config
    data = fill_missing_data(data, config)

    k = 10
    folds = utils.k_fold_split(k, data, seed=11, shuffle=True)

    for data in folds[:1]:
        training_features = data['training']['features']
        training_labels = data['training']['labels']
        testing_features = data['testing']['features']
        testing_labels = data['testing']['labels']

        training_acc = []
        testing_acc = []
        proportions = [5, 10, 15, 20, 30, 50, 80]
        for c in proportions:
            print("+" * 40, "%s %% Training data" % c, "+" * 20)
            data_points = int(np.ceil((c / 100) * training_features.shape[0]))
            training_features_subset = training_features[:data_points]
            training_labels_subset = training_labels[:data_points]

            ada_boost = AdaBoost(DecisionStumpType.OPTIMAL)
            ada_boost.train(training_features_subset, training_labels_subset, testing_features, testing_labels, 20, 5)

            training_predictions = ada_boost.predict(training_features_subset)
            acc, labels, thr = utils.convert_predictions_to_labels(training_labels_subset, training_predictions)
            training_acc.append(acc)

            testing_predictions = ada_boost.predict(testing_features)
            acc, labels, thr = utils.convert_predictions_to_labels(testing_labels, testing_predictions)
            testing_acc.append(acc)

            ada_boost.plot_metrics()
            print()

        print("Training Acc:", training_acc)
        print("Testing Acc:", testing_acc)
        plt.plot(list(range(len(proportions))), training_acc, label="Training Acc", c="Orange")
        plt.plot(list(range(len(proportions))), testing_acc, label="Testing Acc", c="Blue")
        plt.legend(loc=4)
        plt.show()


if __name__ == '__main__':
    np.random.seed(2)
    # demo_uci_crx_optimal_decision_stump()
    # demo_uci_vote_optimal_decision_stump()
    demo_uci_vote_optimal_decision_stump_for_c_percent_data()
