import numpy as np
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
    parser = UciDataParser('crx')
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
        ada_boost.train(training_features, training_labels, testing_features, testing_labels, 100)

        testing_predictions = ada_boost.predict(testing_features)
        ada_boost.plot_metrics()
        utils.plot_roc_curve(testing_labels, testing_predictions)


def demo_uci_vote_optimal_decision_stump():
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
        ada_boost.train(training_features, training_labels, testing_features, testing_labels, 20)

        testing_predictions = ada_boost.predict(testing_features)
        ada_boost.plot_metrics()
        utils.plot_roc_curve(testing_labels, testing_predictions)


if __name__ == '__main__':
    np.random.seed(2)
    # demo_uci_crx_optimal_decision_stump()
    demo_uci_vote_optimal_decision_stump()
