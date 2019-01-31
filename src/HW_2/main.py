import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from HW_2 import regression
from HW_2 import utils
from HW_2.perceptron import Perceptron


def predict_housing_prices(regressor):
    data = utils.get_housing_data()

    training_features = data['training']['features']
    testing_features = data['testing']['features']

    combined_features = np.concatenate((training_features, testing_features))
    normalized_features = utils.normalize_data_using_zero_mean_unit_variance(combined_features)

    training_features = normalized_features[:training_features.shape[0]]
    testing_features = normalized_features[training_features.shape[0]:]

    training_features = utils.prepend_one_to_feature_vectors(training_features)
    testing_features = utils.prepend_one_to_feature_vectors(testing_features)

    model = regressor()
    model.train(training_features, data['training']['prices'])

    training_predictions = model.predict(training_features)
    training_mse = np.square(data['training']['prices'] - training_predictions).mean()
    print('Training MSE for housing prices', training_mse)

    testing_predictions = model.predict(testing_features)
    testing_mse = np.square(data['testing']['prices'] - testing_predictions).mean()
    print('Testing MSE for housing prices', testing_mse)


def plot_roc_curve(y_true, y_pred_prob):
    thresholds = np.linspace(np.min(y_pred_prob), np.max(y_pred_prob), 100)
    fpr_list = []
    tpr_list = []
    for threshold in thresholds:
        y_pred = [1 if t >= threshold else 0 for t in y_pred_prob]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    auc = -np.trapz(tpr_list, fpr_list)
    plt.plot(fpr_list, tpr_list, label="SpamBase Data Set, auc={}".format(auc))
    plt.legend(loc=4)
    plt.show()


def predict_spam_labels(regressor, plot_roc=False):
    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])
    data['features'] = utils.prepend_one_to_feature_vectors(data['features'])

    k = 4
    splits = utils.k_fold_split(k, data, shuffle=True)

    training_accuracy = []
    testing_accuracy = []
    label_threshold = 0.413

    for split in splits[:1]:
        model = regressor()
        model.train(split['training']['features'], split['training']['labels'])
        training_predictions = model.predict(split['training']['features'])
        training_predictions = [1 if t >= label_threshold else 0 for t in training_predictions]

        training_accuracy.append(accuracy_score(split['training']['labels'], training_predictions))

        testing_predictions = model.predict(split['testing']['features'])
        if plot_roc:
            plot_roc_curve(split['testing']['labels'], testing_predictions)

        testing_predictions = [1 if t >= label_threshold else 0 for t in testing_predictions]
        testing_accuracy.append(accuracy_score(split['testing']['labels'], testing_predictions))
        print(confusion_matrix(split['testing']['labels'], testing_predictions))

    # print('\nTraining Accuracy for spam labels', training_accuracy)
    print('Mean Training Accuracy for spam labels', np.mean(training_accuracy))

    # print('\nTesting Accuracy for spam labels', testing_accuracy)
    print('Mean Testing Accuracy for spam labels', np.mean(testing_accuracy))


def demo_regression():
    print("Linear Regression\n")
    producer = lambda: regression.LinearRegression()
    predict_housing_prices(producer)
    predict_spam_labels(producer, True)
    print("\n{}\n".format("=" * 100))
    print("Ridge Regression\n")
    producer = lambda: regression.RidgeRegression(0.034)
    predict_housing_prices(producer)
    predict_spam_labels(producer)
    print("\n{}\n".format("=" * 100))
    print("Linear Regression Using Stochastic Gradient Descent\n")
    producer = lambda: regression.SGDLinearRegression(0.002, 80, 11)
    predict_housing_prices(producer)
    producer = lambda: regression.SGDLinearRegression(0.001, 200, 11)
    predict_spam_labels(producer, True)
    print("\n{}\n".format("=" * 100))
    print("Linear Regression Using Batch Gradient Descent\n")
    producer = lambda: regression.BGDLinearRegression(0.002, 80, 11)
    predict_housing_prices(producer)
    print("\n{}\n".format("=" * 100))
    print("Logistic Regression Using Stochastic Gradient Descent\n")
    producer = lambda: regression.SGDLogisticRegression(0.0005, 200, 1)
    predict_spam_labels(producer)
    print("\n{}\n".format("=" * 100))
    print("Logistic Regression Using Batch Gradient Descent\n")
    producer = lambda: regression.BGDLogisticRegression(0.0005, 200, 1)
    predict_spam_labels(producer)


def demo_perceptron():
    data = utils.get_perceptron_data()
    perceptron = Perceptron(0.001, 11, 0.1)
    features = data['features']
    features = utils.normalize_data_using_zero_mean_unit_variance(features)
    features = utils.prepend_one_to_feature_vectors(features)

    labels = data['labels']
    perceptron.train(features, labels)
    print(np.transpose(perceptron.weights).tolist())


if __name__ == '__main__':
    demo_regression()
    demo_perceptron()
