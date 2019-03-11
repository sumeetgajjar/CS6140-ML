import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from HW_4 import utils
from HW_4.GDA_on_spam import SpamGDA


class NaiveBayesBernoulli:

    def __init__(self, seed, dimension, k) -> None:
        super().__init__()
        self.dimension = dimension
        self.k = k
        self.seed = seed
        self.prior_p_non_spam = None
        self.prior_p_spam = None
        self.zero_non_spam_prob = None
        self.one_non_spam_prob = None
        self.zero_spam_prob = None
        self.one_spam_prob = None

    def train(self, features, labels):
        non_spam_count = labels[labels == 0].shape[0]
        spam_count = labels[labels == 1].shape[0]

        self.prior_p_non_spam = non_spam_count / features.shape[0]
        self.prior_p_spam = spam_count / features.shape[0]

        zero_non_spam_prob = []
        one_non_spam_prob = []
        zero_spam_prob = []
        one_spam_prob = []

        for i in range(self.dimension):
            f_i = features[:, i][labels == 0]
            i_zero_non_spam_count = f_i[f_i == 0].shape[0]
            p_i_zero_non_spam = (i_zero_non_spam_count + self.k) / (non_spam_count + (2 * self.k))

            i_one_non_spam_count = f_i[f_i == 1].shape[0]
            p_i_one_non_spam = (i_one_non_spam_count + self.k) / (non_spam_count + (2 * self.k))

            f_i = features[:, i][labels == 1]
            i_zero_spam_count = f_i[f_i == 0].shape[0]
            p_i_zero_spam = (i_zero_spam_count + self.k) / (spam_count + (2 * self.k))

            i_one_spam_count = f_i[f_i == 1].shape[0]
            p_i_one_spam = (i_one_spam_count + self.k) / (spam_count + (2 * self.k))

            zero_non_spam_prob.append(p_i_zero_non_spam)
            one_non_spam_prob.append(p_i_one_non_spam)
            zero_spam_prob.append(p_i_zero_spam)
            one_spam_prob.append(p_i_one_spam)

        self.zero_non_spam_prob = np.array(zero_non_spam_prob)
        self.one_non_spam_prob = np.array(one_non_spam_prob)
        self.zero_spam_prob = np.array(zero_spam_prob)
        self.one_spam_prob = np.array(one_spam_prob)

    def predict(self, features):

        predicted_non_spam_probs = []
        predicted_spam_probs = []

        for f in features:
            p_non_spam = np.product(self.zero_non_spam_prob[f == 0])
            p_non_spam *= np.product(self.one_non_spam_prob[f == 1]) * self.prior_p_non_spam

            p_spam = np.product(self.zero_spam_prob[f == 0])
            p_spam *= np.product(self.one_spam_prob[f == 1]) * self.prior_p_spam

            predicted_non_spam_probs.append(p_non_spam)
            predicted_spam_probs.append(p_spam)

        return np.column_stack((predicted_non_spam_probs, predicted_spam_probs))

    @staticmethod
    def convert_continuous_features_to_discrete(features):
        features = features.copy()
        mean = features.mean(axis=0)

        dimension = features.shape[1]
        for i in range(dimension):
            f_i = features[:, i]
            f_i[f_i <= mean[i]] = 0
            f_i[f_i > mean[i]] = 1

        return features


class NaiveBayesGaussian:

    def __init__(self) -> None:
        super().__init__()
        self.gda = None

    def train(self, features, labels):
        self.gda = SpamGDA(0.1)
        self.gda.train(features, labels)
        self.gda.sigma[~np.eye(self.gda.sigma.shape[0], dtype=bool)] = 0

    def predict(self, features):
        return self.gda.predict(features)


class NaiveBayesFourBins:

    def __init__(self, seed, dimension, k) -> None:
        super().__init__()
        self.dimension = dimension
        self.k = k
        self.seed = seed
        self.prior_p_non_spam = None
        self.prior_p_spam = None

    def train(self, features, labels):
        non_spam_count = labels[labels == 0].shape[0]
        spam_count = labels[labels == 1].shape[0]

        self.prior_p_non_spam = non_spam_count / features.shape[0]
        self.prior_p_spam = spam_count / features.shape[0]


def print_info_table(info_table):
    table = PrettyTable(["Fold", "FP Rate", "FN Rate", "Error Rate"])

    avg_fp_rate = np.mean([info[1] for info in info_table])
    avg_fn_rate = np.mean([info[2] for info in info_table])
    avg_error_rate = np.mean([info[3] for info in info_table])

    info_table.append(["Avg", avg_fp_rate, avg_fn_rate, avg_error_rate])

    for info in info_table:
        table.add_row(info)

    print(table)


def plot_roc_curve(testing_true_labels, testing_predictions):
    testing_predictions = testing_predictions.copy()

    log_odds = np.log(testing_predictions[:, 1]) - np.log(testing_predictions[:, 0])
    sorted_indices = np.argsort(log_odds)

    fpr, tpr, thresholds = roc_curve(testing_true_labels[sorted_indices], np.sort(log_odds))
    area_under_curve = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % area_under_curve)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def demo_classifier(data, classifier, classifier_name):
    k_folds = utils.k_fold_split(10, data, seed=34928731, shuffle=True)

    training_accuracy = []
    testing_accuracy = []

    info_table = []

    print("+" * 20, classifier_name, "+" * 20)
    i = 1
    for k_fold_data in k_folds:
        classifier.train(k_fold_data['training']['features'], k_fold_data['training']['labels'])

        training_predictions = classifier.predict(k_fold_data['training']['features'])
        training_prediction_labels = np.argmax(training_predictions, axis=1)

        testing_predictions = classifier.predict(k_fold_data['testing']['features'])
        testing_prediction_labels = np.argmax(testing_predictions, axis=1)

        training_accuracy.append(
            accuracy_score(k_fold_data['training']['labels'], training_prediction_labels))

        testing_true_labels = k_fold_data['testing']['labels']
        testing_accuracy.append(accuracy_score(testing_true_labels, testing_prediction_labels))

        tn, fp, fn, tp = confusion_matrix(testing_true_labels, testing_prediction_labels).ravel()
        fp_rate = fp / testing_true_labels.shape[0]
        fn_rate = fn / testing_true_labels.shape[0]
        error_rate = fp_rate + fn_rate
        info_table.append(["{}".format(i), fp_rate, fn_rate, error_rate])

        if i == 1:
            plot_roc_curve(testing_true_labels, testing_predictions)

        i += 1

    print_info_table(info_table)

    print("Training accuracy: ", np.mean(training_accuracy))
    print("Testing accuracy: ", np.mean(testing_accuracy))
    print("+" * 80)
    print()


def demo_naive_bayes_with_bernoulli_features():
    data = utils.get_spam_data()
    data['features'] = NaiveBayesBernoulli.convert_continuous_features_to_discrete(data['features'])
    demo_classifier(data, NaiveBayesBernoulli(1, data['features'].shape[1], 1), "Naive Bayes Bernoulli")


def demo_naive_bayes_gaussian_features():
    data = utils.get_spam_data()
    demo_classifier(data, NaiveBayesGaussian(), "Naive Bayes Gaussian")


if __name__ == '__main__':
    demo_naive_bayes_with_bernoulli_features()
    demo_naive_bayes_gaussian_features()
