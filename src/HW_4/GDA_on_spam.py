from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score

from HW_4 import utils


class GDA:

    def __init__(self, sigma_diagonal_multiplier=1.0) -> None:
        super().__init__()
        self.non_spam_mean = None
        self.spam_mean = None
        self.sigma = None
        self.d = None
        self.p_non_spam = None
        self.p_spam = None
        self.sigma_diagonal_multiplier = sigma_diagonal_multiplier

    def train(self, features, labels):
        counter = Counter(labels)
        self.p_non_spam = counter[0] / labels.shape[0]
        self.p_spam = counter[1] / labels.shape[0]

        self.d = features.shape[1]
        non_spam_features = features[labels == 0]
        spam_features = features[labels == 1]

        self.non_spam_mean = non_spam_features.mean(axis=0)
        self.spam_mean = spam_features.mean(axis=0)

        mean = np.array([self.non_spam_mean, self.spam_mean])

        temp = features - mean[labels]
        sigma = np.matmul(np.transpose(temp), temp) / features.shape[0]
        di = np.diag_indices(self.d)
        sigma[di] = sigma[di] * self.sigma_diagonal_multiplier
        self.sigma = sigma

    def vectorized_predict(self, features):
        common = 1 / (np.sqrt(np.power(2 * np.pi, self.d) * np.linalg.det(self.sigma)))
        sigma_inverse = np.linalg.pinv(self.sigma)

        x_minus_non_spam_mean = features - self.non_spam_mean
        p_x_non_spam = common * np.exp(
            -0.5 * (np.matmul(x_minus_non_spam_mean, sigma_inverse) * x_minus_non_spam_mean).sum(axis=1))
        p_predict_non_spam = p_x_non_spam * self.p_non_spam

        x_minus_spam_mean = features - self.spam_mean
        p_x_spam = common * np.exp(
            -0.5 * (np.matmul(x_minus_spam_mean, sigma_inverse) * x_minus_spam_mean).sum(axis=1))
        p_predict_spam = p_x_spam * self.p_spam

        return np.argmax(np.column_stack((p_predict_non_spam, p_predict_spam)), axis=1)

    def predict(self, features):
        common = 1 / (np.sqrt(np.power(2 * np.pi, self.d) * np.linalg.det(self.sigma)))
        sigma_inverse = np.linalg.pinv(self.sigma)

        predicted_labels = []
        for feature in features:
            x_minus_non_spam_mean = feature - self.non_spam_mean
            p_x_non_spam = common * np.exp(
                -0.5 * np.matmul(np.matmul(x_minus_non_spam_mean, sigma_inverse), np.transpose(x_minus_non_spam_mean)))
            p_predict_non_spam = p_x_non_spam * self.p_non_spam

            x_minus_spam_mean = feature - self.spam_mean
            p_x_spam = common * np.exp(
                -0.5 * np.matmul(np.matmul(x_minus_spam_mean, sigma_inverse), np.transpose(x_minus_spam_mean)))
            p_predict_spam = p_x_spam * self.p_spam

            predicted_labels.append(np.argmax([p_predict_non_spam, p_predict_spam]))

        return predicted_labels


def demo_gda_on_spam_data():
    data = utils.get_spam_data()
    k_folds = utils.k_fold_split(10, data, seed=1, shuffle=True)

    training_accuracy = []
    testing_accuracy = []
    for k_fold_data in k_folds:
        gda = GDA(1)
        gda.train(k_fold_data['training']['features'], k_fold_data['training']['labels'])
        training_predicted_labels = gda.vectorized_predict(k_fold_data['training']['features'])
        testing_predicted_labels = gda.vectorized_predict(k_fold_data['testing']['features'])

        training_accuracy.append(accuracy_score(k_fold_data['training']['labels'], training_predicted_labels))
        testing_accuracy.append(accuracy_score(k_fold_data['testing']['labels'], testing_predicted_labels))

    print("Training accuracy: ", np.mean(training_accuracy))
    print("Testing accuracy: ", np.mean(testing_accuracy))


if __name__ == '__main__':
    demo_gda_on_spam_data()
