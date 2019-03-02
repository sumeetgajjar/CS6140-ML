import numpy as np
from sklearn.metrics import accuracy_score

from HW_4 import utils


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
            p_non_spam *= np.product(self.one_non_spam_prob[f == 1])

            p_spam = np.product(self.zero_spam_prob[f == 0])
            p_spam *= np.product(self.one_spam_prob[f == 1])

            predicted_non_spam_probs.append(p_non_spam)
            predicted_spam_probs.append(p_spam)

        return np.argmax(np.column_stack((predicted_non_spam_probs, predicted_spam_probs)), axis=1)

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


def demo_naive_bayes_with_bernoulli_features():
    data = utils.get_spam_data()
    data['features'] = NaiveBayesBernoulli.convert_continuous_features_to_discrete(data['features'])

    k_folds = utils.k_fold_split(10, data, seed=1, shuffle=True)

    training_accuracy = []
    testing_accuracy = []
    for k_fold_data in k_folds:
        naive_bayes = NaiveBayesBernoulli(1, data['features'].shape[1], 1)
        naive_bayes.train(k_fold_data['training']['features'], k_fold_data['training']['labels'])

        training_predicted_labels = naive_bayes.predict(k_fold_data['training']['features'])
        testing_predicted_labels = naive_bayes.predict(k_fold_data['testing']['features'])

        training_accuracy.append(accuracy_score(k_fold_data['training']['labels'], training_predicted_labels))
        testing_accuracy.append(accuracy_score(k_fold_data['testing']['labels'], testing_predicted_labels))

    print("Training accuracy: ", np.mean(training_accuracy))
    print("Testing accuracy: ", np.mean(testing_accuracy))


if __name__ == '__main__':
    demo_naive_bayes_with_bernoulli_features()
