import heapq
from collections import Counter
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed


class SVM:

    def __init__(self, C, tol, max_passes, max_iterations=100, display=True, _id=1) -> None:
        self._id = _id
        self.display = display
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.max_iterations = max_iterations
        self.alpha = None
        self.b = None
        self.features = None
        self.labels = None
        self.non_zero_alpha_indices = None

    def __f_x(self, x):
        features, labels, alpha = self.features[self.non_zero_alpha_indices], \
                                  self.labels[self.non_zero_alpha_indices], \
                                  self.alpha[self.non_zero_alpha_indices]

        dot_product = (np.multiply(features, x).sum(axis=1))
        summation = (alpha * labels * dot_product).sum()
        return summation + self.b

    def __calculate_E(self, x, y):
        return self.__f_x(x) - y

    @staticmethod
    def __select_j_randomly(i, m):
        while True:
            j = np.random.randint(0, m)
            if i != j:
                return j

    @staticmethod
    def __compute_L_H(y_i, alpha_i, y_j, alpha_j, C):

        if y_i != y_j:
            return max(0, alpha_j - alpha_i), min(C, C + alpha_j - alpha_i)
        else:
            return max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j)

    @staticmethod
    def __compute_n(x_i, x_j):
        return 2 * np.dot(x_i, x_j) - np.dot(x_i, x_i) - np.dot(x_j, x_j)

    @staticmethod
    def __compute_and_clip_alpha_j(alpha_j, y_j, E_i, E_j, n, L, H):
        alpha_j = alpha_j - (y_j * ((E_i - E_j) / n))

        if alpha_j > H:
            return H
        elif L <= alpha_j <= H:
            return alpha_j
        else:
            return L

    @staticmethod
    def __compute_alpha_i(alpha_i, y_i, y_j, alpha_j_old, alpha_j):
        return alpha_i + (y_i * y_j * (alpha_j_old - alpha_j))

    @staticmethod
    def __compute_b1_b2(x_i, y_i, E_i, alpha_i, alpha_i_old, x_j, y_j, E_j, alpha_j, alpha_j_old, b):
        x_i_dot = np.dot(x_i, x_i)
        x_j_dot = np.dot(x_j, x_j)
        x_i_x_j_dot = np.dot(x_i, x_j)

        b1 = b - E_i - (y_i * (alpha_i - alpha_i_old) * x_i_dot) - (y_j * (alpha_j - alpha_j_old) * x_i_x_j_dot)
        b2 = b - E_j - (y_i * (alpha_i - alpha_i_old) * x_i_x_j_dot) - (y_j * (alpha_j - alpha_j_old) * x_j_dot)

        return b1, b2

    def __update_non_zero_alpha_list(self):
        self.non_zero_alpha_indices = self.alpha != 0

    def train(self, features, labels):
        m = features.shape[0]

        self.features, self.labels = features, labels
        self.alpha, self.b, self.non_zero_alpha_indices = np.zeros(m), 0, np.zeros(m, dtype=np.bool)

        passes, iterations = 1, 1
        while passes <= self.max_passes and iterations <= self.max_iterations:
            num_alpha_changed = 0
            for i in range(m):
                x_i, y_i, alpha_i = features[i], labels[i], self.alpha[i]

                E_i = self.__calculate_E(x_i, y_i)
                temp = E_i * y_i
                if (temp < -self.tol and alpha_i < self.C) or (temp > self.tol and alpha_i > self.C):
                    j = self.__select_j_randomly(i, m)
                    x_j, y_j, alpha_j = features[j], labels[j], self.alpha[j]
                    E_j = self.__calculate_E(x_j, y_j)

                    alpha_i_old, alpha_j_old = alpha_i, alpha_j

                    L, H = self.__compute_L_H(y_i, alpha_i, y_j, alpha_j, self.C)
                    if L == H:
                        continue

                    n = self.__compute_n(x_i, x_j)
                    if n >= 0:
                        continue

                    alpha_j = self.__compute_and_clip_alpha_j(alpha_j, y_j, E_i, E_j, n, L, H)

                    if np.abs(alpha_j - alpha_j_old) < 1e-05:
                        continue

                    alpha_i = self.__compute_alpha_i(alpha_i, y_i, y_j, alpha_j_old, alpha_j)
                    b1, b2 = self.__compute_b1_b2(x_i, y_i, E_i, alpha_i, alpha_i_old,
                                                  x_j, y_j, E_j, alpha_j, alpha_j_old, self.b)

                    if 0 < alpha_i < self.C:
                        self.b = b1
                    elif 0 < alpha_j < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    self.alpha[i] = alpha_i
                    self.alpha[j] = alpha_j

                    self.__update_non_zero_alpha_list()

                    num_alpha_changed += 1

            if num_alpha_changed == 0:
                passes += 1
            else:
                passes = 0

            if self.display:
                print(
                    "Id=>{}, Iteration=>{}, Passes=>{}, # of Alpha's Changed=>{}".format(self._id, iterations, passes,
                                                                                         num_alpha_changed))

            iterations += 1

    def predict(self, features):
        return np.array([1 if self.__f_x(feature) >= 0 else -1 for feature in features])


class MultiClassSVM:

    def __init__(self, C, tol, max_passes, max_iterations=100, display=True, no_of_jobs=1) -> None:
        self.no_of_jobs = no_of_jobs
        self.display = display
        self.max_iterations = max_iterations
        self.max_passes = max_passes
        self.tol = tol
        self.C = C
        self.classifiers = {}

    def __make_pairs(self, features, labels):
        pairs = combinations(range(np.unique(labels).shape[0]), 2)

        data = []
        for pair in pairs:
            ix_1 = labels == pair[0]
            ix_2 = labels == pair[1]
            f_1, f_2 = features[ix_1], features[ix_2]

            f = np.append(f_1, f_2, axis=0)
            l = np.append(np.ones(f_1.shape[0], dtype=np.int) * -1, np.ones(f_2.shape[0], dtype=np.int) * 1)

            if f.shape[0] != l.shape[0]:
                raise Exception("Feature and Label shape mismatch: F=>{}, L=>{}".format(f.shape, l.shape))

            temp = {
                'features': f,
                'labels': l,
                'l1': pair[0],
                'l2': pair[1]
            }

            data.append(temp)

        return data

    @staticmethod
    def __train_one_vs_one_classifier_wrapper(args):
        features, labels, l1, l2, C, tol, max_passes, max_iterations, display, _id = args
        return MultiClassSVM.__train_one_vs_one_classifier(features, labels, l1, l2,
                                                           C, tol, max_passes, max_iterations, display, _id)

    @staticmethod
    def __train_one_vs_one_classifier(features, labels, l1, l2, C, tol, max_passes, max_iterations, display, _id):
        classifier = SVM(C, tol, max_passes, max_iterations, display, _id)
        classifier.train(features, labels)
        return classifier, l1, l2

    def train(self, features, labels):
        pairs = self.__make_pairs(features, labels)

        arg_list = []
        for ix, pair in enumerate(pairs):
            arg = pair['features'], pair['labels'], pair['l1'], pair['l2'], \
                  self.C, self.tol, self.max_passes, self.max_iterations, self.display, ix

            arg_list.append(arg)

        classifiers = Parallel(n_jobs=self.no_of_jobs, backend="threading", verbose=49)(
            map(delayed(MultiClassSVM.__train_one_vs_one_classifier_wrapper), arg_list)
        )

        for classifier, l1, l2 in classifiers:
            self.classifiers[(l1, l2)] = classifier

    @staticmethod
    def __predict_wrapper(args):
        l1, l2, classifier, features = args
        predictions = classifier.predict(features)
        predictions[predictions == 1] = l2
        predictions[predictions == -1] = l1
        return predictions, l1, l2

    @staticmethod
    def __get_key(l1, l2):
        return min(l1, l2), max(l1, l2)

    def predict(self, features):

        arg_list = []
        for k, classifier in self.classifiers.items():
            arg_list.append((k[0], k[1], classifier, features))

        results = Parallel(n_jobs=self.no_of_jobs, backend="threading", verbose=49)(
            map(delayed(MultiClassSVM.__predict_wrapper), arg_list)
        )

        result_mapping = {}
        all_predicted_labels = []
        for result in results:
            predictions, l1, l2 = result
            result_mapping[self.__get_key(l1, l2)] = predictions
            all_predicted_labels.append(predictions)

        all_predicted_labels = np.array(all_predicted_labels)

        no_of_data_points = features.shape[0]
        final_predictions = np.ones(no_of_data_points, dtype=np.int) * -1
        for ix in range(no_of_data_points):
            current_prediction = all_predicted_labels[:, ix]

            freq = [(-count, label) for label, count in Counter(current_prediction).items()]
            heapq.heapify(freq)

            c1, l1 = heapq.heappop(freq)
            c2, l2 = heapq.heappop(freq)

            if c1 == c2:
                final_predictions[ix] = result_mapping[self.__get_key(l1, l2)][ix]
            else:
                final_predictions[ix] = l1

        return final_predictions
