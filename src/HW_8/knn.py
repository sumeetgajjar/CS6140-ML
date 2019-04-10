from collections import Counter

import numpy as np
from joblib import Parallel, delayed


class KNN:
    def __init__(self, kernel, training_features, training_label, n_jobs=1) -> None:
        self.n_jobs = n_jobs
        self.kernel = kernel
        self.training_features = training_features
        self.training_label = training_label

    def __get_predicted_label(self, k, distances):
        k_closet_points = np.argsort(distances)[:k]
        freq = Counter(self.training_label[k_closet_points])
        predicted_label = max(freq.items(), key=lambda _tuple: _tuple[1])
        return predicted_label

    def __kernel_wrapper(self, args):
        x, k = args
        distances = self.kernel(x, self.training_features)
        return self.__get_predicted_label(k, distances)

    def predict(self, features, k):
        arg_list = [(x, k) for x in features]
        predictions = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=49)(map(delayed(self.kernel), arg_list))
        return np.array(predictions)


class Kernel:

    @staticmethod
    def cosine_kernel(x, all_points):
        a_mag = np.linalg.norm(x)
        b_mag = np.linalg.norm(all_points, axis=1)
        dot_product = np.multiply(x, all_points).sum(axis=1)
        return 1 - np.cos(dot_product / (a_mag * b_mag))

    @staticmethod
    def gaussian_kernel(x, all_points, sigma):
        x_minus_y = np.square(np.linalg.norm(x - all_points, axis=1))
        return np.exp((- x_minus_y / (2 * sigma * sigma)))

    @staticmethod
    def polynomial_kernel(x, all_points, degree):
        dot_product = np.multiply(x, all_points).sum(axis=1)
        return np.power((dot_product + 1), degree)
