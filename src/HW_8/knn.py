from collections import Counter
from enum import Enum

import numpy as np
from joblib import Parallel, delayed


class Kernel(Enum):
    EUCLIDEAN = 1
    COSINE = 2
    GAUSSIAN = 3
    POLYNOMIAL = 4


class SimilarityMeasures:

    @staticmethod
    def euclidean(x, points):
        return np.sqrt(np.square(x - points).sum(axis=1))

    @staticmethod
    def cosine(x, all_points):
        a_mag = np.linalg.norm(x)
        b_mag = np.linalg.norm(all_points, axis=1)
        dot_product = np.multiply(x, all_points).sum(axis=1)
        return dot_product / (a_mag * b_mag)

    @staticmethod
    def gaussian(x, all_points, sigma=2):
        x_minus_y = np.square(np.linalg.norm(x - all_points, axis=1))
        return np.exp((- x_minus_y / (2 * sigma * sigma)))

    @staticmethod
    def polynomial(x, all_points, degree=2):
        dot_product = np.multiply(x, all_points).sum(axis=1)
        return np.power((dot_product + 1), degree)


class KNNMode(Enum):
    K_POINTS = 1
    RADIUS = 2

    @classmethod
    def is_valid(cls, mode):
        return mode in KNNMode.__members__

    def get_str(self):
        if self is KNNMode.K_POINTS:
            return 'K'
        elif self is KNNMode.RADIUS:
            return 'R'
        else:
            raise Exception("Unreachable Code")


class KNN:
    kernel_map = {
        Kernel.EUCLIDEAN: SimilarityMeasures.euclidean,
        Kernel.COSINE: SimilarityMeasures.cosine,
        Kernel.GAUSSIAN: SimilarityMeasures.gaussian,
        Kernel.POLYNOMIAL: SimilarityMeasures.polynomial
    }

    def __init__(self, kernel, training_features, training_label, mode=KNNMode.K_POINTS, n_jobs=1, verbose=1) -> None:
        self.n_jobs = n_jobs
        if kernel not in self.kernel_map:
            raise Exception("Invalid Kernel", kernel)
        self.kernel = kernel

        if KNNMode.is_valid(mode):
            raise Exception("Invalid KNN Mode", mode)

        self.mode = mode
        self.training_features = training_features
        self.training_label = training_label
        self.verbose = verbose

    def __get_k_closet_points(self, k, distances):
        if self.kernel == Kernel.EUCLIDEAN:
            return np.argsort(distances)[:k]
        else:
            return np.argsort(distances)[::-1][:k]

    def __get_closet_points_in_radius(self, r, distances):
        if self.kernel == Kernel.EUCLIDEAN:
            return distances <= r
        else:
            return distances >= r

    def __get_predicted_label(self, k, distances):
        neighbor_indices = None
        if self.mode == KNNMode.K_POINTS:
            neighbor_indices = self.__get_k_closet_points(k, distances)
        elif self.mode == KNNMode.RADIUS:
            neighbor_indices = self.__get_closet_points_in_radius(k, distances)

        freq = Counter(self.training_label[neighbor_indices])
        predicted_label = max(freq.items(), key=lambda _tuple: _tuple[1])[0]
        return predicted_label

    def __kernel_wrapper(self, args):
        x, k_list = args
        distances = self.kernel_map[self.kernel](x, self.training_features)

        predicted_labels = []
        for k in k_list:
            predicted_labels.append(self.__get_predicted_label(k, distances))

        return predicted_labels

    def predict(self, features, k_list):
        arg_list = [(x, k_list) for x in features]
        predictions = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=self.verbose)(
            map(delayed(self.__kernel_wrapper), arg_list))
        return np.array(predictions)
