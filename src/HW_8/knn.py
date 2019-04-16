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
    def gaussian(x, all_points, sigma=10):
        x_minus_y = np.square(x - all_points).sum(axis=1)
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

    def __init__(self, kernel, mode=KNNMode.K_POINTS, n_jobs=1, verbose=1) -> None:
        self.n_jobs = n_jobs
        if kernel not in self.kernel_map:
            raise Exception("Invalid Kernel", kernel)
        self.kernel = kernel

        if KNNMode.is_valid(mode):
            raise Exception("Invalid KNN Mode", mode)

        self.mode = mode
        self.verbose = verbose
        self.default_class = None

    def __get_k_closet_points(self, ix, k, distances):
        if self.kernel == Kernel.EUCLIDEAN:
            neighbors = np.argsort(distances)[:k + 1]
        else:
            neighbors = np.argsort(distances)[::-1][:k + 1]

        return np.delete(neighbors, np.where(neighbors == ix))

    def __get_closet_points_in_radius(self, ix, r, distances):
        if self.kernel == Kernel.EUCLIDEAN:
            neighbors = distances <= r
        else:
            neighbors = distances >= r

        neighbors[ix] = False
        return neighbors

    def __get_neighbor_indices(self, ix, k, distances):
        neighbor_indices = None
        if self.mode == KNNMode.K_POINTS:
            neighbor_indices = self.__get_k_closet_points(ix, k, distances)
        elif self.mode == KNNMode.RADIUS:
            neighbor_indices = self.__get_closet_points_in_radius(ix, k, distances)

            if np.all(~neighbor_indices):
                neighbor_indices = self.__get_k_closet_points(ix, 1, distances)

        return neighbor_indices

    def __get_predicted_labels(self, labels, neighbor_indices):
        freq = Counter(labels[neighbor_indices])
        if freq:
            predicted_label = max(freq.items(), key=lambda _tuple: _tuple[1])
            return predicted_label[0]
        else:
            print("Default class assigned")
            return self.default_class

    def __kernel_wrapper(self, args):
        ix, x, features, labels, k_list = args
        distances = self.kernel_map[self.kernel](x, features)

        predicted_labels = []
        for k in k_list:
            neighbor_indices = self.__get_neighbor_indices(ix, k, distances)
            predicted_label = self.__get_predicted_labels(labels, neighbor_indices)
            predicted_labels.append(predicted_label)

        return predicted_labels

    def __set_default_class(self, labels):
        numbers, counts = np.unique(labels, return_counts=True)
        self.default_class = numbers[np.argmax(counts)]

    def predict(self, features, labels, k_list):
        self.__set_default_class(labels)
        arg_list = [(ix, x, features, labels, k_list) for ix, x in enumerate(features)]
        predictions = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=self.verbose)(
            map(delayed(self.__kernel_wrapper), arg_list))
        return np.array(predictions)
