from collections import Counter

import numpy as np
from joblib import Parallel, delayed


class KNN:
    def __init__(self, kernel, training_features, training_label, n_jobs=1) -> None:
        self.n_jobs = n_jobs
        self.kernel = kernel
        self.training_features = training_features
        self.training_label = training_label

    def __kernel_wrapper(self, args):
        x, k = args
        distances = self.kernel(x, self.training_features)
        k_closet_points = np.argsort(distances)[:k]
        freq = Counter(self.training_label[k_closet_points])
        predicted_label = max(freq.items(), key=lambda _tuple: _tuple[1])
        return predicted_label

    def predict(self, features, k):
        arg_list = [(x, k) for x in features]
        predictions = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=49)(map(delayed(self.kernel), arg_list))
        return np.array(predictions)
