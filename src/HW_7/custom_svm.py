import numpy as np


class SVM:

    def __init__(self, C, tol, max_iteration, display=True, display_step=1) -> None:
        self.display_step = display_step
        self.display = display
        self.C = C
        self.tol = tol
        self.max_passes = max_iteration
        self.alpha = None
        self.b = None
        self.features = None
        self.labels = None

    def __f_x(self, x):
        dot_product = (np.multiply(self.features, x).sum(axis=1))
        summation = (self.alpha * self.labels * dot_product).sum()
        return summation + self.b

    def train(self, features, labels):
        m = features.shape[0]

        self.features, self.labels = features, labels
        self.alpha, self.b = np.zeros(m), 0

        passes = 1
        while passes <= self.max_passes:
            num_alpha_changed = 0
            for i in range(m):
                E_i =

            if self.display and (passes == 1 or passes % self.max_passes == 0):
                print("Step=>{}".format(passes))

            passes += 1

    def predict(self, features):
        pass
