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

    @staticmethod
    def __select_j_randomly(i, m):
        while True:
            j = np.random.randint(0, m)
            if i != j:
                return j

    def __calculate_E(self, x, y):
        return self.__f_x(x) - y

    def __compute_L_H(self, y_i, alpha_i, y_j, alpha_j):

        if y_i != y_j:
            return np.max(0, alpha_j - alpha_i), np.min(self.C, self.C + alpha_j - alpha_i)
        else:
            return np.max(0, alpha_i + alpha_j - self.C), np.min(self.C, alpha_i + alpha_j)

    def __compute_n(self, x_i, x_j):
        return 2 * np.dot(x_i, x_j) - np.dot(x_i, x_i) - np.dot(x_j, x_j)


    def train(self, features, labels):
        m = features.shape[0]

        self.features, self.labels = features, labels
        self.alpha, self.b = np.zeros(m), 0

        passes = 1
        while passes <= self.max_passes:
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
                    L, H = self.__compute_L_H(y_i, alpha_i, y_j, alpha_j)

                    if L == H:
                        continue

                    n = self.__compute_n()

            if self.display and (passes == 1 or passes % self.max_passes == 0):
                print("Step=>{}".format(passes))

            passes += 1

    def predict(self, features):
        pass
