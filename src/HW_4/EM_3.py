import numpy as np
from scipy.stats import multivariate_normal

from HW_4 import utils


class EM:

    def __init__(self, seed, no_of_gaussian, gaussian_dimensions, cov_diagonal_multiplier) -> None:
        super().__init__()
        self.seed = seed
        self.no_of_gaussian = no_of_gaussian
        self.gaussian_dimensions = gaussian_dimensions
        self.mean = None
        self.cov = None
        self.mixture_coefficient = None
        self.z_i_m = None
        self.cov_diagonal_multiplier = cov_diagonal_multiplier
        np.random.seed(seed)

    def e_step(self, data):
        p0 = multivariate_normal.pdf(data, mean=self.mean[0], cov=self.cov[0])
        p1 = multivariate_normal.pdf(data, mean=self.mean[1], cov=self.cov[1])
        p2 = multivariate_normal.pdf(data, mean=self.mean[2], cov=self.cov[2])
        probs = np.array([p0, p1, p2])
        z_i_m = (probs * self.mixture_coefficient) / np.sum(probs * self.mixture_coefficient, axis=0)
        self.z_i_m = z_i_m

    def m_step(self, data):
        mean0 = np.sum(self.z_i_m[0] * data.T, axis=1) / np.sum(self.z_i_m[0])
        mean1 = np.sum(self.z_i_m[1] * data.T, axis=1) / np.sum(self.z_i_m[1])
        mean2 = np.sum(self.z_i_m[2] * data.T, axis=1) / np.sum(self.z_i_m[2])
        p0, p1, p2 = np.sum(self.z_i_m, axis=1) / data.shape[0]
        sigma0 = np.dot((self.z_i_m[0] * (data - mean0).T), (data - mean0)) / np.sum(self.z_i_m[0])
        sigma1 = np.dot((self.z_i_m[1] * (data - mean1).T), (data - mean1)) / np.sum(self.z_i_m[1])
        sigma2 = np.dot((self.z_i_m[2] * (data - mean1).T), (data - mean2)) / np.sum(self.z_i_m[2])

        self.mean = np.array([mean0, mean1, mean2])
        self.cov = np.array([sigma0, sigma1, sigma2])
        self.mixture_coefficient = np.array([[p0], [p1], [p2]])

    def train(self, data, epochs, display_step):
        self.z_i_m = np.random.random((self.no_of_gaussian, data.shape[0]))
        self.z_i_m = self.z_i_m / np.sum(self.z_i_m, axis=0)

        for i in range(epochs):
            self.m_step(data)
            self.e_step(data)

            # if i % display_step == 0 or i == 0:
            #     self.plot_gaussian(data)

    def plot_gaussian(self, data):
        print("Mean", self.mean)
        print("Cov", self.cov)
        print("Pie", self.mixture_coefficient)


def demo_em_on_mixture_of_two_gaussian_data():
    data = utils.get_mixture_of_three_gaussian_data()
    em = EM(42, 3, 2, 0.01)
    em.train(data, 100, 1000)
    print("Mean", em.mean)
    print("Sigma", em.cov)
    print("Mixture Coefficients", em.mixture_coefficient)


if __name__ == '__main__':
    demo_em_on_mixture_of_two_gaussian_data()
